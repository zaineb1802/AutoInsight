"""
Validator Agent
===============
Validates the dataframe before cleaning and applies pre-cleaning fixes:

  1. Target column viability check
  2. Numeric-as-string detection + coercion
  3. ID-like column detection + drop
     (monotonically increasing integers/strings with full cardinality)
  4. High-cardinality string column detection + drop
     (object columns where unique ratio > threshold AND not useful as-is)
  5. All-null column detection + drop
  6. Duplicate column name detection
  7. LLM severity assessment

High-cardinality string handling
---------------------------------
Columns like Name, Ticket, Description etc. have near-unique values that
make label-encoding meaningless and inject noise into tree-based models.
Detection: object dtype + (nunique / nrows) > HIGH_CARD_RATIO_THRESHOLD.
Action   : drop by default UNLESS a column pattern suggests it contains
           extractable structure (e.g. name titles, date strings, codes).
           Those are flagged for the Feature agent to handle instead of
           being silently dropped.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from automl.state import AutoMLState

logger = logging.getLogger("autoinsight.agents.validator")

# Columns whose unique-value ratio exceeds this are considered high-cardinality
HIGH_CARD_RATIO_THRESHOLD = 0.5   # > 50 % unique values relative to row count
# Minimum rows before high-cardinality rule kicks in (small datasets are exempt)
HIGH_CARD_MIN_ROWS = 50

# Regex patterns that suggest a column has extractable structure worth keeping
_STRUCTURED_PATTERNS = [
    re.compile(r"(date|time|year|month|day|timestamp)", re.I),
    re.compile(r"(code|id|zip|postal|sku|isbn)", re.I),
]

SYSTEM_PROMPT = """You are a data validation expert.
Given a validation report, output a JSON object:
{
  "severity": "ok" | "warning" | "error",
  "action_required": ["list of specific actions to take"],
  "safe_to_proceed": true | false
}
Respond ONLY with valid JSON."""


class ValidatorAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("ValidatorAgent starting ...")
        df = state["dataframe"]
        target = state.get("target_column", "")
        task_type = state.get("task_type", "regression")

        issues: list[str] = []
        fixes: dict[str, str] = {}

        # ── 1. Target viability ──────────────────────────────────────────
        if target not in df.columns:
            issues.append(f"CRITICAL: Target column '{target}' not found in dataframe.")
        else:
            null_pct = df[target].isnull().mean()
            if null_pct > 0.5:
                issues.append(
                    f"Target '{target}' has {null_pct:.1%} missing — too many to model reliably."
                )
            if task_type == "classification" and df[target].nunique() > 50:
                issues.append(
                    f"Target '{target}' has {df[target].nunique()} unique values "
                    f"but task is 'classification' — consider regression."
                )

        # ── 2. Numeric-as-string columns ─────────────────────────────────
        numeric_as_str: list[str] = []
        for col in df.select_dtypes(include="object").columns:
            if col == target:
                continue
            sample = df[col].dropna().head(200)
            if len(sample) == 0:
                continue
            converted = pd.to_numeric(sample, errors="coerce")
            if converted.notna().mean() > 0.9:
                numeric_as_str.append(col)
                fixes[col] = "convert_to_numeric"

        if numeric_as_str:
            issues.append(f"Columns appear numeric but stored as string: {numeric_as_str}")

        # ── 3. ID-like columns (monotonic + near-fully unique) ───────────
        id_cols: list[str] = []
        for col in df.columns:
            if col == target or col in fixes:
                continue
            try:
                unique_ratio = df[col].nunique() / max(len(df), 1)
                is_full_unique = unique_ratio > 0.98
                is_monotonic = (
                    pd.to_numeric(df[col], errors="coerce").dropna().is_monotonic_increasing
                )
                if is_full_unique and is_monotonic:
                    id_cols.append(col)
                    fixes[col] = "drop_id"
            except Exception:
                pass

        if id_cols:
            issues.append(f"Likely ID columns (dropped): {id_cols}")

        # ── 4. High-cardinality string columns ───────────────────────────
        high_card_drop: list[str] = []
        high_card_keep: list[str] = []  # flagged but kept for feature agent

        if len(df) >= HIGH_CARD_MIN_ROWS:
            for col in df.select_dtypes(include="object").columns:
                if col == target or col in fixes:
                    continue
                unique_ratio = df[col].nunique() / max(len(df), 1)
                if unique_ratio > HIGH_CARD_RATIO_THRESHOLD:
                    # Check if the column name suggests extractable structure
                    has_structure = any(p.search(col) for p in _STRUCTURED_PATTERNS)
                    if has_structure:
                        high_card_keep.append(col)
                    else:
                        high_card_drop.append(col)
                        fixes[col] = "drop_high_cardinality"

        if high_card_drop:
            issues.append(
                f"High-cardinality string columns with no extractable structure "
                f"(dropped): {high_card_drop}"
            )
        if high_card_keep:
            issues.append(
                f"High-cardinality string columns with potential structure "
                f"(kept for feature engineering): {high_card_keep}"
            )

        # ── 5. All-null columns ──────────────────────────────────────────
        all_null = [col for col in df.columns if df[col].isnull().all()]
        if all_null:
            issues.append(f"Fully null columns (dropped): {all_null}")
            for col in all_null:
                fixes[col] = "drop_null_col"

        # ── 6. Duplicate column names ────────────────────────────────────
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            issues.append(f"Duplicate column names detected: {dup_cols}")

        # ── LLM severity assessment ──────────────────────────────────────
        llm_assessment = self._llm_assess(issues)

        validation_report = {
            "issues": issues,
            "fixes": fixes,
            "numeric_as_str": numeric_as_str,
            "id_cols": id_cols,
            "high_card_dropped": high_card_drop,
            "high_card_kept": high_card_keep,
            "all_null_cols": all_null,
            "llm_assessment": llm_assessment,
        }

        logger.info(
            "Validation complete — %d issues | severity: %s | dropping: %s",
            len(issues),
            llm_assessment.get("severity", "unknown"),
            [c for c, a in fixes.items() if "drop" in a],
        )

        df_fixed = self._apply_fixes(df.copy(), fixes)
        return {**state, "dataframe": df_fixed, "validation_report": validation_report}

    # ------------------------------------------------------------------

    def _llm_assess(self, issues: list[str]) -> dict:
        import json

        if not issues:
            return {"severity": "ok", "action_required": [], "safe_to_proceed": True}

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content="Validation issues found:\n" + "\n".join(f"- {i}" for i in issues)
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("LLM validation assessment failed: %s", exc)
            return {"severity": "warning", "action_required": issues, "safe_to_proceed": True}

    def _apply_fixes(self, df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
        cols_to_drop = []
        for col, action in fixes.items():
            if col not in df.columns:
                continue
            if action == "convert_to_numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.debug("Converted '%s' to numeric.", col)
            elif action in ("drop_id", "drop_null_col", "drop_high_cardinality"):
                cols_to_drop.append(col)

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
            logger.info("Dropped columns: %s", cols_to_drop)

        return df