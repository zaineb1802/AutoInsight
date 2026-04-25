"""
Strategy Agent
==============
Uses the LLM to create a structured plan for:
  - Data cleaning steps
  - Feature engineering steps
based on the EDA summary and the user's goal.

This version hardens plan quality with:
  - strict post-LLM schema normalization
  - deduplication / pruning of noisy or redundant steps
  - dataset-size-aware efficiency guards
  - target leakage prevention (target never appears in planned feature/cleaning columns)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from automl.state import AutoMLState

logger = logging.getLogger("autoinsight.agents.strategy")

SYSTEM_PROMPT = """You are an expert ML engineer. Based on the EDA findings, 
design a robust and efficient data preparation strategy for tabular ML.

Prefer precision and minimalism: include only high-value steps.

Efficiency rules:
- Small/medium datasets: allow richer features if strongly justified.
- Large datasets (>= 20k rows): avoid expensive transformations unless critical.
- Never add polynomial_features and interaction_features together unless <= 20 numeric columns.
- Never propose more than 2 heavy feature steps total.

Safety rules:
- Never include the target column in any step, unless cleaning the missing rows.
- Only use columns that exist in the dataset.
- If no useful step is needed, return empty lists.

Return ONLY a JSON object with this exact structure:
{
  "cleaning_steps": [
    {"step": "handle_missing", "columns": ["col1"], "method": "median"},
    {"step": "remove_outliers", "columns": ["col2"], "method": "iqr"},
    {"step": "drop_constants", "columns": []},
    {"step": "fix_dtypes", "columns": ["col3"], "target_dtype": "numeric"}
  ],
  "feature_steps": [
    {"step": "encode_categorical", "columns": ["cat_col"], "method": "label"},
    {"step": "polynomial_features", "columns": ["num_col"], "degree": 2},
    {"step": "interaction_features", "columns": ["col_a", "col_b"]},
    {"step": "log_transform", "columns": ["skewed_col"]}
  ],
  "rationale": "Brief explanation of choices"
}

Valid cleaning methods: median, mean, mode, drop_rows, drop_cols, iqr, zscore, clip
Valid feature methods: label, onehot, ordinal
Only include steps that are genuinely needed. Respond with valid JSON only."""

VALID_CLEANING_STEPS = {"handle_missing", "remove_outliers", "drop_constants", "fix_dtypes", "drop_duplicates"}
VALID_FEATURE_STEPS = {
    "encode_categorical",
    "polynomial_features",
    "interaction_features",
    "log_transform",
    "datetime_decompose",
}
VALID_CLEANING_METHODS = {"median", "mean", "mode", "drop_rows", "drop_cols", "iqr", "zscore", "clip"}
VALID_ENCODING_METHODS = {"label", "onehot", "ordinal"}


class StrategyAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("StrategyAgent starting ...")
        backend, model = self._llm_identity()
        logger.info("StrategyAgent LLM in use: %s / %s", backend, model)

        eda_summary = state.get("eda_summary", {})
        profile = eda_summary.get("profile", {})
        insights = eda_summary.get("llm_insights", {})
        goal = state["goal"]
        task_type = state.get("task_type", "unknown")
        target = state.get("target_column", "")
        metric = state.get("metric", "")
        shape = profile.get("shape", [0, 0])
        n_rows = int(shape[0]) if shape and len(shape) > 0 and shape[0] is not None else 0

        prompt_body = (
            f"Goal: {goal}\n"
            f"Task type: {task_type}\n"
            f"Primary metric: {metric}\n"
            f"Target column: {target}\n\n"
            f"EDA findings:\n"
            f"  Shape: {profile.get('shape')}\n"
            f"  Missing columns: {[k for k, v in profile.get('missing_counts', {}).items() if v > 0]}\n"
            f"  High cardinality: {profile.get('high_cardinality', [])}\n"
            f"  Constant cols: {profile.get('constant_cols', [])}\n"
            f"  Skewed cols: {profile.get('skewed_cols', [])}\n"
            f"  Dtype map: {profile.get('dtypes', {})}\n"
            f"  Key findings: {insights.get('key_findings', [])}\n"
            f"  Quality issues: {insights.get('quality_issues', [])}\n"
            "Generate a minimal, high-impact plan only.\n"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_body),
        ]

        try:
            print(f"[Strategy] LLM active: backend={backend}, model={model}")
            response = self.llm.invoke(messages)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            strategy = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM strategy planning failed (%s). Using default strategy.", exc)
            strategy = self._default_strategy(profile, target, task_type, n_rows)

        strategy = self._normalize_strategy(
            strategy=strategy,
            profile=profile,
            target=target,
            task_type=task_type,
            n_rows=n_rows,
        )

        logger.info(
            "Strategy planned — %d cleaning steps, %d feature steps",
            len(strategy.get("cleaning_steps", [])),
            len(strategy.get("feature_steps", [])),
        )

        return {**state, "strategy": strategy}

    # ------------------------------------------------------------------

    def _default_strategy(self, profile: dict, target: str, task_type: str, n_rows: int) -> dict:
        """Minimal safe strategy when LLM fails."""
        missing_cols = [
            col for col, cnt in profile.get("missing_counts", {}).items()
            if cnt > 0 and col != target
        ]
        constant_cols = [c for c in profile.get("constant_cols", []) if c != target]
        high_card = [c for c in profile.get("high_cardinality", []) if c != target]
        skewed = [c for c in profile.get("skewed_cols", []) if c != target]

        cleaning_steps = []
        if constant_cols:
            cleaning_steps.append({"step": "drop_constants", "columns": constant_cols})
        if missing_cols:
            cleaning_steps.append(
                {"step": "handle_missing", "columns": missing_cols, "method": "median"}
            )
        if profile.get("duplicate_rows", 0) > 0:
            cleaning_steps.append({"step": "drop_duplicates", "columns": []})

        feature_steps = []
        if high_card:
            feature_steps.append(
                {"step": "encode_categorical", "columns": high_card, "method": "label"}
            )
        if skewed and n_rows <= 50000:
            feature_steps.append({"step": "log_transform", "columns": skewed})

        return {
            "cleaning_steps": cleaning_steps,
            "feature_steps": feature_steps,
            "rationale": (
                "Default conservative strategy for tabular "
                f"{task_type or 'ml'} tasks."
            ),
        }

    def _normalize_strategy(
        self,
        strategy: dict,
        profile: dict,
        target: str,
        task_type: str,
        n_rows: int,
    ) -> dict:
        """
        Validate and optimize an LLM-generated strategy for precision + efficiency.
        """
        all_cols = set(profile.get("dtypes", {}).keys())
        if not all_cols:
            # Fallback when dtype map is unavailable
            all_cols = set((profile.get("missing_counts") or {}).keys())

        cleaning_in = strategy.get("cleaning_steps", []) if isinstance(strategy, dict) else []
        feature_in = strategy.get("feature_steps", []) if isinstance(strategy, dict) else []

        cleaning_steps: list[dict] = []
        feature_steps: list[dict] = []

        # --- Cleaning normalization ---
        seen_cleaning = set()
        for step_cfg in cleaning_in:
            if not isinstance(step_cfg, dict):
                continue
            step = step_cfg.get("step", "")
            if step not in VALID_CLEANING_STEPS:
                continue

            cols = self._sanitize_columns(step_cfg.get("columns", []), all_cols, target)
            key = (step, tuple(cols), step_cfg.get("method"), step_cfg.get("target_dtype"))
            if key in seen_cleaning:
                continue
            seen_cleaning.add(key)

            normalized = {"step": step, "columns": cols}
            if step == "handle_missing":
                method = step_cfg.get("method", "median")
                normalized["method"] = method if method in VALID_CLEANING_METHODS else "median"
            elif step == "remove_outliers":
                method = step_cfg.get("method", "iqr")
                normalized["method"] = method if method in {"iqr", "zscore", "clip"} else "iqr"
            elif step == "fix_dtypes":
                target_dtype = step_cfg.get("target_dtype", "numeric")
                normalized["target_dtype"] = (
                    target_dtype if target_dtype in {"numeric", "datetime", "string"} else "numeric"
                )

            # Keep step only when meaningful.
            if step in {"drop_duplicates"} or cols:
                cleaning_steps.append(normalized)

        # --- Feature normalization ---
        seen_feature = set()
        heavy_budget = 2
        numeric_cols = set(profile.get("numeric_cols", []))
        cat_cols = set(profile.get("categorical_cols", []))
        skewed_cols = set(profile.get("skewed_cols", []))
        high_card = set(profile.get("high_cardinality", []))

        for step_cfg in feature_in:
            if not isinstance(step_cfg, dict):
                continue
            step = step_cfg.get("step", "")
            if step not in VALID_FEATURE_STEPS:
                continue

            cols = self._sanitize_columns(step_cfg.get("columns", []), all_cols, target)

            # Type-aware pruning
            if step in {"log_transform", "polynomial_features", "interaction_features"}:
                cols = [c for c in cols if c in numeric_cols]
            if step == "encode_categorical":
                cols = [c for c in cols if c in cat_cols]
            if step == "datetime_decompose":
                cols = [c for c in cols if c in all_cols]

            # Auto-scope useful columns when LLM omits them
            if step == "encode_categorical" and not cols:
                cols = [c for c in sorted(cat_cols & (all_cols - {target}))][:20]
            elif step == "log_transform" and not cols:
                cols = [c for c in sorted(skewed_cols & numeric_cols)][:20]

            # Heavy-step budget
            is_heavy = step in {"polynomial_features", "interaction_features"}
            if is_heavy:
                if heavy_budget <= 0:
                    continue
                # On large data, avoid expensive synthetic expansion
                if n_rows >= 20000:
                    continue
                heavy_budget -= 1

            key = (step, tuple(cols), step_cfg.get("method"), step_cfg.get("degree"))
            if key in seen_feature:
                continue
            seen_feature.add(key)

            normalized = {"step": step, "columns": cols}
            if step == "encode_categorical":
                method = step_cfg.get("method", "label")
                # High-cardinality categories are generally safer with label for efficiency.
                if any(c in high_card for c in cols):
                    method = "label"
                normalized["method"] = method if method in VALID_ENCODING_METHODS else "label"
            elif step == "polynomial_features":
                degree = int(step_cfg.get("degree", 2))
                normalized["degree"] = 2 if degree >= 2 else 2

            if cols:
                feature_steps.append(normalized)

        # Keep only one of polynomial or interaction when many numeric columns.
        if len(numeric_cols) > 20:
            feature_steps = self._drop_one_heavy_feature(feature_steps)

        # Add high-value deterministic steps if missing
        if profile.get("duplicate_rows", 0) > 0 and not any(s["step"] == "drop_duplicates" for s in cleaning_steps):
            cleaning_steps.insert(0, {"step": "drop_duplicates", "columns": []})

        if (profile.get("missing_counts") or {}) and not any(s["step"] == "handle_missing" for s in cleaning_steps):
            miss_cols = [
                c for c, cnt in (profile.get("missing_counts") or {}).items()
                if cnt > 0 and c != target and c in all_cols
            ][:50]
            if miss_cols:
                cleaning_steps.append({"step": "handle_missing", "columns": miss_cols, "method": "median"})

        rationale = strategy.get("rationale", "") if isinstance(strategy, dict) else ""
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = (
                f"Normalized strategy for {task_type or 'tabular'} with "
                "precision, leakage safety, and compute efficiency constraints."
            )

        return {
            "cleaning_steps": cleaning_steps,
            "feature_steps": feature_steps,
            "rationale": rationale.strip(),
        }

    def _sanitize_columns(self, columns: Any, all_cols: set[str], target: str) -> list[str]:
        if not isinstance(columns, list):
            return []
        clean: list[str] = []
        for c in columns:
            if isinstance(c, str) and c in all_cols and c != target and c not in clean:
                clean.append(c)
        return clean

    def _drop_one_heavy_feature(self, feature_steps: list[dict]) -> list[dict]:
        """Prefer interaction over polynomial on high-dimensional numeric datasets."""
        has_poly = any(s["step"] == "polynomial_features" for s in feature_steps)
        has_inter = any(s["step"] == "interaction_features" for s in feature_steps)
        if has_poly and has_inter:
            return [s for s in feature_steps if s["step"] != "polynomial_features"]
        return feature_steps

    def _llm_identity(self) -> tuple[str, str]:
        backend = getattr(self.llm, "_autoinsight_backend", None)
        model = getattr(self.llm, "_autoinsight_model", None)

        if not model:
            model = (
                getattr(self.llm, "model_name", None)
                or getattr(self.llm, "model", None)
                or self.llm.__class__.__name__
            )
        if not backend:
            class_name = self.llm.__class__.__name__.lower()
            if "groq" in class_name:
                backend = "groq"
            elif "google" in class_name or "gemini" in class_name:
                backend = "gemini"
            else:
                backend = "unknown"

        return str(backend), str(model)