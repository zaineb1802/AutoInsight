"""
Strategy Agent
==============
Uses the LLM to create a structured plan for:
  - Data cleaning steps
  - Feature engineering steps
based on the EDA summary and the user's goal.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from automl.state import AutoMLState

logger = logging.getLogger("autoinsight.agents.strategy")

SYSTEM_PROMPT = """You are an expert ML engineer. Based on the EDA findings, 
design a data preparation strategy.

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


class StrategyAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("StrategyAgent starting ...")

        eda_summary = state.get("eda_summary", {})
        profile = eda_summary.get("profile", {})
        insights = eda_summary.get("llm_insights", {})
        goal = state["goal"]
        task_type = state.get("task_type", "unknown")
        target = state.get("target_column", "")

        prompt_body = (
            f"Goal: {goal}\n"
            f"Task type: {task_type}\n"
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
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_body),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            strategy = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM strategy planning failed (%s). Using default strategy.", exc)
            strategy = self._default_strategy(profile, target)

        logger.info(
            "Strategy planned — %d cleaning steps, %d feature steps",
            len(strategy.get("cleaning_steps", [])),
            len(strategy.get("feature_steps", [])),
        )

        return {**state, "strategy": strategy}

    # ------------------------------------------------------------------

    def _default_strategy(self, profile: dict, target: str) -> dict:
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

        feature_steps = []
        if high_card:
            feature_steps.append(
                {"step": "encode_categorical", "columns": high_card, "method": "label"}
            )
        if skewed:
            feature_steps.append({"step": "log_transform", "columns": skewed})

        return {
            "cleaning_steps": cleaning_steps,
            "feature_steps": feature_steps,
            "rationale": "Default conservative strategy (LLM fallback).",
        }