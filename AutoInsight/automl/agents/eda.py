"""
EDA Agent
=========
Generates a comprehensive exploratory data analysis summary including:
  - Shape, dtypes, memory usage
  - Descriptive statistics
  - Missing value counts
  - Cardinality and distributions
  - Correlation with target
  - Data quality flags
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from automl.state import AutoMLState
from automl.tools.profiling import profile_dataframe

logger = logging.getLogger("autoinsight.agents.eda")

SYSTEM_PROMPT = """You are a senior data scientist performing exploratory data analysis.
Analyse the provided data profile and return a concise JSON summary with these keys:
{
  "key_findings": ["list of important findings"],
  "quality_issues": ["list of data quality concerns"],
  "recommended_focus": "which features or aspects deserve most attention"
}
Respond ONLY with valid JSON — no markdown, no explanation."""


class EDAAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("EDAAgent starting ...")
        df = state["dataframe"]
        target = state.get("target_column")

        # Compute raw profile
        profile = profile_dataframe(df, target_column=target)

        # Ask LLM to interpret the profile
        llm_insights = self._get_llm_insights(profile)

        eda_summary = {
            "profile": profile,
            "llm_insights": llm_insights,
        }

        logger.info(
            "EDA complete — %d rows × %d cols | missing cells: %d",
            profile["shape"][0],
            profile["shape"][1],
            sum(v for v in profile["missing_counts"].values()),
        )

        return {**state, "eda_summary": eda_summary}

    # ------------------------------------------------------------------

    def _get_llm_insights(self, profile: dict) -> dict:
        import json

        # Truncate profile for the prompt to avoid token overflow
        summary_text = (
            f"Shape: {profile['shape']}\n"
            f"Dtypes: {profile['dtypes']}\n"
            f"Missing counts: {profile['missing_counts']}\n"
            f"Numeric stats (first 5 cols): "
            f"{dict(list(profile.get('numeric_stats', {}).items())[:5])}\n"
            f"High cardinality cols: {profile.get('high_cardinality', [])}\n"
            f"Constant cols: {profile.get('constant_cols', [])}\n"
            f"Duplicate rows: {profile.get('duplicate_rows', 0)}\n"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Data profile:\n{summary_text}"),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("LLM EDA interpretation failed: %s", exc)
            return {
                "key_findings": ["LLM interpretation unavailable."],
                "quality_issues": [],
                "recommended_focus": "All features",
            }