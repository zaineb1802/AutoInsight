"""
Goal Parser Agent (multi-metric)
=================
Parses the user's natural-language objective and extracts:
  - target_column  : the column to predict
  - task_type      : "classification" | "regression"
  - metric         : primary evaluation metric

Metric selection rules (applied both in the LLM prompt and in the
post-LLM safety net so behaviour is consistent regardless of model quality):
  - Regression        -> ["r2", "rmse"]
  - Classification    -> ["accuracy", "f1"]
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from automl.state import AutoMLState

logger = logging.getLogger("autoinsight.agents.goal_parser")

SYSTEM_PROMPT = """You are an expert data scientist assistant.
Given a dataset's column names (with dtype and cardinality) and a user ML goal, extract:

1. target_column : exact column name to predict (must be one of the listed columns).
2. task_type     : "classification" or "regression".
3. metric        : best single evaluation metric, chosen by these rules:
     - Regression        -> ["r2", "rmse"]
     - Classification    -> ["accuracy", "f1"]
   Valid values: "rmse" "mae" "r2" "mape" "f1" "roc_auc" "precision" "recall" "accuracy"

Respond ONLY with valid JSON, no markdown, no explanation:
{"target_column": "...", "task_type": "...", "metric": "..."}
"""


class GoalParserAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("GoalParserAgent starting ...")
        df = state["dataframe"]
        goal = state["goal"]
        backend, model = self._llm_identity()
        logger.info("GoalParserAgent LLM in use: %s / %s", backend, model)

        # Give the LLM per-column context so it can make better decisions
        col_lines = [
            f"  {col}  dtype={df[col].dtype}  unique={df[col].nunique()}"
            for col in df.columns
        ]
        user_message = (
            "Dataset columns:\n" + "\n".join(col_lines)
            + f"\n\nUser goal: {goal}\n\n"
            "Return JSON with target_column, task_type, metric. "
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        target_column, task_type, metric = None, None, None
        try:
            print(f"[GoalParser] LLM active: backend={backend}, model={model}")
            response = self.llm.invoke(messages)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            target_column = parsed["target_column"]
            task_type = parsed["task_type"]
            metric = parsed["metric"]
        except Exception as exc:
            logger.warning("LLM parse failed (%s) — using heuristics.", exc)

        # Validate target column exists; fall back to last column
        if target_column not in df.columns:
            logger.warning(
                "Target '%s' not in columns — falling back to last column.", target_column
            )
            target_column = df.columns[-1]

        # If LLM failed to produce task_type/metric, use heuristics
        if not task_type or not metric:
            _, task_type, metric = self._heuristic_fallback(df, goal, target_column)

        # Post-LLM safety net: enforce sensible metric for every dataset
        metric = self._validate_metric(df, target_column, task_type, metric)

        logger.info(
            "Goal parsed — target: %s | task: %s | metric: %s",
            target_column, task_type, metric,
        )
        return {**state, "target_column": target_column, "task_type": task_type, "metric": metric}

    # ------------------------------------------------------------------
    # Metric validator — runs after LLM, always
    # ------------------------------------------------------------------

    def _validate_metric(self, df: Any, target_column: str, task_type: str, llm_metric: str) -> str:
        valid_clf = {"f1", "roc_auc", "precision", "recall", "accuracy"}
        valid_reg = {"rmse", "mae", "r2", "mape"}

        if task_type == "classification":
            if llm_metric not in valid_clf:
                logger.warning("Invalid clf metric '%s' -> overriding with 'accuracy'.", llm_metric)
                return "accuracy"
            return llm_metric

        else:  # regression
            if llm_metric not in valid_reg:
                logger.warning("Invalid reg metric '%s' -> overriding with 'r2'.", llm_metric)
                return "r2"
            return llm_metric

    # ------------------------------------------------------------------
    # Heuristic fallback (used when LLM fails completely)
    # ------------------------------------------------------------------

    def _heuristic_fallback(self, df: Any, goal: str, target_column: str) -> tuple[str, str, str]:
        goal_lower = goal.lower()

        # If target not yet known, search column names in the goal text
        if target_column not in df.columns:
            target_column = df.columns[-1]
            for col in df.columns:
                if col.lower() in goal_lower:
                    target_column = col
                    break

        n_unique = df[target_column].nunique()
        is_object = df[target_column].dtype == object

        if is_object or n_unique <= 20:
            task_type = "classification"
            metric = "accuracy"
        else:
            task_type = "regression"
            metric = "r2"

        return target_column, task_type, metric

    def _llm_identity(self) -> tuple[str, str]:
        """Best-effort backend/model name for runtime visibility."""
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