"""
Modeling Agent
==============
Trains multiple ML models appropriate for the task type and selects the best one.

Classification models: LogisticRegression, RandomForestClassifier,
                       GradientBoostingClassifier, SVC
Regression models:    LinearRegression, Ridge, RandomForestRegressor,
                      GradientBoostingRegressor, SVR
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl.state import AutoMLState
from automl.tools import modeling as model_tools

logger = logging.getLogger("autoinsight.agents.modeling")


class ModelingAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("ModelingAgent starting ...")

        engineered = state.get("engineered_dataframe")
        cleaned = state.get("cleaned_dataframe")
        df = (
            engineered if engineered is not None
            else cleaned if cleaned is not None
            else state["dataframe"]
        ).copy()

        target = state["target_column"]
        task_type = state["task_type"]
        metric = state["metric"]

        if target not in df.columns:
            logger.error("Target column '%s' not found after feature engineering!", target)
            return {**state, "model_results": [], "best_model": None}

        # Drop non-numeric columns that weren't encoded (safety)
        feature_cols = [
            col for col in df.columns
            if col != target and pd.api.types.is_numeric_dtype(df[col])
        ]
        X = df[feature_cols].fillna(0)
        y = df[target]

        logger.info(
            "Training on %d samples × %d features | task: %s | metric: %s",
            len(X), len(feature_cols), task_type, metric,
        )

        results, best = model_tools.train_and_evaluate(X, y, task_type, metric)

        if best:
            logger.info(
                "Best model: %s (%.4f %s)",
                best["model_name"], best["score"], metric,
            )

        return {**state, "model_results": results, "best_model": best}