"""
Feature Engineering Agent
==========================
Executes the feature plan from the Strategy Agent:
  - Categorical encoding (label / one-hot / ordinal)
  - Log transforms for skewed columns
  - Polynomial features
  - Interaction features
  - Datetime decomposition

General smart features (run on every dataset automatically):
  - Name/title extraction  : detects "Surname, Title. Firstname" columns
  - Family size            : detects sibling/parent count columns
  - Safety encoding        : label-encodes any leftover object columns
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl.state import AutoMLState
from automl.tools import feature_engineering as fe_tools

logger = logging.getLogger("autoinsight.agents.feature")


class FeatureAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("FeatureAgent starting ...")

        cleaned = state.get("cleaned_dataframe")
        df = (cleaned if cleaned is not None else state["dataframe"]).copy()
        strategy = state.get("strategy", {})
        target = state.get("target_column", "")

        feature_steps = strategy.get("feature_steps", [])
        executed: list[str] = []

        for step_cfg in feature_steps:
            step = step_cfg.get("step", "")
            cols = [
                c for c in step_cfg.get("columns", [])
                if c in df.columns and c != target
            ]
            if not cols and step not in ("polynomial_features",):
                continue

            if step == "encode_categorical":
                method = step_cfg.get("method", "label")
                df, log = fe_tools.encode_categorical(df, cols, method, target)
                executed.append(log)

            elif step == "log_transform":
                df, log = fe_tools.log_transform(df, cols)
                executed.append(log)

            elif step == "polynomial_features":
                degree = step_cfg.get("degree", 2)
                df, log = fe_tools.polynomial_features(df, cols, degree, target)
                executed.append(log)

            elif step == "interaction_features":
                df, log = fe_tools.interaction_features(df, cols)
                executed.append(log)

            elif step == "datetime_decompose":
                df, log = fe_tools.datetime_decompose(df, cols)
                executed.append(log)

            else:
                logger.debug("Unknown feature step: %s — skipped.", step)

        # ── General smart features (dataset-agnostic, always run) ────────

        # 1. Title extraction from Name-like formatted columns
        all_obj_cols = [c for c in df.columns if df[c].dtype == object and c != target]
        df, log = fe_tools.extract_name_titles(df, all_obj_cols, target)
        if log and "No name" not in log:
            executed.append(log)

        # 2. FamilySize + IsAlone from sibling/parent count columns
        df, log = fe_tools.add_family_size(df, target)
        if log and "No family" not in log:
            executed.append(log)

        # 3. Safety: label-encode any leftover object columns before modeling
        df, log = fe_tools.encode_remaining_categoricals(df, target)
        if log:
            executed.append(log)

        logger.info(
            "Feature engineering complete — %d steps. Final shape: %s",
            len(executed), df.shape,
        )

        return {**state, "engineered_dataframe": df}