"""
Cleaning Agent
==============
Executes the cleaning plan produced by the Strategy Agent:
  - Missing value imputation (median / mean / mode / drop)
  - Outlier removal (IQR / Z-score / clip)
  - Constant column removal
  - Dtype coercion
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl.state import AutoMLState
from automl.tools import cleaning as cleaning_tools

logger = logging.getLogger("autoinsight.agents.cleaning")


class CleaningAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm  # Reserved for future LLM-guided decisions

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("CleaningAgent starting ...")
        df = state["dataframe"].copy()
        strategy = state.get("strategy", {})
        target = state.get("target_column", "")

        cleaning_steps = strategy.get("cleaning_steps", [])
        executed: list[str] = []

        for step_cfg in cleaning_steps:
            step = step_cfg.get("step", "")
            cols = [c for c in step_cfg.get("columns", []) if c in df.columns and c != target]

            if step == "handle_missing":
                method = step_cfg.get("method", "median")
                df, log = cleaning_tools.handle_missing(df, cols, method)
                executed.append(log)

            elif step == "remove_outliers":
                method = step_cfg.get("method", "iqr")
                df, log = cleaning_tools.remove_outliers(df, cols, method)
                executed.append(log)

            elif step == "drop_constants":
                df, log = cleaning_tools.drop_constant_columns(df, cols)
                executed.append(log)

            elif step == "fix_dtypes":
                target_dtype = step_cfg.get("target_dtype", "numeric")
                df, log = cleaning_tools.fix_dtypes(df, cols, target_dtype)
                executed.append(log)

            elif step == "drop_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                executed.append(f"Dropped {before - len(df)} duplicate rows.")

            else:
                logger.debug("Unknown cleaning step: %s — skipped.", step)

        # Always: drop rows where target is null
        if target and target in df.columns:
            before = len(df)
            df = df[df[target].notna()].reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                executed.append(f"Dropped {dropped} rows with null target '{target}'.")

        # Fill any remaining missing values with median/mode as safety net
        df, log = cleaning_tools.fill_remaining_missing(df, target)
        executed.append(log)

        logger.info("Cleaning complete — %d steps executed. Final shape: %s", len(executed), df.shape)

        return {**state, "cleaned_dataframe": df}