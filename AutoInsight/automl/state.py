"""
Shared LangGraph state definition for the AutoInsight workflow.
All agents read from and write to this TypedDict.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

import pandas as pd


class AutoMLState(TypedDict, total=False):
    # ----- Input -----
    dataframe: pd.DataFrame          # Raw dataframe loaded by main.py
    goal: str                        # User-specified ML objective
    output_path: str                 # Desired path for the final report

    # ----- Goal parsing -----
    task_type: Optional[str]         # "classification" | "regression"
    target_column: Optional[str]     # Name of the target column
    metric: Optional[str]            # Primary evaluation metric

    # ----- EDA -----
    eda_summary: Optional[dict]      # Statistical summary dict from EDA agent

    # ----- Strategy -----
    strategy: Optional[dict]         # Cleaning + feature engineering plan

    # ----- Validation -----
    validation_report: Optional[dict]  # Issues flagged by the Validator

    # ----- Cleaned data -----
    cleaned_dataframe: Optional[pd.DataFrame]

    # ----- Engineered data -----
    engineered_dataframe: Optional[pd.DataFrame]

    # ----- Modeling -----
    model_results: Optional[list[dict]]   # Per-model metrics
    best_model: Optional[dict]            # Best model info + params

    # ----- Report -----
    report_path: Optional[str]       # Actual path where report was written

    # ----- LangGraph message bus -----
    messages: list[Any]
