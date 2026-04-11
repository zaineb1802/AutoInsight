"""
Data profiling utilities for the EDA Agent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """
    Compute a comprehensive profile of the dataframe.

    Returns
    -------
    dict with keys:
        shape, dtypes, missing_counts, missing_pct, numeric_stats,
        categorical_stats, cardinality, high_cardinality, constant_cols,
        duplicate_rows, numeric_cols, categorical_cols, skewed_cols,
        target_distribution (if target provided)
    """
    profile: dict = {}

    profile["shape"] = list(df.shape)
    profile["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing = df.isnull().sum()
    profile["missing_counts"] = missing.to_dict()
    profile["missing_pct"] = (missing / len(df) * 100).round(2).to_dict()

    # Column type buckets
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    profile["numeric_cols"] = numeric_cols
    profile["categorical_cols"] = categorical_cols

    # Numeric descriptive stats
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        profile["numeric_stats"] = desc.to_dict(orient="index")
    else:
        profile["numeric_stats"] = {}

    # Cardinality
    cardinality = {col: int(df[col].nunique()) for col in df.columns}
    profile["cardinality"] = cardinality

    # High cardinality categoricals (>50 unique values)
    profile["high_cardinality"] = [
        col for col in categorical_cols if cardinality[col] > 50
    ]

    # Constant columns (1 unique value)
    profile["constant_cols"] = [
        col for col in df.columns if cardinality[col] <= 1
    ]

    # Duplicate rows
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # Skewed numeric columns (|skewness| > 1)
    skewed = []
    for col in numeric_cols:
        try:
            skew_val = float(df[col].dropna().skew())
            if abs(skew_val) > 1.0:
                skewed.append(col)
        except Exception:
            pass
    profile["skewed_cols"] = skewed

    # Categorical stats (top 5 values)
    cat_stats = {}
    for col in categorical_cols[:20]:  # Limit to avoid huge output
        vc = df[col].value_counts(normalize=True).head(5)
        cat_stats[col] = vc.to_dict()
    profile["categorical_stats"] = cat_stats

    # Correlation with target
    if target_column and target_column in numeric_cols:
        other_num = [c for c in numeric_cols if c != target_column]
        if other_num:
            corr = df[other_num + [target_column]].corr()[target_column].drop(target_column)
            profile["target_correlation"] = corr.sort_values(key=abs, ascending=False).to_dict()

    # Target distribution
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(target_series):
            profile["target_distribution"] = {
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "n_unique": int(target_series.nunique()),
            }
        else:
            vc = target_series.value_counts().head(10)
            profile["target_distribution"] = vc.to_dict()

    return profile
