"""
Cleaning helper functions used by the Cleaning Agent.
Each function returns (modified_df, log_message).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

def handle_missing(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "median",
) -> tuple[pd.DataFrame, str]:
    """
    Impute missing values in the given columns.

    Methods: median, mean, mode, drop_rows, drop_cols
    """
    affected = [c for c in columns if c in df.columns and df[c].isnull().any()]
    if not affected:
        return df, "No missing values to handle."

    if method == "drop_rows":
        before = len(df)
        df = df.dropna(subset=affected).reset_index(drop=True)
        return df, f"Dropped {before - len(df)} rows with NaN in {affected}."

    if method == "drop_cols":
        df = df.drop(columns=affected)
        return df, f"Dropped columns with NaN: {affected}."

    for col in affected:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # For non-numeric columns always use mode
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        elif method == "median":
            fill_val = df[col].median()
        elif method == "mean":
            fill_val = df[col].mean()
        elif method == "mode":
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
        else:
            fill_val = df[col].median()

        df[col] = df[col].fillna(fill_val)

    return df, f"Imputed missing values in {affected} using '{method}'."


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "iqr",
) -> tuple[pd.DataFrame, str]:
    """
    Remove or clip outliers.

    Methods: iqr (drop rows), zscore (drop rows), clip (winsorize)
    """
    num_cols = [
        c for c in columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not num_cols:
        return df, "No numeric columns for outlier removal."

    before = len(df)

    if method == "clip":
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
        return df, f"Clipped outliers (IQR) in {num_cols}."

    if method == "zscore":
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            mask &= z.abs() <= 3
        df = df[mask].reset_index(drop=True)
        return df, f"Removed {before - len(df)} rows with |z-score| > 3 in {num_cols}."

    # Default: IQR drop
    mask = pd.Series([True] * len(df), index=df.index)
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)
    df = df[mask].reset_index(drop=True)
    return df, f"Removed {before - len(df)} outlier rows (IQR) from {num_cols}."


# ---------------------------------------------------------------------------
# Column dropping
# ---------------------------------------------------------------------------

def drop_constant_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, str]:
    to_drop = [c for c in columns if c in df.columns]
    if not to_drop:
        return df, "No constant columns to drop."
    df = df.drop(columns=to_drop)
    return df, f"Dropped constant columns: {to_drop}."


# ---------------------------------------------------------------------------
# Dtype coercion
# ---------------------------------------------------------------------------

def fix_dtypes(
    df: pd.DataFrame,
    columns: list[str],
    target_dtype: str = "numeric",
) -> tuple[pd.DataFrame, str]:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df, "No columns to fix dtypes."

    fixed = []
    for col in cols:
        try:
            if target_dtype == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target_dtype == "string":
                df[col] = df[col].astype(str)
            fixed.append(col)
        except Exception:
            pass

    return df, f"Fixed dtypes to '{target_dtype}' for: {fixed}."


# ---------------------------------------------------------------------------
# Safety net: fill remaining missing
# ---------------------------------------------------------------------------

def fill_remaining_missing(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Fill any leftover NaN values with column median (numeric) or mode (object)."""
    filled = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
            filled.append(col)

    if filled:
        return df, f"Safety fill applied to {len(filled)} column(s): {filled}."
    return df, "No remaining missing values."
