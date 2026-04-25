"""
Feature engineering helper functions used by the Feature Agent.
Each function returns (modified_df, log_message).

General-purpose improvements over v1
--------------------------------------
- log_transform : uses log1p which is valid for zero values (fixes the > 0 bug).
  Columns with any negative values are skipped with a warning.
- extract_name_titles : detects "Name"-like columns and extracts a Title feature
  (Mr / Mrs / Miss / Master / Rare) using regex — works on any dataset that has
  a formatted name column (Titanic-style, some medical datasets, etc.)
- family_size : detects SibSp/Parch-style columns (sibling/spouse count +
  parent/child count) and creates FamilySize = sib + par + 1. Column names are
  matched by common aliases so it generalises beyond Titanic.
- encode_remaining_categoricals : safety net that label-encodes any leftover
  object columns before modeling.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("autoinsight.tools.feature_engineering")


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categorical(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "label",
    target: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Encode categorical columns.

    Methods
    -------
    label   : integer label encoding (default)
    onehot  : one-hot encoding, drops first level
    ordinal : alias for label
    """
    cols = [c for c in columns if c in df.columns and c != target]
    if not cols:
        return df, "No categorical columns to encode."

    # Hard rule: "sex" column must always be label-encoded.
    sex_cols = [c for c in cols if _is_sex_col(c)]
    other_cols = [c for c in cols if c not in sex_cols]

    encoded_parts: list[str] = []
    if sex_cols:
        for col in sex_cols:
            df[col] = df[col].astype("category").cat.codes
        encoded_parts.append(f"Label-encoded mandatory sex column(s): {sex_cols}.")

    if method in ("label", "ordinal"):
        for col in other_cols:
            df[col] = df[col].astype("category").cat.codes
        if other_cols:
            encoded_parts.append(f"Label-encoded: {other_cols}.")
        return df, " ".join(encoded_parts)

    if method == "onehot":
        if not other_cols:
            return df, " ".join(encoded_parts)
        before = set(df.columns)
        df = pd.get_dummies(df, columns=other_cols, drop_first=True)
        new_cols = list(set(df.columns) - before)
        encoded_parts.append(f"One-hot encoded {other_cols} -> {len(new_cols)} new columns.")
        return df, " ".join(encoded_parts)

    # Fallback
    for col in other_cols:
        df[col] = df[col].astype("category").cat.codes
    if other_cols:
        encoded_parts.append(f"Label-encoded (fallback): {other_cols}.")
    return df, " ".join(encoded_parts)


def _is_sex_col(col_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", col_name.lower())
    return normalized == "sex"


# ---------------------------------------------------------------------------
# Log transform  ← FIXED: log1p is valid for zero; skip negatives
# ---------------------------------------------------------------------------

def log_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Apply log1p transform to skewed numeric columns.

    log1p(x) = log(1 + x), which is:
      - valid and well-defined for x >= 0  (zero maps to 0)
      - invalid for x < -1

    Columns that contain negative values are skipped with a warning.
    """
    cols = [
        c for c in columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        return df, "No numeric columns for log transform."

    applied, skipped = [], []
    for col in cols:
        col_min = df[col].dropna().min()
        if col_min < 0:
            skipped.append(col)
            logger.warning(
                "log_transform: skipping '%s' — contains negative values (min=%.4f).", col, col_min
            )
            continue
        df[col] = np.log1p(df[col])
        applied.append(col)

    parts = []
    if applied:
        parts.append(f"Log1p-transformed: {applied}.")
    if skipped:
        parts.append(f"Skipped (negative values): {skipped}.")
    return df, " ".join(parts) if parts else "No columns transformed."


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------

def polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
    target: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Add polynomial features (degree 2 max, capped at 5 source columns).
    Prevents feature explosion on small datasets.
    """
    cols = [
        c for c in columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c != target
    ][:5]  # Cap source columns

    if not cols:
        return df, "No numeric columns for polynomial features."

    degree = min(degree, 2)
    new_cols = []
    for col in cols:
        for d in range(2, degree + 1):
            new_name = f"{col}_pow{d}"
            df[new_name] = df[col] ** d
            new_cols.append(new_name)

    return df, f"Added {len(new_cols)} polynomial feature(s) from {cols}."


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def interaction_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Create pairwise multiplication interaction terms (capped at 10 pairs).
    """
    cols = [
        c for c in columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(cols) < 2:
        return df, "Need >=2 numeric columns for interaction features."

    new_cols: list[str] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            new_name = f"{cols[i]}_x_{cols[j]}"
            df[new_name] = df[cols[i]] * df[cols[j]]
            new_cols.append(new_name)
            if len(new_cols) >= 10:
                break
        if len(new_cols) >= 10:
            break

    return df, f"Created {len(new_cols)} interaction feature(s)."


# ---------------------------------------------------------------------------
# Datetime decomposition
# ---------------------------------------------------------------------------

def datetime_decompose(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Decompose datetime columns into year, month, day, dayofweek, hour.
    Drops the original column after decomposition.
    """
    decomposed = []
    for col in columns:
        if col not in df.columns:
            continue
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().mean() < 0.5:
                continue
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_dayofweek"] = dt.dt.dayofweek
            if (dt.dt.hour != 0).any():
                df[f"{col}_hour"] = dt.dt.hour
            df.drop(columns=[col], inplace=True)
            decomposed.append(col)
        except Exception:
            pass

    return df, (
        f"Decomposed datetime columns: {decomposed}."
        if decomposed else "No datetime columns decomposed."
    )


# ---------------------------------------------------------------------------
# Name/title extraction  ← NEW general-purpose feature
# ---------------------------------------------------------------------------

# Regex that matches "Surname, Title. Firstname" patterns
_TITLE_RE = re.compile(r",\s*([^.]+)\.")

# Mapping of raw titles to grouped categories
_TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Ms": "Miss",
    "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Capt": "Rare", "Sir": "Rare", "Lady": "Rare", "Countess": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Mlle": "Miss",
}


def extract_name_titles(
    df: pd.DataFrame,
    columns: list[str],
    target: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    For any Name-like column that contains comma-separated 'Surname, Title. Firstname'
    patterns, extract a grouped Title feature and then drop the original column.

    This generalises beyond Titanic: any dataset where names are formatted this way
    (common in genealogy, medical, and some HR datasets) will benefit.
    """
    extracted = []
    for col in columns:
        if col not in df.columns or col == target:
            continue
        if df[col].dtype != object:
            continue

        # Check a sample to see if the pattern matches
        sample = df[col].dropna().head(50).astype(str)
        matches = sample.apply(lambda x: bool(_TITLE_RE.search(x)))
        if matches.mean() < 0.7:
            continue  # Column doesn't look like formatted names

        titles = df[col].astype(str).apply(
            lambda x: _TITLE_RE.search(x).group(1).strip() if _TITLE_RE.search(x) else "Unknown"
        )
        grouped = titles.map(lambda t: _TITLE_MAP.get(t, "Rare"))
        df[f"{col}_title"] = grouped.astype("category").cat.codes
        df.drop(columns=[col], inplace=True)
        extracted.append(col)
        logger.info("Extracted title from '%s' -> '%s_title'.", col, col)

    return df, (
        f"Extracted title features from: {extracted}."
        if extracted else "No name-title columns detected."
    )


# ---------------------------------------------------------------------------
# Family size feature  ← NEW general-purpose feature
# ---------------------------------------------------------------------------

# Common aliases for sibling/spouse count and parent/child count columns
_SIB_ALIASES = {"sibsp", "siblings", "sibling_spouse", "sib_sp", "num_siblings"}
_PAR_ALIASES = {"parch", "parents", "parent_child", "par_ch", "num_parents"}


def add_family_size(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Detect sibling/spouse count and parent/child count columns by common name
    aliases and create FamilySize = sib_col + par_col + 1.

    Also adds IsAlone = 1 if FamilySize == 1.

    Works on any dataset that has these patterns — not Titanic-specific.
    """
    sib_col = _find_col(df.columns, _SIB_ALIASES, target)
    par_col = _find_col(df.columns, _PAR_ALIASES, target)

    if sib_col is None or par_col is None:
        return df, "No family-size columns detected."

    df["FamilySize"] = df[sib_col] + df[par_col] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    logger.info("Created FamilySize and IsAlone from '%s' + '%s'.", sib_col, par_col)
    return df, f"Created FamilySize and IsAlone from '{sib_col}' + '{par_col}'."


def _find_col(columns: Any, aliases: set[str], target: Optional[str]) -> Optional[str]:
    for col in columns:
        if col == target:
            continue
        if col.lower() in aliases:
            return col
    return None


# ---------------------------------------------------------------------------
# Safety net: encode remaining categoricals
# ---------------------------------------------------------------------------

def encode_remaining_categoricals(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Label-encode any object/category columns not handled by earlier steps.
    This is the last gate before the data enters the modeling stage.
    """
    obj_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target
    ]
    if not obj_cols:
        return df, ""

    for col in obj_cols:
        df[col] = df[col].astype("category").cat.codes

    return df, f"Safety label-encoding applied to: {obj_cols}."