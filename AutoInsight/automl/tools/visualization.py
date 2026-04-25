"""
Visualization utilities for feature-target relationship analysis.

Automatically selects plot type based on feature/target dtypes:
- numeric feature vs numeric target -> scatter + regression line
- categorical feature vs numeric target -> boxplot
- numeric feature vs categorical target -> boxplot (grouped by target)
- categorical feature vs categorical target -> heatmap (crosstab)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger("autoinsight.tools.visualization")
sns.set_theme(style="whitegrid")


def generate_feature_target_plots(
    df: pd.DataFrame,
    target: str,
    output_dir: str,
    max_features: int = 8,
) -> list[str]:
    if target not in df.columns:
        logger.warning("Target '%s' not found for visualization.", target)
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    candidate_cols = [c for c in df.columns if c != target]
    # Prefer numeric features first, then categoricals.
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in candidate_cols if c not in numeric_cols]
    selected = (numeric_cols + cat_cols)[:max_features]

    generated: list[str] = []
    for feature in selected:
        try:
            fig_path = _plot_feature_vs_target(df, feature, target, output_path)
            if fig_path:
                generated.append(str(fig_path))
        except Exception as exc:
            logger.warning("Failed to generate plot for %s vs %s: %s", feature, target, exc)

    return generated


def _plot_feature_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    output_dir: Path,
) -> Path | None:
    x_num = pd.api.types.is_numeric_dtype(df[feature])
    y_num = pd.api.types.is_numeric_dtype(df[target])

    fig, ax = plt.subplots(figsize=(10, 6))
    title = f"{feature} vs {target}"

    if x_num and y_num:
        sns.regplot(data=df, x=feature, y=target, ax=ax, scatter_kws={"alpha": 0.6, "s": 32})
        ax.set_title(f"Scatter with Trend: {title}", fontsize=13, weight="bold")
    elif (not x_num) and y_num:
        top_levels = df[feature].astype(str).value_counts().head(12).index
        dff = df[df[feature].astype(str).isin(top_levels)].copy()
        sns.boxplot(data=dff, x=feature, y=target, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(f"Boxplot: {title}", fontsize=13, weight="bold")
    elif x_num and (not y_num):
        top_levels = df[target].astype(str).value_counts().head(8).index
        dff = df[df[target].astype(str).isin(top_levels)].copy()
        sns.boxplot(data=dff, x=target, y=feature, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(f"Distribution of {feature} by {target}", fontsize=13, weight="bold")
    else:
        top_x = df[feature].astype(str).value_counts().head(10).index
        top_y = df[target].astype(str).value_counts().head(10).index
        dff = df[df[feature].astype(str).isin(top_x) & df[target].astype(str).isin(top_y)].copy()
        ct = pd.crosstab(dff[target].astype(str), dff[feature].astype(str))
        if ct.empty:
            plt.close(fig)
            return None
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Crosstab Heatmap: {title}", fontsize=13, weight="bold")

    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel(target, fontsize=11)
    fig.tight_layout()

    file_name = f"{_safe_name(feature)}_vs_{_safe_name(target)}.png"
    out_file = output_dir / file_name
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _safe_name(name: Any) -> str:
    text = str(name)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
