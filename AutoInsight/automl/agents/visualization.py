"""
Visualization Agent
===================
Generates clean feature-vs-target visualizations with automatic plot-type selection.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
from automl.state import AutoMLState
from automl.tools.visualization import generate_feature_target_plots
logger = logging.getLogger("autoinsight.agents.visualization")
class VisualizationAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("VisualizationAgent starting ...")
        engineered = state.get("engineered_dataframe")
        cleaned = state.get("cleaned_dataframe")
        df = (
            engineered if engineered is not None
            else cleaned if cleaned is not None
            else state.get("dataframe")
        )
        target = state.get("target_column")
        output_path = state.get("output_path", "report.md")
        if df is None or not target:
            logger.warning("Visualization skipped: dataframe or target missing.")
            return {**state, "visualization_outputs": []}
        report_stem = Path(output_path).stem
        plot_dir = Path("reports") / f"{report_stem}_plots"
        plots = generate_feature_target_plots(
            df=df,
            target=target,
            output_dir=str(plot_dir),
            max_features=8,
        )
        logger.info("Visualization generated %d plot(s).", len(plots))
        return {**state, "visualization_outputs": plots}