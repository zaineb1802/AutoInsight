"""
LangGraph workflow orchestration for AutoInsight.
Wires up all 8 agents into a sequential StateGraph.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from automl.state import AutoMLState
from automl.llm import get_llm

logger = logging.getLogger("autoinsight.graph")


def build_graph(llm_backend: str = "auto") -> StateGraph:
    """Construct and compile the AutoInsight LangGraph workflow."""

    llm = get_llm(llm_backend)

    # Import agents (deferred to avoid circular imports at module level)
    from automl.agents.goal_parser import GoalParserAgent
    from automl.agents.eda import EDAAgent
    from automl.agents.strategy import StrategyAgent
    from automl.agents.validator import ValidatorAgent
    from automl.agents.cleaning import CleaningAgent
    from automl.agents.feature import FeatureAgent
    from automl.agents.modeling import ModelingAgent
    from automl.agents.report import ReportAgent

    # Instantiate agents
    goal_parser = GoalParserAgent(llm)
    eda_agent = EDAAgent(llm)
    strategy_agent = StrategyAgent(llm)
    validator_agent = ValidatorAgent(llm)
    cleaning_agent = CleaningAgent(llm)
    feature_agent = FeatureAgent(llm)
    modeling_agent = ModelingAgent(llm)
    report_agent = ReportAgent(llm)

    # Build the graph
    workflow = StateGraph(AutoMLState)

    workflow.add_node("goal_parser", goal_parser.run)
    workflow.add_node("eda", eda_agent.run)
    workflow.add_node("strategy", strategy_agent.run)
    workflow.add_node("validator", validator_agent.run)
    workflow.add_node("cleaning", cleaning_agent.run)
    workflow.add_node("feature", feature_agent.run)
    workflow.add_node("modeling", modeling_agent.run)
    workflow.add_node("report", report_agent.run)

    # Linear pipeline edges
    workflow.set_entry_point("goal_parser")
    workflow.add_edge("goal_parser", "eda")
    workflow.add_edge("eda", "strategy")
    workflow.add_edge("strategy", "validator")
    workflow.add_edge("validator", "cleaning")
    workflow.add_edge("cleaning", "feature")
    workflow.add_edge("feature", "modeling")
    workflow.add_edge("modeling", "report")
    workflow.add_edge("report", END)

    logger.info("LangGraph workflow compiled successfully.")
    return workflow.compile()
