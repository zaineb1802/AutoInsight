"""
Report Agent
============
Generates a comprehensive Markdown report summarising the full AutoML run.
Uses the LLM to write the executive summary and insights sections.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from automl.state import AutoMLState

logger = logging.getLogger("autoinsight.agents.report")

SYSTEM_PROMPT = """You are a data science technical writer. 
Given a structured AutoML run summary, write a concise executive summary (3-5 paragraphs) 
and 3-5 actionable recommendations for the user.

Return ONLY a JSON object:
{
  "executive_summary": "...",
  "recommendations": ["rec1", "rec2", "rec3"]
}
No markdown, no preamble."""


class ReportAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, state: AutoMLState) -> AutoMLState:
        logger.info("ReportAgent starting ...")

        output_path = state.get("output_path", "report.md")
        llm_section = self._get_llm_section(state)
        report_md = self._build_report(state, llm_section)

        Path(output_path).write_text(report_md, encoding="utf-8")
        logger.info("Report written to: %s", output_path)

        return {**state, "report_path": output_path}

    # ------------------------------------------------------------------

    def _get_llm_section(self, state: AutoMLState) -> dict:
        best = state.get("best_model") or {}
        results = state.get("model_results") or []

        summary_for_llm = {
            "goal": state.get("goal"),
            "task_type": state.get("task_type"),
            "target_column": state.get("target_column"),
            "metric": state.get("metric"),
            "best_model": best.get("model_name"),
            "best_score": best.get("score"),
            "all_models": [
                {"name": r["model_name"], "score": r["score"]} for r in results
            ],
            "key_eda_findings": (
                state.get("eda_summary", {})
                .get("llm_insights", {})
                .get("key_findings", [])
            ),
        }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(summary_for_llm, indent=2)),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("LLM report section failed: %s", exc)
            return {
                "executive_summary": (
                    f"AutoInsight completed an automated ML pipeline for the goal: "
                    f"'{state.get('goal')}'. "
                    f"The best model was {best.get('model_name', 'N/A')} with a score of "
                    f"{float(best.get('score') or 0):.4f} ({state.get('metric')})."
                ),
                "recommendations": [
                    "Review feature importance to understand key drivers.",
                    "Consider hyperparameter tuning on the best model.",
                    "Collect more data if model performance is unsatisfactory.",
                ],
            }

    def _build_report(self, state: AutoMLState, llm_section: dict) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        goal = state.get("goal", "N/A")
        target = state.get("target_column", "N/A")
        task_type = state.get("task_type", "N/A")
        metric = state.get("metric", "N/A")
        best = state.get("best_model") or {}
        results = state.get("model_results") or []
        profile = state.get("eda_summary", {}).get("profile", {})
        strategy = state.get("strategy", {})
        val_report = state.get("validation_report", {})

        # --- Model comparison table ---
        model_table_rows = "\n".join(
            f"| {r['model_name']} | {r['score']:.4f} | {r.get('train_score', 'N/A')} |"
            for r in sorted(results, key=lambda x: x["score"], reverse=(metric not in ("rmse", "mae", "mape")))
        )
        model_table = (
            "| Model | Test Score | Train Score |\n"
            "|-------|-----------|-------------|\n"
            + model_table_rows
        )

        # --- Feature importance ---
        feat_importance = best.get("feature_importance") or {}
        model_name = best.get("model_name", "")
        if feat_importance:
            top_feats = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            feat_imp_section = "## Feature Importance (Top 10)\n\n"
            feat_imp_section += "| Feature | Importance |\n|---------|----------|\n"
            feat_imp_section += "\n".join(
                f"| {f} | {v:.4f} |" for f, v in top_feats
            )
        else:
            # Some models (e.g. SVC with RBF kernel, KNN) do not expose feature importances.
            feat_imp_section = (
                "## Feature Importance\n\n"
                f"> **Not available for `{model_name}`** — this model type does not expose "
                "feature importances (e.g. kernel SVM, KNN).\n>\n"
                "> **Workaround options:**\n"
                "> - Switch to `GradientBoostingClassifier` or `RandomForestClassifier` "
                "for built-in importance scores.\n"
                "> - Use permutation importance (run `autoinsight` with `--explain` flag "
                "— coming soon).\n"
            )

        # --- Cleaning & feature steps ---
        cleaning_steps = strategy.get("cleaning_steps", [])
        feature_steps = strategy.get("feature_steps", [])

        cleaning_list = "\n".join(
            f"- **{s['step']}** on `{s.get('columns', [])}` "
            f"(method: {s.get('method', 'N/A')})"
            for s in cleaning_steps
        ) or "_No cleaning steps._"

        feature_list = "\n".join(
            f"- **{s['step']}** on `{s.get('columns', [])}` "
            f"(method: {s.get('method', s.get('degree', 'N/A'))})"
            for s in feature_steps
        ) or "_No feature engineering steps._"

        # --- Validation issues ---
        val_issues = val_report.get("issues", [])
        val_list = "\n".join(f"- {i}" for i in val_issues) or "_No issues detected._"

        # --- Recommendations ---
        recs = llm_section.get("recommendations", [])
        recs_list = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs))

        report = f"""# 🤖 AutoInsight ML Report

**Generated:** {now}  
**Goal:** {goal}  
**Target Column:** `{target}`  
**Task Type:** {task_type.title()}  
**Primary Metric:** {metric.upper()}

---

## 📋 Executive Summary

{llm_section.get("executive_summary", "")}

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Rows | {profile.get("shape", ["?", "?"])[0]} |
| Columns | {profile.get("shape", ["?", "?"])[1]} |
| Missing Cells | {sum(v for v in profile.get("missing_counts", {}).values())} |
| Duplicate Rows | {profile.get("duplicate_rows", 0)} |
| Numeric Columns | {len(profile.get("numeric_cols", []))} |
| Categorical Columns | {len(profile.get("categorical_cols", []))} |

---

## 🔬 EDA Key Findings

{chr(10).join("- " + f for f in state.get("eda_summary", {}).get("llm_insights", {}).get("key_findings", ["N/A"]))}

---

## [WARN] Validation Issues

{val_list}

---

## 🧹 Data Cleaning Steps

{cleaning_list}

---

## 🛠️ Feature Engineering Steps

{feature_list}

---

## 🏆 Model Performance

**Best Model:** `{best.get("model_name", "N/A")}` — {metric.upper()}: **{best.get("score", 0):.4f}**

### All Models

{model_table}

---

{feat_imp_section}

---

## 💡 Recommendations

{recs_list}

---

## ⚙️ Best Model Configuration

```json
{json.dumps(best.get("params", {}), indent=2)}
```

---

*Report generated by AutoInsight*
"""
        return report