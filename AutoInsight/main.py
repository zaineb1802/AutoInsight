"""
AutoInsight — LLM-driven AutoML entry point.
Handles data loading from multiple sources and launches the LangGraph workflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 on the console stream so Unicode chars render on Windows (cp1252)
import io as _io
_utf8_stream = _io.TextIOWrapper(
    getattr(sys.stdout, "buffer", None) or sys.stdout,
    encoding="utf-8",
    errors="replace",
    line_buffering=True,
) if hasattr(sys.stdout, "buffer") else sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(_utf8_stream),
        logging.FileHandler(Path("logs") / "autoinsight.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("autoinsight.main")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading CSV: %s", path)
    return pd.read_csv(path)


def load_excel(path: str) -> pd.DataFrame:
    logger.info("Loading Excel: %s", path)
    return pd.read_excel(path)


def load_json(path: str) -> pd.DataFrame:
    logger.info("Loading JSON: %s", path)
    return pd.read_json(path)


def load_parquet(path: str) -> pd.DataFrame:
    logger.info("Loading Parquet: %s", path)
    return pd.read_parquet(path)


def load_url(url: str) -> pd.DataFrame:
    """Load a CSV/JSON from a remote URL."""
    import requests

    logger.info("Loading URL: %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        import io
        return pd.read_json(io.StringIO(response.text))
    # Default: assume CSV
    import io
    return pd.read_csv(io.StringIO(response.text))


def load_google_sheet(sheet_url: str) -> pd.DataFrame:
    """Convert a Google Sheets URL to CSV export and load it."""
    import re

    logger.info("Loading Google Sheet: %s", sheet_url)
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("Invalid Google Sheets URL — could not extract sheet ID.")
    sheet_id = match.group(1)
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    return load_url(csv_url)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """Route to the correct loader based on CLI flags."""
    if args.csv:
        return load_csv(args.csv)
    if args.excel:
        return load_excel(args.excel)
    if args.json:
        return load_json(args.json)
    if args.parquet:
        return load_parquet(args.parquet)
    if args.url:
        return load_url(args.url)
    if args.gsheet:
        return load_google_sheet(args.gsheet)
    raise ValueError(
        "No data source provided. Use --csv, --excel, --json, --parquet, --url, or --gsheet."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoinsight",
        description="AutoInsight: LLM-driven AutoML system.",
    )

    # Data source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", metavar="PATH", help="Path to a CSV file")
    source.add_argument("--excel", metavar="PATH", help="Path to an Excel file")
    source.add_argument("--json", metavar="PATH", help="Path to a JSON file")
    source.add_argument("--parquet", metavar="PATH", help="Path to a Parquet file")
    source.add_argument("--url", metavar="URL", help="URL pointing to a CSV/JSON dataset")
    source.add_argument("--gsheet", metavar="URL", help="Google Sheets URL")

    parser.add_argument(
        "--goal",
        required=True,
        metavar="TEXT",
        help='Describe your ML objective, e.g. "Predict crude oil price"',
    )
    parser.add_argument(
        "--output",
        default="report.md",
        metavar="PATH",
        help="Path for the generated markdown report (default: report.md)",
    )
    parser.add_argument(
        "--llm",
        default="auto",
        choices=["auto", "groq", "ollama", "gemini"],
        help="LLM backend to use (default: auto — tries Groq then Ollama then Gemini)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    parser = build_parser()
    args = parser.parse_args()

    # Load data
    try:
        df = load_data(args)
    except Exception as exc:
        logger.error("Failed to load data: %s", exc)
        sys.exit(1)

    logger.info("Dataset loaded — shape: %s", df.shape)

    # Import and run the LangGraph workflow
    from automl.graph import build_graph

    graph = build_graph(llm_backend=args.llm)

    initial_state = {
        "dataframe": df,
        "goal": args.goal,
        "output_path": args.output,
        "task_type": None,
        "target_column": None,
        "metric": None,
        "eda_summary": None,
        "strategy": None,
        "validation_report": None,
        "cleaned_dataframe": None,
        "engineered_dataframe": None,
        "model_results": None,
        "best_model": None,
        "report_path": None,
        "messages": [],
    }

    logger.info("Starting AutoInsight workflow …")
    final_state = graph.invoke(initial_state)

    report_path = final_state.get("report_path", args.output)
    logger.info("[OK] AutoInsight complete. Report saved to: %s", report_path)
    print(f"\n[OK] Done! Report saved to: {report_path}\n")


if __name__ == "__main__":
    main()