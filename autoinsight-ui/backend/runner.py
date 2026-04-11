"""
JobRunner
=========
Manages background AutoML jobs.

Each job runs in a ThreadPoolExecutor worker so the FastAPI event loop
stays non-blocking. Log output is captured via a custom logging handler
and stored in the JobResponse.logs list for SSE streaming.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("autoinsight.runner")
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

from models import DataSource, JobResponse, JobStatus, ModelResult, RunConfig

# ---------------------------------------------------------------------------
# Custom log handler that writes to the job's log buffer
# ---------------------------------------------------------------------------

class JobLogHandler(logging.Handler):
    def __init__(self, job: JobResponse) -> None:
        super().__init__()
        self.job = job
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            self.job.logs.append(line)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage tracker
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "goal_parser",
    "eda",
    "strategy",
    "validator",
    "cleaning",
    "feature",
    "modeling",
    "report",
]

STAGE_LABELS = {
    "goal_parser": "Parsing goal",
    "eda":         "Exploratory data analysis",
    "strategy":    "Planning strategy",
    "validator":   "Validating data",
    "cleaning":    "Cleaning data",
    "feature":     "Engineering features",
    "modeling":    "Training models",
    "report":      "Generating report",
}


# ---------------------------------------------------------------------------
# JobRunner
# ---------------------------------------------------------------------------

class JobRunner:
    def __init__(self, max_workers: int = 4) -> None:
        self._jobs: dict[str, JobResponse] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, job_id: str, config: RunConfig, report_path: str) -> JobResponse:
        job = JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            goal=config.goal,
            created_at=time.time(),
            report_path=report_path,
        )
        with self._lock:
            self._jobs[job_id] = job

        self._executor.submit(self._run, job, config, report_path)
        return job

    def get(self, job_id: str) -> Optional[JobResponse]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobResponse]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            del self._jobs[job_id]
            return True

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self, job: JobResponse, config: RunConfig, report_path: str) -> None:
        # Attach a per-job log handler scoped to the autoinsight logger.
        # Each job gets its own handler instance; they are removed in finally
        # so concurrent jobs do not leak logs into each other's buffers.
        handler = JobLogHandler(job)
        handler.set_name(f"autoinsight-job-{job.job_id}")
        root_logger = logging.getLogger("autoinsight")
        # Remove any stale handler with the same job id (defensive)
        root_logger.handlers = [
            h for h in root_logger.handlers
            if getattr(h, "name", None) != handler.name
        ]
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        try:
            job.status = JobStatus.RUNNING
            job.logs.append(f"[START] Job {job.job_id} started")

            # ── Load data ──────────────────────────────────────────────
            job.current_stage = "Loading data"
            df = self._load_data(config, job)

            # ── Add AutoInsight root to sys.path so automl package is importable ──
            # Priority:
            #   1. AUTOINSIGHT_ROOT env var (set manually or by start.bat)
            #   2. Sibling directory named "AutoInsight" relative to this file
            #   3. parent.parent of this file (legacy fallback)
            autoinsight_root = (
                os.environ.get("AUTOINSIGHT_ROOT")
                or str(Path(__file__).parent.parent.parent / "AutoInsight")
            )
            if autoinsight_root not in sys.path:
                sys.path.insert(0, autoinsight_root)
                logger.info("Added to sys.path: %s", autoinsight_root)

            # ── Build and run LangGraph workflow ───────────────────────
            from automl.graph import build_graph

            graph = build_graph(llm_backend=config.llm.value)

            initial_state = {
                "dataframe": df,
                "goal": config.goal,
                "output_path": report_path,
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

            # Patch graph to track stage progress
            final_state = self._run_with_stage_tracking(graph, initial_state, job)

            # ── Populate job results ───────────────────────────────────
            job.task_type = final_state.get("task_type")
            job.target_column = final_state.get("target_column")
            job.metric = final_state.get("metric")

            raw_results = final_state.get("model_results") or []
            job.model_results = [
                ModelResult(
                    model_name=r["model_name"],
                    score=r["score"],
                    train_score=r.get("train_score", 0.0),
                )
                for r in raw_results
            ]

            best = final_state.get("best_model") or {}
            job.best_model = best.get("model_name")
            job.best_score = best.get("score")
            job.feature_importance = best.get("feature_importance") or {}

            job.finished_at = time.time()
            job.elapsed_seconds = round(job.finished_at - job.created_at, 1)
            job.status = JobStatus.DONE
            job.logs.append(f"[DONE] Finished in {job.elapsed_seconds}s — report at {report_path}")

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.finished_at = time.time()
            job.elapsed_seconds = round(job.finished_at - job.created_at, 1)
            job.logs.append(f"[FAILED] {exc}")
            job.logs.append(traceback.format_exc())

        finally:
            root_logger.removeHandler(handler)

    def _run_with_stage_tracking(self, graph, initial_state: dict, job: JobResponse) -> dict:
        """
        LangGraph's .invoke() is opaque about which node is running.
        We use .stream() instead which yields one event per node,
        letting us update job.current_stage in real-time.
        """
        final_state = initial_state
        for event in graph.stream(initial_state):
            for node_name, state_update in event.items():
                label = STAGE_LABELS.get(node_name, node_name)
                job.current_stage = label
                if label not in job.stages_done:
                    job.stages_done.append(label)
                job.logs.append(f"[STAGE] {label}")
                final_state = {**final_state, **state_update}
        return final_state

    def _load_data(self, config: RunConfig, job: JobResponse) -> pd.DataFrame:
        if config.source_type == DataSource.FILE:
            path = Path(config.file_path or "")
            if not path.exists():
                raise FileNotFoundError(f"Uploaded file not found: {path}")
            suffix = path.suffix.lower()
            job.logs.append(f"[DATA] Loading file: {path.name}")
            if suffix == ".csv":
                return pd.read_csv(path)
            if suffix in (".xlsx", ".xls"):
                return pd.read_excel(path)
            if suffix == ".json":
                return pd.read_json(path)
            if suffix == ".parquet":
                return pd.read_parquet(path)
            raise ValueError(f"Unsupported file type: {suffix}")

        if config.source_type == DataSource.URL:
            job.logs.append(f"[DATA] Fetching URL: {config.url}")
            import io, requests
            r = requests.get(config.url, timeout=30)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "json" in ct:
                return pd.read_json(io.StringIO(r.text))
            return pd.read_csv(io.StringIO(r.text))

        if config.source_type == DataSource.GSHEET:
            import re, io, requests
            match = re.search(r"/d/([a-zA-Z0-9-_]+)", config.gsheet_url or "")
            if not match:
                raise ValueError("Invalid Google Sheets URL")
            sheet_id = match.group(1)
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            job.logs.append(f"[DATA] Fetching Google Sheet: {sheet_id}")
            r = requests.get(csv_url, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))

        raise ValueError(f"Unknown source_type: {config.source_type}")