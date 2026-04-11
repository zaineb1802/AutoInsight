"""
AutoInsight FastAPI Backend
===========================
Endpoints:
  POST /api/upload          — upload a dataset file
  POST /api/run             — start an AutoML job
  GET  /api/jobs/{id}       — poll job status + results
  GET  /api/jobs/{id}/logs  — SSE stream of live log lines
  GET  /api/jobs/{id}/report — fetch the final markdown report
  GET  /api/jobs             — list all jobs (most recent first)
  DELETE /api/jobs/{id}      — remove a job record
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import JobResponse, JobStatus, RunConfig
from runner import JobRunner

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoinsight.api")

app = FastAPI(title="AutoInsight API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

runner = JobRunner()


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Accept a CSV/Excel/JSON/Parquet file and store it server-side."""
    allowed = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    # Quick preview
    try:
        df = _load_preview(dest, suffix)
        preview = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "head": df.head(5).fillna("").to_dict(orient="records"),
        }
    except Exception as exc:
        preview = {"error": str(exc)}

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(dest),
        "suffix": suffix,
        "preview": preview,
    }


def _load_preview(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Cannot load {suffix}")


# ---------------------------------------------------------------------------
# Run a job
# ---------------------------------------------------------------------------

@app.post("/api/run", response_model=JobResponse)
async def run_job(config: RunConfig) -> JobResponse:
    """Queue a new AutoML job and return its ID immediately."""
    job_id = str(uuid.uuid4())
    report_path = str(REPORTS_DIR / f"{job_id}.md")

    job = runner.submit(
        job_id=job_id,
        config=config,
        report_path=report_path,
    )
    return job


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = runner.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    return runner.list_jobs()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    if not runner.delete(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": job_id}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/report")
async def get_report(job_id: str) -> dict:
    job = runner.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=202, detail="Report not ready yet")
    path = Path(job.report_path or "")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")
    return {"markdown": path.read_text(encoding="utf-8"), "job_id": job_id}


# ---------------------------------------------------------------------------
# SSE log stream
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str) -> StreamingResponse:
    """Server-Sent Events endpoint — streams log lines as they arrive."""

    async def generate() -> AsyncGenerator[str, None]:
        cursor = 0
        timeout = 300  # max 5 min stream
        start = time.time()

        while time.time() - start < timeout:
            job = runner.get(job_id)
            if not job:
                yield "data: [JOB NOT FOUND]\n\n"
                break

            lines = job.logs[cursor:]
            for line in lines:
                yield f"data: {line}\n\n"
            cursor += len(lines)

            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                yield f"data: [STATUS:{job.status.value}]\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )