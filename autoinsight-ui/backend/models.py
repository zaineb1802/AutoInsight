"""
Pydantic models for AutoInsight API request/response contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class DataSource(str, Enum):
    FILE = "file"
    URL = "url"
    GSHEET = "gsheet"


class LLMBackend(str, Enum):
    AUTO = "auto"
    GROQ = "groq"
    GEMINI = "gemini"


class RunConfig(BaseModel):
    """Payload for POST /api/run"""

    goal: str = Field(..., min_length=5, description="ML objective in plain English")
    llm: LLMBackend = Field(LLMBackend.AUTO, description="LLM backend to use")

    # Data source — exactly one must be provided
    source_type: DataSource = Field(..., description="How data is provided")
    file_path: Optional[str] = Field(None, description="Server-side path from /api/upload")
    url: Optional[str] = Field(None, description="Remote CSV/JSON URL")
    gsheet_url: Optional[str] = Field(None, description="Google Sheets URL")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ModelResult(BaseModel):
    model_name: str
    score: float
    train_score: float
    r2_score: Optional[float] = None
    r2_train_score: Optional[float] = None


class JobResponse(BaseModel):
    """Returned for every job-related endpoint"""

    job_id: str
    status: JobStatus
    goal: str
    created_at: float
    finished_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None

    # Progress
    current_stage: Optional[str] = None
    stages_done: list[str] = []

    # Results (populated when done)
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    metric: Optional[str] = None
    model_results: list[ModelResult] = []
    best_model: Optional[str] = None
    best_score: Optional[float] = None
    best_r2: Optional[float] = None
    feature_importance: dict[str, float] = {}
    report_path: Optional[str] = None
    best_model_path: Optional[str] = None
    visualization_outputs: list[str] = []

    # Error
    error: Optional[str] = None

    # Live log buffer
    logs: list[str] = []
