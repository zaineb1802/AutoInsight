"""
LLM client factory for AutoInsight.
Priority: Groq → Gemini (or explicit selection via --llm flag).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger("autoinsight.llm")


def _load_env_once() -> None:
    """
    Ensure API keys are loaded regardless of current working directory.
    Priority:
      1) AUTOINSIGHT_DOTENV explicit path
      2) AutoInsight/.env resolved from this module location
      3) default dotenv discovery fallback
    """
    explicit = os.getenv("AUTOINSIGHT_DOTENV")
    if explicit:
        load_dotenv(dotenv_path=explicit, override=False)
        return

    project_env = Path(__file__).resolve().parents[1] / ".env"
    if project_env.exists():
        load_dotenv(dotenv_path=project_env, override=False)
        return

    load_dotenv(override=False)


_load_env_once()


def _tag_llm(llm: Any, backend: str, model: str) -> Any:
    """Attach backend/model metadata so agents can report active LLM."""
    setattr(llm, "_autoinsight_backend", backend)
    setattr(llm, "_autoinsight_model", model)
    return llm


def _try_groq() -> Any | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0,
            max_tokens=4096,
        )
        # No ping — avoids burning rate-limit quota on startup
        logger.info("LLM backend: Groq (llama-3.3-70b-versatile)")
        return _tag_llm(llm, backend="groq", model="llama-3.3-70b-versatile")
    except Exception as exc:
        logger.warning("Groq unavailable: %s", exc)
        return None

def _try_gemini() -> Any | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=4096,
        )
        logger.info("LLM backend: Gemini (gemini-1.5-flash)")
        return _tag_llm(llm, backend="gemini", model="gemini-1.5-flash")
    except Exception as exc:
        logger.warning("Gemini unavailable: %s", exc)
        return None


def get_llm(backend: str = "auto") -> Any:
    """
    Return a LangChain chat model.

    Parameters
    ----------
    backend : str
        "auto" | "groq" | "gemini"
    """
    if backend == "groq":
        llm = _try_groq()
    elif backend == "gemini":
        llm = _try_gemini()
    else:  # "auto"
        llm = _try_groq() or _try_gemini()

    if llm is None:
        has_groq = bool(os.getenv("GROQ_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        raise RuntimeError(
            "No LLM backend available. "
            f"Requested='{backend}', GROQ_API_KEY set={has_groq}, GEMINI/GOOGLE key set={has_gemini}. "
            "Set GROQ_API_KEY or GEMINI_API_KEY in AutoInsight/.env."
        )
    return llm