"""
LLM client factory for AutoInsight.
Priority: Groq → Ollama → Gemini  (or explicit selection via --llm flag).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("autoinsight.llm")


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
        return llm
    except Exception as exc:
        logger.warning("Groq unavailable: %s", exc)
        return None


def _try_ollama() -> Any | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="llama3.2:1b",
            base_url=base_url,
            temperature=0,
        )
        logger.info("LLM backend: Ollama (llama3.2:1b) @ %s", base_url)
        return llm
    except Exception as exc:
        logger.warning("Ollama unavailable: %s", exc)
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
        return llm
    except Exception as exc:
        logger.warning("Gemini unavailable: %s", exc)
        return None


def get_llm(backend: str = "auto") -> Any:
    """
    Return a LangChain chat model.

    Parameters
    ----------
    backend : str
        "auto" | "groq" | "ollama" | "gemini"
    """
    if backend == "groq":
        llm = _try_groq()
    elif backend == "ollama":
        llm = _try_ollama()
    elif backend == "gemini":
        llm = _try_gemini()
    else:  # "auto"
        llm = _try_groq() or _try_ollama() or _try_gemini()

    if llm is None:
        raise RuntimeError(
            "No LLM backend available. Set GROQ_API_KEY, GEMINI_API_KEY, "
            "or start an Ollama server and set OLLAMA_BASE_URL."
        )
    return llm