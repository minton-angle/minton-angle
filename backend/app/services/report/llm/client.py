from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

import httpx


logger_llm_client = logging.getLogger("app.llm")

# ------------------------------------------------------------------
# LLM usage (token counts)
# ------------------------------------------------------------------
_LAST_LLM_USAGE: Dict[str, Any] = {}


def _set_last_llm_usage(u: Any) -> None:
    global _LAST_LLM_USAGE
    if isinstance(u, dict):
        _LAST_LLM_USAGE = u
    else:
        _LAST_LLM_USAGE = {}


def get_last_llm_usage() -> Dict[str, Any]:
    return _LAST_LLM_USAGE if isinstance(_LAST_LLM_USAGE, dict) else {}


# ------------------------------------------------------------------
# LLM Provider Settings (Groq / Hugging Face)
# ------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Groq (OpenAI-compatible)
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Hugging Face (OpenAI-compatible)
HF_BASE_URL = os.getenv("HF_BASE_URL", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "").strip()

# Shared generation params
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", os.getenv("GROQ_MAX_TOKENS", "1600")))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.8")))

# Some providers do not support response_format=json_object. Keep it optional.
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "1").strip() not in ("0", "false", "False")


def _chat_completions_url(base_url: str) -> str:
    """Build a chat-completions URL from an OpenAI-compatible base URL."""
    b = (base_url or "").rstrip("/")
    if not b:
        return ""
    return f"{b}/chat/completions"


def call_llm(messages, model: str = "") -> str:
    """Call the configured provider via OpenAI-compatible chat completions."""
    provider = (LLM_PROVIDER or "groq").strip().lower()

    try:
        effective_model = (
            model
            or (HF_MODEL if provider == "hf" else GROQ_MODEL)
            or "model"
        )
        logger_llm_client.info(
            "LLM call provider=%s base_url=%s model=%s temperature=%.2f max_tokens=%d",
            provider,
            HF_BASE_URL if provider == "hf" else GROQ_BASE_URL,
            effective_model,
            DEFAULT_TEMPERATURE,
            DEFAULT_MAX_TOKENS,
        )
    except Exception:
        pass

    if provider == "hf":
        if not HF_BASE_URL:
            raise RuntimeError("HF_BASE_URL is not set (e.g. https://<your-hf-endpoint>/v1)")
        if not HF_API_KEY:
            raise RuntimeError("HF_API_KEY (or HF_TOKEN) is not set")

        url = _chat_completions_url(HF_BASE_URL)
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }
        chosen_model = model or HF_MODEL or "model"
        body = {
            "model": chosen_model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        timeout = httpx.Timeout(60.0)
        t0 = time.perf_counter()

        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=body)

        logger_llm_client.info(
            "HF status=%s time_ms=%.1f",
            r.status_code,
            (time.perf_counter() - t0) * 1000.0,
        )

        if r.status_code >= 400:
            raise RuntimeError(f"HF API error {r.status_code}: {r.text}")

        data = r.json()
        _set_last_llm_usage(data.get("usage"))
        try:
            if data.get("usage"):
                logger_llm_client.info("HF usage=%s", json.dumps(data.get("usage"), ensure_ascii=False))
        except Exception:
            pass
        return data["choices"][0]["message"]["content"]

    # default: groq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = _chat_completions_url(GROQ_BASE_URL)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    if LLM_JSON_MODE:
        body["response_format"] = {"type": "json_object"}

    timeout = httpx.Timeout(40.0)
    t0 = time.perf_counter()

    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)

        # If provider rejects response_format, retry once without it.
        if r.status_code == 400 and LLM_JSON_MODE and "response_format" in body:
            try:
                txt = r.text or ""
            except Exception:
                txt = ""
            if "response_format" in txt or "json_object" in txt or "response format" in txt.lower():
                body.pop("response_format", None)
                r = client.post(url, headers=headers, json=body)

    logger_llm_client.info(
        "Groq status=%s time_ms=%.1f",
        r.status_code,
        (time.perf_counter() - t0) * 1000.0,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")

    data = r.json()
    _set_last_llm_usage(data.get("usage"))
    try:
        if data.get("usage"):
            logger_llm_client.info("Groq usage=%s", json.dumps(data.get("usage"), ensure_ascii=False))
    except Exception:
        pass
    return data["choices"][0]["message"]["content"]