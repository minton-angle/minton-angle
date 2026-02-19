from __future__ import annotations

import json
import os
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

logger_llm = logging.getLogger("app.llm")

# ------------------------------------------------------------------
# Groq Settings
# ------------------------------------------------------------------
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "800"))
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.4"))


# ------------------------------------------------------------------
# System Prompt (분석 리포트 톤 고정)
# ------------------------------------------------------------------
def _system_prompt(lang: str) -> str:
    return (
        "당신은 배드민턴 스윙의 '성장 분석 리포트'를 작성하는 데이터 분석가입니다. "
        "코칭하거나 훈련을 지시하지 마세요. "
        "데이터 기반으로 변화와 패턴을 설명하세요. "
        "입력은 키프레임(KF1~KF3)의 오차각도와 meta.insights입니다. "
        "반드시 JSON만 반환하세요. "
        "최상위 키는 summary, growth, actions, today_checklist 입니다. "
        "동작 라벨은 반드시 '백스윙 동작', '임팩트 동작', '팔로스루 동작'으로 작성하세요. "
        "각 동작의 내용은 서로 달라야 합니다. "
        "problem_one에는 해당 오차각도 숫자(예: 0.12°)를 반드시 포함하세요. "
        "fix_two는 지시형 문장 금지이며, 분석 포인트 2개로 작성하세요. "
        "절대 '~하세요' 또는 '~하지 마세요' 표현을 사용하지 마세요."
    )


# ------------------------------------------------------------------
# User Prompt
# ------------------------------------------------------------------
def _user_prompt(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]],
    lang: str
) -> str:

    payload = {
        "angles": angles,
        "meta": meta or {},
        "schema": {
            "summary": "string",
            "growth": {
                "direction": "improved|worsened|flat",
                "delta_mean_abs_kf_error": "number",
                "message": "string"
            },
            "actions": {
                "kf1": {
                    "title": "백스윙 동작",
                    "problem_one": "string",
                    "fix_two": "string[] (주의해야 할 분석 포인트 2개)"
                },
                "kf2": {
                    "title": "임팩트 동작",
                    "problem_one": "string",
                    "fix_two": "string[]"
                },
                "kf3": {
                    "title": "팔로스루 동작",
                    "problem_one": "string",
                    "fix_two": "string[]"
                }
            },
            "today_checklist": "string[]"
        }
    }

    return (
        "다음 입력으로 성장 분석 리포트를 생성하세요.\n"
        "규칙:\n"
        "1) 세 동작의 분석 내용은 서로 달라야 합니다.\n"
        "2) problem_one에는 해당 오차각도 값(°)을 반드시 포함하세요.\n"
        "3) fix_two는 분석 포인트 2개이며 지시형 문장 금지.\n"
        "4) today_checklist는 3개 항목으로 작성하세요.\n\n"
        f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
    )


# ------------------------------------------------------------------
# Normalize
# ------------------------------------------------------------------
def _ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else ([] if x is None else [x])


def _normalize_report(report_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report_obj, dict):
        return {}

    report_obj.setdefault("summary", "-")
    report_obj.setdefault(
        "growth",
        {"direction": "flat", "delta_mean_abs_kf_error": 0.0, "message": "-"}
    )

    report_obj.setdefault("actions", {})
    for k, title in [
        ("kf1", "백스윙 동작"),
        ("kf2", "임팩트 동작"),
        ("kf3", "팔로스루 동작")
    ]:
        node = report_obj["actions"].setdefault(
            k,
            {"title": title, "problem_one": "-", "fix_two": []}
        )
        node["fix_two"] = _ensure_list(node.get("fix_two"))

    report_obj.setdefault("today_checklist", [])
    return report_obj


# ------------------------------------------------------------------
# Groq API Call
# ------------------------------------------------------------------
def _call_groq_chat(messages, model: str = GROQ_MODEL) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    timeout = httpx.Timeout(40.0)
    t0 = time.perf_counter()

    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)

    logger_llm.info(
        "Groq status=%s time_ms=%.1f",
        r.status_code,
        (time.perf_counter() - t0) * 1000.0,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def generate_report(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
    lang: str = "ko",
    model: str = GROQ_MODEL,
) -> Dict[str, Any]:

    messages = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": _user_prompt(angles, meta, lang)},
    ]

    raw = _call_groq_chat(messages, model)
    logger_llm.info("LLM raw(head)=%s", raw)

    try:
        report_obj = json.loads(raw)
        report_obj = _normalize_report(report_obj)
    except Exception:
        raise RuntimeError(f"Invalid JSON from LLM: {raw}")

    report_obj.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    report_obj.setdefault("model", model)

    logger_llm.info("LLM report=%s", json.dumps(report_obj, ensure_ascii=False))
    return report_obj