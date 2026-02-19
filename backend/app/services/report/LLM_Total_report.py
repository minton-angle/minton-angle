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
    # NOTE: lang is kept for future extensibility; current prompt is Korean-first.
    return """
당신은 배드민턴 동작 분석 AI 코치입니다.

[절대 규칙]
1) 수치는 반드시 `meta.kf_stats`에 있는 값만 사용하십시오.
   - 사용 가능: meta.kf_stats.kf1_error|kf2_error|kf3_error 의 current_mean, prev_mean, delta, direction
   - 사용 금지: angles(최신 단일 세션 값), raw angle, 임의로 만든 수치/예시 수치
2) 각 동작(kf1/kf2/kf3)의 내용은 서로 달라야 합니다. (같은 문장/같은 수치 반복 금지)
3) direction 판정:
   - delta < 0: improved
   - delta > 0: worsened
   - delta == 0: flat
4) 문구에는 반드시 "이전 기간 대비" 표현이 포함되어야 합니다.
5) 출력은 반드시 JSON 오브젝트 1개이며, 아래 스키마를 정확히 지키십시오.
   - summary: string
   - growth: { direction: improved|worsened|flat, delta_mean_abs_kf_error: number, message: string }
   - actions: { kf1: {title, problem_one, fix_two}, kf2: {...}, kf3: {...} }
   - today_checklist: string[]
6) problem_one에는 아래 3개 값을 반드시 포함하십시오(° 표기 포함, 소수점 2자리 권장):
   - current_mean, prev_mean, delta
7) fix_two는 "주의해야 할 분석 포인트" 2개를 배열로 작성하십시오.
   - 지시형(해라/하세요) 문장 금지, 관찰/포인트 형태로 작성
""".strip()


# ------------------------------------------------------------------
# User Prompt
# ------------------------------------------------------------------
def _user_prompt(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]],
    lang: str,
) -> str:
    m = meta or {}

    # LLM이 반드시 써야 하는 값만 제공(angles는 제공하지 않음: 최신 1건 고정/0.1° 앵커링 방지)
    safe_meta = {
        "post_idx": m.get("post_idx"),
        "range": m.get("range"),
        "summary": m.get("summary", {}),
        "trend": m.get("trend", {}),
        "kf_stats": m.get("kf_stats", {}),
        "insights": m.get("insights", {}),
    }

    schema = {
        "summary": "string",
        "growth": {
            "direction": "improved|worsened|flat",
            "delta_mean_abs_kf_error": "number",
            "message": "string",
        },
        "actions": {
            "kf1": {
                "title": "백스윙 동작",
                "problem_one": "string",
                "fix_two": "string[]",
            },
            "kf2": {
                "title": "임팩트 동작",
                "problem_one": "string",
                "fix_two": "string[]",
            },
            "kf3": {
                "title": "팔로스루 동작",
                "problem_one": "string",
                "fix_two": "string[]",
            },
        },
        "today_checklist": "string[]",
    }

    payload = {
        "meta": safe_meta,
        "schema": schema,
    }

    return (
        "다음 입력(meta.kf_stats, meta.trend)을 사용해 '기간 비교 기반' 성장 분석 리포트를 생성하세요.\n"
        "중요: angles/단일 세션 값은 사용 금지이며 입력에도 제공되지 않습니다.\n"
        "작성 규칙:\n"
        "1) kf1/kf2/kf3 분석 내용은 서로 달라야 합니다.\n"
        "2) 각 kf의 problem_one에는 current_mean, prev_mean, delta(모두 ° 포함) 3개를 반드시 포함하세요.\n"
        "3) fix_two는 2개 항목의 배열이며, 지시형(해라/하세요) 문장 금지.\n"
        "4) today_checklist는 정확히 3개 항목의 배열로 작성하세요.\n"
        "5) 숫자는 meta.kf_stats 값만 사용하세요.\n\n"
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

    try:
        kf_stats = (meta or {}).get("kf_stats", {})
        logger_llm.info(
            "LLM prompt inputs range=%s kf_stats=%s",
            (meta or {}).get("range"),
            json.dumps(kf_stats, ensure_ascii=False),
        )
    except Exception:
        pass

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