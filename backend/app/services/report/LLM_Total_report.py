from __future__ import annotations

import json
import os
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx


# -----------------------------------------------------------------------------
# Logging (LLM 레벨)
# -----------------------------------------------------------------------------
logger_llm = logging.getLogger("app.llm")


# -----------------------------------------------------------------------------
# Groq(OpenAI-compatible) Settings
# -----------------------------------------------------------------------------
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# PoC 추천 모델(속도/품질 밸런스): llama-3.1-8b-instant
# 필요시 환경변수로 교체 가능
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# 안전장치: 너무 긴 출력 방지
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "800"))
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.8"))


# -----------------------------------------------------------------------------
# Prompt Builder
# -----------------------------------------------------------------------------
def _system_prompt(lang: str) -> str:
    if lang.lower().startswith("en"):
        return (
            "You are a biomechanics coaching assistant. "
            "Given keyframe error angles (KF1~KF3, in degrees) relative to a reference posture, "
            "write a concise, actionable posture feedback report. "
            "Return ONLY valid JSON. Do not include markdown."
        )

    # default: Korean
    return (
        "당신은 배드민턴 스윙 자세의 '성장(개선)'을 알려주는 코칭 어시스턴트입니다. "
        "입력은 키프레임(KF1~KF3)의 오차각도(도 단위)와 meta.insights(성장/정체/편차 지표)입니다. "
        "반드시 아래의 최상위 JSON 키를 정확히 사용해 반환하세요(키 이름 변경 금지): summary, growth, actions, today_checklist. JSON 외 텍스트는 절대 포함하지 마세요. 동작 라벨은 반드시 '백스윙 동작', '임팩트 동작', '팔로스루 동작'으로 작성하세요."
    )


def _user_prompt(angles: Dict[str, float], meta: Optional[Dict[str, Any]], lang: str) -> str:
    payload = {
        "angles": angles,
        "meta": meta or {},
        "schema": {
            "summary": "string (전체 성장 요약 한 줄)",
            "growth": {
                "direction": "improved|worsened|flat",
                "delta_mean_abs_kf_error": "number",
                "message": "string (이전 대비 변화 설명)"
            },
            "actions": {
                "kf1": {
                    "title": "백스윙 동작",
                    "problem_one": "string (가장 핵심 문제 1개)",
                    "fix_two": "string[] (구체적 교정 방법 2개)"
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
            "today_checklist": "string[] (오늘 바로 실천할 3개)"
        },
    }

    if lang.lower().startswith("en"):
        return (
            "Analyze the following keyframe error angles (KF1~KF3) and generate a JSON report. "
            "Make fixes practical and measurable.\n\n"
            f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
        )

    return (
        "다음 입력으로 '성장 리포트'를 생성하세요. 반드시 schema의 최상위 키를 그대로 사용해 JSON만 반환하세요. "
        "wins는 1~3개, top_issues는 1~3개, quick_checklist는 3~5개로 작성하세요.\n\n"
        f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
    )


# -----------------------------------------------------------------------------
# Report Normalization Helpers
# -----------------------------------------------------------------------------

def _kf_label_to_key(label: str) -> Optional[str]:
    s = str(label or "")
    if "KF1" in s:
        return "kf1_error"
    if "KF2" in s:
        return "kf2_error"
    if "KF3" in s:
        return "kf3_error"
    return None


def _ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else ([] if x is None else [x])


def _normalize_report(report_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report_obj, dict):
        return {}

    report_obj.setdefault("summary", "-")
    report_obj.setdefault("growth", {
        "direction": "flat",
        "delta_mean_abs_kf_error": 0.0,
        "message": "-"
    })
    report_obj.setdefault("actions", {
        "kf1": {"title": "백스윙 동작", "problem_one": "-", "fix_two": []},
        "kf2": {"title": "임팩트 동작", "problem_one": "-", "fix_two": []},
        "kf3": {"title": "팔로스루 동작", "problem_one": "-", "fix_two": []},
    })
    report_obj.setdefault("today_checklist", [])

    return report_obj


# -----------------------------------------------------------------------------
# Groq API Call (OpenAI-compatible Chat Completions)
# -----------------------------------------------------------------------------

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

    # httpx timeout: connect/read 모두 제한
    timeout = httpx.Timeout(connect=10.0, read=40.0, write=10.0, pool=10.0)

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    logger_llm.info("Groq chat completion status=%s time_ms=%.1f model=%s", r.status_code, dt_ms, model)

    if r.status_code >= 400:
        logger_llm.error("Groq API error body(head)=%s", r.text[:500])
        # Groq는 429(rate limit) 등을 반환할 수 있음
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected Groq response shape: {data}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_report(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
    lang: str = "ko",
    model: str = GROQ_MODEL,
) -> Dict[str, Any]:
    """관절 오차각도 JSON을 받아 리포트(JSON)를 반환합니다."""

    messages = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": _user_prompt(angles, meta, lang)},
    ]

    raw = _call_groq_chat(messages=messages, model=model)
    logger_llm.info("LLM raw(head)=%s", raw)

    # LLM이 JSON만 반환하도록 요구하지만, 안전하게 파싱
    try:
        report_obj = json.loads(raw)
        report_obj = _normalize_report(report_obj)
        logger_llm.info(
            "LLM parsed ok severity=%s keys=%s",
            report_obj.get("overall_severity"),
            list(report_obj.keys()),
        )
        logger_llm.info("LLM report=%s", json.dumps(report_obj, ensure_ascii=False))
    except json.JSONDecodeError:
        logger_llm.warning("LLM JSON decode failed. raw(head)=%s", raw[:800])

        # 마지막 안전장치: JSON 부분만 잘라서 재시도(간단)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"LLM did not return JSON. raw={raw}")
        report_obj = json.loads(raw[start : end + 1])
        report_obj = _normalize_report(report_obj)

        logger_llm.info(
            "LLM parsed(ok after slice) severity=%s keys=%s",
            report_obj.get("overall_severity"),
            list(report_obj.keys()),
        )
    # 서버에서 공통 필드 추가
    report_obj.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    report_obj.setdefault("model", model)

    return report_obj
