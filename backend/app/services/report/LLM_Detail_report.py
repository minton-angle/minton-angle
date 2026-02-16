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
# Prompt Builder: 스윙 상세 피드백용 Prompt/함수
# System Prompt: 항상 고정된 역할을 부여 + 출력 형식 제약
# User Prompt: 실제 점수/키프레임 데이터를 JSON으로 전달 + 분석 요청 (영/한)
# -----------------------------------------------------------------------------

def _system_prompt_swing(lang: str) -> str: # 언어별 시스템 프롬프트
    if lang.lower().startswith("en"): # 소문자로 처리 하여
        return (
            "You are a badminton swing coaching assistant. "
            "Given per-metric swing scores (0~100) and keyframe indices (KF1~KF3), "
            "write concise, actionable coaching feedback. "
            "Return ONLY valid JSON. Do not include markdown."
        )

    return (
        "당신은 배드민턴 스윙 코칭 어시스턴트입니다. "
        "스윙 진단 점수(0~100)와 키프레임(KF1~KF3) 정보를 입력으로 받아, "
        "사용자가 바로 따라할 수 있는 간결하고 실행 가능한 코칭 피드백을 작성하세요. "
        "반드시 JSON만 반환하세요. 마크다운/설명 문장/코드블록을 섞지 마세요."
    )

def _user_prompt_swing(
    scores: Dict[str, float],
    meta: Optional[Dict[str, Any]],
    lang: str,
    kf: Optional[Dict[str, int]] = None,
) -> str:
    # 점수/키프레임/메타 정보를 JSON으로 묶어서 전달 (LLM이 쉽게 파싱 가능하도록)
    payload = {
        "scores": scores,
        "keyframes": kf or {},
        "meta": meta or {},
        "constraints": {
            "score_range": "0~100 (higher is better)",
            "output_format": {
                "overall": "string",
                "details": {"metric": "number"},
                "strengths": ["string"],
                "improvements": ["string"],
                "next_goals": ["string"],
                "drills": [
                    {
                        "title": "string",
                        "how": ["string"],
                        "target_metric": "string",
                        "target_score": "number",
                    }
                ],
            },
        },
    }

    if lang.lower().startswith("en"):
        return (
            "Analyze the following badminton swing scores and generate a JSON coaching report. "
            "Make fixes practical and measurable.\n\n"
            f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
        )

    return (
        "다음 배드민턴 스윙 점수를 분석해 코칭 리포트를 JSON으로 생성하세요. "
        "교정 방법은 구체적이고 측정 가능하게 작성하세요.\n\n"
        f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
    )


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

def generate_swing_detail_feedback(
    scores: Dict[str, float],
    kf1: int,
    kf2: int,
    kf3: int,
    meta: Optional[Dict[str, Any]] = None,
    lang: str = "ko",
    model: str = GROQ_MODEL,
) -> Dict[str, Any]:
    """스윙 점수(0~100) + 키프레임 정보를 받아 LLM 코칭(JSON)을 반환합니다."""
    messages = [
        {"role": "system", "content": _system_prompt_swing(lang)},
        {
            "role": "user",
            "content": _user_prompt_swing(
                scores=scores,
                meta=meta,
                lang=lang,
                kf={"kf1": int(kf1), "kf2": int(kf2), "kf3": int(kf3)},
            ),
        },
    ]

    raw = _call_groq_chat(messages=messages, model=model)
    logger_llm.info("LLM swing raw(head)=%s", raw[:800])

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger_llm.warning("LLM swing JSON decode failed. raw(head)=%s", raw[:800])
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"LLM did not return JSON. raw={raw}")
        obj = json.loads(raw[start : end + 1])

    obj.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    obj.setdefault("model", model)
    obj.setdefault("kind", "swing_detail")
    obj.setdefault("details", scores)

    return obj