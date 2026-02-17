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
        "반드시 아래 형식과 동일한 JSON 구조로만 응답하세요.\n"
        "키 이름을 절대 변경하지 마세요.\n"
        "추가 필드도 만들지 마세요.\n\n"
        "{\n"
        '  "overall": "string",\n'
        '  "metric_feedback": {\n'
        '    "hip_rotation": {"message":"string","tip":"string","next_goal":"string","score":0},\n'
        '    "impact_height": {...},\n'
        '    "elbow_extension": {...},\n'
        '    "follow_through": {...}\n'
        "  }\n"
        "}\n"
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
        "ui_metrics": {
            "hip_rotation": {
                "score_key": "shoulder_rotation",   # 또는 실제 hip score가 있다면 그 key
                "dom_id_feedback": "hip-feedback",
                "dom_id_value": "hip-value"
            },
            "impact_height": {
                "score_key": "hit_position",        # (임시 매핑) 실제 height score가 있다면 그 key로
                "dom_id_feedback": "height-feedback",
                "dom_id_value": "height-value"
            },
            "elbow_extension": {
                "score_key": "elbow_height",
                "dom_id_feedback": "elbow-feedback",
                "dom_id_value": "elbow-value"
            },
            "follow_through": {
                "score_key": "follow_through",
                "dom_id_feedback": "followthrough-feedback",
                "dom_id_value": "followthrough-value"
            }
        },
        "constraints": {
            "output_format": {
                "overall": "string",
                "metric_feedback": {
                    "hip_rotation": {"message": "string", "tip": "string", "next_goal": "string", "score": "number"},
                    "impact_height": {"message": "string", "tip": "string", "next_goal": "string", "score": "number"},
                    "elbow_extension": {"message": "string", "tip": "string", "next_goal": "string", "score": "number"},
                    "follow_through": {"message": "string", "tip": "string", "next_goal": "string", "score": "number"},
                },
                "strengths": ["string"],
                "improvements": ["string"],
                "next_goals": ["string"],
                "drills": [{"title":"string","how":["string"],"target_metric":"string","target_score":"number"}]
            }
        }
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

def _build_metric_message(metric_name: str, score: float, lang: str) -> str:
    """점수 기반 fallback 코멘트(LLM 출력이 스키마를 어겼을 때 프론트 매핑을 보장)."""
    s = float(score) if score is not None else 0.0
    if lang.lower().startswith("en"):
        if s >= 85:
            return f"{metric_name}: Great. Keep the current form and repeat with consistency."
        if s >= 70:
            return f"{metric_name}: Decent. Reduce variability and focus on one clear cue."
        return f"{metric_name}: Needs work. Slow down and fix the main position before adding speed."

    if s >= 85:
        return f"{metric_name}: 좋습니다. 지금 폼을 유지하면서 반복 일관성을 높이세요."
    if s >= 70:
        return f"{metric_name}: 무난합니다. 흔들림을 줄이고 한 가지 큐에 집중해보세요."
    return f"{metric_name}: 개선이 필요합니다. 속도를 낮추고 자세 포인트를 먼저 고정하세요."


def _normalize_swing_feedback(obj: Dict[str, Any], scores: Dict[str, float], lang: str) -> Dict[str, Any]:
    """LLM이 제멋대로 키를 뱉어도 프론트가 기대하는 스키마로 보정."""
    if not isinstance(obj, dict):
        obj = {}

    # 1) 키 공백 제거(예: " 코칭 리포트 ")
    cleaned: Dict[str, Any] = {}
    for k, v in obj.items():
        kk = k.strip() if isinstance(k, str) else k
        cleaned[kk] = v
    obj = cleaned

    # 2) 상위가 "코칭 리포트" 구조인 경우 펼치기
    report = None
    for key in ("코칭 리포트", "코칭리포트", "report", "Report"):
        if key in obj and isinstance(obj[key], dict):
            report = obj[key]
            break
    if report is not None:
        rep2: Dict[str, Any] = {}
        for k, v in report.items():
            kk = k.strip() if isinstance(k, str) else k
            rep2[kk] = v
        report = rep2

    # 3) overall 만들기
    overall = obj.get("overall")
    if (not overall) and report:
        strengths = report.get("강점") or report.get("Strengths")
        improvements = report.get("개선점") or report.get("Improvements")
        if isinstance(strengths, list) and isinstance(improvements, list):
            if lang.lower().startswith("en"):
                overall = f"Strengths: {', '.join(strengths[:2])}. Improve: {', '.join(improvements[:2])}."
            else:
                overall = f"강점: {', '.join(strengths[:2])}. 개선: {', '.join(improvements[:2])}."

    if not isinstance(overall, str) or not overall.strip():
        overall = (
            "스윙 분석 결과를 기반으로 교정 포인트를 정리했습니다."
            if not lang.lower().startswith("en")
            else "Here are actionable coaching points based on your swing analysis."
        )

    # 4) metric_feedback 보장
    mf = obj.get("metric_feedback")
    if not isinstance(mf, dict):
        mf = {}

    # 점수 키 매핑(현재 서비스 스코어 키 기준)
    hip_score = scores.get("shoulder_rotation")
    height_score = scores.get("hit_position")
    elbow_score = scores.get("elbow_height")
    follow_score = scores.get("follow_through")

    def ensure_metric(key: str, label: str, score_val: Optional[float]):
        cur = mf.get(key)
        if not isinstance(cur, dict):
            cur = {}
        if "score" not in cur or cur.get("score") in (None, "number"):
            cur["score"] = float(score_val) if score_val is not None else 0.0
        if not isinstance(cur.get("message"), str) or not cur.get("message").strip():
            cur["message"] = _build_metric_message(label, cur["score"], lang)
        cur.setdefault("tip", "")
        cur.setdefault("next_goal", "")
        mf[key] = cur

    ensure_metric("hip_rotation", "골반 회전", hip_score)
    ensure_metric("impact_height", "임팩트 높이", height_score)
    ensure_metric("elbow_extension", "팔꿈치 신전", elbow_score)
    ensure_metric("follow_through", "팔로우스루", follow_score)

    # 5) 최종 스키마 병합
    out = dict(obj)
    out["overall"] = overall
    out["metric_feedback"] = mf

    # 선택 필드들도 리스트 형태 보장
    if not isinstance(out.get("strengths"), list):
        out["strengths"] = []
    if not isinstance(out.get("improvements"), list):
        out["improvements"] = []
    if not isinstance(out.get("next_goals"), list):
        out["next_goals"] = []
    if not isinstance(out.get("drills"), list):
        out["drills"] = []

    return out
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

    obj = _normalize_swing_feedback(obj=obj, scores=scores, lang=lang)

    obj.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    obj.setdefault("model", model)
    obj.setdefault("kind", "swing_detail")
    obj.setdefault("details", scores)

    return obj