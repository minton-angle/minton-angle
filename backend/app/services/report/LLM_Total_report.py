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
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.8"))


# ------------------------------------------------------------------
# System Prompt (분석 리포트 톤 고정)
# ------------------------------------------------------------------
def _system_prompt(lang: str) -> str:
    # NOTE: lang is kept for future extensibility; current prompt is Korean-first.
    return """
당신은 배드민턴 동작 분석 AI 코치입니다.

[절대 규칙]
1) 수치는 반드시 `meta.score_stats`에 있는 값만 사용하십시오.
   - 사용 가능 키: 1_Ready_Total, 2_Rotation_Total, 3_Backswing_Total, 4_Impact_Total, 5_FollowSwing_SuccessRate, Average_Score
   - 5_FollowSwing_SuccessRate는 성공률 점수(0~100)이며, 추가 필드 false_rate_current/false_rate_prev/risk_level을 함께 제공받을 수 있습니다.
1-1) 팔로스윙 섹션에서 risk_level이 improve 또는 risk인 경우에는 '부상 예방/주의' 관찰 포인트를 반드시 1개 이상 포함하십시오.
   - 단, 의학적 진단/확정 표현 금지(예: "어깨 충돌이다", "부상이다").
   - 허용 톤(관찰/주의): "부담이 커질 수 있어요", "통증이 있으면 강도를 낮출 필요가 있어요", "지속되면 전문가 상담을 고려할 수 있어요".
   - risk_level=ok인 경우에는 부상 위험 언급을 하지 마십시오.
   - 각 키별 사용 가능 값: current_mean, prev_mean, delta, direction
   - 사용 금지: angles(단일 세션 값), raw angle, 임의로 만든 수치/예시 수치
1-2) 각 섹션은 서로 다른 신체/동작 관찰 영역을 다뤄야 합니다.
   - ready(준비): 스탠스, 상체 높이, 팔 위치, 준비 타이밍 중 최소 2개 포함
   - rotation(회전): 골반 회전, 체간 분리, 중심축 유지, 하체-상체 연결 중 최소 2개 포함
   - backswing(백스윙): 팔꿈치 위치, 손목 각도, 라켓 준비 경로 중 최소 2개 포함
   - impact(임팩트): 타점 위치, 라켓 각도, 임팩트 순간 체중 이동 중 최소 2개 포함
   - followswing(팔로스윙): 스윙 마무리 높이, 어깨/팔꿈치 부담 여부, 과회전 여부 중 최소 2개 포함
2) 각 섹션(ready/rotation/backswing/impact/followswing)의 내용은 서로 달라야 합니다. (같은 문장/같은 수치 반복 금지)
3) direction 판정은 입력의 direction 값을 그대로 따르십시오.
   - improved: delta > 0 (점수 상승)
   - worsened: delta < 0 (점수 하락)
   - flat: delta == 0
4) 문구에는 반드시 "이전 기간 대비" 표현이 포함되어야 합니다.
5) 출력은 반드시 JSON 오브젝트 1개이며, 아래 스키마를 정확히 지키십시오.
   - summary: string
   - growth: { direction: improved|worsened|flat, delta_average_score: number, message: string }
   - sections: {
       ready: { title, change_one, focus_two },
       rotation: { title, change_one, focus_two },
       backswing: { title, change_one, focus_two },
       impact: { title, change_one, focus_two },
       followswing: { title, change_one, focus_two }
     }
   - today_checklist: string[]
6) change_one에는 아래 3개 값을 반드시 포함하십시오(점수 표기, 소수점 2자리):
   - current_mean, prev_mean, delta
7) focus_two는 "관찰 포인트" 2개를 배열로 작성하십시오.
   - 지시형(해라/하세요/줄이세요/올리세요/하세요 등) 문장 금지.
   - 형태 예시: "~이 유지되는지", "~이 과하게 되지 않는지", "통증/불편감이 동반되는지" 처럼 관찰 문장으로 작성.
8) today_checklist는 정확히 3개 항목의 배열로 작성하십시오.
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
        "score_stats": m.get("score_stats", {}),
        "insights": m.get("insights", {}),
    }

    schema = {
        "summary": "string",
        "growth": {
            "direction": "improved|worsened|flat",
            "delta_average_score": "number",
            "message": "string",
        },
        "sections": {
            "ready": {"title": "준비", "change_one": "string", "focus_two": "string[]"},
            "rotation": {"title": "회전", "change_one": "string", "focus_two": "string[]"},
            "backswing": {"title": "백스윙", "change_one": "string", "focus_two": "string[]"},
            "impact": {"title": "임팩트", "change_one": "string", "focus_two": "string[]"},
            "followswing": {"title": "팔로스윙", "change_one": "string", "focus_two": "string[]"},
        },
        "today_checklist": "string[]",
    }

    payload = {
        "meta": safe_meta,
        "schema": schema,
    }

    return (
        "다음 입력(meta.score_stats, meta.trend)을 사용해 '기간 비교 기반' 점수 리포트를 생성하세요.\n"
        "중요: angles/단일 세션 값은 사용 금지이며 입력에도 제공되지 않습니다.\n"
        "작성 규칙:\n"
        "1) ready/rotation/backswing/impact/followswing 분석 내용은 서로 달라야 합니다.\n"
        "2) 각 섹션의 change_one에는 current_mean, prev_mean, delta(모두 점수, 소수점 2자리) 3개를 반드시 포함하세요.\n"
        "3) focus_two는 2개 항목의 배열이며, 지시형(해라/하세요) 문장 금지.\n"
        "4) today_checklist는 정확히 3개 항목의 배열로 작성하세요.\n"
        "5) 숫자는 meta.score_stats 값만 사용하세요.\n\n"
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

    # New score-based sections
    report_obj.setdefault("sections", {})
    for key, title in [
        ("ready", "준비"),
        ("rotation", "회전"),
        ("backswing", "백스윙"),
        ("impact", "임팩트"),
        ("followswing", "팔로스윙"),
    ]:
        node = report_obj["sections"].setdefault(
            key,
            {"title": title, "change_one": "-", "focus_two": []},
        )
        node["focus_two"] = _ensure_list(node.get("focus_two"))

    # Backward-compat: map score sections -> legacy actions(kf1/kf2/kf3) if actions missing
    if not report_obj.get("actions"):
        report_obj["actions"] = {
            "kf1": {
                "title": "백스윙 동작",
                "problem_one": report_obj["sections"]["backswing"].get("change_one", "-"),
                "fix_two": report_obj["sections"]["backswing"].get("focus_two", []),
            },
            "kf2": {
                "title": "임팩트 동작",
                "problem_one": report_obj["sections"]["impact"].get("change_one", "-"),
                "fix_two": report_obj["sections"]["impact"].get("focus_two", []),
            },
            "kf3": {
                "title": "팔로스루 동작",
                "problem_one": report_obj["sections"]["followswing"].get("change_one", "-"),
                "fix_two": report_obj["sections"]["followswing"].get("focus_two", []),
            },
        }

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
        score_stats = (meta or {}).get("score_stats", {})
        logger_llm.info(
            "LLM prompt inputs range=%s score_stats=%s",
            (meta or {}).get("range"),
            json.dumps(score_stats, ensure_ascii=False),
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