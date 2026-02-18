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
        "반드시 아래의 최상위 JSON 키를 정확히 사용해 반환하세요(키 이름 변경/한글 키 금지): "
        "summary, overall_severity, growth, plateau, consistency, wins, top_issues, quick_checklist. "
        "overall_severity 값은 low|medium|high 중 하나입니다. "
        "JSON 외 텍스트는 절대 포함하지 마세요."
    )


def _user_prompt(angles: Dict[str, float], meta: Optional[Dict[str, Any]], lang: str) -> str:
    payload = {
        "angles": angles,
        "meta": meta or {},
        "schema": {
            "summary": "string",
            "overall_severity": "low|medium|high",
            "growth": {"direction": "improved|worsened|flat", "delta_mean_abs_kf_error": "number", "message": "string"},
            "plateau": {"kf": "kf1_error|kf2_error|kf3_error|null", "message": "string", "why": "string", "fix": "string[]"},
            "consistency": {"kf": "kf1_error|kf2_error|kf3_error|null", "message": "string", "how_to_practice": "string[]"},
            "wins": "{kf:string,message:string}[]",
            "top_issues": "{joint:string,error_deg:number,interpretation:string,why_it_matters:string,fix:string[]}[]",
            "quick_checklist": "string[]",
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
    """Best-effort normalization.

    Frontend expects:
      summary, overall_severity, growth, plateau, consistency, wins, top_issues, quick_checklist

    Some model outputs may come as Korean/nested, e.g. {"성장 리포트": {...}}.
    """
    if not isinstance(report_obj, dict):
        return {}

    # If already correct shape, keep.
    expected_keys = {"summary", "overall_severity", "growth", "plateau", "consistency", "wins", "top_issues", "quick_checklist"}
    if expected_keys.issubset(set(report_obj.keys())):
        return report_obj

    inner = None
    if "성장 리포트" in report_obj and isinstance(report_obj.get("성장 리포트"), dict):
        inner = report_obj.get("성장 리포트")

    if not inner:
        # Try a minimal coercion (avoid breaking UI)
        report_obj.setdefault("summary", report_obj.get("요약") or "-")
        report_obj.setdefault("overall_severity", "low")
        report_obj.setdefault("growth", {"direction": "flat", "delta_mean_abs_kf_error": 0.0, "message": "-"})
        report_obj.setdefault("plateau", {"kf": None, "message": "-", "why": "", "fix": []})
        report_obj.setdefault("consistency", {"kf": None, "message": "-", "how_to_practice": []})
        report_obj.setdefault("wins", [])
        report_obj.setdefault("top_issues", [])
        report_obj.setdefault("quick_checklist", [])
        return report_obj

    # Map Korean/nested structure
    summary = inner.get("리포트 요약") or inner.get("요약") or "-"

    # severity: if 개선 정도/좋음 etc, default low.
    sev = "low"
    sev_raw = str(inner.get("개선 정도") or "").lower()
    if any(k in sev_raw for k in ["나쁨", "bad", "high"]):
        sev = "high"
    elif any(k in sev_raw for k in ["보통", "medium"]):
        sev = "medium"

    # growth
    growth_obj = {
        "direction": "improved" if "좋" in str(inner.get("개선 정도") or "") else "flat",
        "delta_mean_abs_kf_error": float(inner.get("delta_mean_abs_kf_error") or 0.0),
        "message": str(inner.get("리포트 요약") or "-")[:300],
    }

    # plateau
    plateau_in = inner.get("plateau")
    plateau = {"kf": None, "message": "-", "why": "", "fix": []}
    if isinstance(plateau_in, dict) and plateau_in:
        # take first key
        k0 = next(iter(plateau_in.keys()))
        kf = _kf_label_to_key(k0)
        v0 = plateau_in.get(k0)
        if isinstance(v0, dict):
            prev_m = v0.get("previous_mean")
            rec_m = v0.get("recent_mean")
            delta = v0.get("delta")
            plateau = {
                "kf": kf,
                "message": f"최근 구간에서 {k0} 변화량(Δ) {delta}° (이전 평균 {prev_m}°, 최근 평균 {rec_m}°)",
                "why": "오차 감소가 정체되면 동일 구간에서 반복적으로 폼이 흔들릴 수 있습니다.",
                "fix": [],
            }

    # consistency
    cons_in = inner.get("consistency")
    consistency = {"kf": None, "message": "-", "how_to_practice": []}
    if isinstance(cons_in, dict) and cons_in:
        k0 = next(iter(cons_in.keys()))
        kf = _kf_label_to_key(k0)
        v0 = cons_in.get(k0)
        std = v0.get("std") if isinstance(v0, dict) else None
        consistency = {
            "kf": kf,
            "message": f"{k0}의 편차(std)가 상대적으로 큽니다 (std={std}).",
            "how_to_practice": [],
        }

    # wins
    wins_in = inner.get("wins")
    wins: list[dict] = []
    for item in _ensure_list(wins_in):
        if isinstance(item, dict) and item:
            kk = next(iter(item.keys()))
            kf = _kf_label_to_key(kk)
            vv = item.get(kk)
            if isinstance(vv, dict):
                wins.append({"kf": kf or kk, "message": f"개선(Δ) {vv.get('delta')}° (전 {vv.get('first_half')}°, 후 {vv.get('second_half')}°)"})
            else:
                wins.append({"kf": kf or kk, "message": "개선되었습니다."})

    normalized = {
        "summary": str(summary),
        "overall_severity": sev,
        "growth": growth_obj,
        "plateau": plateau,
        "consistency": consistency,
        "wins": wins,
        "top_issues": [],
        "quick_checklist": [],
    }
    return normalized


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
