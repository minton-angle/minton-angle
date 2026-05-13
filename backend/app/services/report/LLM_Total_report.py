from __future__ import annotations

import json
import os
import logging
import time

from typing import Any, Dict, Optional

import httpx

try:
    from app.services.report.tools.recommended_youtube_tool import recommended_youtube_tool
except Exception:
    recommended_youtube_tool = None

from app.services.report.retrieval.chroma_retriever import retrieve_coaching_evidence
from app.services.report.retrieval.rag_query_builder import metric_query_text


logger_llm = logging.getLogger("app.llm")

# ------------------------------------------------------------------
# LLM usage (token counts)
# ------------------------------------------------------------------
# OpenAI-compatible responses may include `usage` like:
# {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
# We keep the last call's usage in-memory so `generate_report()` can attach it.
_LAST_LLM_USAGE: Dict[str, Any] = {}

def _set_last_llm_usage(u: Any) -> None:
    global _LAST_LLM_USAGE
    if isinstance(u, dict):
        _LAST_LLM_USAGE = u
    else:
        _LAST_LLM_USAGE = {}

def _get_last_llm_usage() -> Dict[str, Any]:
    return _LAST_LLM_USAGE if isinstance(_LAST_LLM_USAGE, dict) else {}

# ------------------------------------------------------------------
# LLM Provider Settings (Groq / Hugging Face)
# ------------------------------------------------------------------
# env 파일에서 LLM_PROVIDER 값을 읽어서 사용할 LLM API를 결정함:
# - LLM_PROVIDER=groq (디폴트값, Groq OpenAI-compatible endpoints)
# - LLM_PROVIDER=hf  (Hugging Face OpenAI-compatible endpoints, e.g. Inference Endpoints/TGI)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Groq (OpenAI-compatible)
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Hugging Face (OpenAI-compatible)
# Examples:
# - HF_BASE_URL=https://<your-endpoint>/v1
# - HF_API_KEY=hf_...  (or provider-specific token)
# - HF_MODEL=<model name> (some endpoints ignore this; keep for compatibility)
HF_BASE_URL = os.getenv("HF_BASE_URL", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "").strip()

# Shared generation params
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", os.getenv("GROQ_MAX_TOKENS", "1600")))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.8")))


# Some providers do not support response_format=json_object. Keep it optional.
# - LLM_JSON_MODE=1 to request JSON mode when supported (Groq supports it).
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "1").strip() not in ("0", "false", "False")

# Debug: dump raw LLM output on JSON parse error
LLM_DUMP_RAW_ON_ERROR = os.getenv("LLM_DUMP_RAW_ON_ERROR", "0").strip() not in ("0", "false", "False")
LLM_DUMP_RAW_DIR = os.getenv("LLM_DUMP_RAW_DIR", "./snapshots/llm_raw").strip() or "./snapshots/llm_raw"


# ------------------------------------------------------------------
# System Prompt (분석 리포트 톤 고정)
# ------------------------------------------------------------------
def _system_prompt(lang: str) -> str:
    # NOTE: lang is kept for future extensibility; current prompt is Korean-first.
    return """
당신은 배드민턴 동작 개선 AI 코치입니다.

[절대 규칙]
0) `meta.retrieved_coaching`가 제공되면, 각 섹션의 analysis는 retrieved_coaching의 stage/metric과 직접 연결되는
    구체적인 신체 움직임 설명을 반드시 포함하십시오.
   - retrieved_coaching의 문구를 그대로 길게 복붙하지 말고, 핵심 근거를 재서술하여 자연스럽게 반영하십시오.
   - retrieved_coaching가 비어있는 경우에만 일반 코칭 지식으로 작성하십시오.
1) 수치는 반드시 `meta.score_stats`에 있는 값만 사용하십시오.
    - 사용 가능 키: 1_Ready_Total, 2_Rotation_Total, 3_Backswing_Total, 4_Impact_Total, 5_FollowSwing_SuccessRate, total_score
    - 각 Total 키(1~4)는 추가로 아래 정보를 포함할 수 있습니다:
     • sub_stats: { 세부키: { current_mean, prev_mean, delta, direction } }
     • worst_sub, worst_sub_current_mean
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
1-3) 각 섹션(ready/rotation/backswing/impact)에서, Total(요약) 점수와 무관하게 worst_sub_current_mean(가장 낮은 세부 항목 점수)가 90 미만이면, 해당 섹션의 meta.score_stats[””].worst_sub(가장 낮은 세부 항목)을 반드시 1회 이상 언급하여
‘Total은 높아도 어떤 세부가 흔들려 보강이 필요한지’를 구체화하십시오.
   - 단, 세부 점수 수치는 sub_stats의 값만 사용하고 임의 추정 금지.
   - worst_sub_current_mean이 90 이상인 경우에는 worst_sub 언급은 선택입니다.
1-4) meta.movement_reasoning이 제공되면, 이는 weak_metrics를 기반으로 생성된 biomechanical movement hypothesis입니다.
   - 각 섹션의 analysis/fix 작성 시 해당 stage와 연결되는 movement_hypotheses와 retrieval_focus를 우선 참고하십시오.
   - movement_reasoning은 원인 가설이며, score_stats에 없는 수치를 새로 만들면 안 됩니다.
   - confidence가 낮은 hypothesis는 확정 표현 대신 "가능성", "경향", "주의가 필요" 수준으로 표현하십시오.
2) 각 섹션(ready/rotation/backswing/impact/followswing)의 내용은 서로 달라야 합니다. (같은 문장/같은 수치 반복 금지)
3) direction 판정은 입력의 direction 값을 그대로 따르십시오.
   - improved: delta > 0 (점수 상승)
   - worsened: delta < 0 (점수 하락)
   - flat: delta == 0
4) 문구에는 반드시 "이전 횟수 대비" 표현이 포함되어야 합니다.
5) 출력은 반드시 JSON 오브젝트 1개이며, 아래 스키마를 정확히 지키십시오.
   - growth: { direction: improved|worsened|flat, delta_average_score: number, message: string }
   - sections: {
        ready: { title, analysis, fix },
        rotation: { title, analysis, fix },
        backswing: { title, analysis, fix },
        impact: { title, analysis, fix },
        followswing: { title, analysis, fix }
     }
6-1) analysis는 해당 Stage의 "최근 N회 기준 비교 기반 동작 분석"만 작성하는 필드입니다.
     - 정확히 2문장 구조를 유지하십시오.
     - 첫 문장: 이전 횟수 대비 세부 동작 흐름(최소 2개 세부 항목)을 객관적으로 요약하십시오.
     - 두 번째 문장: 해당 변화가 경기력 또는 동작 안정성에 어떤 영향을 주는지 설명하십시오.
     - 점수/델타/평균 수치 직접 언급 금지(숫자 금지).
     - worst_sub_current_mean이 90 미만인 경우,
       해당 worst_sub를 1회 이상 언급하여 '세부 보완 필요' 관점으로 포함하십시오.
6-2) fix는 해당 Stage의 "구체적 교정 방법"만 작성하는 필드입니다.
     - retrieved_coaching을 근거로 사용자가 바로 적용할 수 있는 구체적 교정 동작을 작성하십시오.
     - fix에는 구체적인 교정 방법, 연습 방법, 드릴 지시를 작성하시오.
     - 반드시 신체 부위와 움직임 방향을 포함하십시오.
        - 예: 라켓을 몸 앞쪽에 두고, 손목이 어깨선 아래로 떨어지지 않도록 준비 자세를 유지하는 방식으로 보완할 수 있습니다.
     - 명령형이 아닌 설명형 제안 문장으로 작성하십시오.

9) 각 섹션은 current_mean(점수)에 따라 피드백 목적이 달라야 합니다.
   - current_mean >= 90: "유지/강점 확인" 중심으로 작성합니다.
     단, worst_sub_current_mean이 90 미만인 경우에는 '문제 지적'이 아니라
     '보강/흔들림 방지' 관점으로 worst_sub를 1회 이상 언급할 수 있습니다.
   - 80 <= current_mean < 90: "안정화/흔들림 방지" 중심으로 작성
   - current_mean < 80: "개선 필요" 중심으로 작성
   - 모든 경우에 fix는 분석 요약이 아니라 실제 교정 방향이어야 합니다.
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
        "trend": m.get("trend", {}),
        "score_stats": m.get("score_stats", {}),
        "weak_metrics": m.get("weak_metrics", []),
        "movement_reasoning": m.get("movement_reasoning", {}),
        "retrieved_coaching": m.get("retrieved_coaching", []),
    }

    schema = {
        "growth": {
            "direction": "improved|worsened|flat",
            "delta_average_score": "number",
            "message": "string",
        },
        "sections": {
            "ready": {"title": "준비", "analysis": "string", "fix": "string"},
            "rotation": {"title": "회전", "analysis": "string", "fix": "string"},
            "backswing": {"title": "백스윙", "analysis": "string", "fix": "string"},
            "impact": {"title": "임팩트", "analysis": "string", "fix": "string"},
            "followswing": {"title": "팔로스윙", "analysis": "string", "fix": "string"},
        },
    }

    payload = {
        "meta": safe_meta,
        "schema": schema,
    }

    return (
    "다음 INPUT_JSON의 meta.score_stats, meta.trend, meta.weak_metrics, meta.movement_reasoning, meta.retrieved_coaching을 사용해 "
    "'최근 N회 기준 비교 기반' 점수 리포트를 생성하세요.\n"
    "angles/단일 세션 값은 사용 금지입니다.\n\n"
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

    report_obj.setdefault(
        "growth",
        {"direction": "flat", "delta_average_score": 0.0, "message": "-"}
    )

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
            {"title": title, "analysis": "-", "fix": "-"},
        )
        node.setdefault("analysis", "-")
        node.setdefault("fix", "-")

    return report_obj


# ------------------------------------------------------------------
# LLM API Call (OpenAI-compatible)
# ------------------------------------------------------------------
def _chat_completions_url(base_url: str) -> str:
    """Build a chat-completions URL from a base URL.

    Accepts base_url like:
    - https://api.groq.com/openai/v1
    - https://<hf-endpoint>/v1
    - https://<custom-host>

    Returns: <base_url>/chat/completions (with /v1 preserved if provided)
    """
    b = (base_url or "").rstrip("/")
    if not b:
        return ""
    # If caller provided .../v1 already, we still append /chat/completions
    return f"{b}/chat/completions"


def _call_llm_chat(messages, model: str) -> str:
    """Call the configured provider (Groq or Hugging Face) via OpenAI-compatible chat completions."""

    provider = (LLM_PROVIDER or "groq").strip().lower()

    # Log which provider/model is actually being used
    try:
        effective_model = (
            model
            or (HF_MODEL if provider == "hf" else GROQ_MODEL)
            or "model"
        )
        logger_llm.info(
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

        # Some HF OpenAI-compatible endpoints ignore `model` (fixed endpoint model), but it is required by schema.
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

        logger_llm.info(
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
                logger_llm.info("HF usage=%s", json.dumps(data.get("usage"), ensure_ascii=False))
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

    # Groq supports JSON mode; keep optional.
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

    logger_llm.info(
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
            logger_llm.info("Groq usage=%s", json.dumps(data.get("usage"), ensure_ascii=False))
    except Exception:
        pass
    return data["choices"][0]["message"]["content"]


def _strip_markdown_code_fences(s: str) -> str:
    """Remove surrounding Markdown code fences if present (```json ... ```)."""
    s = (s or "").strip()
    if not s:
        return s

    # Handle inline fenced blocks like: ```json { ... }``` or ``` { ... }```
    if s.startswith("```") and s.endswith("```"):
        inner = s[3:-3].strip()
        # Drop optional language tag at the beginning (e.g. json)
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner

    # Handle multi-line fences:
    # ```json\n{...}\n```
    # ```\n{...}\n```
    if s.startswith("```"):
        parts = s.splitlines()
        if parts:
            first = parts[0].strip()
            # If the first line contains JSON right after ```json, keep the remainder.
            # Example: ```json {"a":1}
            if first.startswith("```") and len(first) > 3:
                rest = first[3:].strip()
                # Remove optional language tag
                if rest.lower().startswith("json"):
                    rest = rest[4:].strip()
                if rest:
                    parts = [rest] + parts[1:]
                else:
                    parts = parts[1:]
            else:
                # Plain ``` on first line
                parts = parts[1:]

        s = "\n".join(parts).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
        return s

    return s


def _extract_first_json_object(s: str) -> str:
    """Best-effort extraction of the first top-level JSON object from text."""
    s = (s or "").strip()
    if not s:
        return s

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    if start < 0:
        return s

    # Bracket matching to find the end of the first object
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()

    # If we couldn't match, return from first '{' onward
    return s[start:].strip()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def generate_report(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
    lang: str = "ko",
    model: str = "",
    system_prompt_override: Optional[str] = None,
    user_prompt_override: Optional[str] = None,
) -> Dict[str, Any]:

    # Upgrade meta with RAG retrieved coaching snippets (optional)
    if meta is not None and not (meta.get("retrieved_coaching") or []):
        try:
            meta["retrieved_coaching"] = retrieve_coaching_evidence(
                meta,
                logger=logger_llm,
            )
            # RAG 검색 결과 로그: 검색 결과 파악 및 디버깅용
            logger_llm.info(
                "RAG injected into meta count=%d",
                len(meta.get("retrieved_coaching") or []),
            )
        except Exception as e:
            logger_llm.warning("RAG retrieve failed err=%s", str(e))
            meta["retrieved_coaching"] = []

    # 최종 prompt 입력 로그(rag on/off는 enrichment 이후 상태 기준)
    try:
        score_stats = (meta or {}).get("score_stats", {})
        logger_llm.info(
            "LLM prompt inputs range=%s score_stats=%s rag=%s",
            (meta or {}).get("range"),
            json.dumps(score_stats, ensure_ascii=False),
            "on" if ((meta or {}).get("retrieved_coaching") or []) else "off",
        )
    except Exception:
        pass

    # 오버라이드 허용: 디버깅/실험용으로 system/user prompt를 완전히 교체할 수 있도록 허용
    system_prompt = system_prompt_override if system_prompt_override is not None else _system_prompt(lang)
    user_prompt = user_prompt_override if user_prompt_override is not None else _user_prompt(angles, meta, lang)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = _call_llm_chat(messages, model)
    # Attach token usage (if provider returns it)
    usage = _get_last_llm_usage()
    logger_llm.info("LLM raw(head)=%s", raw)

    raw_clean = _strip_markdown_code_fences(raw)
    raw_clean = _extract_first_json_object(raw_clean)

    try:
        report_obj = json.loads(raw_clean)
    except Exception as e:
        logger_llm.exception(
            "LLM report JSON parse failed err=%s raw_clean=%s raw=%s",
            str(e),
            raw_clean,
            raw,
        )
        raise

    try:
        report_obj = _normalize_report(report_obj)
    except Exception as e:
        logger_llm.exception(
            "LLM report normalize failed err=%s report_obj=%s",
            str(e),
            json.dumps(report_obj, ensure_ascii=False) if isinstance(report_obj, dict) else str(report_obj),
        )
        raise
    try:
        if recommended_youtube_tool is not None:
            report_obj["recommended_youtube"] = recommended_youtube_tool(meta or {})
        else:
            report_obj.setdefault("recommended_youtube", [])
    except Exception as e:
        logger_llm.warning("recommended_youtube_tool failed err=%s", str(e))
        report_obj.setdefault("recommended_youtube", [])
 
    return report_obj