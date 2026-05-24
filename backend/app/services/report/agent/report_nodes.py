from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.services.report.agent.state import ReportAgentState
from app.services.report.llm.client import call_llm


logger_report_node = logging.getLogger("app.report.graph")
logger_report_llm = logging.getLogger("app.llm")

MAX_REPORT_RETRY = 2


REPORT_GRADER_SYSTEM_PROMPT = """
당신은 Report Grounding Grader 입니다.

역할:
- 생성된 최종 리포트가 입력 evidence와 movement_reasoning에 근거했는지 평가합니다.
- 단순 문장 품질이 아니라 grounding, hallucination, usefulness를 평가합니다.
- evidence가 충분한데 리포트가 잘못 작성된 경우 regenerate로 판단합니다.
- evidence 자체가 부족해서 좋은 리포트를 만들기 어려운 경우 rewrite로 판단합니다.

평가 기준:
1. final_report가 retrieved_merged_evidence에 없는 내용을 단정적으로 생성했는가?
2. final_report의 교정 방법이 retrieved_merged_evidence와 연결되는가?
3. final_report가 movement_reasoning의 핵심 biomechanical 문제와 일관되는가?
4. final_report가 score_stats/weak_metrics에 없는 수치나 metric을 만들었는가?
5. evidence가 충분한데 표현/구성만 문제라면 regenerate로 판단하십시오.
6. evidence 자체가 부족하거나 missing concept가 남아 있으면 rewrite로 판단하십시오.
7. 충분히 grounded되어 있고 사용자에게 유용하면 good으로 판단하십시오.

출력은 반드시 JSON 객체 하나만 반환하십시오.
모든 문자열 값은 한국어로 작성하십시오.
markdown code block을 사용하지 마십시오.

JSON schema:
{
  "grade": "good | regenerate | rewrite",
  "grounded": true,
  "useful": true,
  "reason": "string",
  "hallucination_risks": ["string"],
  "missing_evidence": ["string"],
  "regenerate_guidance": ["string"],
  "rewrite_guidance": ["string"]
}
""".strip()


def _safe_str(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def _strip_markdown_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner
    return text


def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def _build_report_meta(state: ReportAgentState) -> Dict[str, Any]:
    """Report Generator/Grader가 사용할 meta payload를 구성합니다."""
    meta = dict(state.get("meta") or {})
    meta["movement_reasoning"] = state.get("movement_reasoning") or {}
    meta["rag_queries"] = state.get("rag_queries") or []
    meta["retrieved_candidates"] = state.get("retrieved_candidates") or []
    meta["retrieved_merged_evidence"] = state.get("retrieved_merged_evidence") or []
    meta["retrieval_grader"] = state.get("retrieval_grader") or {}
    meta["retrieval_history"] = state.get("retrieval_history") or []
    return meta


def report_generator_node(state: ReportAgentState) -> ReportAgentState:
    """retrieved_merged_evidence를 기반으로 최종 리포트를 생성합니다."""
    report_retry_count = int(state.get("report_retry_count", 0))
    meta = _build_report_meta(state)

    # 기존 LLM_Total_report의 최종 리포트 프롬프트를 재사용합니다.
    # 순환 import를 피하기 위해 노드 실행 시점에 import합니다.
    from app.services.report.LLM_Total_report import (  # pylint: disable=import-outside-toplevel
        _system_prompt,
        _user_prompt,
    )

    lang = _safe_str(meta.get("lang") or "ko") or "ko"
    messages = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": _user_prompt(meta, lang)},
    ]

    raw = call_llm(messages, model="")

    try:
        final_report = json.loads(
            _extract_json_object(
                _strip_markdown_code_fences(raw)
            )
        )
    except Exception:
        final_report = {
            "raw": raw,
            "parse_error": True,
        }

    logger_report_node.info(
        "[LangGraph][Report Generator] report_retry_count=%d evidence_count=%d parse_error=%s",
        report_retry_count,
        len(meta.get("retrieved_merged_evidence") or []),
        bool(final_report.get("parse_error")) if isinstance(final_report, dict) else False,
    )

    return {
        **state,
        "meta": meta,
        "final_report": final_report,
        "report_retry_count": report_retry_count,
    }


def report_grader_node(state: ReportAgentState) -> ReportAgentState:
    """최종 리포트가 evidence에 grounded 되었는지 평가합니다."""
    meta = _build_report_meta(state)
    final_report = state.get("final_report") or {}

    user_payload = {
        "movement_reasoning": state.get("movement_reasoning") or {},
        "retrieved_merged_evidence": state.get("retrieved_merged_evidence") or [],
        "retrieval_grader": state.get("retrieval_grader") or {},
        "score_stats": meta.get("score_stats") or {},
        "weak_metrics": meta.get("weak_metrics") or [],
        "final_report": final_report,
    }

    messages = [
        {"role": "system", "content": REPORT_GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        raw = call_llm(messages, model="")
        parsed = json.loads(
            _extract_json_object(
                _strip_markdown_code_fences(raw)
            )
        )

        grade = _safe_str(parsed.get("grade")).lower()
        if grade not in {"good", "regenerate", "rewrite"}:
            grade = "regenerate"

        report_grader = {
            "grade": grade,
            "grounded": bool(parsed.get("grounded", False)),
            "useful": bool(parsed.get("useful", False)),
            "reason": _safe_str(parsed.get("reason")),
            "hallucination_risks": parsed.get("hallucination_risks") or [],
            "missing_evidence": parsed.get("missing_evidence") or [],
            "regenerate_guidance": parsed.get("regenerate_guidance") or [],
            "rewrite_guidance": parsed.get("rewrite_guidance") or [],
        }

    except Exception as exc:
        logger_report_llm.warning(
            "[LLM_REPORT_GRADER] failed err=%s. fallback regenerate used.",
            str(exc),
        )
        report_grader = {
            "grade": "regenerate",
            "grounded": False,
            "useful": False,
            "reason": f"Report Grader 파싱 실패: {str(exc)}",
            "hallucination_risks": [],
            "missing_evidence": [],
            "regenerate_guidance": ["최종 리포트를 evidence 중심으로 다시 작성하십시오."],
            "rewrite_guidance": [],
        }

    logger_report_llm.info(
        "[LLM_REPORT_GRADER] %s",
        json.dumps(report_grader, ensure_ascii=False),
    )

    return {
        **state,
        "meta": meta,
        "report_grader": report_grader,
    }


def youtube_recommendation_node(state: ReportAgentState) -> ReportAgentState:
    """최종 리포트에 YouTube 추천 결과를 결합합니다."""
    final_report = state.get("final_report") or {}
    if not isinstance(final_report, dict):
        return state

    try:
        from app.services.report.LLM_Total_report import recommended_youtube_tool  # pylint: disable=import-outside-toplevel

        final_report = recommended_youtube_tool(final_report)
    except Exception as exc:
        logger_report_node.warning(
            "[LangGraph][YouTube Recommendation] failed err=%s",
            str(exc),
        )

    logger_report_node.info("[LangGraph][YouTube Recommendation] completed")
    return {
        **state,
        "final_report": final_report,
    }


def decide_after_report_grader(state: ReportAgentState) -> str:
    """Report Grader 결과에 따라 regenerate/rewrite/good 분기를 결정합니다."""
    report_grader = state.get("report_grader") or {}
    grade = _safe_str(report_grader.get("grade")).lower()
    report_retry_count = int(state.get("report_retry_count", 0))

    if grade == "good":
        return "good"

    if grade == "rewrite":
        return "rewrite"

    if grade == "regenerate" and report_retry_count < MAX_REPORT_RETRY:
        return "regenerate"

    # 재생성 한도를 넘으면 루프를 종료하고 현재 리포트를 사용합니다.
    return "good"


def increment_report_retry_node(state: ReportAgentState) -> ReportAgentState:
    """Report Generator 재시도 횟수를 증가시킵니다."""
    report_retry_count = int(state.get("report_retry_count", 0)) + 1
    logger_report_node.info(
        "[LangGraph][Report Retry] report_retry_count=%d",
        report_retry_count,
    )
    return {
        **state,
        "report_retry_count": report_retry_count,
    }