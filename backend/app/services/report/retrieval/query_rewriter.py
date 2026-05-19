from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.services.report.llm.client import call_llm


logger_query_rewriter = logging.getLogger("app.llm")


QUERY_REWRITE_SYSTEM_PROMPT = """
당신은 Self-RAG Query Rewriter 입니다.

역할:
- 이전 retrieval이 왜 실패했는지 해석합니다.
- movement_reasoning과 retrieval_grader 결과를 함께 보고 검색 의도를 다시 설계합니다.
- 단순히 기존 query에 missing keyword를 덧붙이지 말고, semantic retrieval intent 자체를 재작성합니다.
- 검색 대상은 배드민턴 코칭 문서이며, 사용자의 weak_metrics와 biomechanical movement hypothesis를 설명할 수 있는 근거를 찾는 것이 목적입니다.

재작성 기준:
1. retrieval_grader.reason, missing_concepts, rewrite_guidance를 분석해 검색 실패 원인을 요약합니다.
2. movement_reasoning의 movement_hypotheses, coaching_focus, retrieval_focus를 우선 반영합니다.
3. 기존 retrieved_docs가 일반론이거나 stage/metric과 직접 연결되지 않으면 더 구체적인 동작 원인/교정 intent로 바꿉니다.
4. metric 이름만 반복하지 말고, 해당 metric이 의미하는 신체 움직임, 원인, 교정 방향을 포함합니다.
5. query는 한국어 semantic query로 작성합니다.
6. "badminton", "배드민턴" 같은 넓은 키워드는 과도하게 반복하지 마십시오.
7. 입력에 없는 수치나 metric을 만들지 마십시오.

출력은 반드시 JSON 객체 하나만 반환하십시오.
markdown code block을 사용하지 마십시오.

JSON schema:
{
  "failure_analysis": "string",
  "rewrite_strategy": "string",
  "rewritten_queries": [
    {
      "stage": "string",
      "metric": "string",
      "q": "string",
      "query_intent": "string",
      "reason": "string"
    }
  ]
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


def _fallback_rewrite_queries(
    *,
    queries: List[Dict[str, Any]],
    grader_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Rule-based fallback when LLM query rewrite fails."""
    rewrite_guidance = grader_result.get("rewrite_guidance") or []
    missing_concepts = grader_result.get("missing_concepts") or []

    suffix_parts = []
    suffix_parts.extend(_safe_str(x) for x in rewrite_guidance if _safe_str(x))
    suffix_parts.extend(_safe_str(x) for x in missing_concepts if _safe_str(x))

    suffix = " ".join(suffix_parts).strip()
    if not suffix:
        return queries

    rewritten_queries: List[Dict[str, Any]] = []
    for item in queries or []:
        q = _safe_str(item.get("q"))
        rewritten_queries.append(
            {
                **item,
                "q": f"{q} {suffix}".strip(),
                "query_source": "rewrite_fallback",
                "rewrite_guidance": rewrite_guidance,
                "missing_concepts": missing_concepts,
            }
        )

    return rewritten_queries


def rewrite_rag_queries_with_llm(
    *,
    queries: List[Dict[str, Any]],
    grader_result: Dict[str, Any],
    movement_reasoning: Dict[str, Any] | None = None,
    retrieved_docs: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Rewrite RAG queries using movement reasoning and retrieval failure feedback.

    This is a Self-RAG style query rewrite step:
    1. Interpret why retrieval failed.
    2. Compare movement_reasoning with retrieval_grader feedback.
    3. Rebuild the semantic retrieval intent instead of appending keywords.
    """
    movement_reasoning = movement_reasoning or {}
    retrieved_docs = retrieved_docs or []

    retrieved_preview = [
        {
            "stage": item.get("stage"),
            "metric": item.get("metric"),
            "source": item.get("source"),
            "content": _safe_str(item.get("content"))[:800],
        }
        for item in retrieved_docs[:6]
        if isinstance(item, dict)
    ]

    user_payload = {
        "previous_queries": queries or [],
        "retrieval_grader": grader_result or {},
        "movement_reasoning": movement_reasoning,
        "retrieved_docs_preview": retrieved_preview,
    }

    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        raw = call_llm(messages, model="")
        parsed = json.loads(
            _extract_json_object(
                _strip_markdown_code_fences(raw)
            )
        )

        rewritten = parsed.get("rewritten_queries") or []
        failure_analysis = _safe_str(parsed.get("failure_analysis"))
        rewrite_strategy = _safe_str(parsed.get("rewrite_strategy"))

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(rewritten):
            if not isinstance(item, dict):
                continue

            q = _safe_str(item.get("q") or item.get("query_intent"))
            if not q:
                continue

            previous = (queries or [{}])[min(idx, max(len(queries or []) - 1, 0))] if queries else {}
            normalized.append(
                {
                    **previous,
                    "stage": _safe_str(item.get("stage") or previous.get("stage")),
                    "metric": _safe_str(item.get("metric") or previous.get("metric")),
                    "q": q,
                    "query_intent": _safe_str(item.get("query_intent") or q),
                    "query_source": "llm_rewrite",
                    "rewrite_reason": _safe_str(item.get("reason")),
                    "failure_analysis": failure_analysis,
                    "rewrite_strategy": rewrite_strategy,
                    "rewrite_guidance": grader_result.get("rewrite_guidance") or [],
                    "missing_concepts": grader_result.get("missing_concepts") or [],
                }
            )

        if normalized:
            logger_query_rewriter.info(
                "[LLM_QUERY_REWRITE] failure_analysis=%s strategy=%s queries=%s",
                failure_analysis,
                rewrite_strategy,
                json.dumps(normalized, ensure_ascii=False),
            )
            return normalized

        raise ValueError("LLM query rewriter returned no valid rewritten_queries")

    except Exception as exc:
        logger_query_rewriter.warning(
            "[LLM_QUERY_REWRITE] failed err=%s. fallback rewrite used.",
            str(exc),
        )
        return _fallback_rewrite_queries(
            queries=queries,
            grader_result=grader_result,
        )