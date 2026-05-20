from __future__ import annotations

import json
import logging
from typing import Any, Dict

from app.services.report.llm.client import call_llm
from app.services.report.agent.prompts import (
    MOVEMENT_REASONING_SYSTEM_PROMPT,
    build_movement_reasoning_user_prompt,
)
from app.services.report.retrieval.retrieval_pipeline import (
    MAX_RETRY,
    filter_docs_by_grader,
    rewrite_rag_queries,
    run_retrieval_attempt,
)
from app.services.report.retrieval.retrieval_grader import grade_retrieval_results
from app.services.report.agent.state import ReportAgentState


logger_agent = logging.getLogger("app.report.agent")
logger_graph_node = logging.getLogger("app.report.graph")


def _strip_markdown_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
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
                return text[start : i + 1]

    return text[start:]


def _fallback_movement_reasoning(state: ReportAgentState, reason: str = "") -> Dict[str, Any]:
    weak_metrics = state.get("weak_metrics") or []
    hypotheses = []

    for item in weak_metrics:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "")
        metric = str(item.get("metric") or "")
        score = item.get("score")
        if not stage or not metric:
            continue

        hypotheses.append(
            {
                "name": f"{stage}_{metric}_weakness",
                "description": f"{stage} 단계의 {metric} 세부 지표가 기준치 미만으로 관찰되었습니다.",
                "related_stages": [stage],
                "related_metrics": [metric],
                "evidence": [f"{metric} score={score}"],
                "coaching_focus": f"{stage} 단계에서 {metric} 안정화가 필요합니다.",
                "confidence": 0.5,
            }
        )

    return {
        "summary": "LLM movement reasoning 생성에 실패하여 weak_metrics 기반 기본 가설을 사용합니다.",
        "movement_hypotheses": hypotheses,
        "retrieval_focus": [
            {
                "stage": str(item.get("stage") or ""),
                "metric": str(item.get("metric") or ""),
                "query_intent": "해당 세부 동작의 배드민턴 코칭 근거 검색",
            }
            for item in weak_metrics
            if isinstance(item, dict)
        ],
        "risk_notes": [],
        "fallback_reason": reason,
    }

# metrics간 인과관계 및 패턴을 LLM으로 분석하여 movement_hypotheses를 생성하는 노드
def movement_reasoning_node(state: ReportAgentState) -> ReportAgentState:
    """Infer biomechanical movement hypotheses from weak_metrics.

    This node is the first LLM reasoning step in the agent workflow.
    It does not retrieve documents. It only interprets the user's normalized
    movement observations and creates reasoning targets for later retrieval.
    """
    meta = state.get("meta") or {}
    weak_metrics = meta.get("weak_metrics")
    score_stats = meta.get("score_stats", {})

    if not weak_metrics:
        logger_agent.info("movement_reasoning skipped: weak_metrics empty")
        movement_reasoning = {
            "summary": "기준치 미만의 세부 동작이 없어 별도 movement weakness reasoning을 수행하지 않았습니다.",
            "movement_hypotheses": [],
            "retrieval_focus": [],
            "risk_notes": [],
        }
        return {
            **state,
            "meta": meta,
            "movement_reasoning": movement_reasoning,
        }

    messages = [
        {"role": "system", "content": MOVEMENT_REASONING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_movement_reasoning_user_prompt(
                weak_metrics=weak_metrics,
                score_stats=score_stats,
            ),
        },
    ]

    try:
        raw = call_llm(messages, model="")
        parsed_text = _extract_json_object(_strip_markdown_code_fences(raw))
        movement_reasoning = json.loads(parsed_text)
    except Exception as exc:
        logger_agent.warning("movement_reasoning failed err=%s", str(exc))
        movement_reasoning = _fallback_movement_reasoning(state, reason=str(exc))

    hypotheses = movement_reasoning.get("movement_hypotheses", []) if isinstance(movement_reasoning, dict) else []
    hypothesis_summary = [
        {
            "name": item.get("name"),
            "related_stages": item.get("related_stages"),
            "related_metrics": item.get("related_metrics"),
            "confidence": item.get("confidence"),
        }
        for item in hypotheses
        if isinstance(item, dict)
    ]
    logger_agent.info(
        "[Movement Reasoning] weak_metric관계 chain counts=%d details=%s",
        len(hypothesis_summary),
        json.dumps(hypothesis_summary, ensure_ascii=False),
    )

    return {
        **state,
        "meta": meta,
        "movement_reasoning": movement_reasoning,
    }


def retrieval_node(state: ReportAgentState) -> ReportAgentState:
    """Run one retrieval attempt and attach candidate retrieval documents.

    LangGraph controls retry/branching. This node performs retrieval only.
    Retrieval grading is handled by retrieval_grader_node.
    """
    meta = state.get("meta") or {}
    retry_count = int(state.get("retry_count") or 0)
    movement_reasoning = state.get("movement_reasoning") or {}
    if movement_reasoning:
        # Backward compatibility: rag_query_builder currently reads movement_reasoning from meta.
        # The graph-level source of truth remains state["movement_reasoning"].
        meta["movement_reasoning"] = movement_reasoning

    docs = run_retrieval_attempt(
        meta=meta,
        attempt=retry_count,
        logger=logger_graph_node,
    )
    logger_graph_node.info(
        "[LangGraph] node=retrieval_rag attempt=%d candidate_doc_count=%d",
        retry_count,
        len(docs or []),
    )

    return {
        **state,
        "meta": meta,
        "retrieved_candidates": docs,
        "retrieval_history": meta.get("retrieval_history") or [],
        "rag_queries": meta.get("rag_queries") or [],
    }


def retrieval_grader_node(state: ReportAgentState) -> ReportAgentState:
    """Evaluate retrieved documents and keep only relevant evidence.

    This follows the notebook-style Self-RAG flow:
    state["retrieved_candidates"] -> grade documents -> state["retrieved_coaching"]
    """
    meta = state.get("meta") or {}
    retry_count = int(state.get("retry_count") or 0)
    docs = state.get("retrieved_candidates") or []
    logger_graph_node.info(
        "[LangGraph][retrieval_grader_node] input candidate_doc_count=%d",
        len(docs or []),
    )
    rag_queries = state.get("rag_queries") or meta.get("rag_queries") or []
    movement_reasoning = state.get("movement_reasoning") or {}

    grader = grade_retrieval_results(
        query=json.dumps(rag_queries, ensure_ascii=False),
        retrieved_docs=docs,
        movement_reasoning=movement_reasoning,
    )

    filtered_docs = filter_docs_by_grader(
        docs=docs,
        grader=grader,
    )

    retrieval_history = state.get("retrieval_history") or meta.get("retrieval_history") or []
    if retrieval_history:
        retrieval_history[-1]["grader"] = grader
        retrieval_history[-1]["filtered_doc_count"] = len(filtered_docs or [])

    meta["retrieval_grader"] = grader
    meta["retrieved_coaching"] = filtered_docs
    meta["retrieval_history"] = retrieval_history

    logger_graph_node.info(
        "[LangGraph] retrieval grading attempt=%d candidate_doc_count=%d filtered_doc_count=%d relevant=%s needs_retry=%s coverage=%s missing_concepts=%s",
        retry_count,
        len(docs or []),
        len(filtered_docs or []),
        grader.get("relevant"),
        grader.get("needs_retry"),
        grader.get("coverage"),
        grader.get("missing_concepts") or [],
    )

    return {
        **state,
        "meta": meta,
        "retrieval_grader": grader,
        "retrieved_coaching": filtered_docs,
        "retrieval_history": retrieval_history,
        "rag_queries": rag_queries,
    }


def query_rewrite_node(state: ReportAgentState) -> ReportAgentState:
    """Rewrite RAG queries from grader feedback for the next retrieval attempt."""
    meta = state.get("meta") or {}
    grader = state.get("retrieval_grader") or {}
    current_queries = state.get("rag_queries") or meta.get("rag_queries") or []
    retry_count = int(state.get("retry_count") or 0) + 1

    rewrite_context_docs = (
        state.get("retrieved_candidates")
        or state.get("retrieved_coaching")
        or []
    )

    rewritten_queries = rewrite_rag_queries(
        queries=current_queries,
        grader_result=grader,
        movement_reasoning=state.get("movement_reasoning") or {},
        retrieved_docs=rewrite_context_docs,
    )

    meta["rag_queries"] = rewritten_queries

    logger_graph_node.info(
        "[LangGraph] LLM query rewrite retry_count=%d query_count=%d context_doc_count=%d queries=%s",
        retry_count,
        len(rewritten_queries or []),
        len(rewrite_context_docs or []),
        json.dumps(rewritten_queries, ensure_ascii=False),
    )

    return {
        **state,
        "meta": meta,
        "retry_count": retry_count,
        "rag_queries": rewritten_queries,
    }


def decide_to_generate_self(state: ReportAgentState) -> str:
    """Route graph execution based on LLM retrieval grader result."""
    grader = state.get("retrieval_grader") or {}
    retry_count = int(state.get("retry_count") or 0)

    if grader.get("needs_retry") and retry_count < MAX_RETRY:
        return "rewrite"

    return "end"