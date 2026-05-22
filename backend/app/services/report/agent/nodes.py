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
    merge_evidence_docs,
    rewrite_rag_queries,
    run_retrieval_attempt,
)
from app.services.report.retrieval.retrieval_grader import grade_retrieval_results
from app.services.report.retrieval.rag_query_builder import (
    METRIC_QUERY_MAP,
    STAGE_QUERY_MAP,
    build_rag_queries,
)
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
    """weak_metrics를 기반으로 biomechanical movement hypothesis를 추론합니다.

    이 노드는 에이전트 흐름의 첫 번째 LLM reasoning 단계입니다.
    문서 검색은 수행하지 않고, 사용자의 정규화된 동작 관찰값을 해석하여
    이후 RAG 검색에 사용할 reasoning target을 생성합니다.
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
                metric_query_reference=METRIC_QUERY_MAP,
                stage_query_reference=STAGE_QUERY_MAP,
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
    """RAG 검색을 1회 수행하고 후보 문서를 state에 저장합니다.

    retry와 분기는 LangGraph가 제어합니다.
    이 노드는 검색만 수행하며, 문서 평가는 retrieval_grader_node에서 처리합니다.
    """
    meta = state.get("meta") or {}
    retrieval_count = int(state.get("retrieval_count", 0)) + 1
    movement_reasoning = state.get("movement_reasoning") or {}

    # retrieval_node는 항상 rag_queries 상태를 기준으로 검색을 수행합니다.
    #
    # - 첫 검색:
    #   movement_reasoning.retrieval_focus 기반으로 rag_queries 생성
    #
    # - rewrite 이후 재검색:
    #   query_rewrite_node에서 갱신한 rag_queries를 그대로 재사용
    #
    # 즉 build_rag_queries()는 최종적으로
    # state에 유지할 rag_queries 리스트를 반환합니다.
    rag_queries = build_rag_queries(
        meta,
        movement_reasoning=movement_reasoning,
        rag_queries=state.get("rag_queries") or [],
        logger=logger_graph_node,
    )

    logger_graph_node.info(
        "[LangGraph] retrieval_node 쿼리 개수=%d 쿼리 리스트=%s",
        len(rag_queries or []),
        json.dumps(rag_queries, ensure_ascii=False),
    )

    docs = run_retrieval_attempt(
        meta=meta,
        movement_reasoning=movement_reasoning,
        rag_queries=rag_queries,
        retrieval_count=retrieval_count,
        logger=logger_graph_node,
    )

    logger_graph_node.info(
        "[LangGraph] retrieval_node 전체 후보 검색 회수=%d 평가 받을 후보 문서 개수=%d",
        retrieval_count,
        len(docs or []),
    )
    retrieval_history = state.get("retrieval_history") or []
    retrieval_history = [
        *retrieval_history,
        {
            "retrieval_count": retrieval_count,
            "query_source": "rewrite" if retrieval_count > 1 else "initial",
            "doc_count": len(docs or []),
        },
    ]

    return {
        **state,
        "meta": meta,
        "retrieved_candidates": docs,
        "retrieval_history": retrieval_history,
        "rag_queries": rag_queries,
        "retrieval_count": retrieval_count,
    }


def retrieval_grader_node(state: ReportAgentState) -> ReportAgentState:
    """검색 후보 문서를 평가하고 관련성 있는 evidence만 state에 저장합니다.

    주피터 Self-RAG 예제 흐름과 동일하게,
    state["retrieved_candidates"]를 평가한 뒤
    통과 문서만 state["retrieved_merged_evidence"]에 저장합니다.
    """
    meta = state.get("meta") or {}
    retrieval_count = int(state.get("retrieval_count", 0))
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

    # Evidence Merge: retry 과정에서 통과한 evidence를 누적 보존합니다.
    previous_evidence_docs = state.get("retrieved_merged_evidence") or []
    merged_evidence_docs = merge_evidence_docs(
        previous_docs=previous_evidence_docs,
        new_docs=filtered_docs,
    )

    retrieval_history = state.get("retrieval_history") or []
    if retrieval_history:
        retrieval_history[-1]["grader"] = grader
        retrieval_history[-1]["filtered_doc_count"] = len(filtered_docs or [])
        retrieval_history[-1]["merged_evidence_count"] = len(merged_evidence_docs or [])

    logger_graph_node.info(
        "[LangGraph][Evidence Merge] retrieval_count=%d candidate_doc_count=%d filtered_doc_count=%d merged_evidence_count=%d relevant=%s needs_retry=%s coverage=%s missing_concepts=%s",
        retrieval_count,
        len(docs or []),
        len(filtered_docs or []),
        len(merged_evidence_docs or []),
        grader.get("relevant"),
        grader.get("needs_retry"),
        grader.get("coverage"),
        grader.get("missing_concepts") or [],
    )

    return {
        **state,
        "meta": meta,
        "retrieval_grader": grader,
        "retrieved_merged_evidence": merged_evidence_docs,
        "retrieval_history": retrieval_history,
        "rag_queries": rag_queries,
        "retrieval_count": retrieval_count,
    }


def query_rewrite_node(state: ReportAgentState) -> ReportAgentState:
    """retrieval_grader 피드백을 기반으로 다음 검색에 사용할 RAG 쿼리를 재작성합니다."""
    meta = state.get("meta") or {}
    grader = state.get("retrieval_grader") or {}
    current_queries = state.get("rag_queries") or meta.get("rag_queries") or []
    retrieval_count = int(state.get("retrieval_count", 0))

    rewrite_context_docs = (
        state.get("retrieved_candidates")
        or state.get("retrieved_merged_evidence")
        or []
    )

    rewritten_queries = rewrite_rag_queries(
        queries=current_queries,
        grader_result=grader,
        movement_reasoning=state.get("movement_reasoning") or {},
        retrieved_docs=rewrite_context_docs,
    )

    logger_graph_node.info(
        "[LangGraph] LLM query rewrite after_retrieval_count=%d query_count=%d context_doc_count=%d queries=%s",
        retrieval_count,
        len(rewritten_queries or []),
        len(rewrite_context_docs or []),
        json.dumps(rewritten_queries, ensure_ascii=False),
    )

    return {
        **state,
        "meta": meta,
        "rag_queries": rewritten_queries,
    }


def decide_to_generate_self(state: ReportAgentState) -> str:
    """Retrieval Grader 결과를 기반으로 rewrite 여부를 결정합니다."""
    grader = state.get("retrieval_grader") or {}
    retrieval_count = int(state.get("retrieval_count", 0))

    # MAX_RETRY는 허용되는 rewrite 횟수입니다.
    # 첫 검색은 retrieval_count=1이므로 최대 검색 회수는 MAX_RETRY + 1입니다.
    if grader.get("needs_retry") and retrieval_count <= MAX_RETRY:
        return "rewrite"

    return "end"