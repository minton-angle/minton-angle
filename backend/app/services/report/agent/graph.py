from __future__ import annotations

import logging
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph, END

from app.services.report.agent.nodes import movement_reasoning_node
from app.services.report.retrieval.retrieval_pipeline import (
    MAX_RETRY,
    rewrite_rag_queries,
    run_retrieval_attempt,
)

logger_graph = logging.getLogger("app.report.graph")


class AdaptiveRAGState(TypedDict, total=False):
    meta: Dict[str, Any]
    movement_reasoning: Dict[str, Any]
    retrieved_coaching: list
    retrieval_grader: Dict[str, Any]
    retrieval_history: list
    retry_count: int
    rag_queries: list



def adaptive_rag_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    meta = state.get("meta") or {}
    retry_count = int(state.get("retry_count") or 0)

    docs = run_retrieval_attempt(
        meta=meta,
        attempt=retry_count,
        logger=logger_graph,
    )

    return {
        **state,
        "meta": meta,
        "retrieved_coaching": docs,
        "retrieval_grader": meta.get("retrieval_grader") or {},
        "retrieval_history": meta.get("retrieval_history") or [],
        "rag_queries": meta.get("rag_queries") or [],
    }


def query_rewrite_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    meta = state.get("meta") or {}
    grader = state.get("retrieval_grader") or {}
    current_queries = state.get("rag_queries") or meta.get("rag_queries") or []
    retry_count = int(state.get("retry_count") or 0) + 1

    rewritten_queries = rewrite_rag_queries(
        queries=current_queries,
        grader_result=grader,
    )

    meta["rag_queries"] = rewritten_queries

    logger_graph.info(
        "LangGraph query rewrite retry_count=%d queries=%s",
        retry_count,
        rewritten_queries,
    )

    return {
        **state,
        "meta": meta,
        "retry_count": retry_count,
        "rag_queries": rewritten_queries,
    }



def should_retry(state: AdaptiveRAGState) -> str:
    grader = state.get("retrieval_grader") or {}
    retry_count = int(state.get("retry_count") or 0)

    if grader.get("needs_retry") and retry_count < MAX_RETRY:
        return "rewrite"

    return "end"



def build_report_graph():
    graph = StateGraph(AdaptiveRAGState)

    graph.add_node(
        "movement_reasoning",
        movement_reasoning_node,
    )

    graph.add_node(
        "adaptive_rag",
        adaptive_rag_node,
    )

    graph.add_node(
        "query_rewrite",
        query_rewrite_node,
    )

    graph.set_entry_point("movement_reasoning")

    graph.add_edge(
        "movement_reasoning",
        "adaptive_rag",
    )

    graph.add_conditional_edges(
        "adaptive_rag",
        should_retry,
        {
            "rewrite": "query_rewrite",
            "end": END,
        },
    )

    graph.add_edge(
        "query_rewrite",
        "adaptive_rag",
    )
    
    return graph.compile()