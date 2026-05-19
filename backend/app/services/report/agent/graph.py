from __future__ import annotations

import logging
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph, END

from app.services.report.agent.nodes import (
    adaptive_rag_node,
    movement_reasoning_node,
    query_rewrite_node,
    retrieval_grader_node,
    should_retry,
)

logger_graph = logging.getLogger("app.report.graph")


class AdaptiveRAGState(TypedDict, total=False):
    meta: Dict[str, Any]
    retrieved_coaching: list
    retrieved_candidates: list
    retrieval_grader: Dict[str, Any]
    retrieval_history: list
    retry_count: int
    rag_queries: list


def build_report_graph():
    graph = StateGraph(AdaptiveRAGState)

    # 노드 정의 
    graph.add_node(
        "movement_reasoning",
        movement_reasoning_node,
    )

    graph.add_node(
        "adaptive_rag",
        adaptive_rag_node,
    )

    graph.add_node(
        "retrieval_grader",
        retrieval_grader_node,
    )

    graph.add_node(
        "query_rewrite",
        query_rewrite_node,
    )

    graph.set_entry_point("movement_reasoning")

    # 그래프 구축
    graph.add_edge(
        "movement_reasoning",
        "adaptive_rag",
    )
    
    graph.add_edge(
        "adaptive_rag",
        "retrieval_grader",
    )

    # 조건부 엣지 추가: 문서 평가 후 결정   
    graph.add_conditional_edges(
        "retrieval_grader",
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