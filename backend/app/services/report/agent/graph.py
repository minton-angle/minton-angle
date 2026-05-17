from __future__ import annotations

import logging
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph, END

from app.services.report.agent.nodes import (
    adaptive_rag_node,
    movement_reasoning_node,
    query_rewrite_node,
    should_retry,
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