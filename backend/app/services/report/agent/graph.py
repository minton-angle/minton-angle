from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from app.services.report.agent.state import ReportAgentState

from app.services.report.agent.nodes import (
    retrieval_node,
    movement_reasoning_node,
    query_rewrite_node,
    retrieval_grader_node,
    decide_to_generate_self,
)
from app.services.report.agent.report_nodes import (
    report_generator_node,
    report_grader_node,
    youtube_recommendation_node,
    decide_after_report_grader,
    increment_report_retry_node,
)

logger_graph = logging.getLogger("app.report.graph")


def build_report_graph():
    graph = StateGraph(ReportAgentState)

    # 노드 정의 
    graph.add_node(
        "movement_reasoning",
        movement_reasoning_node,
    )

    graph.add_node(
        "retrieval_rag",
        retrieval_node,
    )

    graph.add_node(
        "retrieval_grader",
        retrieval_grader_node,
    )

    graph.add_node(
        "query_rewrite",
        query_rewrite_node,
    )

    graph.add_node(
        "report_generator",
        report_generator_node,
    )

    graph.add_node(
        "report_grader",
        report_grader_node,
    )

    graph.add_node(
        "report_retry",
        increment_report_retry_node,
    )

    graph.add_node(
        "youtube_recommendation",
        youtube_recommendation_node,
    )

    graph.set_entry_point("movement_reasoning")

    # 그래프 구축
    graph.add_edge(
        "movement_reasoning",
        "retrieval_rag",
    )
    
    graph.add_edge(
        "retrieval_rag",
        "retrieval_grader",
    )

    # 조건부 엣지 추가: 문서 평가 후 결정   
    graph.add_conditional_edges(
        "retrieval_grader",
        decide_to_generate_self,
        {
            "rewrite": "query_rewrite",
            "end": "report_generator",
        },
    )

    graph.add_edge(
        "query_rewrite",
        "retrieval_rag",
    )

    graph.add_edge(
        "report_generator",
        "report_grader",
    )

    # 조건부 엣지 추가: 보고서 평가 후 결정
    # - regenerate: 같은 movement_reasoning으로 Report 재생성
    # - rewrite: Query Rewrite 노드로 이동 후 Retrieval 루프 재진입
    # - good: YouTube Recommendation 후 종료
    graph.add_conditional_edges(
        "report_grader",
        decide_after_report_grader,
        {
            "regenerate": "report_retry",
            "rewrite": "query_rewrite",
            "good": "youtube_recommendation",
        },
    )

    # Report Retry 노드는 재시도 횟수를 증가시키고 다시 Report Generator 실행
    graph.add_edge(
        "report_retry",
        "report_generator",
    )

    graph.add_edge(
        "youtube_recommendation",
        END,
    )

    return graph.compile()