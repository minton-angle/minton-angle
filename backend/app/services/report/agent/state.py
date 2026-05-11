from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ReportAgentState(TypedDict, total=False):
    """State carried across the report reasoning workflow.

    This state is initialized from router/service-generated meta.
    It is intentionally separated from DB models so LangGraph nodes can operate
    on normalized dictionaries only.
    """

    meta: Dict[str, Any]
    score_stats: Dict[str, Any]
    weak_metrics: List[Dict[str, Any]]

    # LLM-generated biomechanical interpretation of weak_metrics.
    movement_reasoning: Dict[str, Any]

    # Future fields for adaptive retrieval workflow.
    rag_queries: List[Dict[str, Any]]
    rag_results: List[Dict[str, Any]]
    retrieval_grade: Dict[str, Any]
    web_results: List[Dict[str, Any]]
    merged_evidence: List[Dict[str, Any]]
    final_report: Dict[str, Any]