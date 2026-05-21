from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ReportAgentState(TypedDict, total=False):
    """LangGraph 리포트 에이전트 상태 스키마.

    meta는 DB에서 조회된 원본 입력값을 담고,
    나머지 필드는 그래프 실행 중 생성되는 중간 상태를 담습니다.
    """

    # DB 조회 기반 원본 입력값
    meta: Dict[str, Any]
    score_stats: Dict[str, Any]
    weak_metrics: List[Dict[str, Any]]

    # Movement Reasoning 결과
    movement_reasoning: Dict[str, Any]

    # RAG 검색 쿼리 목록
    rag_queries: List[Dict[str, Any]]

    # RAG 검색 후보 문서와 최종 주입 문서
    retrieved_candidates: List[Dict[str, Any]]
    retrieved_coaching: List[Dict[str, Any]]

    # Retrieval Grader 결과와 검색 이력
    retrieval_grader: Dict[str, Any]
    retrieval_history: List[Dict[str, Any]]

    # RAG 검색 실행 횟수. 첫 검색은 1부터 시작합니다.
    retrieval_count: int

    # 향후 Report Grader / Evidence Merge 확장용 필드
    rag_results: List[Dict[str, Any]]
    retrieval_grade: Dict[str, Any]
    merged_evidence: List[Dict[str, Any]]
    final_report: Dict[str, Any]