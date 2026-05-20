from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.services.report.retrieval.chroma_retriever import (
    retrieve_coaching_evidence,
)
from app.services.report.retrieval.query_rewriter import rewrite_rag_queries_with_llm
from app.services.report.retrieval.retrieval_grader import grade_retrieval_results


logger_pipeline = logging.getLogger("app.llm")

MAX_RETRY = 2


def _safe_str(v: Any) -> str:
    try:
        return "" if v is None else str(v)
    except Exception:
        return ""
    

# 문서 필터링 실행 함수
def filter_docs_by_grader(
    *,
    docs: List[Dict[str, Any]],
    grader: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """선택된 문서 인덱스를 retrieval_grader 결과에서 받아서 실제 문서 리스트를 필터링"""
    accepted = grader.get("filtered_doc_indices")
    if not isinstance(accepted, list):
        return docs

    accepted_set = set()
    for idx in accepted:
        try:
            idx_int = int(idx)
        except Exception:
            continue
        if 0 <= idx_int < len(docs or []):
            accepted_set.add(idx_int)

    if not accepted_set:
        return []

    filtered_docs: List[Dict[str, Any]] = []
    doc_judgements = grader.get("doc_judgements") or []
    judgement_by_index = {}
    if isinstance(doc_judgements, list):
        for item in doc_judgements:
            if not isinstance(item, dict):
                continue
            try:
                judgement_by_index[int(item.get("index"))] = item
            except Exception:
                continue

    for idx, doc in enumerate(docs or []):
        if idx not in accepted_set:
            continue
        judgement = judgement_by_index.get(idx) or {}
        filtered_docs.append(
            {
                **doc,
                "retrieval_grader_accepted": True,
                "retrieval_grader_reason": _safe_str(judgement.get("reason")),
            }
        )

    return filtered_docs


def rewrite_rag_queries(
    *,
    queries: List[Dict[str, Any]],
    grader_result: Dict[str, Any],
    movement_reasoning: Dict[str, Any] | None = None,
    retrieved_docs: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Rewrite existing RAG queries using LLM grader guidance.

    LangGraph controls when this is called and whether another retrieval attempt
    is needed. The actual semantic rewrite is delegated to query_rewriter.py.
    """
    return rewrite_rag_queries_with_llm(
        queries=queries,
        grader_result=grader_result,
        movement_reasoning=movement_reasoning,
        retrieved_docs=retrieved_docs,
    )

# 검색만 실행하고 docs 반환
def run_retrieval_attempt(
    *,
    meta: Dict[str, Any],
    movement_reasoning: Optional[Dict[str, Any]] = None,
    rag_queries: Optional[List[Dict[str, Any]]] = None,
    attempt: int,
    logger: logging.Logger | None = None,
) -> List[Dict[str, Any]]:
    """Run exactly one retrieval attempt and return candidate documents only.

    State updates such as retrieval_history are handled by LangGraph nodes.
    """
    logger = logger or logger_pipeline

    docs = retrieve_coaching_evidence(
        meta,
        movement_reasoning=movement_reasoning or {},
        rag_queries=rag_queries or [],
        logger=logger,
    )

    # 초기 검색인 경우 retrieve_coaching_evidence/build_rag_queries 내부에서 쿼리를 생성
    # 재검색 루프에서는 LangGraph state의 rag_queries를 유지하고,
    # 쿼리 목록은 query_rewrite_node에서 재작성

    logger.info(
        "[LangGraph] RAG retrieval attempt=%d candidate_doc_count=%d",
        attempt,
        len(docs or []),
    )

    return docs