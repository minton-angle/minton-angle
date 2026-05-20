from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

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

# 검색 실행 하고 기록하는 함수
def run_retrieval_attempt(
    *,
    meta: Dict[str, Any],
    attempt: int,
    logger: logging.Logger | None = None,
) -> List[Dict[str, Any]]:
    """Run exactly one retrieval attempt.

    This function only retrieves candidate documents. Retrieval grading and
    document filtering are handled by the LangGraph retrieval_grader node.
    """
    logger = logger or logger_pipeline

    docs = retrieve_coaching_evidence(
        meta,
        logger=logger,
    )

    history = meta.get("retrieval_history") or []
    history.append(
        {
            "attempt": attempt,
            "query_source": "rewrite" if attempt > 0 else "initial",
            "doc_count": len(docs or []),
        }
    )
    meta["retrieval_history"] = history

    logger.info(
        "[LangGraph] RAG retrieval attempt=%d candidate_doc_count=%d",
        attempt,
        len(docs or []),
    )

    return docs