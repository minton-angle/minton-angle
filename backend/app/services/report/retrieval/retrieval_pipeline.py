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


def run_retrieval_attempt(
    *,
    meta: Dict[str, Any],
    attempt: int,
    logger: logging.Logger | None = None,
) -> List[Dict[str, Any]]:
    """Run exactly one retrieval + grading attempt.

    Retry loop orchestration belongs to LangGraph, not this function.
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
            "grader": meta.get("retrieval_grader") or {},
            "doc_count": len(docs or []),
        }
    )
    meta["retrieval_history"] = history

    grader = grade_retrieval_results(

        query=json.dumps(meta.get("rag_queries") or [], ensure_ascii=False),

        retrieved_docs=docs,

        movement_reasoning=meta.get("movement_reasoning") or {},

    )
    meta["retrieval_grader"] = grader
    
    logger.info(
        "[Adaptive RAG] 현재 검색 시도 회수 =%d 주입 대상 문서 개수=%d 검색 결과와 질문 관련성=%s 재검색 필요 여부=%s 요구사항 커버 정도(1이 완벽)=%s 현재 검색 결과에서 부족한 개념 목록=%s",
        attempt,
        len(docs or []),
        grader.get("relevant"),
        grader.get("needs_retry"),
        grader.get("coverage"),
        grader.get("missing_concepts") or [],
    )
    return docs