from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.services.report.retrieval.chroma_retriever import (
    retrieve_coaching_evidence,
)


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
) -> List[Dict[str, Any]]:
    """Rewrite existing RAG queries using LLM grader guidance.

    This function does not run retrieval. LangGraph controls when this is called
    and whether another retrieval attempt is needed.
    """
    rewrite_guidance = grader_result.get("rewrite_guidance") or []
    missing_concepts = grader_result.get("missing_concepts") or []

    suffix_parts = []
    if rewrite_guidance:
        suffix_parts.extend(_safe_str(x) for x in rewrite_guidance if _safe_str(x))
    if missing_concepts:
        suffix_parts.extend(_safe_str(x) for x in missing_concepts if _safe_str(x))

    suffix = " ".join(suffix_parts).strip()
    if not suffix:
        return queries

    rewritten_queries: List[Dict[str, Any]] = []
    for item in queries or []:
        q = _safe_str(item.get("q"))
        rewritten_queries.append(
            {
                **item,
                "q": f"{q} {suffix}".strip(),
                "query_source": "rewrite",
                "rewrite_guidance": rewrite_guidance,
                "missing_concepts": missing_concepts,
            }
        )

    return rewritten_queries


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

    grader = meta.get("retrieval_grader") or {}
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