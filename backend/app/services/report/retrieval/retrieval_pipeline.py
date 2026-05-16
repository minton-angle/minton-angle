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

    logger.info(
        "Adaptive RAG retrieval attempt=%d docs=%d grader=%s",
        attempt,
        len(docs or []),
        json.dumps(meta.get("retrieval_grader") or {}, ensure_ascii=False),
    )

    return docs