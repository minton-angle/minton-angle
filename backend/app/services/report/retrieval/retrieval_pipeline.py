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
    

def _filter_docs_by_grader(
    *,
    docs: List[Dict[str, Any]],
    grader: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Keep only documents accepted by the retrieval grader.
    """
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

    filtered_docs = _filter_docs_by_grader(
        docs=docs,
        grader=grader,
    )

    meta["retrieval_grader"] = grader
    meta["retrieved_coaching"] = filtered_docs

    logger.info(
        "[Adaptive RAG] 현재 검색 시도 회수 =%d 주입 대상 문서 개수=%d 검색 결과와 질문 관련성=%s 재검색 필요 여부=%s 요구사항 커버 정도(1이 완벽)=%s 현재 검색 결과에서 부족한 개념 목록=%s",
        attempt,
        len(filtered_docs or []),
        grader.get("relevant"),
        grader.get("needs_retry"),
        grader.get("coverage"),
        grader.get("missing_concepts") or [],
    )
    return filtered_docs