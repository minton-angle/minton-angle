from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.services.report.retrieval.chroma_retriever import (
    retrieve_coaching_evidence,
)
from app.services.report.retrieval.query_rewriter import rewrite_rag_queries_with_llm


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


# Evidence Merge: attempt별 Retrieval Grader 통과 문서를 누적 병합합니다.
def merge_evidence_docs(
    *,
    previous_docs: List[Dict[str, Any]],
    new_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evidence Merge 단계입니다.

    이전 attempt의 통과 문서와 이번 attempt의 통과 문서를 누적 병합합니다.
    동일 문서가 여러 번 검색될 수 있으므로 id/source/page/chunk/stage/metric 조합으로 중복을 제거합니다.
    """
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for doc in [*(previous_docs or []), *(new_docs or [])]:
        if not isinstance(doc, dict):
            continue

        key = _safe_str(doc.get("id"))
        if not key:
            key = "::".join(
                [
                    _safe_str(doc.get("stage")),
                    _safe_str(doc.get("metric")),
                    _safe_str(doc.get("source_file")),
                    _safe_str(doc.get("page")),
                    _safe_str(doc.get("chunk")),
                ]
            )

        if not key or key in seen:
            continue

        seen.add(key)
        merged.append(doc)

    return merged


def rewrite_rag_queries(
    *,
    queries: List[Dict[str, Any]],
    grader_result: Dict[str, Any],
    movement_reasoning: Dict[str, Any] | None = None,
    retrieved_docs: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """retrieval_grader 피드백을 기반으로 기존 RAG 쿼리를 재작성합니다.

    호출 시점과 재검색 여부는 LangGraph가 제어하고,
    실제 semantic rewrite는 query_rewriter.py에 위임합니다.
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
    """RAG 검색을 1회 실행하고 후보 문서만 반환합니다.

    retrieval_history와 rag_queries 같은 상태 갱신은 LangGraph 노드에서 처리합니다.
    """
    logger = logger or logger_pipeline

    docs = retrieve_coaching_evidence(
        meta,
        movement_reasoning=movement_reasoning or {},
        rag_queries=rag_queries or [],
        logger=logger,
    )

    # rag_queries는 retrieval_node에서 확정한 뒤 전달됩니다.
    # run_retrieval_attempt는 전달받은 rag_queries로 검색만 수행합니다.
    # 재검색 루프에서는 LangGraph state의 rag_queries를 유지하고,
    # 쿼리 목록은 query_rewrite_node에서 재작성

    logger.info(
        "[LangGraph] RAG retrieval attempt=%d candidate_doc_count=%d",
        attempt,
        len(docs or []),
    )

    return docs