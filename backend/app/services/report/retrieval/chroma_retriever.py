from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.services.report.retrieval.retrieval_query_builder import build_rag_queries, metric_query_text
# from app.services.report.retrieval.reranker import rerank_with_cross_encoder



logger_retrieval = logging.getLogger("app.llm")


# ------------------------------------------------------------------
# RAG (Chroma) Settings
# ------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "app/chroma_coach_pdf")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "coach_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
COACH_RAG_TOPK = int(os.getenv("COACH_RAG_TOPK", "22"))
COACH_RAG_MAX_CHARS = int(os.getenv("COACH_RAG_MAX_CHARS", "0"))
COACH_RAG_CANDIDATE_K = int(os.getenv("COACH_RAG_CANDIDATE_K", "8"))
COACH_RAG_PER_QUERY_TOPK = int(os.getenv("COACH_RAG_PER_QUERY_TOPK", "2"))

# 리포트 생성 시점에는 임베딩 적재를 수행하지 않는다.
# backend/scripts/ingest_rag.py를 먼저 실행해 Chroma를 준비한 뒤, 여기서는 검색만 수행한다.

def _safe_str(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


class _E5LangChainEmbeddings:
    """LangChain-compatible embedding wrapper for multilingual-e5 models."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [f"passage: {_safe_str(text)}" for text in texts]
        return self.model.encode(passages, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        query = f"query: {_safe_str(text)}"
        return self.model.encode([query], normalize_embeddings=True).tolist()[0]


@lru_cache(maxsize=1)
def _get_chroma():
    """Lazy-load LangChain Chroma vectorstore for retrieval only."""
    try:
        try:
            from langchain_chroma import Chroma
        except Exception:
            from langchain_community.vectorstores import Chroma
    except Exception as exc:
        logger_retrieval.warning(
            "LangChain RAG deps missing. install langchain langchain-community langchain-chroma chromadb sentence-transformers. err=%s",
            str(exc),
        )
        return None

    try:
        embeddings = _E5LangChainEmbeddings(EMBED_MODEL)
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    except Exception as exc:
        logger_retrieval.warning("LangChain RAG init failed err=%s", str(exc))
        return None

    try:
        existing = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else 0
    except Exception:
        existing = 0

    if not existing:
        logger_retrieval.warning(
            "LangChain RAG collection is empty. Run `python backend/scripts/ingest_rag.py --reset` before generating reports. dir=%s collection=%s",
            CHROMA_DIR,
            CHROMA_COLLECTION,
        )
    else:
        logger_retrieval.info("LangChain RAG collection loaded count=%d dir=%s", existing, CHROMA_DIR)

    return vectorstore




def retrieve_coaching_evidence(
    meta: Dict[str, Any],
    *,
    movement_reasoning: Optional[Dict[str, Any]] = None,
    rag_queries: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Retrieve coaching snippets from Chroma for prompt injection.

    Responsibilities:
    - state에서 전달받은 movement_reasoning / rag_queries와 meta의 weak_metrics를 기반으로 RAG 쿼리를 생성
    - execute Chroma similarity search
    - 현재는 rerank 없이 Chroma similarity 결과를 사용
    - normalize selected documents into retrieved_merged_evidence evidence
    """
    log = logger or logger_retrieval
    vectorstore = _get_chroma()
    if vectorstore is None:
        return []

    queries = build_rag_queries(
        meta or {},
        movement_reasoning=movement_reasoning or {},
        rag_queries=rag_queries or [],
        logger=log,
    )

    log.info(
        "[RAG] 검색 쿼리 개수=%d 쿼리 내용=%s",
        len(queries),
        json.dumps([
            {
                "stage": item.get("stage"),
                "metric": item.get("metric"),
                "q": item.get("q"),
            }
            for item in queries
        ], ensure_ascii=False),
    )

    if not queries:
        return []

    results: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    candidate_k = max(COACH_RAG_CANDIDATE_K, COACH_RAG_PER_QUERY_TOPK)
    per_q = COACH_RAG_PER_QUERY_TOPK

    for query_item in queries:
        text = _safe_str(query_item.get("q"))
        query_stage = _safe_str(query_item.get("stage"))
        query_metric = _safe_str(query_item.get("metric"))

        try:
            retrieved_pairs = vectorstore.similarity_search_with_score(
                text,
                k=candidate_k,
            )
        except Exception as exc:
            log.warning("LangChain RAG query failed q=%s err=%s", text, str(exc))
            continue

        # LangGraph Self-RAG 루프 검증 중에는 rerank를 임시로 사용하지 않습니다.
        # Chroma similarity 결과를 그대로 사용합니다.
        reranked_pairs = [
            (doc_obj, distance, None)
            for doc_obj, distance in retrieved_pairs
        ]

        for rank, (doc_obj, distance, rerank_score) in enumerate(reranked_pairs, start=1):
            metadata = doc_obj.metadata if isinstance(getattr(doc_obj, "metadata", None), dict) else {}
            raw_doc = _safe_str(getattr(doc_obj, "page_content", ""))
            preview = raw_doc.replace("\n", " ").strip()[:300]
            # log.info(
            #     "[RAG] candidate rank=%d stage=%s metric=%s source=%s page=%s chunk=%s distance=%s rerank_score=%s preview=%s",
            #     rank,
            #     query_stage,
            #     query_metric,
            #     _safe_str(metadata.get("source_file")),
            #     _safe_str(metadata.get("page")),
            #     _safe_str(metadata.get("chunk")),
            #     distance,
            #     rerank_score,
            #     preview,
            # )

        selected_pairs = reranked_pairs[:per_q]

        log.info(
        "[RAG query별 검색 결과] stage=%s metric=%s retrieved=%d selected=%d query='%s'",
            query_stage,
            query_metric,
            len(retrieved_pairs),
            len(selected_pairs),
            text,
        )

        for doc_obj, distance, rerank_score in selected_pairs:
            inject_allowed = len(results) < COACH_RAG_TOPK

            metadata = doc_obj.metadata if isinstance(getattr(doc_obj, "metadata", None), dict) else {}
            source_file = _safe_str(metadata.get("source_file"))
            page = _safe_str(metadata.get("page"))
            chunk = _safe_str(metadata.get("chunk"))
            sid = _safe_str(metadata.get("id"))
            if not sid:
                sid = (
                    f"{query_stage}:"
                    f"{query_metric}:"
                    f"{_safe_str(query_item.get('score_band'))}:"
                    f"{source_file}:"
                    f"{page}:"
                    f"{chunk}"
                )
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            raw_doc = _safe_str(getattr(doc_obj, "page_content", ""))
            doc = raw_doc
            preview = raw_doc.replace("\n", " ").strip()

            if COACH_RAG_MAX_CHARS > 0 and len(doc) > COACH_RAG_MAX_CHARS:
                doc = doc[:COACH_RAG_MAX_CHARS].rstrip() + "…"

            log.info(
                "[최종 입력 문서]RAG hit stage=%s metric=%s source=%s page=%s chunk=%s distance=%s rerank_score=%s injected=%s raw_len=%d preview=%s",
                query_stage,
                query_metric,
                source_file,
                page,
                chunk,
                distance,
                rerank_score,
                inject_allowed,
                len(raw_doc),
                preview,
            )

            inj_title = _safe_str(metadata.get("title"))
            inj_summary = _safe_str(metadata.get("summary"))
            inj_cause = _safe_str(metadata.get("cause"))
            inj_impact = _safe_str(metadata.get("impact"))
            inj_fix = _safe_str(metadata.get("fix"))
            inj_check = _safe_str(metadata.get("checklist"))
            inj_drills = _safe_str(metadata.get("drills"))

            parts = []
            if inj_summary:
                parts.append(f"요약: {inj_summary}")
            if inj_cause:
                parts.append(f"원인: {inj_cause}")
            if inj_impact:
                parts.append(f"영향: {inj_impact}")
            if inj_fix:
                parts.append(f"교정: {inj_fix}")
            if inj_check:
                parts.append(f"체크: {inj_check}")
            if inj_drills:
                parts.append(f"개선방법: {inj_drills}")

            inj_content = "\n".join(parts).strip() or doc

            if inject_allowed:
                results.append(
                    {
                        "id": sid,
                        "stage": query_stage,
                        "metric": query_metric,
                        "metric_query": metric_query_text(query_stage, query_metric),
                        "score_band": _safe_str(metadata.get("score_band")),
                        "title": inj_title or source_file,
                        "rerank_score": rerank_score,
                        "content": inj_content,
                        "distance": distance,
                        "doc_type": _safe_str(metadata.get("doc_type")),
                        "source_file": source_file,
                        "page": page,
                        "chunk": chunk,
                    }
                )

    log.info(
        "[RAG] 최종 누적(주입문서) 개수=%d ids=%s",
        len(results),
        [item.get("id") for item in results],
    )
    try:
        stage_counts: Dict[str, int] = {}
        for item in results:
            stage = _safe_str(item.get("stage"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        log.info("[RAG] 주입된 stage_counts=%s", json.dumps(stage_counts, ensure_ascii=False))
    except Exception:
        pass

    return results