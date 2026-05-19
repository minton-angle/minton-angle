from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, List, Tuple


logger_reranker = logging.getLogger("app.llm")

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


def _safe_str(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


@lru_cache(maxsize=1)
def get_cross_encoder():
    """Lazy-load CrossEncoder reranker."""
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        logger_reranker.warning("CrossEncoder deps missing err=%s", str(exc))
        return None

    try:
        model = CrossEncoder(CROSS_ENCODER_MODEL)
        logger_reranker.info("CrossEncoder loaded model=%s", CROSS_ENCODER_MODEL)
        return model
    except Exception as exc:
        logger_reranker.warning(
            "CrossEncoder load failed model=%s err=%s",
            CROSS_ENCODER_MODEL,
            str(exc),
        )
        return None


def rerank_with_cross_encoder(
    query: str,
    retrieved_pairs: list,
    *,
    logger: logging.Logger | None = None,
) -> list:
    """Rerank Chroma retrieved pairs with a CrossEncoder.

    Args:
        query: Search query text.
        retrieved_pairs: List of `(doc_obj, distance)` pairs returned by Chroma.
        logger: Optional caller logger.

    Returns:
        List of `(doc_obj, distance, rerank_score)` sorted by rerank_score desc.
        If reranker is unavailable, rerank_score is set to 0.0 and original order is preserved.
    """
    log = logger or logger_reranker

    if not retrieved_pairs:
        return []

    reranker = get_cross_encoder()
    if reranker is None:
        return [(doc_obj, distance, 0.0) for doc_obj, distance in retrieved_pairs]

    pairs = [
        (query, _safe_str(getattr(doc_obj, "page_content", "")))
        for doc_obj, _ in retrieved_pairs
    ]

    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        log.warning("CrossEncoder rerank failed query=%s err=%s", query, str(exc))
        return [(doc_obj, distance, 0.0) for doc_obj, distance in retrieved_pairs]

    reranked = []
    for (doc_obj, distance), score in zip(retrieved_pairs, scores):
        reranked.append((doc_obj, distance, float(score)))

    reranked.sort(key=lambda item: item[2], reverse=True)
    return reranked