

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scripts.ingest_rag")


COACH_KB_PATH = os.getenv("COACH_KB_PATH", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_coach_kb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "coach_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")


def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _join_list(v: Any) -> str:
    if isinstance(v, list):
        return " / ".join([_safe_str(x) for x in v if _safe_str(x)])
    return _safe_str(v)


def _doc_text(d: Dict[str, Any]) -> str:
    stage = _safe_str(d.get("stage"))
    metric = _safe_str(d.get("metric"))
    band = _safe_str(d.get("score_band"))
    title = _safe_str(d.get("title"))
    content = _safe_str(d.get("content"))
    summary = _safe_str(d.get("summary"))
    cause = _join_list(d.get("cause"))
    impact = _join_list(d.get("impact"))
    fix = _join_list(d.get("fix"))
    checklist = _join_list(d.get("checklist"))
    drills = _join_list(d.get("drills"))

    extra = " ".join(
        [
            f"요약:{summary}" if summary else "",
            f"원인:{cause}" if cause else "",
            f"영향:{impact}" if impact else "",
            f"교정:{fix}" if fix else "",
            f"체크:{checklist}" if checklist else "",
            f"개선방안:{drills}" if drills else "",
        ]
    ).strip()

    base = f"[{stage}] [{metric}] [{band}] {title} {content}".strip()
    return f"{base} {extra}".strip()


def _read_kb(path: str) -> list[Dict[str, Any]]:
    path = _safe_str(path).strip()
    if not path:
        raise ValueError("COACH_KB_PATH is empty. Pass --kb-path or set COACH_KB_PATH.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except Exception as e:
        raise RuntimeError(f"RAG KB read failed path={path} err={e}") from e

    if not raw:
        return []

    if "\n" in raw and not raw.lstrip().startswith("["):
        docs: list[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                docs.append(item)
        return docs

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"RAG KB parse failed path={path} err={e}") from e

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("documents"), list):
        return [x for x in obj.get("documents") if isinstance(x, dict)]
    return []


class E5LangChainEmbeddings:
    """LangChain-compatible embedding wrapper for multilingual-e5 models."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [f"passage: {_safe_str(t)}" for t in texts]
        return self.model.encode(passages, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        query = f"query: {_safe_str(text)}"
        return self.model.encode([query], normalize_embeddings=True).tolist()[0]


def _to_documents(docs: list[Dict[str, Any]]):
    from langchain_core.documents import Document

    documents: list[Document] = []
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            continue

        did = _safe_str(d.get("id") or f"kb_{i}").strip() or f"kb_{i}"
        metadata = {
            "id": did,
            "stage": _safe_str(d.get("stage")),
            "metric": _safe_str(d.get("metric")),
            "score_band": _safe_str(d.get("score_band")),
            "title": _safe_str(d.get("title")),
            "summary": _safe_str(d.get("summary")),
            "cause": _join_list(d.get("cause")),
            "impact": _join_list(d.get("impact")),
            "fix": _join_list(d.get("fix")),
            "checklist": _join_list(d.get("checklist")),
            "drills": _join_list(d.get("drills")),
            "tags": _safe_str(d.get("tags")),
            "doc_type": "json_kb",
        }
        documents.append(Document(page_content=_doc_text(d), metadata=metadata))
    return documents


def ingest_json_kb(
    kb_path: str,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    reset: bool = False,
    batch_size: int = 64,
) -> int:
    try:
        try:
            from langchain_chroma import Chroma
        except Exception:
            from langchain_community.vectorstores import Chroma
    except Exception as e:
        raise RuntimeError(
            "LangChain RAG deps missing. install langchain langchain-community langchain-chroma chromadb sentence-transformers."
        ) from e

    chroma_path = Path(chroma_dir)
    if reset and chroma_path.exists():
        logger.info("Removing existing Chroma directory: %s", chroma_path)
        shutil.rmtree(chroma_path)

    chroma_path.mkdir(parents=True, exist_ok=True)

    raw_docs = _read_kb(kb_path)
    documents = _to_documents(raw_docs)
    if not documents:
        logger.warning("No documents to ingest. kb_path=%s", kb_path)
        return 0

    embeddings = E5LangChainEmbeddings(embed_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )

    for start in range(0, len(documents), batch_size):
        vectorstore.add_documents(documents[start : start + batch_size])

    logger.info(
        "RAG KB ingested count=%d chroma_dir=%s collection=%s model=%s",
        len(documents),
        chroma_path,
        collection_name,
        embed_model,
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSON/JSONL coaching KB into Chroma for RAG.")
    parser.add_argument("--kb-path", default=COACH_KB_PATH, help="JSON/JSONL KB file path")
    parser.add_argument("--chroma-dir", default=CHROMA_DIR, help="Chroma persistence directory")
    parser.add_argument("--collection", default=CHROMA_COLLECTION, help="Chroma collection name")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="SentenceTransformer embedding model")
    parser.add_argument("--batch-size", type=int, default=64, help="VectorStore add batch size")
    parser.add_argument("--reset", action="store_true", help="Delete existing Chroma directory before ingest")
    args = parser.parse_args()

    ingest_json_kb(
        kb_path=args.kb_path,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        reset=args.reset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()