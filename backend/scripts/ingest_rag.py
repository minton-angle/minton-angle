from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scripts.ingest_rag")


# backend/scripts/ingest_rag.py -> backend/
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


PDF_DIR = os.getenv("PDF_DIR", os.getenv("COACH_PDF_DIR", "app/data/pdfs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "app/chroma_coach_pdf")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "coach_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "800"))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "120"))


def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _resolve_backend_path(path: str) -> Path:
    """Resolve relative paths from backend/.env against the backend directory."""
    p = Path(_safe_str(path).strip())
    if p.is_absolute():
        return p
    return BASE_DIR / p


def _detect_lang(text: str) -> str:
    """Small heuristic for metadata only. Retrieval uses multilingual embeddings."""
    t = _safe_str(text)
    if not t:
        return "unknown"
    ko_count = sum(1 for ch in t if "가" <= ch <= "힣")
    en_count = sum(1 for ch in t if "a" <= ch.lower() <= "z")
    if ko_count > en_count:
        return "ko"
    if en_count > 0:
        return "en"
    return "unknown"


def _clean_text(text: str) -> str:
    text = _safe_str(text)
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


class E5LangChainEmbeddings:
    """LangChain-compatible embedding wrapper for multilingual-e5 models.

    E5 계열은 문서에는 `passage:`, 쿼리에는 `query:` prefix를 붙이는 방식이 권장된다.
    """

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


def _load_and_split_pdfs(pdf_dir: Path, chunk_size: int, chunk_overlap: int):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found. pdf_dir=%s", pdf_dir)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(chunk_size, 200),
        chunk_overlap=max(min(chunk_overlap, chunk_size // 2), 0),
        separators=["\n\n", "\n", "다. ", "요. ", ". ", " ", ""],
    )

    all_splits = []
    for pdf_path in pdf_paths:
        logger.info("Loading PDF: %s", pdf_path)
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
        except Exception as e:
            logger.warning("PDF load failed path=%s err=%s", pdf_path, e)
            continue

        rel_source = str(pdf_path.relative_to(pdf_dir))
        for doc in docs:
            doc.page_content = _clean_text(doc.page_content)
            doc.metadata = doc.metadata or {}
            doc.metadata["source_file"] = rel_source
            doc.metadata["doc_type"] = "pdf"
            doc.metadata["sport"] = "badminton"
            if "page" in doc.metadata:
                try:
                    doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
                except Exception:
                    pass

        splits = splitter.split_documents(docs)
        for chunk_idx, split in enumerate(splits):
            split.page_content = _clean_text(split.page_content)
            if not split.page_content:
                continue
            split.metadata = split.metadata or {}
            split.metadata["source_file"] = rel_source
            split.metadata["doc_type"] = "pdf"
            split.metadata["sport"] = "badminton"
            split.metadata["chunk"] = chunk_idx + 1
            split.metadata["language"] = _detect_lang(split.page_content)
            all_splits.append(split)

    logger.info("PDF chunks prepared count=%d pdf_dir=%s", len(all_splits), pdf_dir)
    return all_splits


def ingest_pdf_rag(
    pdf_dir: str,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP,
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
            "LangChain RAG deps missing. install langchain langchain-community langchain-chroma chromadb sentence-transformers pypdf."
        ) from e

    pdf_path = _resolve_backend_path(pdf_dir)
    chroma_path = _resolve_backend_path(chroma_dir)

    if reset and chroma_path.exists():
        logger.info("Removing existing Chroma directory: %s", chroma_path)
        shutil.rmtree(chroma_path)

    chroma_path.mkdir(parents=True, exist_ok=True)

    documents = _load_and_split_pdfs(
        pdf_dir=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not documents:
        logger.warning("No PDF chunks to ingest. pdf_dir=%s", pdf_path)
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
        "PDF RAG ingested count=%d pdf_dir=%s chroma_dir=%s collection=%s model=%s chunk_size=%d overlap=%d",
        len(documents),
        pdf_path,
        chroma_path,
        collection_name,
        embed_model,
        chunk_size,
        chunk_overlap,
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF files into Chroma for badminton RAG.")
    parser.add_argument("--pdf-dir", default=PDF_DIR, help="PDF directory path. Relative paths are resolved from backend/.")
    parser.add_argument("--chroma-dir", default=CHROMA_DIR, help="Chroma persistence directory")
    parser.add_argument("--collection", default=CHROMA_COLLECTION, help="Chroma collection name")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="SentenceTransformer embedding model")
    parser.add_argument("--chunk-size", type=int, default=PDF_CHUNK_SIZE, help="PDF text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=PDF_CHUNK_OVERLAP, help="PDF text chunk overlap")
    parser.add_argument("--batch-size", type=int, default=64, help="VectorStore add batch size")
    parser.add_argument("--reset", action="store_true", help="Delete existing Chroma directory before ingest")
    args = parser.parse_args()

    ingest_pdf_rag(
        pdf_dir=args.pdf_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()