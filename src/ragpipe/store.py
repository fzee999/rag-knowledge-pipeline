"""Vector store: index chunks in Chroma and run semantic search over them."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragpipe.chunking import split_documents
from ragpipe.config import Settings, get_settings
from ragpipe.embeddings import get_embeddings
from ragpipe.loaders import load_directory, load_file
from ragpipe.retrieval import get_reranker, keyword_index_for, rrf_fuse

_BATCH_SIZE = 128


def make_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    """The default store, built from environment settings (cached per process)."""
    settings = get_settings()
    return make_vectorstore(settings, get_embeddings(settings.embedding_model))


def ingest_path(path: str | Path, vectorstore: Chroma | None = None, settings: Settings | None = None) -> dict:
    """Load -> chunk -> embed -> index a folder or a single file. Returns counts."""
    settings = settings or get_settings()
    vectorstore = vectorstore or get_vectorstore()
    path = Path(path)
    docs = load_directory(path) if path.is_dir() else load_file(path)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    index_documents(chunks, vectorstore)
    return {
        "files": len({doc.metadata["source"] for doc in docs}),
        "pages_or_docs": len(docs),
        "chunks": len(chunks),
    }


def index_documents(chunks: list[Document], vectorstore: Chroma) -> int:
    """Add chunks, first removing older chunks of the same source files (idempotent re-ingest)."""
    for source in sorted({chunk.metadata["source"] for chunk in chunks}):
        existing = vectorstore.get(where={"source": source}, include=[])
        if existing["ids"]:
            vectorstore.delete(ids=existing["ids"])
    for start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[start:start + _BATCH_SIZE]
        vectorstore.add_documents(batch, ids=[chunk.metadata["chunk_id"] for chunk in batch])
    return len(chunks)


RETRIEVAL_MODES = ("vector", "hybrid", "rerank", "hybrid_rerank")


def search(
    query: str,
    k: int | None = None,
    vectorstore: Chroma | None = None,
    source: str | None = None,
    mode: str | None = None,
) -> list[tuple[Document, float]]:
    """Top-k chunks for `query`, best first. `source` restricts to one document, e.g. "nist/NIST.CSWP.29.pdf".

    Modes (RETRIEVAL_MODE):
      vector         cosine similarity only; score = relevance in [0, 1]
      hybrid         BM25 + vector, fused by reciprocal rank; score = RRF score
      rerank         vector top-N candidates rescored by a cross-encoder; score = cross-encoder logit
      hybrid_rerank  hybrid top-N candidates rescored by the cross-encoder
    """
    settings = get_settings()
    k = k or settings.top_k
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}, got {mode!r}")
    vectorstore = vectorstore or get_vectorstore()

    n_candidates = k if mode == "vector" else max(settings.retrieval_candidates, k)
    candidates = vectorstore.similarity_search_with_relevance_scores(
        query, k=n_candidates, filter={"source": source} if source else None,
    )
    if mode in ("hybrid", "hybrid_rerank"):
        keyword_hits = keyword_index_for(vectorstore).search(query, n_candidates, source=source)
        candidates = rrf_fuse(candidates, keyword_hits, limit=n_candidates)
    if mode in ("rerank", "hybrid_rerank"):
        return get_reranker(settings.rerank_model).rerank(query, (doc for doc, _ in candidates), k)
    return candidates[:k]


def list_sources(vectorstore: Chroma | None = None) -> list[dict]:
    """One row per indexed document: source, title, url and chunk count."""
    vectorstore = vectorstore or get_vectorstore()
    rows: dict[str, dict] = {}
    for meta in vectorstore.get(include=["metadatas"])["metadatas"]:
        row = rows.setdefault(meta["source"], {
            "source": meta["source"],
            "title": meta.get("title", meta["source"]),
            "page_url": meta.get("page_url", ""),
            "chunks": 0,
        })
        row["chunks"] += 1
    return sorted(rows.values(), key=lambda row: row["source"])


def get_source_chunks(source: str, vectorstore: Chroma | None = None) -> list[Document]:
    """All chunks of one document, in reading order."""
    vectorstore = vectorstore or get_vectorstore()
    data = vectorstore.get(where={"source": source}, include=["documents", "metadatas"])
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(data["documents"], data["metadatas"])]
    return sorted(docs, key=lambda doc: (doc.metadata.get("page", 0), doc.metadata.get("start_index", 0)))
