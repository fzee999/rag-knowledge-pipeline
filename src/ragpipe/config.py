"""Runtime settings, read from environment variables (and a .env file in the working directory)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import find_dotenv, load_dotenv

DEFAULT_MODELS = {"ollama": "llama3.1:8b", "anthropic": "claude-opus-5-5"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_effort: str
    ollama_num_ctx: int
    embedding_model: str
    chroma_dir: str
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_mode: str = "hybrid_rerank"
    retrieval_candidates: int = 20
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache
def get_settings() -> Settings:
    # usecwd=True: read the .env of the app being run, not the one next to this package.
    load_dotenv(find_dotenv(usecwd=True))
    # RAG_* prefix: generic names like CLAUDE_EFFORT can collide with variables set by other tools.
    provider = os.getenv("RAG_PROVIDER", "ollama").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"RAG_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {provider!r}")
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("RAG_MODEL") or DEFAULT_MODELS[provider],
        llm_effort=os.getenv("RAG_EFFORT", "medium"),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chroma_dir=os.getenv("CHROMA_DIR", ".chroma"),
        collection=os.getenv("CHROMA_COLLECTION", "documents"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "250")),
        top_k=int(os.getenv("TOP_K", "5")),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid_rerank"),
        retrieval_candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "20")),
        rerank_model=os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
    )
