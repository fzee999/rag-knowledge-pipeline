"""Embeddings: a LangChain `Embeddings` backed by FastEmbed (ONNX on CPU, no API key).

Anthropic doesn't offer an embeddings model, so embeddings run locally. The default
model, BAAI/bge-small-en-v1.5 (384 dims), is small and good for English retrieval.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        cache_dir = cache_dir or str(Path.home() / ".cache" / "fastembed")
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the model's query-side formatting (bge uses a query instruction).
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embeddings(model_name: str) -> LocalEmbeddings:
    return LocalEmbeddings(model_name)
