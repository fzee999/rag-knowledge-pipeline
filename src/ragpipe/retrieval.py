"""Retrieval upgrades on top of vector search.

- KeywordIndex: BM25 over every indexed chunk. Catches exact terms that embeddings blur
  ("15 characters", "single-factor", "Tier 4").
- rrf_fuse: reciprocal rank fusion, which merges ranked lists without needing comparable scores.
- Reranker: a cross-encoder that reads query and passage together and rescores candidates.
  Slower than embeddings, so it only sees the top candidates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its may must of on or our "
    "should that the their there these this to was what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class KeywordIndex:
    """BM25 index over the chunks stored in a Chroma collection."""

    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        self.ids = ids
        self.docs = [Document(id=i, page_content=t, metadata=m) for i, t, m in zip(ids, texts, metadatas)]
        self._bm25 = BM25Okapi([tokenize(text) for text in texts]) if texts else None

    @classmethod
    def from_vectorstore(cls, vectorstore: Chroma) -> KeywordIndex:
        data = vectorstore.get(include=["documents", "metadatas"])
        return cls(data["ids"], data["documents"], data["metadatas"])

    def search(self, query: str, k: int, source: str | None = None) -> list[tuple[Document, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        hits = [
            (self.docs[i], float(scores[i]))
            for i in ranked
            if scores[i] > 0 and (source is None or self.docs[i].metadata.get("source") == source)
        ]
        return hits[:k]


_keyword_cache: dict[int, tuple[frozenset[str], KeywordIndex]] = {}


def keyword_index_for(vectorstore: Chroma) -> KeywordIndex:
    """Build the BM25 index once per store; rebuild when the set of chunk IDs changes (after ingest)."""
    ids = frozenset(vectorstore.get(include=[])["ids"])
    cached = _keyword_cache.get(id(vectorstore))
    if cached is None or cached[0] != ids:
        cached = (ids, KeywordIndex.from_vectorstore(vectorstore))
        _keyword_cache[id(vectorstore)] = cached
    return cached[1]


def rrf_fuse(*ranked_lists: list[tuple[Document, float]], k: int = 60, limit: int | None = None) -> list[tuple[Document, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank). Rank-based, so BM25 and
    cosine scores never need to be on the same scale. k=60 is the value from the original paper."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked, start=1):
            key = doc.metadata.get("chunk_id") or doc.id or doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    fused = sorted(scores, key=scores.get, reverse=True)
    return [(docs[key], scores[key]) for key in fused[:limit]]


class Reranker:
    """Cross-encoder reranking. `model` needs a `rerank(query, texts) -> scores` method (FastEmbed's
    TextCrossEncoder has one), so tests can pass a fake."""

    def __init__(self, model):
        self._model = model

    def rerank(self, query: str, docs: Iterable[Document], k: int) -> list[tuple[Document, float]]:
        docs = list(docs)
        if not docs:
            return []
        scores = list(self._model.rerank(query, [doc.page_content for doc in docs]))
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:k]]


@lru_cache
def get_reranker(model_name: str) -> Reranker:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return Reranker(TextCrossEncoder(model_name=model_name, cache_dir=str(Path.home() / ".cache" / "fastembed")))
