import pytest
from langchain_core.documents import Document

import ragpipe.store
from ragpipe.retrieval import KeywordIndex, Reranker, rrf_fuse, tokenize
from ragpipe.store import search


def doc(chunk_id: str, text: str = "") -> Document:
    return Document(page_content=text or chunk_id, metadata={"chunk_id": chunk_id, "source": "a.md"})


class FakeCrossEncoder:
    """Scores a passage 1.0 if it contains every query word, else 0.0."""

    def rerank(self, query, texts):
        words = query.lower().split()
        return [float(all(w in text.lower() for w in words)) for text in texts]


def test_tokenize_drops_stopwords_and_keeps_numbers():
    assert tokenize("What is the minimum of 15 characters?") == ["minimum", "15", "characters"]


def test_bm25_ranks_exact_terms_and_skips_non_matches():
    index = KeywordIndex(
        ["1", "2", "3"],
        ["passwords need 15 characters", "expense claims within 90 days", "the framework has six functions"],
        [{"source": "nist.pdf"}, {"source": "expenses.md"}, {"source": "csf.pdf"}],
    )
    hits = index.search("15 characters", k=3)
    assert [d.page_content for d, _ in hits] == ["passwords need 15 characters"]
    assert index.search("15 characters", k=3, source="expenses.md") == []


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = rrf_fuse([(a, 0.9), (b, 0.8), (c, 0.7)], [(c, 12.0), (a, 3.0)])
    assert [d.metadata["chunk_id"] for d, _ in fused] == ["a", "c", "b"]


def test_reranker_orders_by_cross_encoder_score():
    docs = [doc("x", "six functions overview"), doc("y", "govern identify protect detect respond recover")]
    ranked = Reranker(FakeCrossEncoder()).rerank("govern recover", docs, k=1)
    assert ranked[0][0].metadata["chunk_id"] == "y"


@pytest.mark.parametrize("mode", ["vector", "hybrid", "rerank", "hybrid_rerank"])
def test_every_search_mode_returns_k_results_and_respects_source(indexed, monkeypatch, mode):
    monkeypatch.setattr(ragpipe.store, "get_reranker", lambda name: Reranker(FakeCrossEncoder()))
    hits = search("15 characters", k=2, vectorstore=indexed, mode=mode)
    assert len(hits) == 2
    filtered = search("15 characters", k=2, vectorstore=indexed, source="policy/expenses.md", mode=mode)
    assert {d.metadata["source"] for d, _ in filtered} == {"policy/expenses.md"}


def test_rerank_puts_the_exact_passage_first(indexed, monkeypatch):
    monkeypatch.setattr(ragpipe.store, "get_reranker", lambda name: Reranker(FakeCrossEncoder()))
    top, _ = search("15 characters", k=1, vectorstore=indexed, mode="hybrid_rerank")[0]
    assert "15 characters" in top.page_content


def test_unknown_mode_is_rejected(indexed):
    with pytest.raises(ValueError):
        search("anything", vectorstore=indexed, mode="magic")
