import os

# Tests stay offline: plain vector search unless a test picks a mode (the default
# hybrid_rerank mode would load the real cross-encoder model).
os.environ["RETRIEVAL_MODE"] = "vector"

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragpipe.chunking import split_documents
from ragpipe.config import Settings
from ragpipe.store import index_documents, make_vectorstore


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="ollama",
        llm_model="test-model",
        llm_effort="low",
        ollama_num_ctx=4096,
        embedding_model="fake",
        chroma_dir=str(tmp_path / "chroma"),
        collection="test",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
    )


@pytest.fixture
def vectorstore(settings):
    """A throwaway Chroma index with fake embeddings, so tests need no model download or API key."""
    return make_vectorstore(settings, DeterministicFakeEmbedding(size=32))


@pytest.fixture
def indexed(vectorstore, settings):
    docs = [
        Document(page_content="Expense claims over 90 days will not be reimbursed. " * 8,
                 metadata={"source": "policy/expenses.md", "title": "Expenses"}),
        Document(page_content="Passwords must be at least 15 characters. " * 8,
                 metadata={"source": "nist/passwords.pdf", "title": "Passwords", "page": 2}),
        Document(page_content="Authenticators are disabled after 100 failed attempts. " * 8,
                 metadata={"source": "nist/passwords.pdf", "title": "Passwords", "page": 1}),
    ]
    index_documents(split_documents(docs, settings.chunk_size, settings.chunk_overlap), vectorstore)
    return vectorstore
