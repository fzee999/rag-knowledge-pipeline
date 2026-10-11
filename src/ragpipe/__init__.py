"""RAG knowledge retrieval pipeline: ingest -> chunk -> embed -> index -> retrieve -> answer."""

from ragpipe.chunking import split_documents
from ragpipe.config import Settings, get_settings
from ragpipe.loaders import load_directory, load_file
from ragpipe.qa import Answer, answer_question
from ragpipe.store import (
    get_source_chunks,
    get_vectorstore,
    index_documents,
    ingest_path,
    list_sources,
    search,
)

__all__ = [
    "Answer",
    "Settings",
    "answer_question",
    "get_settings",
    "get_source_chunks",
    "get_vectorstore",
    "index_documents",
    "ingest_path",
    "list_sources",
    "load_directory",
    "load_file",
    "search",
    "split_documents",
]
