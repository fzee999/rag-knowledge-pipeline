"""Chunking: split Documents into overlapping, retrievable passages."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prefer to break at markdown headings, then paragraphs, then sentence ends (regexes).
SEPARATORS = [r"\n## ", r"\n### ", r"\n#### ", r"\n\n", r"\n", r"(?<=[.!?]) ", " ", ""]


def split_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 250,
    min_chunk_chars: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        is_separator_regex=True,
        add_start_index=True,
    )
    chunks = merge_small_chunks(splitter.split_documents(docs), min_chunk_chars)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = chunk_id(chunk)
        # Contextual chunk header: the embedding sees which document a passage
        # belongs to, which helps short passages like "Tier 2 approval" retrieve well.
        title = chunk.metadata.get("title")
        if title:
            chunk.page_content = f"{title}\n\n{chunk.page_content}"
    return chunks


def merge_small_chunks(chunks: list[Document], min_chars: int) -> list[Document]:
    """Fold chunks shorter than `min_chars` (a lone heading, a figure caption) into the next
    chunk of the same document, or the previous one if there is no next.

    On their own they carry too little meaning to retrieve well, and they crowd out real passages.
    """
    merged: list[Document] = []
    carry: Document | None = None
    for chunk in chunks:
        if carry is not None:
            if carry.metadata.get("source") == chunk.metadata.get("source"):
                chunk = Document(page_content=f"{carry.page_content}\n\n{chunk.page_content}", metadata=chunk.metadata)
            else:
                _append_to_previous(merged, carry)
            carry = None
        if len(chunk.page_content) < min_chars:
            carry = chunk
        else:
            merged.append(chunk)
    if carry is not None:
        _append_to_previous(merged, carry)
    return merged


def _append_to_previous(merged: list[Document], small: Document) -> None:
    if merged and merged[-1].metadata.get("source") == small.metadata.get("source"):
        merged[-1].page_content = f"{merged[-1].page_content}\n\n{small.page_content}"
    else:
        merged.append(small)  # a whole document that's short: keep it


def chunk_id(chunk: Document) -> str:
    """Stable ID so re-ingesting the same file overwrites instead of duplicating."""
    meta = chunk.metadata
    key = f"{meta.get('source')}|{meta.get('page', 0)}|{meta.get('start_index', 0)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
