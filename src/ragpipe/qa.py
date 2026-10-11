"""Grounded question answering: retrieve passages, then have Claude answer only from them, with citations."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ragpipe.config import get_settings
from ragpipe.llm import get_chat_model, is_refusal
from ragpipe.store import search

SYSTEM_PROMPT = """You answer questions using only the numbered passages provided from an organization's documents.

- Base every statement on the passages and cite the passage numbers you used in square brackets, e.g. [1] or [2][4].
- If the passages don't contain the answer, say the documents don't cover it. Don't fill gaps from general knowledge.
- Lead with a short, direct answer, then add supporting detail only if it helps."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict] = field(default_factory=list)
    passages: str = ""


def format_passages(hits: list[tuple[Document, float]]) -> tuple[str, list[dict]]:
    """Number the retrieved chunks for the prompt and build the matching citation list."""
    blocks, sources = [], []
    for n, (doc, score) in enumerate(hits, start=1):
        meta = doc.metadata
        location = meta["source"] + (f", page {meta['page']}" if "page" in meta else "")
        blocks.append(f"[{n}] ({location})\n{doc.page_content}")
        sources.append({
            "n": n,
            "source": meta["source"],
            "title": meta.get("title", ""),
            "page": meta.get("page"),
            "page_url": meta.get("page_url", ""),
            "score": round(score, 3),
        })
    return "\n\n".join(blocks), sources


def answer_question(
    question: str,
    k: int | None = None,
    vectorstore: Chroma | None = None,
    llm: BaseChatModel | None = None,
) -> Answer:
    hits = search(question, k or get_settings().top_k, vectorstore=vectorstore)
    if not hits:
        return Answer(question, "Nothing is indexed yet. Run `ragpipe ingest <folder>` first.")

    passages, sources = format_passages(hits)
    llm = llm or get_chat_model()
    response = llm.invoke([
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}"),
    ])
    text = "Claude declined to answer this request." if is_refusal(response) else response.text
    return Answer(question, text, sources, passages)
