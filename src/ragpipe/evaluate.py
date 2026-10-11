"""Evaluation.

Retrieval (free, no API calls):
  - hit_rate@k: share of questions where an expected source appears in the top-k chunks
  - MRR: mean reciprocal rank of the first chunk from an expected source
  - answer_in_context: share of questions whose top-k chunks contain the expected key phrases

Answers (optional, calls Claude):
  - answer_match: the answer contains every expected key phrase
  - faithfulness (--judge): an LLM judge checks every claim is supported by the retrieved passages
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from ragpipe.llm import get_chat_model
from ragpipe.qa import answer_question
from ragpipe.store import search

JUDGE_PROMPT = """You are grading a RAG system. Decide whether every factual claim in the ANSWER is supported by the PASSAGES.
Ignore citation formatting. A statement that the documents don't cover something counts as supported.

<passages>
{passages}
</passages>

<question>{question}</question>

<answer>
{answer}
</answer>"""


class FaithfulnessVerdict(BaseModel):
    unsupported_claims: list[str] = Field(description="Claims in the answer that the passages don't support")
    faithful: bool = Field(description="True if every claim is supported by the passages")


def load_eval_set(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def answer_matches(answer: str, expected: list[str]) -> bool:
    """Every expected phrase must appear (case-insensitive); "a|b" accepts either alternative."""
    text = answer.lower()
    return all(any(alt.lower() in text for alt in phrase.split("|")) for phrase in expected)


def evaluate_retrieval(items: list[dict], k: int, vectorstore: Chroma | None = None) -> dict:
    rows = []
    for item in items:
        hits = search(item["question"], k, vectorstore=vectorstore)
        retrieved = [doc.metadata["source"] for doc, _ in hits]
        rank = next((i for i, src in enumerate(retrieved, start=1) if src in item["expected_sources"]), None)
        # Stricter than hit rate: do the retrieved passages actually contain the answer?
        context = " ".join(doc.page_content for doc, _ in hits)
        rows.append({
            "question": item["question"],
            "rank": rank,
            "answer_in_context": answer_matches(context, item.get("answer_contains", [])),
            "retrieved": retrieved,
        })
    n = len(rows) or 1
    return {
        "k": k,
        "hit_rate": sum(row["rank"] is not None for row in rows) / n,
        "mrr": sum(1 / row["rank"] for row in rows if row["rank"]) / n,
        "answer_in_context": sum(row["answer_in_context"] for row in rows) / n,
        "rows": rows,
    }


def evaluate_answers(items: list[dict], k: int, vectorstore: Chroma | None = None, judge: bool = False) -> dict:
    judge_llm = get_chat_model(effort="low").with_structured_output(FaithfulnessVerdict, method="json_schema") if judge else None
    rows = []
    for item in items:
        answer = answer_question(item["question"], k, vectorstore=vectorstore)
        row = {
            "question": item["question"],
            "answer": answer.text,
            "answer_match": answer_matches(answer.text, item.get("answer_contains", [])),
        }
        if judge_llm:
            verdict = judge_llm.invoke(JUDGE_PROMPT.format(
                passages=answer.passages, question=item["question"], answer=answer.text,
            ))
            row["faithful"] = verdict.faithful
            row["unsupported_claims"] = verdict.unsupported_claims
        rows.append(row)
    n = len(rows) or 1
    summary = {"answer_match": sum(row["answer_match"] for row in rows) / n, "rows": rows}
    if judge:
        summary["faithfulness"] = sum(row["faithful"] for row in rows) / n
    return summary
