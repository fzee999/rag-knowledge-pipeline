"""REST API (local development): `ragpipe serve`, then open http://127.0.0.1:8000/docs."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ragpipe.qa import answer_question
from ragpipe.store import ingest_path, list_sources, search

app = FastAPI(title="RAG Knowledge Pipeline")


class IngestRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str
    k: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sources")
def sources() -> list[dict]:
    return list_sources()


@app.get("/search")
def search_endpoint(q: str, k: int = 5, source: str | None = None) -> list[dict]:
    return [
        {"score": round(score, 3), "text": doc.page_content, **doc.metadata}
        for doc, score in search(q, k, source=source)
    ]


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    path = Path(request.path).resolve()
    # Only allow ingesting files under the folder the server was started from.
    if not path.is_relative_to(Path.cwd().resolve()):
        raise HTTPException(400, "path must be inside the server's working directory")
    if not path.exists():
        raise HTTPException(404, f"{request.path} not found")
    return ingest_path(path)


@app.post("/query")
def query(request: QueryRequest) -> dict:
    answer = answer_question(request.question, request.k)
    return {"answer": answer.text, "sources": answer.sources}
