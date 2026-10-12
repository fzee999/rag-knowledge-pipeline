# RAG Knowledge Retrieval Pipeline

Document ingestion, chunking, embeddings and vector search over real enterprise content, with
grounded, cited answers from a local LLM (Ollama) or Claude. Built with LangChain, Chroma and FastEmbed.

```mermaid
flowchart LR
    A[PDF / Markdown / HTML] --> B[Loaders<br/>clean text + citation metadata]
    B --> C[Chunker<br/>heading-aware, small-chunk merge]
    C --> D[Embeddings<br/>bge-small, local ONNX]
    D --> E[(Chroma<br/>cosine index)]
    Q[Question] --> F[Hybrid retriever<br/>vector + BM25, rank fusion]
    E --> F
    F --> R[Cross-encoder rerank<br/>top 20 → top 5]
    R --> G
    G[LLM: Ollama llama3.1 or Claude<br/>answers only from passages, cites them]
    G --> H[Answer + sources]
```

## What's inside

| Stage | Module | Notes |
|---|---|---|
| Ingestion | `ragpipe/loaders.py` | PDF (one doc per page, running headers/footers removed), Markdown (front matter and Hugo shortcodes stripped), HTML (main content only). Every document gets `source`, `title`, `page`, `page_url` metadata. |
| Chunking | `ragpipe/chunking.py` | Recursive splitter that breaks at headings, then paragraphs, then sentence ends. Chunks under 200 characters are merged into a neighbour. Each chunk gets a title header and a stable ID (re-ingesting replaces chunks, never duplicates them). |
| Embeddings | `ragpipe/embeddings.py` | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX on CPU, no API key), with separate query and passage encoders. |
| Vector search | `ragpipe/store.py` | Persistent Chroma with cosine distance, top-k search, per-document filter. |
| Hybrid search + reranking | `ragpipe/retrieval.py` | BM25 keyword search fused with vector search by reciprocal rank, then the top 20 rescored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`, local ONNX). `RETRIEVAL_MODE` = `vector`, `hybrid`, `rerank` or `hybrid_rerank` (default). |
| Grounded Q&A | `ragpipe/qa.py`, `ragpipe/llm.py` | The LLM answers only from numbered passages and cites them, or says the documents don't cover the question. Provider is switchable: **Ollama** (default, local, free) or **Claude** (`claude-opus-5-5`, adaptive thinking, server-side refusal fallback). |
| Evaluation | `ragpipe/evaluate.py`, `eval/questions.jsonl` | hit-rate@k, MRR and answer-in-context (free); answer match and LLM-judged faithfulness (optional). |
| Interfaces | `ragpipe/cli.py`, `ragpipe/api.py` | `ragpipe` CLI and a FastAPI service (`/ingest`, `/query`, `/search`, `/sources`). |

## Data: real, public documents

`data/sources.json` lists 12 real documents. `scripts/download_corpus.py` fetches them into `data/raw/`; they aren't committed.

| Source | Format | License |
|---|---|---|
| [GitLab Handbook](https://handbook.gitlab.com/): values, travel & expense policy, spending company money, confidentiality levels, password guidelines, ransomware response, acceptable use, anti-harassment, mission, time-off types | Markdown, HTML | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), © GitLab B.V. |
| [NIST Cybersecurity Framework 2.0](https://doi.org/10.6028/NIST.CSWP.29) (CSWP 29) | PDF, 32 pages | Public domain (U.S. Government work) |
| [NIST SP 800-63B-4](https://doi.org/10.6028/NIST.SP.800-63B-4), Digital Identity Guidelines: Authentication | PDF, 129 pages | Public domain (U.S. Government work) |

Together: 12 files, 171 pages/docs, **954 chunks** (1,000-character chunks, 250-character overlap).

## Quickstart (Windows)

Prerequisites: Python 3.11 or 3.12, and [Ollama](https://ollama.com) with a tool-capable model:

```bash
ollama pull llama3.1:8b
```

Then:

```bash
py -3.12 -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install -e ".[dev]"
```
```bash
python scripts/download_corpus.py
```
```bash
ragpipe ingest data/raw
```
```bash
ragpipe ask "How long do I have to submit an expense report?"
```

No API key is needed. To use Claude instead, copy `.env.example` to `.env` and set
`RAG_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.

Other commands: `ragpipe search "<query>"` (retrieval only), `ragpipe sources`, `ragpipe eval`,
`ragpipe serve` (API docs at http://127.0.0.1:8000/docs).

## Evaluation

`eval/questions.jsonl` has 17 questions about the corpus. Each one names the document(s) that answer it
and key phrases the answer must contain. The questions were hand-written; every expected phrase was
checked against the downloaded source text.

```bash
ragpipe eval                    # retrieval metrics, free
ragpipe eval --answers --judge  # + answer match and faithfulness (LLM calls)
```

Results (k=5, chunk size 1000, `llama3.1:8b` on an RTX 4060 laptop GPU; answers are checked for key
phrases as whole words):

| Configuration | hit_rate@5 | MRR | answer_in_context | answer_match | faithfulness |
|---|---|---|---|---|---|
| vector, overlap 150 (baseline) | 1.00 | 0.97 | 0.88 | 0.88 | 1.00 |
| hybrid_rerank, overlap 150 | 1.00 | 0.97 | 0.94 | 0.94 | 0.94* |
| **hybrid_rerank, overlap 250** (default) | 1.00 | 0.97 | **1.00** | 0.94** | 0.94* |

\* The 8B judge flagged the same answer in both runs; every flagged claim is in the source and the
retrieved passages, so verified faithfulness is 1.00.
\** The one miss answered "No [1][6]" to "Should we force users to change their passwords periodically?",
which is correct but too terse for the phrase check.

- **answer_in_context** is the strictest retrieval metric: do the top-5 chunks contain the answer text?
- **Hybrid + rerank fixed "the six functions of CSF 2.0"**, where vector search returned table-of-contents
  pages. Neither BM25 nor reranking alone helped; the combination did. Per-mode retrieval at overlap
  150 / 250: vector 0.88 / 0.88, hybrid 0.88 / 0.94, rerank 0.88 / 1.00, hybrid_rerank 0.94 / 1.00.
- **A 250-character chunk overlap fixed "minimum password length (single-factor)".** With 150, the NIST
  rule pair "15 characters (single-factor) / eight characters (multi-factor)" was split so that the
  better-ranking chunk held only the 8-character rule. Chosen by sweeping 150/250/300, with no regressions.
- **Header/footer stripping and small-chunk merging** cut wasted top-5 slots (tiny header-only chunks)
  from 11/85 to 0/85.
- 17 questions is a small set: these are strong signals, not proof that the system generalizes.

## Next improvements (measure each with `ragpipe eval`)

- A bigger eval set (including unanswerable and multi-document questions) to confirm the gains generalize.
- A stronger, independent judge model for faithfulness.
- Strip table-of-contents dot leaders at ingest.

## Tests

```bash
pytest
```

22 tests, all offline: fake embeddings, a fake cross-encoder and a fake chat model, so no model
download, Ollama or API key is needed.

## Related

[agentic-doc-assistant](../agentic-doc-assistant) builds a LangGraph agent on top of this pipeline.
