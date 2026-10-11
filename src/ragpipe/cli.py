"""Command line interface: `ragpipe ingest | search | ask | sources | eval | serve`."""

from __future__ import annotations

import argparse
import sys
import textwrap

from ragpipe.config import get_settings


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="ragpipe", description="RAG knowledge retrieval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="load, chunk, embed and index a folder or file")
    p.add_argument("path")
    p.add_argument("--reset", action="store_true", help="delete the whole index first")

    p = sub.add_parser("search", help="retrieval only, no LLM call")
    p.add_argument("query")
    p.add_argument("-k", type=int)
    p.add_argument("--source", help="restrict to one indexed file")

    p = sub.add_parser("ask", help="answer a question with Claude, grounded in retrieved passages")
    p.add_argument("question")
    p.add_argument("-k", type=int)

    sub.add_parser("sources", help="list indexed documents")

    p = sub.add_parser("eval", help="score retrieval (and optionally answers) on an eval set")
    p.add_argument("--file", default="eval/questions.jsonl")
    p.add_argument("-k", type=int)
    p.add_argument("--answers", action="store_true", help="also generate answers with Claude (uses API credits)")
    p.add_argument("--judge", action="store_true", help="with --answers: LLM-judge faithfulness")

    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    {
        "ingest": _ingest,
        "search": _search,
        "ask": _ask,
        "sources": _sources,
        "eval": _eval,
        "serve": _serve,
    }[args.command](args)


def _ingest(args: argparse.Namespace) -> None:
    from ragpipe.store import get_vectorstore, ingest_path

    if args.reset:
        get_vectorstore().reset_collection()
    print(f"Ingesting {args.path} (first run downloads the embedding model) ...")
    counts = ingest_path(args.path)
    print(f"Indexed {counts['chunks']} chunks from {counts['files']} files "
          f"({counts['pages_or_docs']} pages/docs) into {get_settings().chroma_dir}")


def _search(args: argparse.Namespace) -> None:
    from ragpipe.store import search

    for n, (doc, score) in enumerate(search(args.query, args.k, source=args.source), start=1):
        page = f" p.{doc.metadata['page']}" if "page" in doc.metadata else ""
        print(f"\n[{n}] score={score:.3f}  {doc.metadata['source']}{page}")
        print(textwrap.indent(textwrap.shorten(doc.page_content, 400), "    "))


def _ask(args: argparse.Namespace) -> None:
    from ragpipe.qa import answer_question

    answer = answer_question(args.question, args.k)
    print(answer.text)
    print("\nSources:")
    for src in answer.sources:
        page = f", page {src['page']}" if src["page"] else ""
        print(f"  [{src['n']}] {src['title']} ({src['source']}{page}) {src['page_url']}")


def _sources(args: argparse.Namespace) -> None:
    from ragpipe.store import list_sources

    rows = list_sources()
    if not rows:
        print("Index is empty. Run `ragpipe ingest data/raw`.")
    for row in rows:
        print(f"{row['chunks']:>5} chunks  {row['source']:<45} {row['title']}")


def _eval(args: argparse.Namespace) -> None:
    from ragpipe.evaluate import evaluate_answers, evaluate_retrieval, load_eval_set

    items = load_eval_set(args.file)
    k = args.k or get_settings().top_k
    retrieval = evaluate_retrieval(items, k)
    for row in retrieval["rows"]:
        status = f"rank {row['rank']}" if row["rank"] else "MISS  "
        context = "" if row["answer_in_context"] else "  (answer text not in retrieved chunks)"
        print(f"{status:<7} {row['question']}{context}")
    print(f"\nRetrieval over {len(items)} questions: hit_rate@{k} = {retrieval['hit_rate']:.2f}, "
          f"MRR = {retrieval['mrr']:.2f}, answer_in_context = {retrieval['answer_in_context']:.2f}")

    if args.answers:
        answers = evaluate_answers(items, k, judge=args.judge)
        for row in answers["rows"]:
            flags = "match" if row["answer_match"] else "NO MATCH"
            if "faithful" in row:
                flags += ", faithful" if row["faithful"] else f", UNSUPPORTED: {row['unsupported_claims']}"
            print(f"- {row['question']}  [{flags}]")
        print(f"\nAnswer match = {answers['answer_match']:.2f}", end="")
        if "faithfulness" in answers:
            print(f", faithfulness = {answers['faithfulness']:.2f}", end="")
        print()


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("ragpipe.api:app", port=args.port)


if __name__ == "__main__":
    main()
