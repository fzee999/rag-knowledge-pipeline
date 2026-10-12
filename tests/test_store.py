from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragpipe.chunking import merge_small_chunks, split_documents
from ragpipe.evaluate import answer_matches
from ragpipe.loaders import _strip_running_headers
from ragpipe.qa import answer_question
from ragpipe.store import get_source_chunks, index_documents, list_sources, search


def test_chunks_get_stable_unique_ids_and_title_header():
    doc = Document(page_content="word " * 200, metadata={"source": "a.md", "title": "Doc A"})
    first = split_documents([doc], chunk_size=100, chunk_overlap=10, min_chunk_chars=0)
    second = split_documents([doc], chunk_size=100, chunk_overlap=10, min_chunk_chars=0)
    ids = [chunk.metadata["chunk_id"] for chunk in first]
    assert len(first) > 1
    assert len(set(ids)) == len(ids)
    assert ids == [chunk.metadata["chunk_id"] for chunk in second]
    assert all(chunk.page_content.startswith("Doc A\n\n") for chunk in first)


def test_small_chunks_are_merged_into_neighbours():
    chunks = [
        Document(page_content="## Procedure", metadata={"source": "a.md", "start_index": 0}),
        Document(page_content="Step one. " * 30, metadata={"source": "a.md", "start_index": 13}),
        Document(page_content="Short tail.", metadata={"source": "a.md", "start_index": 400}),
        Document(page_content="Tiny doc.", metadata={"source": "b.md", "start_index": 0}),
    ]
    merged = merge_small_chunks(chunks, min_chars=50)
    assert [c.metadata["source"] for c in merged] == ["a.md", "b.md"]
    assert merged[0].page_content.startswith("## Procedure\n\nStep one.")
    assert merged[0].page_content.endswith("Short tail.")
    assert merged[1].page_content == "Tiny doc."


def test_pdf_running_headers_are_removed():
    pages = [f"NIST CSWP 29\nFebruary 26, 2024\nBody text of page {n}." for n in range(4)]
    cleaned = _strip_running_headers(pages)
    assert cleaned == [f"Body text of page {n}." for n in range(4)]


def test_chunks_break_after_sentences_not_before():
    doc = Document(page_content="Claims expire after 90 days. " * 30, metadata={"source": "a.md"})
    for chunk in split_documents([doc], chunk_size=120, chunk_overlap=0):
        assert chunk.page_content.startswith("Claims")


def test_reingesting_a_source_replaces_its_chunks(indexed, settings):
    before = len(indexed.get()["ids"])
    doc = Document(page_content="Expense claims over 90 days will not be reimbursed. " * 8,
                   metadata={"source": "policy/expenses.md", "title": "Expenses"})
    index_documents(split_documents([doc], settings.chunk_size, settings.chunk_overlap), indexed)
    assert len(indexed.get()["ids"]) == before


def test_list_sources_and_reading_order(indexed):
    rows = {row["source"]: row for row in list_sources(indexed)}
    assert set(rows) == {"policy/expenses.md", "nist/passwords.pdf"}
    assert rows["nist/passwords.pdf"]["chunks"] > 1

    pages = [doc.metadata["page"] for doc in get_source_chunks("nist/passwords.pdf", indexed)]
    assert pages == sorted(pages)


def test_search_can_filter_by_source(indexed):
    hits = search("anything", k=10, vectorstore=indexed, source="policy/expenses.md")
    assert hits
    assert {doc.metadata["source"] for doc, _ in hits} == {"policy/expenses.md"}


def test_answer_question_builds_numbered_citations(indexed):
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage("Claims older than 90 days aren't reimbursed [1].")]))
    answer = answer_question("How long do I have to file expenses?", k=2, vectorstore=indexed, llm=fake_llm)
    assert answer.text.startswith("Claims older than 90 days")
    assert [src["n"] for src in answer.sources] == [1, 2]
    assert answer.passages.startswith("[1] (")


def test_answer_matching_supports_alternatives():
    assert answer_matches("The limit is $10,000 per month.", ["10K|10,000"])
    assert not answer_matches("The limit is high.", ["10K|10,000"])
    assert answer_matches("Govern and Identify", ["govern", "identify"])
    assert not answer_matches("risk governance", ["govern"])           # whole words only
    assert not answer_matches("published in 2015", ["15"])
    assert answer_matches("Use 1Password.", ["1Password"])
