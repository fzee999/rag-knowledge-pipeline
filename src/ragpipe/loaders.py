"""Document ingestion: turn files on disk into LangChain Documents with citation metadata.

Every Document carries `source` (path relative to the ingested folder), `title`,
`file_type`, and `page` for PDFs, so answers can cite exactly where a fact came from.
If a `sources.json` manifest sits in or next to the folder, its title/url/license
fields are attached too.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm"}

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FRONT_MATTER_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
# Hugo shortcodes such as {{% panel ... %}} or {{< youtube ... >}}
_SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}")
_BLANK_LINES = re.compile(r"\n{3,}")


def load_directory(root: str | Path, manifest: dict[str, dict] | None = None) -> list[Document]:
    """Load every supported file under `root` (recursively)."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    if manifest is None:
        manifest = find_manifest(root)

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            source = path.relative_to(root).as_posix()
            docs.extend(load_file(path, source=source, extra=manifest.get(source)))
    return docs


def load_file(path: str | Path, source: str | None = None, extra: dict | None = None) -> list[Document]:
    """Load one file. PDFs yield one Document per page; other formats yield one Document."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = _load_pdf(path)
    elif suffix in {".html", ".htm"}:
        docs = [_load_html(path)]
    elif suffix in {".md", ".markdown", ".txt"}:
        docs = [_load_markdown(path)]
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    base = {"source": source or path.name, "file_type": suffix.lstrip(".")}
    for key in ("title", "page_url", "publisher", "license"):
        if extra and extra.get(key):
            base[key] = extra[key]
    for doc in docs:
        doc.metadata = _clean_metadata({**doc.metadata, **base})
    return [doc for doc in docs if doc.page_content.strip()]


def find_manifest(root: Path) -> dict[str, dict]:
    """Return {filename: metadata} from a sources.json in `root` or its parent, if present."""
    for candidate in (root / "sources.json", root.parent / "sources.json"):
        if candidate.is_file():
            entries = json.loads(candidate.read_text(encoding="utf-8"))
            return {entry["filename"]: entry for entry in entries}
    return {}


def _load_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = path.stem.replace("-", " ").title()
    if match := _FRONT_MATTER.match(text):
        if title_match := _FRONT_MATTER_TITLE.search(match.group(1)):
            title = title_match.group(1)
        text = text[match.end():]
    text = _SHORTCODE.sub("", text)
    return Document(page_content=_tidy(text), metadata={"title": title})


def _load_html(path: Path) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "noscript", "nav", "aside", "header", "footer", "form", "svg"]):
        tag.decompose()
    # Prefer the main content container over the whole page chrome.
    body = soup.select_one(".td-content") or soup.find("main") or soup.find("article") or soup.body or soup
    return Document(page_content=_tidy(body.get_text("\n")), metadata={"title": title})


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    pages = _strip_running_headers([page.extract_text() or "" for page in reader.pages])
    return [
        Document(page_content=_tidy(text), metadata={"title": title, "page": number})
        for number, text in enumerate(pages, start=1)
    ]


def _strip_running_headers(pages: list[str], min_share: float = 0.5) -> list[str]:
    """Drop short lines that repeat on at least `min_share` of pages (running headers/footers).

    Otherwise every page carries e.g. "NIST CSWP 29 ... February 26, 2024", and figure-only
    pages become header-only chunks that match any query naming the document.
    """
    if len(pages) < 3:
        return pages
    counts = Counter(line.strip() for text in pages for line in set(text.splitlines()) if line.strip())
    repeated = {line for line, n in counts.items() if n >= len(pages) * min_share and len(line) <= 120}
    return ["\n".join(line for line in text.splitlines() if line.strip() not in repeated) for text in pages]


def _tidy(text: str) -> str:
    lines = (line.rstrip() for line in text.replace("\r\n", "\n").splitlines())
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _clean_metadata(metadata: dict) -> dict:
    """Chroma only stores str/int/float/bool metadata values."""
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
