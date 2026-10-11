import json

from ragpipe.loaders import load_directory, load_file


def test_markdown_front_matter_and_shortcodes_are_stripped(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(
        '---\ntitle: "Travel Policy"\nweight: 3\n---\n\n## Flights\n\n{{% alert title="Note" %}}\nBook economy.\n{{% /alert %}}\n',
        encoding="utf-8",
    )
    [doc] = load_file(path, source="policy.md")
    assert doc.metadata["title"] == "Travel Policy"
    assert doc.metadata["source"] == "policy.md"
    assert "weight:" not in doc.page_content
    assert "{{" not in doc.page_content
    assert "Book economy." in doc.page_content


def test_html_keeps_main_content_and_drops_page_chrome(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Time Off</title><script>var x = 1;</script></head><body>"
        "<nav>Home | About</nav><main><h1>Time Off</h1><p>Take the time you need.</p></main>"
        "<footer>Copyright</footer></body></html>",
        encoding="utf-8",
    )
    [doc] = load_file(path)
    assert doc.metadata["title"] == "Time Off"
    assert "Take the time you need." in doc.page_content
    for chrome in ("Home | About", "Copyright", "var x"):
        assert chrome not in doc.page_content


def test_directory_attaches_manifest_metadata(tmp_path):
    raw = tmp_path / "raw"
    (raw / "docs").mkdir(parents=True)
    (raw / "docs" / "a.md").write_text("Hello world", encoding="utf-8")
    (raw / "docs" / "ignored.xyz").write_text("not loaded", encoding="utf-8")
    manifest = [{"filename": "docs/a.md", "title": "Doc A", "page_url": "https://example.org/a", "license": "CC BY-SA 4.0"}]
    (tmp_path / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")

    [doc] = load_directory(raw)
    assert doc.metadata["source"] == "docs/a.md"
    assert doc.metadata["title"] == "Doc A"
    assert doc.metadata["page_url"] == "https://example.org/a"
