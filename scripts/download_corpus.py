"""Download the real document corpus listed in data/sources.json into data/raw/.

The corpus is public, openly licensed enterprise content:
  - GitLab Handbook policy pages (CC BY-SA 4.0)
  - NIST publications (public domain, U.S. Government works)

Files are fetched rather than committed so the repo stays small and the
original licenses/attribution stay with their sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources.json"
DEST = ROOT / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    headers = {"User-Agent": "rag-knowledge-pipeline/0.1 (corpus download)"}

    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        for src in sources:
            target = DEST / src["filename"]
            if target.exists() and not args.force:
                print(f"skip   {src['filename']} (already downloaded)")
                continue
            response = client.get(src["url"])
            response.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            print(f"saved  {src['filename']} ({len(response.content) / 1024:,.0f} KB)")

    print(f"\nCorpus ready in {DEST.relative_to(ROOT)}/ - next: ragpipe ingest data/raw")


if __name__ == "__main__":
    main()
