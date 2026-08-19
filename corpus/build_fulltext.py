"""One-off script: re-fetches the real, full PDF text for every paper_id
already in corpus/corpus.jsonl (deterministic arXiv PDF URL from paper_id),
and writes corpus/corpus_fulltext.jsonl. Run ONCE, its output committed
pre-built like corpus.jsonl itself -- A1_chunking_study.ipynb loads the
committed file and makes zero live network calls at notebook-execution
time. Free (no API key), reuses corpus/build_corpus.py's existing
download_pdf_text().
"""

from __future__ import annotations

import json
from pathlib import Path

from corpus.build_corpus import download_pdf_text


def paper_id_to_pdf_url(paper_id: str) -> str:
    # "arxiv:2601.00086" -> "https://arxiv.org/pdf/2601.00086"
    arxiv_id = paper_id.removeprefix("arxiv:")
    return f"https://arxiv.org/pdf/{arxiv_id}"


def main() -> None:
    corpus_path = Path(__file__).parent / "corpus.jsonl"
    with open(corpus_path, encoding="utf-8") as f:
        paper_ids = sorted({json.loads(line)["paper_id"] for line in f if line.strip()})

    out_path = Path(__file__).parent / "corpus_fulltext.jsonl"
    n_written = 0
    n_skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for paper_id in paper_ids:
            print(f"Fetching {paper_id}...")
            try:
                full_text = download_pdf_text(paper_id_to_pdf_url(paper_id))
                if len(full_text.strip()) < 500:
                    raise ValueError("extracted text too short, likely a bad parse")
            except Exception as exc:
                print(f"  SKIPPED ({exc})")
                n_skipped += 1
                continue
            f.write(json.dumps({"paper_id": paper_id, "full_text": full_text}) + "\n")
            n_written += 1

    print()
    print(f"Done. Papers written: {n_written}, skipped: {n_skipped} -> {out_path}")


if __name__ == "__main__":
    main()
