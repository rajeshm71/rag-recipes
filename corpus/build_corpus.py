"""Builds corpus/corpus.jsonl from real arXiv papers.

Requires the `corpus-build` optional dependency group:
    uv sync --extra corpus-build

Queries arXiv's public Atom API for papers in the given categories, downloads
each PDF, extracts text, and chunks it per SPEC.md §4's fixed policy: 512
tokens per chunk (tiktoken-counted), 64-token overlap, paragraph-boundary
aware, at most 3 chunks kept per paper to avoid one-paper dominance.

This script is shipped for transparency (SPEC.md §4) -- corpus.jsonl is
committed pre-built and does NOT need to be regenerated on every install.

If a paper's PDF fails to download or its text fails to extract, that paper
is skipped with a logged warning; the script keeps going and reports at the
end how many papers/chunks it actually produced versus the target.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
MAX_CHUNKS_PER_PAPER = 3


@dataclass
class ArxivPaper:
    paper_id: str
    title: str
    authors: list[str]
    year: int
    category: str
    pdf_url: str


def _get_tokenizer():
    import tiktoken

    try:
        return tiktoken.encoding_for_model("gpt-4.1-mini")
    except KeyError:
        # gpt-4.1-mini may not be recognized by an older installed tiktoken;
        # cl100k_base is the encoding family GPT-4-class chat models use.
        return tiktoken.get_encoding("cl100k_base")


def search_arxiv(
    category: str,
    max_results: int,
    start: int = 0,
    date_from: str = "20240101000000",
    date_to: str = "20251231235959",
) -> list[ArxivPaper]:
    import requests

    # Filter by submission date server-side rather than fetching the newest
    # results and discarding client-side. Fetching-then-filtering breaks
    # silently whenever "today" has moved far enough that the newest page
    # of results no longer contains any papers in the target window --
    # exactly what happened when this was first tried against arXiv's
    # actual current date.
    search_query = f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]"
    query = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    resp = requests.get(f"{ARXIV_API}?{query}", timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id_url = entry.find("atom:id", ATOM_NS).text
        arxiv_id = arxiv_id_url.rsplit("/", 1)[-1]
        # Strip version suffix (e.g. "2410.12345v2" -> "2410.12345") so
        # chunk_id stays stable if the paper gets revised upstream.
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title = entry.find("atom:title", ATOM_NS).text.strip().replace("\n", " ")
        authors = [
            a.find("atom:name", ATOM_NS).text
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        published = entry.find("atom:published", ATOM_NS).text
        year = int(published[:4])

        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None:
            continue

        papers.append(
            ArxivPaper(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                authors=authors,
                year=year,
                category=category,
                pdf_url=pdf_url,
            )
        )
    return papers


def download_pdf_text(pdf_url: str) -> str:
    import requests
    from pypdf import PdfReader
    import io

    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text)


def _split_paragraphs(text: str) -> list[str]:
    # Collapse excess whitespace, split on blank-line boundaries.
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_oversized_paragraph(
    para: str, tokenizer, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Token-level fallback for a single paragraph that alone exceeds
    chunk_tokens. PDF text extraction frequently loses blank-line breaks
    (an entire page can come back as one "paragraph"), so paragraph-only
    splitting is not sufficient on its own -- without this fallback, chunks
    silently blow past the target from SPEC.md §4.

    chunk_tokens/overlap_tokens default to the module constants (P1's
    fixed 512/64 policy for the main corpus); P5's A1_chunking_study.ipynb
    passes different values to build the fixed-256/512/1024 variants --
    see corpus/chunking_strategies.py.
    """
    tokens = tokenizer.encode(para)
    if len(tokens) <= chunk_tokens:
        return [para]

    pieces = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < len(tokens):
        piece_tokens = tokens[start : start + chunk_tokens]
        pieces.append(tokenizer.decode(piece_tokens))
        start += step
    return pieces


def chunk_text(
    text: str, tokenizer, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Paragraph-boundary-aware chunking to ~chunk_tokens tokens per chunk,
    with overlap_tokens of trailing overlap carried into the next chunk.
    Paragraphs that individually exceed chunk_tokens are split at the
    token level as a fallback (see _split_oversized_paragraph).

    chunk_tokens/overlap_tokens default to the module constants; see
    _split_oversized_paragraph's docstring for why they're parametrized.
    """
    raw_paragraphs = _split_paragraphs(text)
    # Expand any paragraph too big to fit in one chunk on its own.
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        paragraphs.extend(_split_oversized_paragraph(para, tokenizer, chunk_tokens, overlap_tokens))

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    def flush():
        if current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))
        if current_tokens + para_tokens > chunk_tokens and current_paragraphs:
            flush()
            # Carry the tail of the previous chunk forward as overlap.
            overlap_text = current_paragraphs[-1]
            overlap_tok_count = len(tokenizer.encode(overlap_text))
            if overlap_tok_count > overlap_tokens:
                current_paragraphs = []
                current_tokens = 0
            else:
                current_paragraphs = [overlap_text]
                current_tokens = overlap_tok_count
        current_paragraphs.append(para)
        current_tokens += para_tokens

    flush()
    return chunks


def guess_section(paragraph_index: int, total_paragraphs: int) -> str:
    """Very rough section heuristic based on position in the paper, since
    extracted PDF text loses heading structure. Good enough for the pilot
    corpus; a real implementation would parse headings.
    """
    fraction = paragraph_index / max(1, total_paragraphs)
    if fraction < 0.15:
        return "abstract"
    if fraction < 0.45:
        return "introduction"
    if fraction < 0.75:
        return "methods"
    return "results"


def build_corpus(
    categories: list[str],
    papers_per_category: int,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    tokenizer = _get_tokenizer()
    all_chunks: list[dict] = []
    papers_used = 0
    papers_skipped = 0
    # arXiv papers can be cross-listed under multiple categories (a paper
    # tagged both cs.CL and cs.AI would otherwise be fetched and chunked
    # once per category, producing colliding chunk_ids). Track paper_ids
    # already used across the WHOLE run, not per category.
    seen_paper_ids: set[str] = set()

    for category in categories:
        print(f"Searching arXiv category {category}...")
        candidates = search_arxiv(category, max_results=papers_per_category * 2)

        collected_for_category = 0
        for paper in candidates:
            if collected_for_category >= papers_per_category:
                break
            if paper.paper_id in seen_paper_ids:
                continue

            print(f"  Fetching {paper.paper_id}: {paper.title[:70]}...")
            try:
                text = download_pdf_text(paper.pdf_url)
                if len(text.strip()) < 500:
                    raise ValueError("extracted text too short, likely a bad parse")
            except Exception as exc:
                print(f"    SKIPPED ({exc})")
                papers_skipped += 1
                time.sleep(1)
                continue

            raw_chunks = chunk_text(text, tokenizer)
            kept = raw_chunks[:MAX_CHUNKS_PER_PAPER]
            total_paragraphs = len(_split_paragraphs(text))

            for i, chunk_text_str in enumerate(kept):
                n_tokens = len(tokenizer.encode(chunk_text_str))
                all_chunks.append(
                    {
                        "chunk_id": f"{paper.paper_id}#{i}",
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "year": paper.year,
                        "category": paper.category,
                        "section": guess_section(i, max(1, len(kept))),
                        "chunk_index": i,
                        "text": chunk_text_str,
                        "n_tokens": n_tokens,
                    }
                )

            seen_paper_ids.add(paper.paper_id)
            papers_used += 1
            collected_for_category += 1
            time.sleep(1)  # be polite to arXiv's servers

    # FIX (plan deviation, review): --dry-run proves the pipeline is
    # reproducible against live arXiv data (fetch, extract, chunk) without
    # overwriting the committed corpus.jsonl that actually ships.
    if dry_run:
        print()
        print(f"[dry run] Would write {len(all_chunks)} chunks from "
              f"{papers_used} papers to {output_path} (papers skipped: {papers_skipped}).")
        if all_chunks:
            preview = all_chunks[0]
            print(f"[dry run] Sample chunk_id: {preview['chunk_id']} "
                  f"({preview['n_tokens']} tokens, section={preview['section']})")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print()
    print(f"Done. Papers used: {papers_used}, skipped: {papers_skipped}, "
          f"chunks written: {len(all_chunks)} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.CL", "cs.LG", "cs.AI"],
        help="arXiv categories to pull from",
    )
    parser.add_argument(
        "--papers-per-category",
        type=int,
        default=6,
        help="how many papers to fetch per category (pilot default: ~18 total)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "corpus.jsonl"),
        help="output path for corpus.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and chunk as normal but don't write the output file; "
        "proves the pipeline still works against live arXiv data without "
        "touching the committed corpus.jsonl",
    )
    args = parser.parse_args()

    build_corpus(
        categories=args.categories,
        papers_per_category=args.papers_per_category,
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
