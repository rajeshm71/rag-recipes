"""Builds corpus/corpus.jsonl from real arXiv papers.

Requires the `corpus-build` optional dependency group:
    uv sync --extra corpus-build

Queries arXiv's public Atom API for papers in the given categories, downloads
each PDF, extracts text, and chunks it per this project's fixed policy: 512
tokens per chunk (tiktoken-counted), 64-token overlap, paragraph-boundary
aware, at most 3 chunks kept per paper to avoid one-paper dominance.

This script is shipped for transparency -- corpus.jsonl is committed
pre-built and does NOT need to be regenerated on every install.

If a paper's PDF fails to download or its text fails to extract, that paper
is skipped with a logged warning; the script keeps going and reports at the
end how many papers/chunks it actually produced versus the target.
"""

from __future__ import annotations

import argparse
import json
import re
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
        # gpt-4.1-mini may not be recognized by an older installed tiktoken.
        # It resolves to o200k_base on versions that know it, and the
        # committed corpus.jsonl's n_tokens were counted with o200k_base, so
        # fall back to that same encoding (cl100k_base would count ~1-5%
        # differently and move chunk boundaries).
        return tiktoken.get_encoding("o200k_base")


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
    import io

    import requests
    from pypdf import PdfReader

    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text)


def _split_paragraphs(text: str) -> list[str]:
    # Collapse excess whitespace, split on blank-line boundaries.
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


# What decode() yields for a token slice that cuts a multi-byte character.
_REPLACEMENT_CHAR = chr(0xFFFD)
# Tokens a piece leaves free beside the carried overlap: the blank-line
# join and re-tokenizing across the seam can add a few.
_SEAM_TOKENS = 8


def _split_oversized_paragraph(
    para: str, tokenizer, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Token-level fallback for a paragraph too long to fit in a chunk
    after the carried overlap. PDF text extraction frequently loses blank-line breaks
    (an entire page can come back as one "paragraph"), so paragraph-only
    splitting is not sufficient on its own -- without this fallback, chunks
    silently blow past the target chunk size.

    Pieces are chunk_tokens - overlap_tokens - _SEAM_TOKENS long, so a piece
    plus the overlap chunk_text() carries in front of it fits in one chunk. They are
    consecutive slices of `para` (concatenated, they give it back exactly,
    each later piece keeping its leading whitespace), cut on whole
    characters: a token boundary can fall inside a multi-byte character.

    chunk_tokens/overlap_tokens default to the module constants (the fixed
    512/64 policy used for the main corpus); A1_chunking_study.ipynb passes
    different values to build the fixed-256/512/1024 variants -- see
    corpus/chunking_strategies.py.
    """
    tokens = tokenizer.encode(para)
    size = max(1, chunk_tokens - overlap_tokens - _SEAM_TOKENS)
    if len(tokens) <= size:
        return [para]

    pieces = []
    start = offset = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        prefix = tokenizer.decode(tokens[:end])
        while end < len(tokens) and end > start + 1 and not para.startswith(prefix):
            end -= 1
            prefix = tokenizer.decode(tokens[:end])
        cut = len(prefix) if para.startswith(prefix) else len(para)
        if end == len(tokens):
            cut = len(para)
        pieces.append(para[offset:cut])
        start, offset = end, cut
    return [p for p in pieces if p.strip()]


def _overlap_tail(text: str, tokenizer, overlap_tokens: int) -> str:
    """The last overlap_tokens tokens of `text`, starting at a whole
    character."""
    if overlap_tokens <= 0:
        return ""
    tokens = tokenizer.encode(text)
    start = max(0, len(tokens) - overlap_tokens)
    while start < len(tokens) - 1 and _REPLACEMENT_CHAR in tokenizer.decode(tokens[start:]):
        start += 1
    # Leading whitespace only: a trailing space separates the tail from a
    # piece that continues the same paragraph.
    return tokenizer.decode(tokens[start:]).lstrip()


def _join(prefix: str, parts: list[tuple[str, bool]]) -> str:
    """Append (text, continues) parts to prefix: a part that continues the
    previous one (a later piece of a split paragraph) is appended as is,
    anything else starts a new paragraph. Trailing whitespace is kept for
    the same reason as in _overlap_tail; callers strip what they store."""
    out = prefix
    for text, continues in parts:
        if not out.strip():
            out = text.lstrip()
        elif continues:
            out += text
        else:
            out = out.rstrip() + "\n\n" + text.lstrip()
    return out


def chunk_text(
    text: str,
    tokenizer,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_new_tokens: int | None = None,
) -> list[str]:
    """Paragraph-boundary-aware chunking to at most chunk_tokens tokens per
    chunk. Every chunk after the first starts with the last overlap_tokens
    tokens of the chunk before it. Paragraphs that individually exceed the
    budget are split at the token level (see _split_oversized_paragraph).

    A chunk whose own new text (excluding the carried overlap) is shorter
    than min_new_tokens is folded into the chunk before it instead, so a
    one-line paragraph or a short tail never ships as a near-copy of its
    neighbour, so a folded chunk can exceed chunk_tokens by about
    min_new_tokens. The default is overlap_tokens, capped at half the new
    text a chunk has room for, so small chunk sizes still split.

    chunk_tokens/overlap_tokens default to the module constants; see
    _split_oversized_paragraph's docstring for why they're parametrized.
    """
    if min_new_tokens is None:
        min_new_tokens = min(overlap_tokens, (chunk_tokens - overlap_tokens) // 2)
    parts: list[tuple[str, bool]] = []
    for para in _split_paragraphs(text):
        pieces = _split_oversized_paragraph(para, tokenizer, chunk_tokens, overlap_tokens)
        parts.extend((piece, i > 0) for i, piece in enumerate(pieces))

    chunks: list[str] = []
    last_raw = ""  # the last chunk before stripping, for overlap and folding
    overlap = ""
    new: list[tuple[str, bool]] = []

    def flush() -> None:
        nonlocal last_raw, overlap, new
        if not new:
            return
        if chunks and len(tokenizer.encode(_join("", new).strip())) < min_new_tokens:
            last_raw = _join(last_raw, new)
            chunks[-1] = last_raw.strip()
        else:
            last_raw = _join(overlap, new)
            chunks.append(last_raw.strip())
        overlap = _overlap_tail(last_raw, tokenizer, overlap_tokens)
        new = []

    for part in parts:
        if new and len(tokenizer.encode(_join(overlap, new + [part]).strip())) > chunk_tokens:
            flush()
        new.append(part)

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

    # --dry-run proves the pipeline is reproducible against live arXiv data
    # (fetch, extract, chunk) without overwriting the committed corpus.jsonl
    # that actually ships.
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
        f.writelines(json.dumps(chunk) + "\n" for chunk in all_chunks)

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
