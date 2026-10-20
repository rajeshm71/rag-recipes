"""Builds corpus/corpus.jsonl from the committed full paper texts.

Requires the `corpus-build` optional dependency group (tiktoken):
    uv sync --extra corpus-build
    uv run python corpus/build_from_fulltext.py

Inputs, both committed:
- corpus/corpus_fulltext.jsonl: {"paper_id", "full_text"} per paper, produced
  once by corpus/build_fulltext.py.
- corpus/papers.jsonl: {"paper_id", "title", "authors", "year", "category"}
  per paper, the arXiv metadata fetched by corpus/build_corpus.py.

Policy:
- Body text only: everything from the "References" heading onward is dropped.
- 512-token chunks on paragraph boundaries, each starting with the last 64
  tokens of the chunk before it (chunk_text() in corpus/build_corpus.py).
- `section` is derived from the paper's own headings ("1 Introduction",
  "3.2 Datasets", "Conclusion"), normalized to a small fixed vocabulary.
- A chunk adding fewer than 64 new tokens is merged into the one before it.
- At most MAX_CHUNKS_PER_PAPER chunks per paper.

No network access. Deterministic for a given tiktoken encoding.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_corpus import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKENS,
    _get_tokenizer,
    chunk_text,
)

CORPUS_DIR = Path(__file__).resolve().parent
MAX_CHUNKS_PER_PAPER = 20

SECTIONS = (
    "abstract",
    "introduction",
    "related_work",
    "methods",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "other",
)

# A heading is a line that is either a bare title ("Abstract", "Conclusion")
# or a numbered one ("2 Method", "3.1 Datasets"). Short, Title-case, no
# trailing period: that excludes ordinary sentences and figure captions.
_HEADING_RE = re.compile(r"^(?:(?P<num>\d+(?:\.\d+)*)\.?\s+)?(?P<title>[A-Z][^\n.]{1,70}?)\s*$")
_BARE_TITLES = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
}
_STOP_TITLES = {"references", "bibliography"}
# Un-numbered titles still accepted once a paper's numbered outline has
# started (back matter, including the unnumbered Limitations section many
# venues require). Any other bare title after that point is a table cell or a figure
# label ("Method" as a column header), not a section.
_BACK_MATTER_TITLES = _STOP_TITLES | {"acknowledgments", "acknowledgements", "appendix", "limitations"}
# A numbered heading may skip ahead by at most this much, so one heading the
# detector missed does not reject every heading after it, while a figure
# label like "199 Request ID" is still rejected.
_MAX_OUTLINE_STEP = 2

_LABEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("abstract", ("abstract",)),
    ("introduction", ("introduction",)),
    ("related_work", ("related work", "background", "preliminar", "prior work")),
    ("conclusion", ("conclusion", "future work", "summary")),
    ("discussion", ("discussion", "limitation", "broader impact", "ethic")),
    ("results", ("result", "finding", "analysis", "ablation", "performance", "comparison")),
    (
        "experiments",
        ("experiment", "evaluation", "setup", "dataset", "baseline", "benchmark", "simulation", "implementation"),
    ),
    (
        "methods",
        ("method", "approach", "model", "framework", "formulation", "architecture", "algorithm", "design", "system"),
    ),
)


def normalize_section(title: str) -> str:
    """Map a heading title to one of SECTIONS ("other" if nothing matches)."""
    # PDF extraction often drops or inserts spaces ("RELATEDWORKS",
    # "F ramework"), so match with all spaces removed.
    t = title.lower().replace(" ", "")
    for label, needles in _LABEL_RULES:
        if any(n.replace(" ", "") in t for n in needles):
            return label
    return "other"


def _looks_like_affiliation(title: str) -> bool:
    """Author affiliations come out of PDFs as numbered, Title-case lines
    ("1 University of Southern California, Los Angeles, CA 90007, USA").
    Section titles almost never combine a comma with a digit."""
    return "," in title and any(ch.isdigit() for ch in title)


def _continues_outline(parts: list[int], outline: list[int]) -> bool:
    """Whether a heading numbered `parts` (2.1 -> [2, 1]) can follow the
    numbered headings seen so far (`outline`, the last accepted number).
    Each level must move forward by 1 to _MAX_OUTLINE_STEP from the level
    above it, which rejects numbered list items ("1. Equality and
    Symmetry:" inside section 3) and figure labels."""
    depth = len(parts)
    if parts[: depth - 1] != outline[: depth - 1] or len(outline) < depth - 1:
        return False
    previous = outline[depth - 1] if len(outline) >= depth else 0
    return 1 <= parts[depth - 1] - previous <= _MAX_OUTLINE_STEP


def find_headings(text: str) -> list[tuple[int, str, str]]:
    """Return (char_offset, raw_title, label) for every heading line in
    reading order. `label` is already resolved for the section the heading
    opens:

    - Numbered lines count when they continue the paper's outline
      (_continues_outline) and are not affiliations.
    - Un-numbered lines count when they are a conventional bare section
      title, and only before the numbered outline starts or as back matter.
    - A top-level heading gets its own label ("other" if it matches no
      rule). A numbered sub-heading takes its section's label, or its own
      when the section's label is "other".
    """
    headings: list[tuple[int, str, str]] = []
    outline: list[int] = []
    section_label: str | None = None
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        match = _HEADING_RE.match(stripped)
        if match:
            title = match.group("title").strip()
            num = match.group("num")
            own = normalize_section(title)
            if num is not None:
                parts = [int(p) for p in num.split(".")]
                if not _looks_like_affiliation(title) and _continues_outline(parts, outline):
                    outline = parts
                    if len(parts) == 1:
                        section_label = own
                        label = own
                    else:
                        label = section_label if section_label not in (None, "other") else own
                    headings.append((offset, title, label))
            elif title.lower() in _BARE_TITLES and (not outline or title.lower() in _BACK_MATTER_TITLES):
                headings.append((offset, title, own))
        offset += len(line) + 1
    return headings


def body_text(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Cut the text at its References heading (the first one that appears
    in the back half of the paper, so a "References" mention inside the
    introduction doesn't truncate the body) and return it with its
    headings."""
    headings = find_headings(text)
    for offset, title, _ in headings:
        if title.lower() in _STOP_TITLES and offset >= len(text) * 0.3:
            text = text[:offset]
            headings = [h for h in headings if h[0] < offset]
            break
    return text, headings


def section_for_offset(offset: int, headings: list[tuple[int, str, str]]) -> str:
    """Label of the nearest heading at or before `offset`. Front matter
    before the first heading is the title/author/abstract block."""
    label = "abstract"
    for h_offset, _, h_label in headings:
        if h_offset > offset:
            break
        label = h_label
    return label


def chunk_paper(full_text: str, tokenizer, max_chunks: int = MAX_CHUNKS_PER_PAPER) -> list[dict]:
    """Chunk one paper's body text. Returns dicts with text, section,
    n_tokens; chunk_index is assigned by the caller."""
    text, headings = body_text(full_text)
    # chunk_text folds fragments (a one-line equation, a short tail) into
    # the chunk before them, so no chunk is a near-copy of its neighbour.
    chunks = chunk_text(text, tokenizer, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)[:max_chunks]

    # Chunks re-join paragraphs with blank lines, which can differ from the
    # source's own whitespace, so they are located with whitespace collapsed.
    flat, offsets = _collapse_whitespace(text)
    out: list[dict] = []
    search_from = 0
    for chunk in chunks:
        pos = flat.find(_collapse_whitespace(chunk)[0], search_from)
        if pos == -1:
            raise ValueError(f"chunk not found in its source text: {chunk[:80]!r}")
        out.append(
            {
                "text": chunk,
                "section": section_for_offset(offsets[pos], headings),
                "n_tokens": len(tokenizer.encode(chunk)),
            }
        )
        search_from = pos + 1
    return out


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """`text` with every whitespace run replaced by one space, plus the
    offset in `text` of each character of the result."""
    chars: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        offsets.append(i)
    return "".join(chars), offsets


def build(
    fulltext_path: Path = CORPUS_DIR / "corpus_fulltext.jsonl",
    papers_path: Path = CORPUS_DIR / "papers.jsonl",
    out_path: Path = CORPUS_DIR / "corpus.jsonl",
    max_chunks_per_paper: int = MAX_CHUNKS_PER_PAPER,
) -> list[dict]:
    tokenizer = _get_tokenizer()
    papers = {}
    with open(papers_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                papers[record["paper_id"]] = record

    all_chunks: list[dict] = []
    with open(fulltext_path, encoding="utf-8") as f:
        texts = [json.loads(line) for line in f if line.strip()]
    for record in sorted(texts, key=lambda r: r["paper_id"]):
        paper = papers[record["paper_id"]]
        for i, chunk in enumerate(chunk_paper(record["full_text"], tokenizer, max_chunks_per_paper)):
            all_chunks.append(
                {
                    "chunk_id": f"{paper['paper_id']}#{i}",
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "category": paper["category"],
                    "section": chunk["section"],
                    "chunk_index": i,
                    "text": chunk["text"],
                    "n_tokens": chunk["n_tokens"],
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return all_chunks


def main() -> None:
    chunks = build()
    by_paper: dict[str, int] = {}
    by_section: dict[str, int] = {}
    for c in chunks:
        by_paper[c["paper_id"]] = by_paper.get(c["paper_id"], 0) + 1
        by_section[c["section"]] = by_section.get(c["section"], 0) + 1
    n_tokens = [c["n_tokens"] for c in chunks]
    print(f"Wrote {len(chunks)} chunks from {len(by_paper)} papers")
    print(f"chunks per paper: min {min(by_paper.values())}, max {max(by_paper.values())}")
    print(f"n_tokens: min {min(n_tokens)}, max {max(n_tokens)}, mean {sum(n_tokens) / len(n_tokens):.0f}")
    print("sections:", dict(sorted(by_section.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
