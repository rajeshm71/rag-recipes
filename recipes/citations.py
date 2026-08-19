"""Extracts inline [chunk_id] citations from an LLM's generated answer text.

Shared by pattern 01 (naive dense) and pattern 07 (contextual retrieval),
since both patterns' generation prompt (prompts/generation_prompt.txt)
asks the model to cite sources as [chunk_id] inline. Kept as its own module
rather than inline in a specific pattern file so it's a shared import, not
duplicated logic across the two patterns that need it.
"""

from __future__ import annotations

import re

# Matches [chunk_id] where chunk_id looks like our corpus schema's IDs,
# e.g. "arxiv:2601.11580#0". Deliberately permissive about the inner
# character set (colons, dots, hashes, hyphens) rather than hardcoding the
# arXiv-specific shape, so it still works if the corpus source changes later.
_CITATION_RE = re.compile(r"\[([\w:.\-#/]+)\]")


def extract_citations(answer_text: str) -> list[str]:
    """Returns the chunk_ids cited in `answer_text`, in the order they first
    appear, with duplicates removed.
    """
    seen: list[str] = []
    for match in _CITATION_RE.findall(answer_text):
        if match not in seen:
            seen.append(match)
    return seen
