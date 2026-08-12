"""Enforces SPEC.md §8's mandatory 8-section template on the 10 pattern
notebooks (01-10). Appendix/baseline/leaderboard notebooks (00, 00b, A1,
A2, 11) are explicitly exempt -- SPEC.md §7 states they are "not in the
main leaderboard" / not one of "the 10 patterns."
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
PATTERN_NOTEBOOK_RE = re.compile(r"^(0[1-9]|10)_.*\.ipynb$")

# (section number, required keyword to appear in that section's title,
# case-insensitive substring match)
REQUIRED_SECTIONS = [
    (1, "what this pattern does"),
    (2, "when to use"),
    (3, "when not to use"),
    (4, "implementation"),
    (5, "run on"),
    (6, "example"),
    (7, "fails"),
    (8, "copy-paste"),
]

HEADER_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.MULTILINE)


def find_pattern_notebooks(notebooks_dir: Path = NOTEBOOKS_DIR) -> list[Path]:
    return sorted(p for p in notebooks_dir.glob("*.ipynb") if PATTERN_NOTEBOOK_RE.match(p.name))


def extract_headers(notebook_path: Path) -> list[tuple[int, str]]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    headers = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        for match in HEADER_RE.finditer(text):
            headers.append((int(match.group(1)), match.group(2).strip()))
    return headers


def lint_notebook(notebook_path: Path) -> list[str]:
    """Returns a list of human-readable problems; empty means it passes."""
    headers = extract_headers(notebook_path)
    problems = []
    numbers_seen = [n for n, _ in headers]

    if numbers_seen != sorted(numbers_seen):
        problems.append(f"section numbers out of order: {numbers_seen}")

    if len(numbers_seen) != len(set(numbers_seen)):
        # A dict built from `headers` would silently keep only the LAST
        # occurrence of a repeated number, masking a real authoring bug
        # (e.g. two "## 4." headers) -- check explicitly before that
        # collapse happens below.
        dupes = sorted({n for n in numbers_seen if numbers_seen.count(n) > 1})
        problems.append(f"duplicate section number(s): {dupes}")

    by_number = dict(headers)
    for expected_number, keyword in REQUIRED_SECTIONS:
        if expected_number not in by_number:
            problems.append(
                f"missing section {expected_number} (expected a title containing {keyword!r})"
            )
        elif keyword.lower() not in by_number[expected_number].lower():
            problems.append(
                f"section {expected_number}'s title {by_number[expected_number]!r} "
                f"doesn't contain expected keyword {keyword!r}"
            )
    return problems


def main() -> None:
    notebooks = find_pattern_notebooks()
    if not notebooks:
        print("No pattern notebooks found -- nothing to lint.", file=sys.stderr)
        sys.exit(1)

    any_failed = False
    for path in notebooks:
        problems = lint_notebook(path)
        if problems:
            any_failed = True
            print(f"FAIL {path.name}:")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK   {path.name}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
