# Contributing to rag-recipes

Thanks for your interest in rag-recipes. This guide covers both "how do I use this" (for people
who just want to run a pattern) and "how do I contribute" (for people who want to submit a PR).

## How to use this repo (non-contributors)

1. `git clone https://github.com/rajeshm71/rag-recipes && cd rag-recipes`
2. `uv sync`
3. `cp .env.example .env` and fill in `OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` if running pattern 07)
4. `jupyter lab notebooks/` and open any pattern notebook
5. To use a pattern in your own project: copy the relevant `recipes/<pattern>.py` file directly
   into your codebase. This repo is a recipe collection, not an installable library --
   `pip install rag-recipes` is **not** supported and there is no PyPI package.

## Ways to contribute

- **Add a new RAG pattern** (most valued contribution)
- **Improve an existing pattern's failure-mode analysis** -- found a query where a pattern breaks
  in an interesting way not yet documented
- **Fix a bug** in `recipes/`, `evals/`, or the CI lint script
- **Improve `docs/choosing.md`** with a decision node the maintainer missed
- **Report an issue**: a metric that looks wrong, a notebook that fails to run, a broken link

Things that will **not** be merged without prior discussion in an issue first: new dependencies,
framework integrations (LangChain/LlamaIndex), UI/dashboard additions, changes to the held-constant
generation prompt, changes to the eval set composition. Open an issue before investing time in any
of these.

## Adding a new pattern: step-by-step

1. Open an issue first using the "New pattern" issue template, describing the pattern and why it's
   not covered by the existing 10. Wait for a maintainer nod before writing code, to avoid wasted
   work.
2. Fork the repo, create a branch named `pattern/<short-name>` off `main`.
3. Add `recipes/<name>.py` with function `retrieve_and_answer(question: str, k: int = 5) ->
   AnswerWithCitations`. Follow the shape of an existing recipe file (e.g. `recipes/hybrid.py`) for
   the interface. See `docs/adding_a_pattern.md` for a full template and checklist.
4. Add `notebooks/<NN>_<name>.ipynb` following the mandatory 8-section template (see
   `docs/adding_a_pattern.md`), including the required "Where this pattern FAILS" section.
5. Extend `notebooks/11_leaderboard.ipynb` to include the new row.
6. Add a decision node to `docs/choosing.md` if the pattern changes the recommendation logic.
7. Run locally before opening a PR: `ruff check .`, `python scripts/lint_notebooks.py`, `pytest
   tests/`, and a `papermill` smoke run with `RAG_RECIPES_LLM=mock`.
8. Open a PR against `main` using the PR template. Link the issue from step 1.
9. CI must be green. A maintainer reviews for: does it stay within the project's hard rules
   (dependency budget of ≤12 top-level runtime deps, no LangChain/LlamaIndex/framework
   abstraction, pinned dated model snapshots, the held-constant generation prompt used verbatim,
   cost printed at the end of the notebook), does the notebook read well, is the failure-mode
   section honest.
10. Once approved, a maintainer merges via squash merge to keep `main` history clean.

## Fixing a bug

1. Check open issues first, comment to claim it, or open a new issue if none exists.
2. Branch off `main` as `fix/<short-description>`.
3. Smallest possible diff. Do not refactor unrelated code in the same PR.
4. Add or update a test that would have caught the bug, when practical.
5. Open a PR referencing the issue (`Fixes #123`).

## Pull request expectations

- One logical change per PR. A new pattern and a bug fix are two PRs, not one.
- PR description must state: what changed, why, and how it was tested (which commands were run
  locally).
- CI (lint + notebook smoke + unit tests) must pass before review. Maintainers will not review a
  red PR.
- Expect review within roughly a week for a well-scoped PR. This is a side project maintained by
  one person; larger or more ambiguous PRs may take longer or get requested changes.
- No `Co-Authored-By: Claude` or AI-tool attribution trailers on commits. PRs written substantially
  by an AI coding tool are welcome, but the commit author should be the human submitting the PR.
- Squash merge is the default merge strategy for this repo -- don't worry about crafting a pristine
  commit history inside your branch.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Be respectful, no
harassment. Report issues to the maintainer's email listed there.
