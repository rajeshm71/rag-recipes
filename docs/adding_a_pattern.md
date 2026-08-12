# Adding a new pattern: implementation guide

This is the implementation-level companion to [CONTRIBUTING.md](../CONTRIBUTING.md)'s step-by-step
overview. Read that first for the process (open an issue, branch naming, PR expectations); this
page covers the actual code and notebook structure.

Before writing anything: open an issue and get a maintainer nod. New dependencies, framework
integrations, and changes to the held-constant generation prompt all require prior discussion --
see `SPEC.md` §22's R1-R12 for the full list of hard rules every pattern must follow.

## 1. The `recipes/<name>.py` contract

Every pattern module exposes one factory function with this exact shape (per `SPEC.md` §16.3):

```python
def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    llm: LLM,
    # ...pattern-specific params (embedder, reranker, etc.)...
    generation_model: str = GENERATION_MODEL,
) -> Callable[..., AnswerWithCitations]:
    ...
    return retrieve_and_answer
```

`retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations` is the returned closure --
this is what `evals.run.run_pattern()` and every notebook's "Implementation" section call. See
`recipes/__init__.py` for the `AnswerWithCitations` dataclass every pattern must return.

**Reuse existing infrastructure -- don't reimplement retrieval primitives:**

- Dense retrieval: `recipes/dense_index.py`'s `build_dense_index()`/`dense_search()`.
- Lexical retrieval: `recipes/bm25.py`'s `BM25Index`.
- RRF fusion: `recipes/hybrid.py`'s `rrf_fuse()`.
- Reranking: `recipes/rerank.py`'s `get_reranker()`/`Reranker` protocol.
- If your pattern composes several of these without needing generation (like an appendix study,
  not one of the 10 patterns), look at `recipes/hybrid_rerank.py`'s `build_retriever()` for the
  retrieval-only composition shape.

A minimal skeleton for a new dense-retrieval-based pattern:

```python
"""Pattern NN: <name>. <one-line key claim -- what does this pattern win at, what does it cost>."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from recipes import AnswerWithCitations
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.llm import LLM

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
) -> Callable[..., AnswerWithCitations]:
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    prompt_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()
        retrieved_ids = dense_search(store, embedder, embedding_model, question, k)
        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}" for cid in retrieved_ids if cid in corpus_by_id
        )
        prompt = prompt_template.format(context=context, question=question)  # R4: held constant
        response = llm.complete(prompt=prompt, model=generation_model, temperature=0.0)
        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerWithCitations(
            answer=response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
        )

    return retrieve_and_answer
```

**R4 (`SPEC.md` §5):** the final answer-generation call must use `prompts/generation_prompt.txt`
verbatim. If your pattern needs an auxiliary LLM call BEFORE retrieval (a query rewrite, a
hypothetical-document generation, a decomposition) -- that's fine and not covered by R4, since it
runs before "the generation step," not during it. Give it its own prompt file
(`prompts/<name>_prompt.txt`) and say so explicitly in your module's docstring, matching the
convention in `recipes/hyde.py`/`recipes/multi_query.py`.

**Defensive parsing:** if your pattern's auxiliary prompt asks the LLM for structured output (JSON),
write a small `_parse_*()` helper that degrades gracefully on any parse failure rather than raising
-- see `recipes/multi_query.py`'s `_parse_subqueries()` for the established pattern (strip code
fences, `json.loads`, fall back to a safe default on any exception).

## 2. Tests (`tests/test_<name>.py`)

Every pattern's tests use `MockEmbedder`/`MockLLM` (`recipes/embeddings.py`/`recipes/llm.py`) --
never real API calls in the test suite. At minimum:

- End-to-end: `make_retrieve_and_answer()` returns a working closure, `retrieved_chunk_ids` come
  from the real corpus.
- Any auxiliary parser: valid input, malformed input (falls back gracefully, doesn't raise).
- If your pattern makes more than one LLM call per question: a token-summing test proving all
  calls' tokens are counted (not just the last one) -- see `tests/test_hyde.py` for the pattern.

## 3. The notebook (`notebooks/<NN>_<name>.ipynb`)

Mandatory 8 sections, in this order, enforced by `scripts/lint_notebooks.py`:

1. **What this pattern does** (3 sentences, plain English)
2. **When to use it** (bullet list)
3. **When NOT to use it** (bullet list, at least 2 items)
4. **Implementation** (imports from `recipes/<name>.py` -- no logic redefined here)
5. **Run on our eval set** (calls `evals.run.run_pattern()`, prints all metrics + CIs)
6. **Example query walkthrough** (one query per eval-set category)
7. **Where this pattern FAILS** (at least 2 real, analyzed failures -- required, cannot be skipped)
8. **Copy-paste snippet** (a markdown fenced code block, NOT a live-executed cell -- `corpus_by_id
   = {}` as pseudo-code is a Python *set* literal if left as a real cell, not an empty dict; this
   bit developers in P2, keep it a markdown block)

Precede section 1 with a reproducibility header cell (platform/python/key dep versions/git commit
sha) and a dedicated Setup cell (loads `corpus_by_id`, `qa_set`, `llm`, `judge_llm` -- see any
existing pattern notebook's Setup cell for the exact shape, including the mock-judge-backend split
needed because `MockLLM`'s canned generation text isn't valid JSON for the judge prompts).

Run `python scripts/lint_notebooks.py` locally before opening a PR -- it checks section presence,
order, and title wording against this list.

## 4. Wire it into the leaderboard

Add your pattern to `notebooks/11_leaderboard.ipynb`'s `PATTERNS` list (one import + one lambda
factory entry, matching the existing 10 entries' shape).

## 5. Local verification checklist

```bash
ruff check .
python scripts/lint_notebooks.py
pytest tests/
RAG_RECIPES_LLM=mock uv run papermill notebooks/<NN>_<name>.ipynb notebooks/<NN>_<name>.ipynb --cwd notebooks
```

All four must pass before opening a PR.
