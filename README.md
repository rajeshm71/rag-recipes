# rag-recipes 🧪

![CI](https://github.com/rajeshm71/rag-recipes/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

rag-recipes is a benchmark suite for RAG (retrieval-augmented generation) patterns. It runs 10
distinct patterns, from naive dense retrieval through agentic RAG, against the same corpus, the
same eval set, and the same held-constant generation prompt. Every pattern ships as both a runnable notebook and
an importable Python function under `recipes/`, ready to copy into another project (together with
the small shared modules it imports and the `prompts/` folder).

## Contents

- [The 10 patterns](#the-10-patterns)
- [Which pattern should I use?](#which-pattern-should-i-use)
- [Retrieval leaderboard](#retrieval-leaderboard)
- [Generation and judge metrics](#generation-and-judge-metrics)
- [Quick start](#quick-start)
- [Appendices](#appendices)
- [How the eval works](#how-the-eval-works)
- [Non-goals](#non-goals)
- [Contributing and license](#contributing-and-license)

---

## The 10 patterns

| # | Pattern | Notebook | Key claim to test |
|---|---|---|---|
| 01 | Naive dense retrieval | [01_naive_dense.ipynb](notebooks/01_naive_dense.ipynb) | Baseline. Fails on rare terms |
| 02 | BM25 | [02_bm25.ipynb](notebooks/02_bm25.ipynb) | Baseline. Fails on paraphrase |
| 03 | Hybrid + RRF | [03_hybrid_rrf.ipynb](notebooks/03_hybrid_rrf.ipynb) | Beats either alone almost always |
| 04 | Cross-encoder rerank | [04_rerank.ipynb](notebooks/04_rerank.ipynb) | Single best add-on. Costs latency |
| 05 | HyDE | [05_hyde.ipynb](notebooks/05_hyde.ipynb) | Helps when query and doc language differ |
| 06 | Multi-query decomposition | [06_multi_query.ipynb](notebooks/06_multi_query.ipynb) | Helps ambiguous queries. Doubles cost |
| 07 | Contextual retrieval (Anthropic) | [07_contextual.ipynb](notebooks/07_contextual.ipynb) | For chunks that lose meaning out of context. One-time indexing cost |
| 08 | Metadata + self-query | [08_self_query.ipynb](notebooks/08_self_query.ipynb) | Needed when corpus has structure |
| 09 | Multi-hop retrieval | [09_multi_hop.ipynb](notebooks/09_multi_hop.ipynb) | For chained-fact questions |
| 10 | Agentic RAG | [10_agentic.ipynb](notebooks/10_agentic.ipynb) | Most flexible. Hardest to debug |

Each pattern is both a notebook, which includes a "Where this pattern FAILS" section, and an
importable function in `recipes/<pattern>.py`. To use one in your own
project, copy that file plus the shared modules it imports (`recipes/store.py`, `dense_index.py`,
`embeddings.py`, `llm.py`, and for hybrid patterns `bm25.py`/`hybrid.py`) and the `prompts/`
folder -- this repo is a recipe collection, not an installable library.

## Which pattern should I use?

- **General default:** hybrid retrieval + cross-encoder rerank (patterns 03 + 04) -- strong,
  low-surprise baseline for most corpora.
- **Jargon-heavy corpora** (queries phrased very differently from your documents' language):
  HyDE (pattern 05).
- **Chunks that lose meaning in isolation** (pronouns, "the model above"): contextual retrieval
  (pattern 07) as a preprocessing step under whichever pattern you pick.
- **Small corpora** (under ~200 pages): the long-context baseline (00b) can beat retrieval
  entirely, with none of the engineering overhead.

The full decision tree lives in [`docs/choosing.md`](docs/choosing.md).

---

## Retrieval leaderboard 🏆

Every retrieval pattern that needs no LLM, scored on the full eval set with local models
(`BAAI/bge-small-en-v1.5` embeddings, `BAAI/bge-reranker-v2-m3` reranker). No API key, and CI
re-computes this table whenever the corpus, the eval set or the retrieval code changes, and
fails if it drifts from what is committed.

<!-- retrieval-leaderboard:start -->
| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | mrr_vs_top | p50_ms |
|---|---|---|---|---|---|---|---|
| 1 | 03+04_hybrid_rerank | 0.894 [0.818, 0.955] | 0.955 [0.894, 1.000] | 0.807 [0.726, 0.879] | 0.924 [0.864, 0.970] | -- | 1762.8 |
| 2 | 04_bm25_rerank | 0.864 [0.773, 0.939] | 0.894 [0.818, 0.955] | 0.785 [0.690, 0.867] | 0.856 [0.780, 0.932] | -0.022 [-0.060, +0.000] (tie) | 1741.1 |
| 3 | 02_bm25 | 0.727 [0.621, 0.833] | 0.848 [0.758, 0.924] | 0.673 [0.570, 0.769] | 0.826 [0.735, 0.909] | -0.134 [-0.218, -0.053] | 1.2 |
| 4 | 03_hybrid_rrf | 0.697 [0.576, 0.803] | 0.955 [0.894, 1.000] | 0.605 [0.515, 0.699] | 0.924 [0.864, 0.977] | -0.202 [-0.285, -0.125] | 12.5 |
| 5 | 01_naive_dense | 0.652 [0.530, 0.773] | 0.894 [0.818, 0.955] | 0.554 [0.459, 0.651] | 0.848 [0.773, 0.917] | -0.253 [-0.361, -0.153] | 9.4 |
<!-- retrieval-leaderboard:end -->

Columns: hit@k is whether any relevant chunk is in the top k; MRR is the reciprocal rank of the
first relevant chunk; recall@10 is the fraction of a question's relevant chunks in the top 10, which
is what separates patterns on the multi-hop questions. Brackets are 95% bootstrap confidence
intervals. mrr_vs_top is each pattern's MRR minus the top pattern's, with an interval bootstrapped
over the per-question differences; "(tie)" means the eval set cannot tell the two apart. Latency is
per query on the machine that produced the table. Reproduce it with:

```bash
uv run python evals/retrieval_leaderboard.py
```

Per-question ranks for every pattern are one flag away (`--per-question ranks.jsonl`), and the
same numbers live in [`outputs/retrieval_leaderboard.md`](outputs/retrieval_leaderboard.md).

## Generation and judge metrics

[`notebooks/11_leaderboard.ipynb`](notebooks/11_leaderboard.ipynb) scores all 10 patterns plus
the two baselines end to end: retrieval metrics, LLM-judged faithfulness, answer relevance and
citation accuracy, filter accuracy for the self-query pattern, p50/p95 latency, and cost per query.
It needs `OPENAI_API_KEY` (embeddings, generation, judging) and `ANTHROPIC_API_KEY` (pattern 07's
contextualization step), and writes `outputs/leaderboard.md` and `outputs/leaderboard.png`. Judge
calls are cached on disk, so re-running after a change re-bills only what changed.

---

## Quick start

```bash
git clone https://github.com/rajeshm71/rag-recipes && cd rag-recipes
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY (and ANTHROPIC_API_KEY for pattern 07)
jupyter lab notebooks/
```

## Appendices

| Notebook | What it studies |
|---|---|
| [00: No-RAG baseline](notebooks/00_baseline_no_rag.ipynb) | Same eval set, no retrieval, parametric memory only |
| [00b: Long-context baseline](notebooks/00b_long_context_baseline.ipynb) | Stuff the whole corpus into context instead of retrieving |
| [A1: Chunking study](notebooks/A1_chunking_study.ipynb) | Fixed-256/512/1024, semantic, document-aware, late chunking, retrieval held constant |
| [A2: Embedding swap](notebooks/A2_embedding_swap.ipynb) | `text-embedding-3-small` vs. `voyage-4` vs. local `bge-large-en-v1.5`, retrieval held constant |

---

## How the eval works

- **Corpus:** [`corpus/corpus.jsonl`](corpus/corpus.jsonl) -- 333 chunks from 18 arXiv papers
  (`cs.CL`/`cs.LG`/`cs.AI`), 11 to 20 chunks per paper. Body text only (everything before the
  references), 512-token chunks with a 64-token overlap on paragraph boundaries, and a `section`
  label derived from each paper's own headings. Built by
  [`corpus/build_from_fulltext.py`](corpus/build_from_fulltext.py) from the committed full texts
  and shipped pre-built, so there is no download step.
- **Eval set:** [`evals/qa_set.jsonl`](evals/qa_set.jsonl) -- 66 hand-authored questions: 24
  keyword, 24 paraphrase, 9 multi-hop (two chunks required, mostly across two papers), and 9
  filter-dependent (a category constraint is needed to disambiguate). Most target a body chunk
  whose answer is not in the paper's abstract, so a pattern has to find the right chunk, not just
  the right paper.
- **Metrics:** hit@3, hit@10, MRR, recall@10 (deterministic), faithfulness/answer
  relevance/citation accuracy (LLM-as-judge), p50/p95 latency, $/query (generation cost only), eval
  cost (generation plus judges) -- every quality metric reported with a 95% bootstrap confidence
  interval (1000 resamples). See [`evals/metrics.py`](evals/metrics.py).

---

## Non-goals

Out of scope for this repo: multimodal RAG (images, tables inside PDFs), Graph RAG, fine-tuning
embeddings, production infrastructure (drift monitoring, feature stores, online evals),
serving/streaming benchmarks.

## Contributing and license

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to use this repo, report a bug, or add a new
pattern. Licensed under [MIT](LICENSE).
