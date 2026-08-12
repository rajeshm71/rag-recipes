# LinkedIn launch draft

**Status: TEMPLATE. Do not post as-is.** The bracketed line needs the real leaderboard winner and
number, which requires a real-key run (see `tasks/todo.md`) -- not yet done as of this draft.
Everything else is ready.

---

I benchmarked 10 RAG patterns on the same corpus and eval set. Here's what won.

Most "which RAG pattern should I use" advice is anecdotal -- one blog post's benchmark, on their
corpus, with their prompt. I wanted a single, fair comparison: same 54-chunk corpus, same 18
hand-written eval questions, same held-constant generation prompt, every pattern scored on hit@k,
MRR, faithfulness, answer relevance, citation accuracy, latency, and cost -- with 95% confidence
intervals, not just point estimates.

[FILL IN once the real-key run lands: 2-3 sentences on the actual winning pattern, the most
surprising failure mode found, and the cost/latency tradeoff of the top contenders.]

What's in the repo:
→ 10 RAG patterns (naive dense, BM25, hybrid+RRF, cross-encoder rerank, HyDE, multi-query,
  contextual retrieval, self-query, multi-hop, agentic) -- each a runnable notebook AND a
  copy-paste-able Python function
→ 2 baselines (no-RAG, long-context) to sanity-check that retrieval is actually helping
→ 2 appendix studies: a chunking-strategy comparison and an embedding-model swap, both holding
  retrieval constant
→ Every pattern's notebook includes a "where this pattern fails" section with real, analyzed
  failure cases -- not just the wins

No LangChain, no LlamaIndex, no framework abstraction. The point is showing the primitives
clearly enough that you can lift exactly the piece you need into your own project.

Repo: [link]

#RAG #LLM #retrievalaugmentedgeneration #genai
