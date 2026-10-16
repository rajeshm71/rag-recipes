"""Pattern 10: agentic RAG. An LLM-driven loop decides, turn by turn, which
tool to call (dense search, BM25 search, or stop) rather than following a
fixed retrieval strategy. The agent's job is retrieval orchestration ONLY:
`finish` ends the search loop but does not produce the answer, so the final
answer still goes through the exact same held-constant generation prompt
every other pattern uses (no pattern gets a custom generation prompt --
this keeps every pattern's answer quality attributable to its retrieval
strategy, not to a better-tuned prompt). This pattern's own flexibility --
and fragility -- is entirely in the RETRIEVAL loop: "most flexible, hardest
to debug."

Every step (thought, action, action_input, observation) is recorded into
the returned AnswerWithCitations.tool_call_trace so readers can see the
agent's reasoning. That trace write to outputs/agentic_traces.jsonl is NOT
done here (this function stays pure/side-effect-free, consistent with
every other recipe module -- and with keeping any on-disk state fully
test-isolated); it's done by evals/run.py's trace_output_path param, which
the 10_agentic.ipynb notebook passes in.

Final ranking: the chunks gathered across every search are fused with
reciprocal rank fusion (recipes/hybrid.py's rrf_fuse) before slicing to k.
Returning them in discovery order instead would let the FIRST search fill
ranks 1-5 regardless of what later searches found, making hit@3/MRR
indistinguishable from a single dense search.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from recipes import AnswerWithCitations
from recipes.bm25 import BM25Index
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.hybrid import rrf_fuse
from recipes.llm import LLM

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"
AGENTIC_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "agentic_prompt.txt"
MAX_ITERATIONS = 4
TOOL_K = 5
_VALID_ACTIONS = {"search_dense", "search_bm25", "finish"}


def _parse_agent_action(text: str) -> dict:
    """Defensive parse: any failure (bad JSON, invalid action name, missing
    keys) degrades to a forced `finish`, never raises and never loops
    forever on malformed output -- important here specifically, since an
    agentic loop with no bound on malformed-output retries is exactly the
    "most flexible, hardest to debug" failure mode this pattern is known for.
    """
    fallback = {"thought": "(unparseable response, stopping)", "action": "finish", "action_input": ""}
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        parsed = json.loads(stripped)
        action = parsed.get("action")
        if action not in _VALID_ACTIONS:
            return fallback
        thought = str(parsed.get("thought", ""))
        action_input = str(parsed.get("action_input") or "").strip()
        if action != "finish" and not action_input:
            # A search with an empty query would reach the embedder as ""
            # (which OpenAI's embeddings endpoint rejects) and throw the
            # whole question out; treat it as "nothing left to search for".
            return {"thought": thought, "action": "finish", "action_input": ""}
        return {"thought": thought, "action": action, "action_input": action_input}
    except Exception:
        return fallback


def _render_observation(ids: list[str], corpus_by_id: dict[str, dict]) -> str:
    """What the agent sees after a search. It must include the chunk TEXT:
    an observation of bare chunk ids gives the next decision call nothing
    to reason about, so "have I gathered enough context?" would be answered
    blind.
    """
    if not ids:
        return "Retrieved 0 chunks."
    rendered = "\n".join(f"[{cid}] {corpus_by_id[cid]['text']}" for cid in ids if cid in corpus_by_id)
    return f"Retrieved {len(ids)} chunks:\n{rendered}"


def _render_transcript(transcript: list[dict]) -> str:
    if not transcript:
        return "(none yet)"
    lines = []
    for step in transcript:
        lines.append(f"Thought: {step['thought']}")
        lines.append(f"Action: {step['action']}({step['action_input']!r})")
        if step.get("observation") is not None:
            lines.append(f"Observation: {step['observation']}")
    return "\n".join(lines)


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    tool_k: int = TOOL_K,
) -> Callable[..., AnswerWithCitations]:
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    chunk_ids = list(corpus_by_id.keys())
    bm25_index = BM25Index()
    bm25_index.build(chunk_ids, [corpus_by_id[cid]["text"] for cid in chunk_ids])
    agentic_template = AGENTIC_PROMPT_PATH.read_text(encoding="utf-8")
    gen_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()
        transcript: list[dict] = []
        ranked_lists: list[list[str]] = []  # one ranked id list per search call
        total_input = total_output = total_cached = total_cache_creation = 0

        for _ in range(max_iterations):
            prompt = agentic_template.format(question=question, transcript=_render_transcript(transcript))
            response = llm.complete(prompt=prompt, model=generation_model, temperature=0.0)
            total_input += response.input_tokens
            total_output += response.output_tokens
            total_cached += response.cached_input_tokens
            total_cache_creation += response.cache_creation_input_tokens

            step = _parse_agent_action(response.text)
            if step["action"] == "finish":
                transcript.append({**step, "observation": None})
                break

            if step["action"] == "search_dense":
                ids = dense_search(store, embedder, embedding_model, step["action_input"], tool_k)
            else:  # search_bm25
                ids = [r.chunk_id for r in bm25_index.search(step["action_input"], tool_k)]
            ranked_lists.append(ids)
            transcript.append({**step, "observation": _render_observation(ids, corpus_by_id)})

        if not ranked_lists:
            # The agent stopped without searching (an unparseable first
            # reply, or an immediate finish). Answering from empty context
            # helps nobody, so fall back to one dense search on the question
            # and record it as an automatic step, not the agent's choice.
            ids = dense_search(store, embedder, embedding_model, question, tool_k)
            ranked_lists.append(ids)
            transcript.append(
                {
                    "thought": "(no search was run; automatic dense search on the question)",
                    "action": "search_dense",
                    "action_input": question,
                    "observation": _render_observation(ids, corpus_by_id),
                }
            )

        retrieved_ids = rrf_fuse(ranked_lists, k)
        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}" for cid in retrieved_ids if cid in corpus_by_id
        )
        final_prompt = gen_template.format(context=context, question=question)
        final_response = llm.complete(prompt=final_prompt, model=generation_model, temperature=0.0)
        total_input += final_response.input_tokens
        total_output += final_response.output_tokens
        total_cached += final_response.cached_input_tokens
        total_cache_creation += final_response.cache_creation_input_tokens

        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerWithCitations(
            answer=final_response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=total_input,
            output_tokens=total_output,
            cached_input_tokens=total_cached,
            cache_creation_input_tokens=total_cache_creation,
            tool_call_trace=transcript,
        )

    return retrieve_and_answer
