"""Pattern 10: agentic RAG. An LLM-driven loop decides, turn by turn, which
tool to call (dense search, BM25 search, or stop) rather than following a
fixed retrieval strategy. The agent's job is retrieval orchestration ONLY:
`finish` ends the search loop but does not produce the answer, so the final
answer still goes through the exact same R4 held-constant generation prompt
every other pattern uses. This keeps R4 airtight (no pattern gets a custom
generation prompt) while still letting the RETRIEVAL strategy itself be the
flexible, agentic part, which is SPEC.md's actual key claim for this
pattern.

Every step (thought, action, action_input, observation) is recorded into
the returned AnswerWithCitations.tool_call_trace. Per SPEC.md §6, these
traces must be logged to outputs/agentic_traces.jsonl -- that file write is
NOT done here (this function stays pure/side-effect-free, consistent with
every other recipe module and with the judge-cache test-isolation lesson
in tasks/lessons.md); it's done by evals/run.py's trace_output_path param,
which the 10_agentic.ipynb notebook passes in.
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
    agent loop is exactly where SPEC.md calls out "hardest to debug."
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
        return {
            "thought": str(parsed.get("thought", "")),
            "action": action,
            "action_input": str(parsed.get("action_input", "")),
        }
    except Exception:
        return fallback


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
        accumulated_ids: list[str] = []
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
            for cid in ids:
                if cid not in accumulated_ids:
                    accumulated_ids.append(cid)
            transcript.append({**step, "observation": f"Retrieved: {ids}"})

        retrieved_ids = accumulated_ids[:k]
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
