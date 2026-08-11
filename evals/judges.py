"""LLM-as-judge callers for faithfulness, answer relevance, and citation
accuracy (SPEC.md §5's metrics table).

Every judge call is cached to disk keyed by a hash of its exact inputs, so
re-running an eval doesn't re-bill the judge model (SPEC.md R5). This is
required, not an optimization: repeated full-leaderboard runs during
development would otherwise multiply real API cost for no new information.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from recipes.llm import LLM
from recipes.pricing import cost_usd

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / ".judge_cache"

JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"


@dataclass
class JudgeResult:
    score: float
    reasoning: str
    usd_cost: float
    from_cache: bool


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _cache_key(judge_type: str, **fields: str) -> str:
    payload = json.dumps({"judge_type": judge_type, "model": JUDGE_MODEL, **fields}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _write_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")


def _parse_json_response(text: str) -> dict:
    """Judge prompts ask for a bare JSON object. Models occasionally wrap it
    in a code fence anyway, so strip that defensively before parsing.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    return json.loads(stripped)


def _call_judge(
    llm: LLM, judge_type: str, prompt_file: str, prompt_vars: dict, cache_fields: dict
) -> JudgeResult:
    key = _cache_key(judge_type, **cache_fields)
    cached = _read_cache(key)
    if cached is not None:
        return JudgeResult(
            score=cached["score"],
            reasoning=cached["reasoning"],
            usd_cost=0.0,
            from_cache=True,
        )

    template = _load_prompt(prompt_file)
    prompt = template.format(**prompt_vars)
    response = llm.complete(prompt=prompt, model=JUDGE_MODEL, temperature=0.0)
    parsed = _parse_json_response(response.text)
    usd = cost_usd(
        JUDGE_MODEL,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_input_tokens=response.cached_input_tokens,
    )
    result = {"score": float(parsed["score"]), "reasoning": str(parsed["reasoning"])}
    _write_cache(key, result)
    return JudgeResult(score=result["score"], reasoning=result["reasoning"], usd_cost=usd, from_cache=False)


def judge_faithfulness(llm: LLM, question: str, context: str, answer: str) -> JudgeResult:
    return _call_judge(
        llm,
        judge_type="faithfulness",
        prompt_file="judge_faithfulness.txt",
        prompt_vars={"context": context, "question": question, "answer": answer},
        cache_fields={"question": question, "context": context, "answer": answer},
    )


def judge_answer_relevance(llm: LLM, question: str, answer: str) -> JudgeResult:
    return _call_judge(
        llm,
        judge_type="answer_relevance",
        prompt_file="judge_relevance.txt",
        prompt_vars={"question": question, "answer": answer},
        cache_fields={"question": question, "answer": answer},
    )


def judge_citation_accuracy(
    llm: LLM, chunks_with_ids: str, answer: str
) -> JudgeResult:
    return _call_judge(
        llm,
        judge_type="citation_accuracy",
        prompt_file="judge_citation.txt",
        prompt_vars={"chunks_with_ids": chunks_with_ids, "answer": answer},
        cache_fields={"chunks_with_ids": chunks_with_ids, "answer": answer},
    )
