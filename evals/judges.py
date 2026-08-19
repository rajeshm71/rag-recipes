"""LLM-as-judge callers for faithfulness, answer relevance, and citation
accuracy.

Every judge call is cached to disk keyed by a hash of its exact inputs, so
re-running an eval doesn't re-bill the judge model. This is required, not
an optimization: repeated full-leaderboard runs during development would
otherwise multiply real API cost for no new information.
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


class JudgeParseError(Exception):
    """Raised when a judge model's response can't be parsed as the expected
    JSON shape. Callers (evals/run.py) catch this per-question rather than
    letting one bad response crash an entire eval run.
    """


@dataclass
class JudgeResult:
    score: float
    reasoning: str
    usd_cost: float
    from_cache: bool


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _cache_key(judge_type: str, prompt_template: str, **fields: str) -> str:
    # Include the prompt template's own content in the cache key, not just
    # its variable inputs. Without this, editing a judge prompt (wording
    # fix, stricter rubric) would silently reuse stale cached scores
    # computed under the OLD prompt, undermining the reproducibility this
    # cache exists to guarantee.
    payload = json.dumps(
        {
            "judge_type": judge_type,
            "model": JUDGE_MODEL,
            "prompt_template": prompt_template,
            **fields,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(key: str) -> dict | None:
    # A cache file left truncated by an interrupted write (Ctrl-C mid-run)
    # would otherwise raise JSONDecodeError here and crash the caller. Treat
    # an unparseable cache file as a miss instead -- the judge call just
    # re-runs and rewrites a good cache entry.
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(key: str, data: dict) -> None:
    # Write atomically. Writing directly to the final path risks leaving a
    # truncated/corrupt file if the process is killed mid-write;
    # write-to-temp-then-replace makes the write all-or-nothing.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final_path = CACHE_DIR / f"{key}.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    tmp_path.replace(final_path)


def _parse_json_response(text: str, judge_type: str) -> dict:
    """Judge prompts ask for a bare JSON object. Models occasionally wrap it
    in a code fence anyway, so strip that defensively before parsing.

    Raises a JudgeParseError naming the judge type and the raw text (rather
    than letting json.JSONDecodeError / KeyError propagate with no context),
    which evals/run.py catches per-question instead of losing the entire
    run to one bad judge response.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
        return {"score": float(parsed["score"]), "reasoning": str(parsed["reasoning"])}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise JudgeParseError(
            f"{judge_type} judge returned unparseable response: {exc}. Raw text: {text!r}"
        ) from exc


def _call_judge(
    llm: LLM, judge_type: str, prompt_file: str, prompt_vars: dict, cache_fields: dict
) -> JudgeResult:
    template = _load_prompt(prompt_file)
    key = _cache_key(judge_type, prompt_template=template, **cache_fields)
    cached = _read_cache(key)
    if cached is not None:
        return JudgeResult(
            score=cached["score"],
            reasoning=cached["reasoning"],
            usd_cost=0.0,
            from_cache=True,
        )

    prompt = template.format(**prompt_vars)
    response = llm.complete(prompt=prompt, model=JUDGE_MODEL, temperature=0.0)
    # May raise JudgeParseError -- intentionally not caught here, so the
    # caller (evals/run.py) can decide how to handle a bad judge response
    # per-question instead of this module silently swallowing it.
    result = _parse_json_response(response.text, judge_type)
    usd = cost_usd(
        JUDGE_MODEL,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_input_tokens=response.cached_input_tokens,
    )
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
