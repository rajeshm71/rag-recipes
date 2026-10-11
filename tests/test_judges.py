"""Coverage for evals/judges.py: cache-key template versioning, robust
JSON parsing, score range checking, atomic cache writes, and the rule that
mock judge calls never touch the on-disk cache.
"""

import pytest

from evals.judges import (
    JudgeParseError,
    _cache_key,
    _parse_json_response,
    _read_cache,
    _write_cache,
    judge_answer_relevance,
)
from recipes.llm import MockLLM


def test_cache_key_changes_when_prompt_template_changes():
    key_a = _cache_key("faithfulness", prompt_template="version A", question="q")
    key_b = _cache_key("faithfulness", prompt_template="version B", question="q")
    assert key_a != key_b


def test_cache_key_stable_for_same_inputs():
    key1 = _cache_key("faithfulness", prompt_template="v1", question="q")
    key2 = _cache_key("faithfulness", prompt_template="v1", question="q")
    assert key1 == key2


def test_parse_json_response_plain():
    result = _parse_json_response('{"score": 1, "reasoning": "ok"}', "faithfulness")
    assert result == {"score": 1.0, "reasoning": "ok"}


def test_parse_json_response_strips_code_fence():
    text = '```json\n{"score": 0, "reasoning": "bad"}\n```'
    result = _parse_json_response(text, "faithfulness")
    assert result == {"score": 0.0, "reasoning": "bad"}


def test_parse_json_response_raises_judge_parse_error_on_garbage():
    with pytest.raises(JudgeParseError):
        _parse_json_response("not json at all", "faithfulness")


def test_parse_json_response_raises_judge_parse_error_on_missing_fields():
    with pytest.raises(JudgeParseError):
        _parse_json_response('{"unexpected": "shape"}', "faithfulness")


@pytest.mark.parametrize("score", ["5", "-0.5", "1.01"])
def test_parse_json_response_rejects_out_of_range_score(score):
    with pytest.raises(JudgeParseError):
        _parse_json_response('{"score": ' + score + ', "reasoning": "r"}', "faithfulness")


def test_mock_judge_calls_never_touch_the_cache(tmp_path, monkeypatch):
    import evals.judges as judges_module

    monkeypatch.setattr(judges_module, "CACHE_DIR", tmp_path)
    llm = MockLLM(default_response='{"score": 1, "reasoning": "canned"}')

    first = judge_answer_relevance(llm, "q", "a")
    second = judge_answer_relevance(llm, "q", "a")

    assert first.from_cache is False
    assert second.from_cache is False  # a real backend would have hit the cache here
    assert len(llm.calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_read_cache_treats_corrupt_file_as_miss(tmp_path, monkeypatch):
    import evals.judges as judges_module

    monkeypatch.setattr(judges_module, "CACHE_DIR", tmp_path)
    key = "deadbeef"
    (tmp_path / f"{key}.json").write_text("{not valid json", encoding="utf-8")

    assert _read_cache(key) is None


def test_write_cache_then_read_cache_round_trips(tmp_path, monkeypatch):
    import evals.judges as judges_module

    monkeypatch.setattr(judges_module, "CACHE_DIR", tmp_path)
    key = "abc123"
    _write_cache(key, {"score": 1.0, "reasoning": "fine"})

    assert _read_cache(key) == {"score": 1.0, "reasoning": "fine"}
    # No leftover .tmp file after a successful write.
    assert not (tmp_path / f"{key}.json.tmp").exists()
