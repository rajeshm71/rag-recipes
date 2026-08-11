"""Regression coverage for the review fixes applied to evals/judges.py:
cache-key template versioning (#1), robust JSON parsing (#2), and atomic
cache writes (#5).
"""

import pytest

from evals.judges import (
    JudgeParseError,
    _cache_key,
    _parse_json_response,
    _read_cache,
    _write_cache,
)


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
