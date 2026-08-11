"""Shared pytest fixtures.

Root-cause fix for a test-isolation bug found while building P2's
02_bm25.ipynb: evals/judges.py's disk cache lives at a real, persistent
filesystem path (outputs/.judge_cache/) by default. Without isolation,
tests (and manual/notebook runs against the real repo) share that cache,
so a test asserting "the judge call actually happened and failed" can
silently observe a stale cache hit from an unrelated earlier run instead.

This autouse fixture redirects evals.judges.CACHE_DIR to a fresh tmp_path
for every single test automatically, so no test needs to remember to do
this itself (tests/test_judges.py's existing explicit monkeypatch calls
still work fine on top of this -- they just redirect to a second, equally
isolated tmp_path, which is harmless).
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_judge_cache(tmp_path, monkeypatch):
    import evals.judges as judges_module

    monkeypatch.setattr(judges_module, "CACHE_DIR", tmp_path / "judge_cache")
