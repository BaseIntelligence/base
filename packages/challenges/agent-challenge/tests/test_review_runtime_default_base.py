"""Contract: measured review runtime default API base is joinbase challenge URL."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_review_runtime():
    path = Path(__file__).resolve().parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_review_api_base_is_joinbase_challenge_path() -> None:
    runtime = _load_review_runtime()
    assert runtime.DEFAULT_REVIEW_API_BASE_URL == (
        "https://chain.joinbase.ai/challenges/agent-challenge"
    )
    # Historical dead host must not be the product default (live residual: 502).
    assert "platform.network" not in runtime.DEFAULT_REVIEW_API_BASE_URL
