"""Former LLM hard-gate tests: gateway/review removed; see deterministic absence suite."""

import importlib

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "LLM review removed; covered by deterministic admission tests (test_prism_llm_hard_gate.py)"
    )
)


def test_llm_modules_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("prism_challenge.evaluator.llm_review")
