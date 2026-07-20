"""Former LLM hold-expiry tests: gateway/held status removed.

Legacy LLM held quarantine is terminal rejected under deterministic admission.
Covered by gw/migration/admission suites; keep this module as an import-safe skip stub.
"""

import importlib

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "LLM held expiry removed; covered by deterministic admission and migration "
        "tests (test_held_expiry.py)"
    )
)


def test_held_status_module_absent() -> None:
    from prism_challenge.models import SubmissionStatus

    assert not hasattr(SubmissionStatus, "HELD")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("prism_challenge.evaluator.llm_review")
