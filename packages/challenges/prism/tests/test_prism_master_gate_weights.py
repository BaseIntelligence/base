"""Former gateway-era master-gate/weights suite.

``prism_challenge.evaluator.llm_review`` is gone. LLM gate assertions live in
``test_gateway_absence_and_deterministic_admission.py``. Score-owner and
finalization contracts for VAL-WEIGHT-093/094 live in:

* ``tests/test_score_owner_policy.py``
* ``tests/test_score_owner_finalization.py``
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "LLM review and master gateway gate removed; "
        "covered by test_score_owner_* and deterministic admission suites"
    )
)


def test_llm_review_module_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("prism_challenge.evaluator.llm_review")
