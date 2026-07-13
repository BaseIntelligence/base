"""Former gateway-coupled decentralized e2e tests removed with LLM gateway.

Gateway-era fixtures imported removed symbols
(WorkAssignmentLifecycleResolver, base.master.llm_gateway).
Covered by tests/unit/test_gateway_absence.py and non-gateway agent suites.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "LLM gateway removed; covered by test_gateway_absence.py "
        "(test_end_to_end_decentralized.py)"
    )
)


def test_gateway_module_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("base.master.llm_gateway")
