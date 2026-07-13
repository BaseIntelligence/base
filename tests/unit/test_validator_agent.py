"""Former gateway-coupled validator agent tests removed with LLM gateway.

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
        "(test_validator_agent.py)"
    )
)


def test_gateway_module_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("base.master.llm_gateway")
