"""Former gateway tests collapsed: LLM gateway removed (see test_gateway_absence)."""
from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.skip(reason="LLM gateway removed; covered by test_gateway_absence.py (test_assignment_gateway_token_issuance.py)")

def test_gateway_module_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("base.master.llm_gateway")
