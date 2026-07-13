"""Removed: llm_usage_records table/migration is forward-only dropped."""

import pytest

pytestmark = pytest.mark.skip(reason="LLM gateway usage metering removed")


def test_placeholder() -> None:
    assert True
