"""Offline E2E path for checkbox 27 — real modules, fixture Lium (no LIUM_API_KEY)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "e2e_lium_attestation.py"
)


def _load_e2e_module():
    spec = importlib.util.spec_from_file_location("e2e_lium_attestation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Ensure script path resolution works when loaded as module
    sys.modules["e2e_lium_attestation"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e2e():
    return _load_e2e_module()


@pytest.mark.asyncio
async def test_offline_honest_writes_score_tier1_attestation_mode(e2e) -> None:
    bag = await e2e.run_offline("honest")
    errors = e2e._assert_honest(bag)
    assert not errors, errors
    assert bag["constation_ok"] is True
    assert bag["score_written"] is True
    assert bag["effective_tier"] == 1
    assert bag["attestation_mode"] == e2e.ATTESTATION_MODE_V1
    assert bag["score_row"] is not None and bag["score_row"] > 0.0


@pytest.mark.asyncio
async def test_offline_adversarial_midrun_swap_no_score_miner_fault(e2e) -> None:
    bag = await e2e.run_offline("adversarial")
    errors = e2e._assert_adversarial(bag)
    assert not errors, errors
    assert bag["run_record_ok"] is False
    assert bag["run_record_reason"] == "corroboration_mismatch"
    assert bag["constation_ok"] is False
    assert bag["score_written"] is False
    assert bag["score_row"] is None
    assert str(bag["ingest_reason"] or "").startswith("miner_fault:")


def test_cli_offline_honest_exit_zero(e2e) -> None:
    code = e2e.main(["--mode", "honest", "--offline"])
    assert code == 0


def test_cli_offline_adversarial_exit_zero(e2e) -> None:
    code = e2e.main(["--mode", "adversarial", "--offline"])
    assert code == 0


def test_live_unavailable_without_key(e2e, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    assert e2e._live_available() is False
