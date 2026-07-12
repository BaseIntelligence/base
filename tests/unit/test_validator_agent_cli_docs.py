"""Tests for the `base validator agent` CLI entrypoint and its documentation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from typer import BadParameter
from typer.testing import CliRunner

from base.cli_app.main import (
    _build_validator_agent,
    _build_validator_weight_submitter,
    _require_validator_master_url,
    _require_validator_protocol_identity,
    _run_validator_agent_runtime,
    app,
)
from base.config.settings import Settings, ValidatorAgentSettings
from base.validator.weight_submitter import ValidatorWeightSubmitter

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS_DOC = ROOT / "docs" / "operations" / "validator.md"


class _FakeKeypair:
    ss58_address = "agent-hotkey"

    def sign(self, message: bytes) -> bytes:
        return b"sig"


def test_validator_agent_command_is_registered() -> None:
    result = CliRunner().invoke(app, ["validator", "agent", "--help"])
    assert result.exit_code == 0
    assert "agent" in result.output.lower()


def test_build_validator_agent_wires_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _FakeKeypair(),
    )
    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(
        master_url="http://master:8081",
        capabilities=["cpu", "gpu"],
        version="9.9.9",
        heartbeat_interval_seconds=30,
        broker_url="http://127.0.0.1:8082",
    )

    agent = _build_validator_agent(settings)

    assert agent.hotkey == "agent-hotkey"
    assert agent.heartbeat_interval == 30


def test_build_validator_agent_threads_self_declared_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _FakeKeypair(),
    )
    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(
        master_url="http://master:8081",
        broker_url="http://127.0.0.1:8082",
        display_name="Acme Validator",
        logo_url="https://acme/logo.png",
    )

    agent = _build_validator_agent(settings)
    meta = agent._meta()

    assert meta["display_name"] == "Acme Validator"
    assert meta["logo_url"] == "https://acme/logo.png"


def test_build_validator_agent_omits_identity_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _FakeKeypair(),
    )
    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(
        master_url="http://master:8081",
        broker_url="http://127.0.0.1:8082",
    )

    agent = _build_validator_agent(settings)
    meta = agent._meta()

    assert "display_name" not in meta
    assert "logo_url" not in meta


def test_build_validator_weight_submitter_is_gated_off_by_default() -> None:
    # Default settings: submit_on_chain_enabled=False -> the per-validator
    # submitter is a no-op (and never builds a live submit runtime / Subtensor).
    submitter = _build_validator_weight_submitter(Settings())
    assert isinstance(submitter, ValidatorWeightSubmitter)
    assert submitter.submit_enabled is False
    # Production force-provenance even when gate is off.
    assert submitter._require_provenance is True
    assert submitter._observation_reporter is not None


def test_build_validator_weight_submitter_enabled_when_gate_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _KP:
        ss58_address = "bound-hotkey"

    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _KP(),
    )
    settings = Settings()
    settings.validator.submit_on_chain_enabled = True
    submitter = _build_validator_weight_submitter(settings)
    assert submitter.submit_enabled is True
    assert submitter._require_provenance is True
    assert submitter._expected_hotkey == "bound-hotkey"
    assert submitter._observation_reporter is not None


def test_build_validator_weight_submitter_fail_closed_on_identity_bind_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(settings: Any) -> Any:
        raise RuntimeError("no wallet")

    monkeypatch.setattr("base.cli_app.main.create_validator_keypair", _boom)
    settings = Settings()
    settings.validator.submit_on_chain_enabled = True
    submitter = _build_validator_weight_submitter(settings)
    assert submitter.submit_enabled is True
    assert submitter._expected_hotkey is None
    assert getattr(submitter, "_identity_unbound", False) is True


async def test_run_validator_agent_runtime_runs_own_submit_loop() -> None:
    # The validator runtime runs the agent loop AND this node's OWN weight-submit
    # loop concurrently; the submit loop is cancelled when the agent loop exits.
    submit_calls: list[int] = []

    class _FakeSubmitter:
        async def run_once(self) -> None:
            submit_calls.append(1)

    class _FakeAgent:
        async def run_forever(self) -> None:
            for _ in range(200):
                if submit_calls:
                    return
                await asyncio.sleep(0.01)

    await asyncio.wait_for(
        _run_validator_agent_runtime(
            cast(Any, _FakeAgent()), cast(Any, _FakeSubmitter()), 0
        ),
        timeout=5,
    )

    assert submit_calls  # the per-validator submit loop ran alongside the agent


def test_operations_doc_documents_validator_agent() -> None:
    content = OPERATIONS_DOC.read_text(encoding="utf-8")
    lowered = content.lower()
    assert "base validator agent" in content
    assert "own docker broker" in lowered
    assert "scoped gateway token" in lowered
    assert "holds no provider key" in lowered
    assert "heartbeat" in lowered


def test_require_validator_master_url_rejects_missing() -> None:
    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(master_url=None)
    with pytest.raises(BadParameter, match="master_url"):
        _require_validator_master_url(settings)


def test_require_validator_master_url_rejects_non_absolute() -> None:
    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(master_url="master:8081")
    with pytest.raises(BadParameter, match="absolute"):
        _require_validator_master_url(settings)


def test_require_validator_master_url_accepts_explicit_http() -> None:
    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(
        master_url="http://127.0.0.1:3180"
    )
    assert _require_validator_master_url(settings) == "http://127.0.0.1:3180"


def test_require_validator_protocol_identity_rejects_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(settings: Any) -> Any:
        raise RuntimeError("missing wallet")

    monkeypatch.setattr("base.cli_app.main.create_validator_keypair", _boom)
    with pytest.raises(BadParameter, match="protocol signing identity"):
        _require_validator_protocol_identity(Settings())


def test_require_validator_protocol_identity_accepts_hotkey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _FakeKeypair(),
    )
    keypair = _require_validator_protocol_identity(Settings())
    assert keypair.ss58_address == "agent-hotkey"
