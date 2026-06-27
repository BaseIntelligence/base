"""Tests for the `base validator agent` CLI entrypoint and its documentation."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from base.cli_app.main import _build_validator_agent, app
from base.config.settings import Settings, ValidatorAgentSettings

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


def test_operations_doc_documents_validator_agent() -> None:
    content = OPERATIONS_DOC.read_text(encoding="utf-8")
    lowered = content.lower()
    assert "base validator agent" in content
    assert "own docker broker" in lowered
    assert "scoped gateway token" in lowered
    assert "holds no provider key" in lowered
    assert "heartbeat" in lowered
