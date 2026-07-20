"""Validator weight-only default (VAL-MEMB-007 / VAL-MEMB-008).

Normal validators pull weights from chain.joinbase.ai and set_weights; they
never host challenge writer DBs, master/postgres, or challenge-* services.
Challenge execution adapters default off.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from base.config.settings import Settings, ValidatorAgentSettings
from base.validator.agent import ChallengeDispatchExecutor, ValidatorAgent
from base.validator.agent.challenge_dispatch import DEFAULT_CHALLENGE_EXECUTOR_FACTORIES
from base.validator.agent.executor import BrokerConfig

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.validator.yml"
INSTALL_VALIDATOR = ROOT / "deploy" / "compose" / "install-validator.sh"
ENV_EXAMPLE = ROOT / "deploy" / "compose" / ".env.validator.example"
VALIDATOR_GUIDE = ROOT / "docs" / "validator" / "README.md"
VALIDATOR_OPS = ROOT / "docs" / "operations" / "validator.md"
VALIDATOR_EXAMPLE = ROOT / "config" / "validator.example.yaml"
SETTINGS_PY = ROOT / "src" / "base" / "config" / "settings.py"
CLI_MAIN = ROOT / "src" / "base" / "cli_app" / "main.py"


def _render_compose(tmp_path: Path, project_name: str) -> dict[str, Any]:
    config = tmp_path / f"{project_name}.yaml"
    identity = tmp_path / f"{project_name}-identity"
    broker_token = tmp_path / f"{project_name}-broker-token"
    config.write_text("{}\n", encoding="utf-8")
    identity.mkdir()
    broker_token.write_text("test-token\n", encoding="utf-8")
    environment = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": project_name,
        "BASE_VALIDATOR_IMAGE_REPOSITORY": "registry.example/base-validator-runtime",
        "BASE_VALIDATOR_IMAGE_DIGEST": "a" * 64,
        "BASE_VALIDATOR_CONFIG": str(config),
        "BASE_VALIDATOR_PROTOCOL_IDENTITY": str(identity),
        "BASE_VALIDATOR_BROKER_TOKEN": str(broker_token),
    }
    rendered = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(rendered.stdout)


# -- VAL-MEMB-007: default weight-only + chain.joinbase.ai -------------------


def test_settings_default_challenge_execution_off() -> None:
    settings = Settings()
    assert settings.validator.agent.challenge_execution_enabled is False
    assert settings.validator.registry_url == "https://chain.joinbase.ai"
    assert settings.validator.submit_on_chain_enabled is False
    assert settings.validator.resolved_weights_url == "https://chain.joinbase.ai"


def test_install_validator_defaults_weight_only_joinbase() -> None:
    content = INSTALL_VALIDATOR.read_text(encoding="utf-8")
    assert "challenge_execution_enabled: false" in content
    assert "https://chain.joinbase.ai" in content
    assert "weight-only" in content.lower() or "Weight-only" in content
    assert (
        "weights/latest" in content
        or "v1/weights" in content
        or "set_weights" in content
    )
    # No master/postgres/challenge bootstrap.
    assert "install-master.sh" not in content
    assert "master-postgres" not in content
    assert "challenge-prism" not in content
    assert "challenge-agent-challenge" not in content


def test_compose_validator_is_agent_only_no_writer_services(tmp_path: Path) -> None:
    rendered = _render_compose(tmp_path, "weight-only-a")
    services = set(rendered["services"])
    assert services == {"validator"}
    for forbidden in (
        "base-master-validator",
        "master-postgres",
        "postgres",
        "challenge-prism",
        "challenge-agent-challenge",
        "evaluator",
        "submitter",
    ):
        assert forbidden not in services
    blob = json.dumps(rendered).lower()
    assert "postgres" not in blob
    assert "challenge-prism" not in blob
    assert "challenge-agent-challenge" not in blob


def test_compose_header_documents_weight_only() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    assert "weight-only" in text.lower()
    assert "chain.joinbase.ai" in text
    assert "never run master" in text.lower() or "NEVER run master" in text


def test_env_example_weight_only_joinbase() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "https://chain.joinbase.ai" in text
    assert "weight-only" in text.lower()
    assert "challenge_execution_enabled" in text
    assert "chain.platform.network" not in text


def test_docs_weight_only_default_and_joinbase() -> None:
    guide = VALIDATOR_GUIDE.read_text(encoding="utf-8")
    ops = VALIDATOR_OPS.read_text(encoding="utf-8")
    blob = guide + "\n" + ops
    assert "weight-only" in blob.lower()
    assert "https://chain.joinbase.ai" in blob
    assert "/v1/weights/latest" in blob
    assert "set_weights" in blob
    assert "challenge_execution_enabled" in blob
    # Sole writer / no submissions-leaderboard writer on validator.
    assert "sole writer" in blob.lower() or "never write" in blob.lower()
    assert "submissions" in blob.lower() and "leaderboard" in blob.lower()
    assert "chain.platform.network" not in blob


def test_validator_example_yaml_weight_only() -> None:
    text = VALIDATOR_EXAMPLE.read_text(encoding="utf-8")
    assert "challenge_execution_enabled: false" in text
    assert "https://chain.joinbase.ai" in text
    assert "submit_on_chain_enabled: false" in text


# -- VAL-MEMB-008: no challenge writer; adapters default off -----------------


def test_build_validator_agent_weight_only_disables_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _KP:
        ss58_address = "weight-only-hotkey"

        def sign(self, message: bytes) -> bytes:
            return b"sig"

    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _KP(),
    )
    from base.cli_app.main import _build_validator_agent

    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(
        master_url="https://chain.joinbase.ai",
        broker_url="http://127.0.0.1:9",
        # default challenge_execution_enabled=False
    )
    agent = _build_validator_agent(settings)
    assert agent._execute_assignments is False
    executor = agent._executor
    assert isinstance(executor, ChallengeDispatchExecutor)
    # Factories empty => prism/agent-challenge adapters not wired.
    assert executor._factories == {}
    for slug in DEFAULT_CHALLENGE_EXECUTOR_FACTORIES:
        assert slug not in executor._factories


def test_build_validator_agent_opt_in_enables_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _KP:
        ss58_address = "executor-hotkey"

        def sign(self, message: bytes) -> bytes:
            return b"sig"

    monkeypatch.setattr(
        "base.cli_app.main.create_validator_keypair",
        lambda settings: _KP(),
    )
    from base.cli_app.main import _build_validator_agent

    settings = Settings()
    settings.validator.agent = ValidatorAgentSettings(
        master_url="https://chain.joinbase.ai",
        broker_url="http://127.0.0.1:9",
        challenge_execution_enabled=True,
    )
    agent = _build_validator_agent(settings)
    assert agent._execute_assignments is True
    executor = agent._executor
    assert isinstance(executor, ChallengeDispatchExecutor)
    for slug in DEFAULT_CHALLENGE_EXECUTOR_FACTORIES:
        assert slug in executor._factories


@pytest.mark.asyncio
async def test_weight_only_agent_skips_assignment_pull() -> None:
    pulls: list[str] = []

    class _Client:
        hotkey = "hk"

        async def register(self, **kwargs: Any) -> Any:
            del kwargs
            return SimpleNamespace(heartbeat_interval_seconds=60)

        async def heartbeat(self, **kwargs: Any) -> None:
            del kwargs
            return None

        async def pull(self) -> list[Any]:
            pulls.append("pulled")
            return []

        async def progress(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

        async def post_result(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

    agent = ValidatorAgent(
        client=_Client(),  # type: ignore[arg-type]
        executor=ChallengeDispatchExecutor(factories={}),
        broker=BrokerConfig(broker_url="http://127.0.0.1:9"),
        capabilities=["cpu"],
        version="test",
        execute_assignments=False,
    )
    summary = await agent.process_pending_assignments()
    assert summary.pulled == 0
    assert summary.completed == 0
    assert pulls == []


@pytest.mark.asyncio
async def test_weight_only_run_forever_heartbeats_without_assignments() -> None:
    heartbeats = 0

    class _Client:
        hotkey = "hk"

        async def register(self, **kwargs: Any) -> Any:
            del kwargs
            return SimpleNamespace(heartbeat_interval_seconds=1)

        async def heartbeat(self, **kwargs: Any) -> None:
            del kwargs
            nonlocal heartbeats
            heartbeats += 1
            if heartbeats >= 1:
                raise asyncio.CancelledError()

        async def pull(self) -> list[Any]:
            raise AssertionError("weight-only must not pull assignments")

    agent = ValidatorAgent(
        client=_Client(),  # type: ignore[arg-type]
        executor=ChallengeDispatchExecutor(factories={}),
        broker=BrokerConfig(broker_url="http://127.0.0.1:9"),
        capabilities=["cpu"],
        version="test",
        heartbeat_interval_seconds=0,
        execute_assignments=False,
    )
    with pytest.raises(asyncio.CancelledError):
        await agent.run_forever()
    assert heartbeats >= 1


def test_cli_docstring_and_settings_document_weight_only() -> None:
    cli = CLI_MAIN.read_text(encoding="utf-8")
    settings = SETTINGS_PY.read_text(encoding="utf-8")
    assert "challenge_execution_enabled" in settings
    assert "Weight-only" in settings or "weight-only" in settings
    assert "challenge_execution_enabled" in cli
    assert "weight-only" in cli.lower() or "Weight-only" in cli


def test_docs_note_optional_audit_reexec_non_write() -> None:
    blob = (
        VALIDATOR_GUIDE.read_text(encoding="utf-8")
        + "\n"
        + VALIDATOR_OPS.read_text(encoding="utf-8")
    )
    assert "audit" in blob.lower()
    assert "non-write" in blob.lower() or "non write" in blob.lower()


def test_install_validator_still_executable() -> None:
    assert INSTALL_VALIDATOR.is_file()
    mode = INSTALL_VALIDATOR.stat().st_mode
    assert mode & stat.S_IXUSR
    # Generated config always binds agent.master_url to the operator flag.
    content = INSTALL_VALIDATOR.read_text(encoding="utf-8")
    assert "master_url: ${MASTER_URL}" in content
    assert re.search(r"challenge_execution_enabled:\s*false", content)
