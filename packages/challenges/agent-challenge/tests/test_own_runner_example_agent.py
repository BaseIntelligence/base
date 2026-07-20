"""Tests for the canonical example submission agent (``scripts/example_agent/agent.py``).

The own-runner driver constructs the submitted ``Agent`` as
``Agent(logs_dir=, model_name=, **extra)`` (``extra`` may carry ``extra_env``),
then drives ``setup`` before ``run``. The committed example agent is the canonical
minimal submission, so it must satisfy that contract exactly and prove
in-container execution via ``environment.exec`` while making no model calls.

These are pure no-docker unit tests: the example module is loaded from its file
location (it lives at a ZIP/script root, not in the package), constructed and
driven directly, and also driven through the REAL :class:`AgentDriver` to prove
it satisfies the construction + invocation contract end to end.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner.driver import AgentDriver

_EXAMPLE_AGENT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "example_agent" / "agent.py"


def _load_example_agent_class() -> type:
    spec = importlib.util.spec_from_file_location("example_agent_agent", _EXAMPLE_AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


class _RecordingEnv:
    """A recording exec-bridge stand-in (duck-typed DockerExecEnvironment)."""

    def __init__(self, *, stdout: str | None = "agent-challenge-ok", return_code: int = 0) -> None:
        self.exec_calls: list[tuple[str, dict[str, Any]]] = []
        self.removed = False
        self._stdout = stdout
        self._return_code = return_code

    async def exec(self, command: str, **kwargs: Any) -> Any:
        self.exec_calls.append((command, kwargs))
        return type(
            "R",
            (),
            {"return_code": self._return_code, "stdout": self._stdout, "stderr": None},
        )()

    def remove(self) -> None:
        self.removed = True


def test_example_agent_constructs_per_driver_contract() -> None:
    Agent = _load_example_agent_class()

    # The driver constructs Agent(logs_dir=, model_name=, **extra). Unknown
    # kwargs (e.g. extra_env, harbor factory extras) must be tolerated.
    agent = Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="no-llm",
        extra_env={"TOKEN": "abc"},
        unexpected_extra="ignored",
    )
    assert agent is not None


def test_example_agent_constructs_with_defaults() -> None:
    Agent = _load_example_agent_class()
    agent = Agent(logs_dir=None, model_name=None)
    assert agent is not None


async def test_example_agent_setup_is_noop_and_does_not_exec() -> None:
    Agent = _load_example_agent_class()
    env = _RecordingEnv()
    agent = Agent(logs_dir=None, model_name=None)

    await agent.setup(env)

    assert env.exec_calls == []


async def test_example_agent_run_proves_execution_via_exec() -> None:
    Agent = _load_example_agent_class()
    env = _RecordingEnv()
    agent = Agent(logs_dir=None, model_name=None)

    await agent.setup(env)
    output = await agent.run("do the task", env, context=None)

    # Exactly one exec proving in-container execution; output is the marker.
    assert len(env.exec_calls) == 1
    assert output == "agent-challenge-ok"


async def test_example_agent_run_returns_fallback_when_exec_empty() -> None:
    Agent = _load_example_agent_class()
    env = _RecordingEnv(stdout="")
    agent = Agent(logs_dir=None, model_name=None)

    output = await agent.run("go", env, context=None)

    assert output  # non-empty fallback summary even when the exec stdout is empty


async def test_example_agent_drives_to_completed_via_real_driver() -> None:
    Agent = _load_example_agent_class()
    env = _RecordingEnv()
    driver = AgentDriver(agent_class=Agent)

    result = await driver.drive(
        environment=env,
        instruction="prove execution",
        start_session=False,
    )

    assert result.status == "completed"
    # The driver constructed + drove the agent, which proved execution via exec.
    assert len(env.exec_calls) == 1


def test_example_agent_makes_no_model_calls() -> None:
    # The canonical example agent must be no-LLM: its source references no model
    # client / network SDK, only environment.exec.
    source = _EXAMPLE_AGENT_PATH.read_text()
    for forbidden in ("openai", "anthropic", "httpx", "requests", "litellm", "openrouter"):
        assert forbidden not in source.lower()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
