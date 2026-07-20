"""Tests for the own-runner PRODUCTION reference agents (Task 22, Phase A).

:mod:`agent_challenge.evaluation.own_runner.reference_agents` provides the two
production reference agents that drive the full-set oracle parity gate WITHOUT
importing harbor:

* :class:`OracleAgent` -- reproduces harbor 0.13.1's ``OracleAgent.run``: it
  executes the task's reference ``solve.sh`` (staged into the live container by
  the preparer) through ``environment.exec``, with harbor's
  ``DEBIAN_FRONTEND=noninteractive`` + the agent's ``extra_env``, so the verifier
  later observes a solved ``/app``. It is NOT the trivial ``_OracleAgent`` test
  double (which merely returns ``"DONE"`` and solves nothing).
* :class:`NopAgent` -- a pure no-op (the resolved=0 floor), reproducing
  ``harbor/agents/nop.py``.

Plus :func:`stage_solution_into`, the per-trial seam that copies a host solution
dir into the container (the agent class is fixed per job but the solution is per
task, so staging cannot flow through agent construction).

These are pure no-docker unit tests: a recording fake environment captures the
exec/upload calls and the agents are also driven through the REAL
:class:`AgentDriver` to prove they satisfy its construction + invocation contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_challenge.evaluation.own_runner.driver import AgentDriver
from agent_challenge.evaluation.own_runner.reference_agents import (
    SOLUTION_CONTAINER_DIR,
    NopAgent,
    OracleAgent,
    stage_solution_into,
)


class _RecordingEnv:
    """A recording exec-bridge stand-in (duck-typed DockerExecEnvironment)."""

    def __init__(self, *, stdout: str | None = "solved-output", return_code: int = 0) -> None:
        self.exec_calls: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[tuple[Path, str]] = []
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

    def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        self.uploads.append((Path(source_dir), target_dir))

    def remove(self) -> None:
        self.removed = True


# ===========================================================================
# OracleAgent
# ===========================================================================
async def test_oracle_agent_execs_staged_solve_script() -> None:
    env = _RecordingEnv()
    agent = OracleAgent(logs_dir=None, model_name=None)

    await agent.setup(env)
    output = await agent.run("solve the task", env, context=None)

    # Exactly one exec: bash <solution>/solve.sh.
    assert len(env.exec_calls) == 1
    command, kwargs = env.exec_calls[0]
    assert command == f"bash {SOLUTION_CONTAINER_DIR}/solve.sh"
    # harbor passes DEBIAN_FRONTEND=noninteractive in the solution exec env.
    assert kwargs["env"]["DEBIAN_FRONTEND"] == "noninteractive"
    # The agent returns the solve.sh output (becomes the agent.log evidence).
    assert output == "solved-output"


async def test_oracle_agent_merges_extra_env() -> None:
    env = _RecordingEnv()
    agent = OracleAgent(logs_dir=None, model_name=None, extra_env={"TOKEN": "abc"})

    await agent.run("go", env, context=None)

    _command, kwargs = env.exec_calls[0]
    assert kwargs["env"]["TOKEN"] == "abc"
    # DEBIAN_FRONTEND is still present alongside the merged extra env.
    assert kwargs["env"]["DEBIAN_FRONTEND"] == "noninteractive"


async def test_oracle_agent_setup_is_noop_and_does_not_exec() -> None:
    env = _RecordingEnv()
    agent = OracleAgent(logs_dir=None, model_name=None)
    await agent.setup(env)
    assert env.exec_calls == []


# ===========================================================================
# NopAgent
# ===========================================================================
async def test_nop_agent_does_nothing() -> None:
    env = _RecordingEnv()
    agent = NopAgent(logs_dir=None, model_name=None)

    await agent.setup(env)
    output = await agent.run("anything", env, context=None)

    # The no-op floor: no exec, no upload, no output.
    assert env.exec_calls == []
    assert output is None


# ===========================================================================
# stage_solution_into
# ===========================================================================
def test_stage_solution_into_uploads_to_solution_dir(tmp_path: Path) -> None:
    env = _RecordingEnv()
    src = tmp_path / "solution"
    src.mkdir()
    (src / "solve.sh").write_text("#!/bin/bash\necho hi\n")

    stage_solution_into(env, src)

    assert env.uploads == [(src, SOLUTION_CONTAINER_DIR)]


# ===========================================================================
# Driver-contract integration (no docker): the agents construct + drive.
# ===========================================================================
async def test_oracle_agent_drives_to_completed_via_real_driver() -> None:
    env = _RecordingEnv()
    driver = AgentDriver(agent_class=OracleAgent)

    result = await driver.drive(
        environment=env,
        instruction="solve",
        start_session=False,
    )

    assert result.status == "completed"
    # The driver invoked the agent, which ran the staged solve.sh.
    assert any(c.startswith("bash ") and "solve.sh" in c for c, _ in env.exec_calls)


async def test_oracle_agent_receives_agent_env_via_driver() -> None:
    env = _RecordingEnv()
    driver = AgentDriver(agent_class=OracleAgent)

    await driver.drive(
        environment=env,
        instruction="solve",
        agent_env={"SECRET": "xyz"},
        start_session=False,
    )

    # The driver forwards agent_env as extra_env to the constructor; the oracle
    # then injects it into the solve.sh exec env.
    _command, kwargs = env.exec_calls[0]
    assert kwargs["env"]["SECRET"] == "xyz"


async def test_nop_agent_drives_to_completed_without_exec_via_real_driver() -> None:
    env = _RecordingEnv()
    driver = AgentDriver(agent_class=NopAgent)

    result = await driver.drive(
        environment=env,
        instruction="noop",
        start_session=False,
    )

    assert result.status == "completed"
    assert env.exec_calls == []
