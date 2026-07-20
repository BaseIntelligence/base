"""Unit tests pinning the stable ``base.own_runner=1`` docker label on every
per-trial sandbox container the own-runner backend creates via ``docker run``.

Both sandbox creation sites -- the task sandbox
(:meth:`TaskContainerBuilder.run_container` -> ``_run_container_attempt``) and
the exec sandbox (:meth:`DockerExecEnvironment.launch`) -- must emit
``--label base.own_runner=1`` in their ``docker run`` argv so a host-level sweep
can identify own-runner sandboxes. These are pure unit tests: ``subprocess.run``
is monkeypatched (mirroring the fake-runner style in the sibling own-runner
tests), no real docker is invoked.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_challenge.evaluation.own_runner import container_builder, exec_bridge
from agent_challenge.evaluation.own_runner.container_builder import TaskContainerBuilder
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
from agent_challenge.evaluation.own_runner.taskdefs import ResourceLimits

_LABEL = ("--label", "base.own_runner=1")


def _cp(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return type(
        "CP",
        (),
        {"returncode": returncode, "stdout": stdout, "stderr": stderr},
    )()


class _FakeRun:
    """Records argv and returns a verdict decided per-call by ``decide``."""

    def __init__(self, decide: Any) -> None:
        self.calls: list[list[str]] = []
        self._decide = decide

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(argv))
        return self._decide(argv)

    def run_attempts(self) -> list[list[str]]:
        return [a for a in self.calls if "run" in a and "-d" in a]


def _has_consecutive(argv: list[str], tokens: tuple[str, ...]) -> bool:
    n = len(tokens)
    return any(tuple(argv[i : i + n]) == tokens for i in range(len(argv) - n + 1))


def test_task_sandbox_run_container_carries_own_runner_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "run" in argv and "-d" in argv:
            return _cp(0, stdout="deadbeefcafe\n")
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    builder.run_container(
        "img:tag",
        resources=ResourceLimits(allow_internet=True),
        container_name="own-runner-task-x",
    )

    attempts = fake.run_attempts()
    assert len(attempts) == 1
    argv = attempts[0]
    assert _has_consecutive(argv, _LABEL)
    # The label sits right after ``--name <name>`` and before ``-w``.
    name_idx = argv.index("--name")
    assert argv[name_idx + 2 : name_idx + 4] == list(_LABEL)


def test_exec_sandbox_launch_carries_own_runner_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRun(lambda argv: _cp(0))
    monkeypatch.setattr(exec_bridge.subprocess, "run", fake)

    DockerExecEnvironment.launch("img:tag", container_name="own-runner-exec-x")

    attempts = fake.run_attempts()
    assert len(attempts) == 1
    argv = attempts[0]
    assert _has_consecutive(argv, _LABEL)
    name_idx = argv.index("--name")
    assert argv[name_idx + 2 : name_idx + 4] == list(_LABEL)
