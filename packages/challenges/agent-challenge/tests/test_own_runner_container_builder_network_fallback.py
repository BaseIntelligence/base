"""Unit tests for ``TaskContainerBuilder.run_container``'s host-network fallback.

The build-time path (``_docker_build``) already retries on the host network when
the daemon cannot create the default-bridge endpoint (a broken/sibling-docker
host whose ``docker0`` is absent; never real DinD). The RUNTIME ``docker run``
path needs the same defect-gated fallback so an ``allow_internet=True`` task --
which runs on the default bridge -- can still start on such a host.

Parity contract (mirrors the build fallback):

* The fallback triggers ONLY when (a) the no-flag attempt fails with the
  bridge-endpoint defect AND (b) the task's network is the default bridge
  (``allow_internet=True`` -> ``network_arg`` is ``None``). It retries once with
  ``--network host``.
* A non-bridge ``docker run`` failure never retries (raises immediately).
* An ``allow_internet=False`` task (``--network none``) never falls back to host
  -- isolation must not be silently widened.

These are pure unit tests: ``subprocess.run`` is monkeypatched, no real docker.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_challenge.evaluation.own_runner import container_builder
from agent_challenge.evaluation.own_runner.container_builder import (
    ContainerBuildError,
    TaskContainerBuilder,
)
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
from agent_challenge.evaluation.own_runner.taskdefs import ResourceLimits

_BRIDGE_ERR = (
    "docker: Error response from daemon: failed to create endpoint "
    "own-runner-task-x on network bridge: ... docker0 ... Device does not exist."
)


def _cp(returncode: int, stderr: str = "") -> Any:
    return type("CP", (), {"returncode": returncode, "stdout": "", "stderr": stderr})()


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


def _has_network(argv: list[str], value: str) -> bool:
    return any(argv[i] == "--network" and argv[i + 1] == value for i in range(len(argv) - 1))


def test_run_container_falls_back_to_host_network_on_bridge_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "run" in argv and "-d" in argv:
            if _has_network(argv, "host"):
                return _cp(0)
            return _cp(125, _BRIDGE_ERR)  # default-bridge attempt hits the defect
        return _cp(0)  # rm -f cleanup between attempts

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    env = builder.run_container(
        "img:tag",
        resources=ResourceLimits(allow_internet=True),
        container_name="own-runner-task-x",
    )

    assert isinstance(env, DockerExecEnvironment)
    assert env.container_name == "own-runner-task-x"
    attempts = fake.run_attempts()
    assert len(attempts) == 2
    assert not _has_network(attempts[0], "host")
    assert _has_network(attempts[1], "host")
    # The partial container is removed between attempts (reuse of the same name).
    assert any("rm" in a and "-f" in a for a in fake.calls)


def test_run_container_does_not_retry_on_non_bridge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "run" in argv and "-d" in argv:
            return _cp(125, "docker: Error response from daemon: no such image: img:tag")
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError):
        builder.run_container("img:tag", resources=ResourceLimits(allow_internet=True))

    assert len(fake.run_attempts()) == 1  # no host-network retry


def test_run_container_isolated_task_never_falls_back_to_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "run" in argv and "-d" in argv:
            return _cp(125, _BRIDGE_ERR)
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError):
        builder.run_container("img:tag", resources=ResourceLimits(allow_internet=False))

    attempts = fake.run_attempts()
    # The single attempt is isolated (--network none); never widened to host.
    assert len(attempts) == 1
    assert _has_network(attempts[0], "none")
    assert not any(_has_network(a, "host") for a in attempts)
