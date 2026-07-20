"""Unit tests for the own-runner builder honoring each image's ACTUAL WORKDIR.

Parity contract (mirrors harbor 0.13.1's Docker backend):

* harbor resolves the task's working directory as
  ``effective_cwd = cwd or task_env_config.workdir`` where
  ``task_env_config.workdir`` is the image's *actual* ``WORKDIR`` (read from the
  image config). The own-runner backend calls ``prepare(task)`` with NO explicit
  ``cwd``, so the builder must inspect the image's configured ``WORKDIR`` and
  thread it into the ``docker run`` (``-w <workdir>``) and the returned
  :class:`DockerExecEnvironment`.
* For the ``fix-git`` task the image's ``WORKDIR`` is ``/app/personal-site`` --
  its ``solve.sh`` uses relative paths (``cat .git/logs/HEAD``) and only works
  when cwd is the repo root. Hardcoding ``/app`` silently breaks it.
* The 88 other tasks expose ``WORKDIR /app``, so threading the real ``WORKDIR``
  is a no-op for them (zero regression). The ``/app`` (:data:`TASK_WORKDIR`)
  value remains ONLY as the fallback when ``docker inspect`` fails or returns an
  empty ``WorkingDir``.

These are pure unit tests: ``subprocess.run`` is monkeypatched (a per-argv
decider), no real docker is invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner import container_builder
from agent_challenge.evaluation.own_runner.container_builder import (
    BuiltTaskContainer,
    TaskContainerBuilder,
)
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
from agent_challenge.evaluation.own_runner.taskdefs import (
    ParsedTask,
    ResourceLimits,
    parse_task,
)

_FIX_GIT_WORKDIR = "/app/personal-site"


def _cp(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    """A duck-typed ``subprocess.CompletedProcess`` (returncode/stdout/stderr)."""
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


def _has_w(argv: list[str], value: str) -> bool:
    return any(argv[i] == "-w" and argv[i + 1] == value for i in range(len(argv) - 1))


def _write_task(
    root: Path,
    *,
    dockerfile: str,
    docker_image: str | None = None,
    allow_internet: bool | None = None,
) -> ParsedTask:
    """Synthesize a real on-disk task tree and parse it via the Task-5 parser.

    Mirrors the fixture shape in ``tests/test_own_runner_container_builder.py``.
    """
    (root / "environment").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "instruction.md").write_text("do the thing")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    env_lines = []
    if docker_image is not None:
        env_lines.append(f'docker_image = "{docker_image}"')
    if allow_internet is not None:
        env_lines.append(f"allow_internet = {str(allow_internet).lower()}")
    env_block = "\n".join(env_lines)
    (root / "task.toml").write_text(f'[task]\nname = "t/sample"\n\n[environment]\n{env_block}\n')
    return parse_task(root, task_id="sample-task")


# --------------------------------------------------------------------------- #
# inspect_image_workdir (harbor's task_env_config.workdir == image WORKDIR)
# --------------------------------------------------------------------------- #
def test_inspect_image_workdir_returns_actual_workdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "inspect" in argv and "--format" in argv:
            # docker inspect --format '{{.Config.WorkingDir}}' -> trailing newline.
            return _cp(0, stdout=f"{_FIX_GIT_WORKDIR}\n")
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    assert builder.inspect_image_workdir("alexgshaw/fix-git:20260403") == _FIX_GIT_WORKDIR


def test_inspect_image_workdir_falls_back_on_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "inspect" in argv and "--format" in argv:
            return _cp(0, stdout="")  # image declares no WORKDIR
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    assert builder.inspect_image_workdir("img:tag") == container_builder.TASK_WORKDIR
    assert builder.inspect_image_workdir("img:tag") == "/app"


def test_inspect_image_workdir_falls_back_on_inspect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "inspect" in argv and "--format" in argv:
            return _cp(1, stderr="No such image: img:tag")
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    assert builder.inspect_image_workdir("img:tag") == container_builder.TASK_WORKDIR
    assert builder.inspect_image_workdir("img:tag") == "/app"


# --------------------------------------------------------------------------- #
# run_container threads workdir into argv AND the returned env
# --------------------------------------------------------------------------- #
def test_run_container_threads_workdir_into_argv_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(argv: list[str]) -> Any:
        if "run" in argv and "-d" in argv:
            return _cp(0, stdout="deadbeefcafe\n")  # fake container id
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    env = builder.run_container(
        "img:tag",
        resources=ResourceLimits(allow_internet=True),
        workdir=_FIX_GIT_WORKDIR,
    )

    assert isinstance(env, DockerExecEnvironment)
    assert env.workdir == _FIX_GIT_WORKDIR
    attempts = fake.run_attempts()
    assert len(attempts) == 1
    assert _has_w(attempts[0], _FIX_GIT_WORKDIR)


# --------------------------------------------------------------------------- #
# END-TO-END through prepare(): inspected WORKDIR is honored
# --------------------------------------------------------------------------- #
def test_prepare_honors_inspected_image_workdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Prebuilt docker_image set -> build_image takes the _ensure_prebuilt path;
    # the local ``docker image inspect <tag>`` succeeds so no pull happens.
    task = _write_task(
        tmp_path / "task",
        dockerfile="FROM alexgshaw/fix-git:20260403\n",
        docker_image="alexgshaw/fix-git:20260403",
        allow_internet=True,
    )

    def decide(argv: list[str]) -> Any:
        # docker image inspect <tag> -> present locally (no pull).
        if argv[1:3] == ["image", "inspect"]:
            return _cp(0)
        # docker inspect --format '{{.Config.WorkingDir}}' <image>
        if "inspect" in argv and "--format" in argv:
            return _cp(0, stdout=f"{_FIX_GIT_WORKDIR}\n")
        # docker run -d ...
        if "run" in argv and "-d" in argv:
            return _cp(0, stdout="deadbeefcafe\n")
        return _cp(0)

    fake = _FakeRun(decide)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)

    builder = TaskContainerBuilder()
    built = builder.prepare(task)

    assert isinstance(built, BuiltTaskContainer)
    assert built.env.workdir == _FIX_GIT_WORKDIR
    attempts = fake.run_attempts()
    assert len(attempts) == 1
    assert _has_w(attempts[0], _FIX_GIT_WORKDIR)
