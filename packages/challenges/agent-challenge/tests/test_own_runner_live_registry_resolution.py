"""DooD task-image resolution: live-registry override (fail-closed to legacy).

The in-CVM DooD orchestrator resolves each task's container image via
``TaskContainerBuilder``. For the live subset a pullable ``repo@sha256`` ref is
supplied through ``live_registry_refs``; the builder then ``docker pull``s that
exact pinned ref instead of building from the task Dockerfile or pulling the
task.toml floating ``docker_image`` tag.

Pinned invariant: with NO live refs configured the resolution is byte-identical
to the legacy behavior (build from Dockerfile / pull the task.toml tag), so
flag-off / offline runs are unchanged. These tests spy on the docker CLI to
assert exactly which image ref the builder resolves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent_challenge.evaluation.own_runner import container_builder as cb
from agent_challenge.evaluation.own_runner.taskdefs import ParsedTask, parse_task

_LIVE_REF = "docker.io/mathiiss/agent-challenge-tb21-foo@sha256:" + ("a" * 64)


def _write_task(root: Path, *, docker_image: str | None) -> ParsedTask:
    (root / "environment").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (root / "instruction.md").write_text("do the thing")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    env = f'docker_image = "{docker_image}"\n' if docker_image else ""
    (root / "task.toml").write_text(f'[task]\nname = "terminal-bench/foo"\n\n[environment]\n{env}')
    return parse_task(root, task_id="terminal-bench/foo")


class _DockerSpy:
    """Records docker argv; makes ``image inspect`` miss so a pull is forced."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, *args, **kwargs):
        self.calls.append(list(argv))
        rc = 0
        # `docker image inspect <ref>` -> miss (1) so the builder pulls.
        if argv[:3] == ["docker", "image", "inspect"]:
            rc = 1
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="")

    def commands(self) -> list[str]:
        return [" ".join(c) for c in self.calls]


def test_live_ref_is_pulled_when_configured(tmp_path, monkeypatch):
    task = _write_task(tmp_path, docker_image="alexgshaw/foo:20251031")
    spy = _DockerSpy()
    monkeypatch.setattr(cb.subprocess, "run", spy)

    builder = cb.TaskContainerBuilder(live_registry_refs={"foo": _LIVE_REF})
    image = builder.build_image(task)

    assert image == _LIVE_REF
    joined = spy.commands()
    # The live pinned ref is pulled; the floating task.toml tag is NOT used.
    assert any(c == f"docker pull {_LIVE_REF}" for c in joined), joined
    assert not any("alexgshaw/foo:20251031" in c for c in joined), joined
    assert not any(c.startswith("docker build") for c in joined), joined


def test_no_live_ref_falls_back_to_prebuilt_tag(tmp_path, monkeypatch):
    # Fail-closed: no live refs => legacy behavior (pull the task.toml tag).
    task = _write_task(tmp_path, docker_image="alexgshaw/foo:20251031")
    spy = _DockerSpy()
    monkeypatch.setattr(cb.subprocess, "run", spy)

    builder = cb.TaskContainerBuilder()  # no live refs
    image = builder.build_image(task)

    assert image == "alexgshaw/foo:20251031"
    joined = spy.commands()
    assert any(c == "docker pull alexgshaw/foo:20251031" for c in joined), joined
    assert not any(_LIVE_REF in c for c in joined), joined


def test_live_ref_absent_for_this_task_falls_back(tmp_path, monkeypatch):
    # A live registry configured for OTHER tasks must not affect this task.
    task = _write_task(tmp_path, docker_image="alexgshaw/foo:20251031")
    spy = _DockerSpy()
    monkeypatch.setattr(cb.subprocess, "run", spy)

    builder = cb.TaskContainerBuilder(live_registry_refs={"other-task": _LIVE_REF})
    image = builder.build_image(task)

    assert image == "alexgshaw/foo:20251031"
    assert not any(_LIVE_REF in c for c in spy.commands())


def test_live_ref_pulled_even_for_dockerfile_only_task(tmp_path, monkeypatch):
    # A task with no prebuilt docker_image would normally build; a live ref
    # short-circuits that to a deterministic pull of the pinned registry image.
    task = _write_task(tmp_path, docker_image=None)
    spy = _DockerSpy()
    monkeypatch.setattr(cb.subprocess, "run", spy)

    builder = cb.TaskContainerBuilder(live_registry_refs={"foo": _LIVE_REF})
    image = builder.build_image(task)

    assert image == _LIVE_REF
    assert not any(c.startswith("docker build") for c in spy.commands())
