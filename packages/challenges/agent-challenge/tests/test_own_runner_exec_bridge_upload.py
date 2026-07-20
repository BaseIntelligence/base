"""Unit tests for ``DockerExecEnvironment.upload_dir`` (Task 22, Phase A).

The own-runner oracle stages the reference solution into the live container via
``environment.upload_dir(source_dir, target_dir)`` (called by
``reference_agents.stage_solution_into``). This mirrors harbor's
``environment.upload_dir`` and the sibling ``verifier_runner.upload_tests``:
``mkdir -p <target>`` as root, then ``docker cp <src>/. <container>:<target>``
(``/.`` copies directory *contents*, preserving file modes).

These are pure unit tests: ``subprocess.run`` is monkeypatched so no real docker
is required; they assert the exact argv sequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner import exec_bridge
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment


class _RecordingRun:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(argv))
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def test_upload_dir_mkdirs_then_docker_cp_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _RecordingRun()
    monkeypatch.setattr(exec_bridge.subprocess, "run", recorder)

    env = DockerExecEnvironment("ctr-xyz", docker_bin="docker")
    src = tmp_path / "solution"
    src.mkdir()
    (src / "solve.sh").write_text("#!/bin/bash\necho hi\n")

    env.upload_dir(src, "/solution")

    assert recorder.calls == [
        ["docker", "exec", "-u", "root", "ctr-xyz", "mkdir", "-p", "/solution"],
        ["docker", "cp", f"{src}/.", "ctr-xyz:/solution"],
    ]


def test_upload_dir_strips_trailing_slash_on_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _RecordingRun()
    monkeypatch.setattr(exec_bridge.subprocess, "run", recorder)

    env = DockerExecEnvironment("ctr-1", docker_bin="docker")
    src = tmp_path / "sol"
    src.mkdir()

    env.upload_dir(f"{src}/", "/solution")

    # Source trailing slash is stripped so the cp arg is always ``<src>/.``.
    assert recorder.calls[1] == ["docker", "cp", f"{src}/.", "ctr-1:/solution"]


def test_upload_dir_honors_custom_docker_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _RecordingRun()
    monkeypatch.setattr(exec_bridge.subprocess, "run", recorder)

    env = DockerExecEnvironment("c", docker_bin="/usr/bin/docker")
    src = tmp_path / "sol"
    src.mkdir()

    env.upload_dir(src, "/solution")

    assert recorder.calls[0][0] == "/usr/bin/docker"
    assert recorder.calls[1][0] == "/usr/bin/docker"
