"""Isolation-invariant tests for the in-CVM orchestrator's task-container launch.

These behavioral, black-box tests pin the isolation invariants the M2 in-CVM
orchestrator MUST preserve on every Terminal-Bench task container it launches via
:meth:`TaskContainerBuilder.run_container` (the ``docker run`` argument vector):

* ``--network none`` by default; an ``allow_internet`` task keeps network access
  (VAL-ORCH-013 / VAL-ORCH-015);
* the golden dataset + task cache is bind-mounted read-only into every task
  container, and writes to it fail (VAL-ORCH-016);
* hardened posture: ``cap-drop ALL``, ``no-new-privileges``, a bounded
  ``pids-limit``, writable ``tmpfs`` for ``/tmp``, and a writable workspace
  volume — rootfs stays writable for harbor ``/tests`` + ``/logs`` (VAL-ORCH-023);
* a task requesting a GPU is rejected and never launched on the CPU-only CVM
  (VAL-ORCH-024).

The pure-argv tests monkeypatch ``subprocess.run`` (a per-argv decider, mirroring
the sibling own-runner tests) so no real docker is invoked; the ``@_docker``
tests launch a throwaway ``python:3.12-slim`` container and prove the posture
against a real daemon (they skip when docker / the image is unavailable).
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner import container_builder
from agent_challenge.evaluation.own_runner.container_builder import (
    AGENT_WORKSPACE_TARGET,
    TASK_PIDS_LIMIT,
    ContainerBuildError,
    ReadOnlyMount,
    TaskContainerBuilder,
    hardening_run_args,
)
from agent_challenge.evaluation.own_runner.taskdefs import ParsedTask, ResourceLimits, parse_task

_IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", _IMAGE],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


_docker = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"docker + {_IMAGE} image required for the isolation-invariant integration tests",
)


# --------------------------------------------------------------------------- #
# fakes / helpers
# --------------------------------------------------------------------------- #
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


def _run_ok(argv: list[str]) -> Any:
    if "run" in argv and "-d" in argv:
        return _cp(0, stdout="deadbeefcafe\n")
    return _cp(0)


def _has_consecutive(argv: list[str], tokens: tuple[str, ...]) -> bool:
    n = len(tokens)
    return any(tuple(argv[i : i + n]) == tokens for i in range(len(argv) - n + 1))


def _single_run_argv(monkeypatch: pytest.MonkeyPatch, **run_kwargs: Any) -> list[str]:
    fake = _FakeRun(_run_ok)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)
    builder = run_kwargs.pop("builder", None) or TaskContainerBuilder()
    builder.run_container("img:tag", **run_kwargs)
    attempts = fake.run_attempts()
    assert len(attempts) == 1
    return attempts[0]


def _write_task(root: Path, *, gpus: int | None = None) -> ParsedTask:
    (root / "environment").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "instruction.md").write_text("do the thing")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    env_lines = []
    if gpus is not None:
        env_lines.append(f"gpus = {gpus}")
    (root / "task.toml").write_text(
        f'[task]\nname = "t/sample"\n\n[environment]\n{chr(10).join(env_lines)}\n'
    )
    return parse_task(root, task_id="sample-task")


# --------------------------------------------------------------------------- #
# VAL-ORCH-013 — --network none by default
# --------------------------------------------------------------------------- #
def test_default_task_launched_network_none(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _single_run_argv(monkeypatch, resources=ResourceLimits())
    assert _has_consecutive(argv, ("--network", "none"))


def test_task_without_allow_internet_is_network_none(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _single_run_argv(monkeypatch, resources=ResourceLimits(allow_internet=False))
    assert _has_consecutive(argv, ("--network", "none"))


# --------------------------------------------------------------------------- #
# VAL-ORCH-015 — allow_internet opt-in honored (and sibling stays isolated)
# --------------------------------------------------------------------------- #
def test_optin_task_gets_network_sibling_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # opt-in task: no `--network none` (default bridge => network access).
    optin = _single_run_argv(monkeypatch, resources=ResourceLimits(allow_internet=True))
    assert not _has_consecutive(optin, ("--network", "none"))
    # a sibling default task is still --network none (opt-in never leaks).
    sibling = _single_run_argv(monkeypatch, resources=ResourceLimits())
    assert _has_consecutive(sibling, ("--network", "none"))


# --------------------------------------------------------------------------- #
# VAL-ORCH-016 — golden/task cache mounted read-only into every task container
# --------------------------------------------------------------------------- #
def test_golden_cache_readonly_mount_in_launch_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "harbor-cache"
    cache.mkdir()
    builder = TaskContainerBuilder(readonly_mounts=(ReadOnlyMount(source=cache, target="/cache"),))
    argv = _single_run_argv(monkeypatch, builder=builder, resources=ResourceLimits())
    assert _has_consecutive(argv, ("-v", f"{cache}:/cache:ro"))


def test_readonly_mount_arg_is_ro() -> None:
    mount = ReadOnlyMount(source=Path("/data/golden"), target="/golden")
    assert mount.arg == "/data/golden:/golden:ro"


@_docker
def test_golden_cache_mount_is_readonly_in_real_container(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dataset-digest.json").write_text('{"tasks": {}}\n')
    builder = TaskContainerBuilder(readonly_mounts=(ReadOnlyMount(source=cache, target="/golden"),))
    name = f"own-runner-iso-{uuid.uuid4().hex[:10]}"
    env = builder.run_container(_IMAGE, resources=ResourceLimits(), container_name=name)
    try:
        # the golden file is readable ...
        read = subprocess.run(
            ["docker", "exec", name, "cat", "/golden/dataset-digest.json"],
            capture_output=True,
            text=True,
        )
        assert read.returncode == 0
        assert "tasks" in read.stdout
        # ... but writing to the mount fails (read-only bind).
        write = subprocess.run(
            ["docker", "exec", name, "sh", "-c", "echo x > /golden/pwned"],
            capture_output=True,
            text=True,
        )
        assert write.returncode != 0
        mounts = subprocess.run(
            ["docker", "inspect", "-f", "{{json .Mounts}}", name],
            capture_output=True,
            text=True,
        )
        assert '"RW":false' in mounts.stdout.replace(" ", "")
    finally:
        env.remove()


# --------------------------------------------------------------------------- #
# VAL-ORCH-023 — hardened container posture on every task container
# --------------------------------------------------------------------------- #
def test_hardening_flags_present_in_launch_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _single_run_argv(monkeypatch, resources=ResourceLimits())
    assert "--read-only" not in argv  # task guest rootfs must stay writable
    assert _has_consecutive(argv, ("--cap-drop", "ALL"))
    assert _has_consecutive(argv, ("--security-opt", "no-new-privileges"))
    assert _has_consecutive(argv, ("--pids-limit", str(TASK_PIDS_LIMIT)))


def test_tmpfs_is_tmp_only(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _single_run_argv(monkeypatch, resources=ResourceLimits())
    tmpfs_specs = [argv[i + 1] for i, tok in enumerate(argv) if tok == "--tmpfs"]
    assert tmpfs_specs, "expected a writable tmpfs mount"
    for spec in tmpfs_specs:
        mount_point = spec.split(":", 1)[0]
        assert mount_point == "/tmp", f"only /tmp may be tmpfs, got {mount_point!r}"


def test_hardening_run_args_shape() -> None:
    args = hardening_run_args()
    assert "--read-only" not in args
    assert "--cap-drop" in args and args[args.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in args
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert "--pids-limit" in args
    # workspace volume remains for agent staging (not a tmpfs).
    assert _has_consecutive(args, ("-v", AGENT_WORKSPACE_TARGET))


@_docker
def test_task_container_hardened_in_real_daemon(tmp_path: Path) -> None:
    builder = TaskContainerBuilder()
    name = f"own-runner-iso-{uuid.uuid4().hex[:10]}"
    env = builder.run_container(_IMAGE, resources=ResourceLimits(), container_name=name)
    try:
        posture = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.HostConfig.ReadonlyRootfs}} {{.HostConfig.CapDrop}} "
                "{{.HostConfig.PidsLimit}} {{.HostConfig.SecurityOpt}} "
                "{{.HostConfig.Privileged}}",
                name,
            ],
            capture_output=True,
            text=True,
        )
        assert posture.returncode == 0
        out = posture.stdout
        assert out.startswith("false ")  # writable rootfs (harbor paths)
        assert "[ALL]" in out  # cap-drop ALL
        assert str(TASK_PIDS_LIMIT) in out  # bounded pids
        assert "no-new-privileges" in out
        assert "false" in out.split()[-1].lower() or out.rstrip().endswith("false")
        # /tmp remains a writable tmpfs; rootfs itself is also writable.
        ok_tmp = subprocess.run(
            ["docker", "exec", name, "sh", "-c", "echo x > /tmp/ok && echo OK"],
            capture_output=True,
            text=True,
        )
        assert ok_tmp.returncode == 0
        assert "OK" in ok_tmp.stdout
        ok_root = subprocess.run(
            ["docker", "exec", name, "sh", "-c", "echo x > /root_probe && echo OK"],
            capture_output=True,
            text=True,
        )
        assert ok_root.returncode == 0
        assert "OK" in ok_root.stdout
    finally:
        env.remove()



# --------------------------------------------------------------------------- #
# Harbor verifier/agent paths must be writable on task containers
# --------------------------------------------------------------------------- #
# Production crash (2026-07-29): upload_tests did `docker exec -u root … mkdir -p
# /tests` and failed with "Read-only file system" because task containers
# incorrectly inherited the *job*-container RO posture. Task containers must
# keep cap-drop / nnp / pids hardening but leave the rootfs writable so harbor
# paths (/tests, /logs/verifier, /solution, /app) and apt-based test.sh work.


def test_hardening_run_args_does_not_force_readonly_rootfs() -> None:
    """Task containers are not the DooD job client — rootfs must stay writable."""
    args = hardening_run_args()
    assert "--read-only" not in args, (
        "task containers must not use --read-only; that blocks harbor "
        "mkdir /tests and apt-based verifiers (Read-only file system)"
    )
    assert _has_consecutive(args, ("--cap-drop", "ALL"))
    assert _has_consecutive(args, ("--security-opt", "no-new-privileges"))
    assert _has_consecutive(args, ("--pids-limit", str(TASK_PIDS_LIMIT)))


@_docker
def test_task_container_allows_harbor_verifier_paths() -> None:
    """Live: mkdir -p /tests and /logs/verifier succeed under production hardening."""
    builder = TaskContainerBuilder()
    name = f"own-runner-harbor-paths-{uuid.uuid4().hex[:10]}"
    env = builder.run_container(_IMAGE, resources=ResourceLimits(), container_name=name)
    try:
        for path in ("/tests", "/logs/verifier", "/solution", "/app"):
            proc = subprocess.run(
                ["docker", "exec", "-u", "root", name, "mkdir", "-p", path],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, (
                f"mkdir -p {path} failed rc={proc.returncode} "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        write = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "root",
                name,
                "sh",
                "-c",
                "echo 1 > /logs/verifier/reward.txt && "
                "echo hi > /app/run.py && "
                "echo ok > /tests/probe && cat /tests/probe",
            ],
            capture_output=True,
            text=True,
        )
        assert write.returncode == 0, (
            f"harbor path writes failed rc={write.returncode} "
            f"stdout={write.stdout!r} stderr={write.stderr!r}"
        )
        assert "ok" in write.stdout
        # Hardening that must remain: cap-drop ALL + nnp + pids + not privileged.
        posture = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.HostConfig.ReadonlyRootfs}} {{.HostConfig.CapDrop}} "
                "{{.HostConfig.PidsLimit}} {{json .HostConfig.SecurityOpt}} "
                "{{.HostConfig.Privileged}}",
                name,
            ],
            capture_output=True,
            text=True,
        )
        assert posture.returncode == 0
        out = posture.stdout.strip()
        assert out.startswith("false "), f"expected writable rootfs, got {out!r}"
        assert "[ALL]" in out
        assert str(TASK_PIDS_LIMIT) in out
        assert "no-new-privileges" in out
        assert out.endswith(" false") or out.rstrip().endswith("false")
    finally:
        env.remove()


# --------------------------------------------------------------------------- #
# VAL-ORCH-024 — GPU-requesting tasks are rejected (never launched)
# --------------------------------------------------------------------------- #
def test_gpu_task_rejected_no_run_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRun(_run_ok)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)
    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError):
        builder.run_container("img:tag", resources=ResourceLimits(gpus=1))
    assert fake.run_attempts() == [], "a GPU task must never reach docker run"


def test_gpu_task_rejected_via_prepare_no_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = _write_task(tmp_path / "gpu-task", gpus=1)
    fake = _FakeRun(_run_ok)
    monkeypatch.setattr(container_builder.subprocess, "run", fake)
    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError):
        builder.prepare(task)
    # neither a build nor a run is attempted for a GPU task.
    assert fake.run_attempts() == []
    assert not any("build" in a for a in fake.calls)
