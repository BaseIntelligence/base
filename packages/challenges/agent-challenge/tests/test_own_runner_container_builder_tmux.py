"""Tests for the build-time tmux bake (own-runner container builder, P0 fix).

The eval runtime is network-isolated (``--network none``), so tmux can only be
made available at BUILD time. ``TaskContainerBuilder.ensure_tmux_image`` restores
upstream harbor's "tmux baked into the image" invariant by building a thin
derived image (``FROM <task_image>`` + a guarded package-manager ``RUN``) before
the task container is started.

Unit tests (monkeypatched ``subprocess``) pin the derived-build contract: the
Dockerfile shape, the install snippet, the typed error / timeout mapping, the
host-network fallback, and the ``prepare()`` wiring. One Docker-gated test builds
a real derived image and proves tmux is usable inside a ``--network none``
container (skips when no build-time network is available).
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
    BUILD_FAILED_REASON_CODE,
    BUILD_TIMEOUT_REASON_CODE,
    BuiltTaskContainer,
    ContainerBuildError,
    TaskContainerBuilder,
    sanitize_image_name,
)
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
from agent_challenge.evaluation.own_runner.taskdefs import ParsedTask, parse_task

_IMAGE = "python:3.12-slim"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _ScriptedRun:
    """Fake ``subprocess.run``: returns queued results (or raises) per call.

    Records each call's argv + kwargs so the Dockerfile (passed via ``input``)
    and the chosen ``--network`` can be asserted.
    """

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append({"argv": list(argv), "kwargs": kwargs})
        if not self._results:
            return _cp(0)
        item = self._results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _write_task(
    root: Path,
    *,
    dockerfile: str,
    docker_image: str | None = None,
    build_timeout_sec: float | None = None,
) -> ParsedTask:
    (root / "environment").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "instruction.md").write_text("do the thing")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    env_lines = []
    if docker_image is not None:
        env_lines.append(f'docker_image = "{docker_image}"')
    if build_timeout_sec is not None:
        env_lines.append(f"build_timeout_sec = {build_timeout_sec}")
    env_block = "\n".join(env_lines)
    (root / "task.toml").write_text(f'[task]\nname = "t/sample"\n\n[environment]\n{env_block}\n')
    return parse_task(root, task_id="sample-task")


def _task(tmp_path: Path, *, build_timeout_sec: float | None = None) -> ParsedTask:
    return _write_task(
        tmp_path / uuid.uuid4().hex,
        dockerfile=f"FROM {_IMAGE}\n",
        build_timeout_sec=build_timeout_sec,
    )


# --------------------------------------------------------------------------- #
# install snippet: guarded + multi-package-manager
# --------------------------------------------------------------------------- #
def test_tmux_install_snippet_is_guarded_and_multi_pkgmgr() -> None:
    snippet = container_builder.TMUX_INSTALL_SNIPPET
    # Idempotent no-op when tmux already exists (parity-safe).
    assert "command -v tmux" in snippet
    # Installs via whichever package manager the base image ships.
    for manager in ("apt-get", "apk", "dnf", "yum"):
        assert manager in snippet
    assert "tmux" in snippet


# --------------------------------------------------------------------------- #
# ensure_tmux_image: derived build contract (monkeypatched docker)
# --------------------------------------------------------------------------- #
def test_ensure_tmux_image_builds_derived_from_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = _ScriptedRun([_cp(0)])
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    task = _task(tmp_path, build_timeout_sec=600.0)
    derived = builder.ensure_tmux_image("alexgshaw/foo:20251031", task)

    assert derived == sanitize_image_name("alexgshaw/foo:20251031-tmux")
    assert len(scripted.calls) == 1
    argv = scripted.calls[0]["argv"]
    assert argv[:2] == ["docker", "build"]
    assert "-t" in argv and derived in argv
    # ``-`` => Dockerfile read from stdin with an empty build context.
    assert argv[-1] == "-"
    dockerfile = scripted.calls[0]["kwargs"]["input"]
    assert dockerfile.startswith("FROM alexgshaw/foo:20251031\n")
    assert f"RUN {container_builder.TMUX_INSTALL_SNIPPET}" in dockerfile
    # build timeout (timeouts.build_sec) is enforced on the derived build too.
    assert scripted.calls[0]["kwargs"]["timeout"] == 600.0


def test_ensure_tmux_image_build_failure_raises_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = _ScriptedRun([_cp(1, stderr="boom: pull access denied")])
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError) as exc:
        builder.ensure_tmux_image("base:1", _task(tmp_path))

    assert exc.value.reason_code == BUILD_FAILED_REASON_CODE
    assert exc.value.stage == "tmux-layer"
    # A non-bridge failure must NOT trigger the host-network retry.
    assert len(scripted.calls) == 1


def _build_attempts(scripted: _ScriptedRun) -> list[dict[str, Any]]:
    """The recorded ``docker build`` calls (drops the ``image rm`` cleanups)."""
    return [c for c in scripted.calls if c["argv"][:2] == ["docker", "build"]]


def test_ensure_tmux_image_timeout_on_both_paths_maps_reason_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A build TIMEOUT now triggers a single host-network retry; only when that
    # retry ALSO times out does the bake fail. The reason code still maps to the
    # bounded build-timeout sentinel, and the bake never hangs unboundedly.
    scripted = _ScriptedRun(
        [
            subprocess.TimeoutExpired(cmd="docker build", timeout=2),
            _cp(0),  # bridge-attempt cleanup (image rm -f)
            subprocess.TimeoutExpired(cmd="docker build", timeout=2),
            _cp(0),  # host-attempt cleanup (image rm -f)
        ]
    )
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError) as exc:
        builder.ensure_tmux_image("base:1", _task(tmp_path, build_timeout_sec=2.0))

    assert exc.value.reason_code == BUILD_TIMEOUT_REASON_CODE
    assert exc.value.stage == "tmux-layer"
    attempts = _build_attempts(scripted)
    assert len(attempts) == 2  # bounded: bridge + exactly one host retry
    assert "--network" not in attempts[0]["argv"]
    assert "host" in attempts[1]["argv"]


def test_ensure_tmux_image_retries_on_build_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The headline next-terrier failure: the no-NAT default bridge makes apt-get
    # HANG until the build timeout (not a fast endpoint error). A bridge TIMEOUT
    # must trigger exactly one --network host retry, which succeeds.
    scripted = _ScriptedRun(
        [
            subprocess.TimeoutExpired(cmd="docker build", timeout=2),
            _cp(0),  # bridge-attempt cleanup (image rm -f)
            _cp(0),  # host retry succeeds
        ]
    )
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    derived = builder.ensure_tmux_image("base:1", _task(tmp_path, build_timeout_sec=2.0))

    assert derived == sanitize_image_name("base:1-tmux")
    attempts = _build_attempts(scripted)
    assert len(attempts) == 2
    assert "--network" not in attempts[0]["argv"]
    assert attempts[1]["argv"].count("--network") == 1
    assert "host" in attempts[1]["argv"]


def test_ensure_tmux_image_retries_on_dns_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When the no-NAT bridge fails fast with a DNS/connect error instead of
    # hanging, that is still a network failure and must fall back to host.
    dns = _cp(1, stderr="E: Could not resolve 'deb.debian.org': bad address")
    scripted = _ScriptedRun([dns, _cp(0)])
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    derived = builder.ensure_tmux_image("base:1", _task(tmp_path))

    assert derived == sanitize_image_name("base:1-tmux")
    attempts = _build_attempts(scripted)
    assert len(attempts) == 2
    assert "--network" not in attempts[0]["argv"]
    assert "host" in attempts[1]["argv"]


def test_ensure_tmux_image_host_retry_timeout_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A DNS failure on the bridge followed by a TIMEOUT on the host retry must
    # raise (bounded) rather than retry endlessly.
    dns = _cp(1, stderr="Temporary failure resolving 'deb.debian.org'")
    scripted = _ScriptedRun([dns, subprocess.TimeoutExpired(cmd="docker build", timeout=2), _cp(0)])
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    with pytest.raises(ContainerBuildError) as exc:
        builder.ensure_tmux_image("base:1", _task(tmp_path, build_timeout_sec=2.0))

    assert exc.value.reason_code == BUILD_TIMEOUT_REASON_CODE
    assert exc.value.stage == "tmux-layer"
    assert len(_build_attempts(scripted)) == 2  # bounded: no third attempt


def test_ensure_tmux_image_retries_on_bridge_endpoint_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bridge = _cp(1, stderr="failed to create endpoint x on network bridge: docker0 missing")
    scripted = _ScriptedRun([bridge, _cp(0)])
    monkeypatch.setattr(container_builder.subprocess, "run", scripted)

    builder = TaskContainerBuilder()
    derived = builder.ensure_tmux_image("base:1", _task(tmp_path))

    assert derived == sanitize_image_name("base:1-tmux")
    assert len(scripted.calls) == 2
    # First attempt: default bridge (no --network). Second: forced host network.
    assert "--network" not in scripted.calls[0]["argv"]
    assert scripted.calls[1]["argv"].count("--network") == 1
    assert "host" in scripted.calls[1]["argv"]


def test_prepare_starts_container_from_derived_tmux_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = _write_task(tmp_path / "task", dockerfile=f"FROM {_IMAGE}\n", docker_image="org/x:1")
    builder = TaskContainerBuilder()
    monkeypatch.setattr(builder, "build_image", lambda t, force_build=False: "org/x:1")
    monkeypatch.setattr(builder, "inspect_image_workdir", lambda image: "/app")
    monkeypatch.setattr(container_builder.subprocess, "run", _ScriptedRun([_cp(0)]))

    captured: dict[str, Any] = {}

    def _fake_run_container(
        image: str, *, resources: Any, container_name: str | None = None, workdir: str
    ) -> DockerExecEnvironment:
        captured["image"] = image
        return DockerExecEnvironment("fake-container", workdir=workdir)

    monkeypatch.setattr(builder, "run_container", _fake_run_container)

    built = builder.prepare(task)
    derived = sanitize_image_name("org/x:1-tmux")

    assert isinstance(built, BuiltTaskContainer)
    # The container is started from the derived (tmux-baked) image, not the base.
    assert captured["image"] == derived
    assert built.image == derived


# --------------------------------------------------------------------------- #
# real docker: the derived image actually contains a usable tmux
# --------------------------------------------------------------------------- #
def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", _IMAGE], capture_output=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


_docker = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"docker + {_IMAGE} image required for derived-image tmux build test",
)


@_docker
def test_ensure_tmux_image_real_derived_runs_tmux_offline(tmp_path: Path) -> None:
    # Bake a tmux-containing base using build-time network (as in the real eval
    # DooD); skip cleanly when no build network is available in this sandbox.
    base = f"ac-tmux-base-{uuid.uuid4().hex[:8]}"
    dockerfile = (
        f"FROM {_IMAGE}\n"
        "RUN set -e; if ! command -v tmux >/dev/null 2>&1; then "
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux; fi\n"
    )
    try:
        setup = subprocess.run(
            ["docker", "build", "--network", "host", "-t", base, "-"],
            input=dockerfile,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("no build-time network to bake a tmux base image")
    if setup.returncode != 0:
        pytest.skip(f"no build-time network to bake a tmux base image: {setup.stderr[-200:]}")

    derived = sanitize_image_name(f"{base}-tmux")
    try:
        task = _write_task(
            tmp_path / "task", dockerfile=f"FROM {_IMAGE}\n", build_timeout_sec=300.0
        )
        builder = TaskContainerBuilder()
        result = builder.ensure_tmux_image(base, task)
        assert result == derived
        # tmux is baked in and runs even when the container has NO network.
        probe = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", derived, "tmux", "-V"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert probe.returncode == 0, probe.stdout
        assert "tmux" in probe.stdout
    finally:
        subprocess.run(["docker", "image", "rm", "-f", derived], capture_output=True, text=True)
        subprocess.run(["docker", "image", "rm", "-f", base], capture_output=True, text=True)


def _host_network_build_ok() -> bool:
    """Whether a host-network ``docker build`` can reach the internet here.

    Gates the host-network-fallback real-docker test: if not even host
    networking can fetch packages, there is no point exercising the fallback.
    """
    try:
        probe = subprocess.run(
            ["docker", "build", "--network", "host", "-"],
            input=(
                f"FROM {_IMAGE}\n"
                "RUN getent hosts deb.debian.org >/dev/null 2>&1 || "
                "python -c \"import socket; socket.gethostbyname('deb.debian.org')\"\n"
            ),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


@_docker
def test_ensure_tmux_image_real_host_network_fallback(tmp_path: Path) -> None:
    # End-to-end proof of the fallback on a no-NAT-bridge host: start from a base
    # that LACKS tmux so the derived build MUST run apt-get. On next-terrier the
    # default bridge has no DNS/NAT, so the bridge attempt fails/hangs and the
    # bake only succeeds via the bounded ``--network host`` retry. Skip when even
    # host networking cannot reach the internet to install tmux.
    if not _host_network_build_ok():
        pytest.skip("no host-network build connectivity to install tmux")

    builder = TaskContainerBuilder()
    # Bounded build-timeout ceiling: on a no-NAT bridge the first attempt hangs
    # until this timeout, then the host retry installs tmux. Each attempt is
    # bounded, so the bake never hangs unboundedly.
    task = _write_task(tmp_path / "task", dockerfile=f"FROM {_IMAGE}\n", build_timeout_sec=120.0)
    derived = sanitize_image_name(f"{_IMAGE}{container_builder.TMUX_IMAGE_SUFFIX}")
    try:
        result = builder.ensure_tmux_image(_IMAGE, task)
        assert result == derived
        # The derived image ships tmux and runs with NO network (eval runtime).
        probe = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", derived, "tmux", "-V"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert probe.returncode == 0, probe.stdout
        assert "tmux" in probe.stdout
    finally:
        subprocess.run(["docker", "image", "rm", "-f", derived], capture_output=True, text=True)
