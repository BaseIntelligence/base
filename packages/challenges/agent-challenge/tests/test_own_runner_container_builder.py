"""Tests for the own-runner task-container builder (Task 11).

These tests pin the container-builder contract against harbor 0.13.1's Docker
backend behavior (extracted from the wheel == the runner image's
``pip install harbor==0.13.1``):

  - image selection mirrors ``should_use_prebuilt_docker_image``
    (``definition.py``): a task with ``docker_image`` uses the prebuilt tag,
    otherwise the task's ``environment/Dockerfile`` is built.
  - the built image name mirrors harbor's ``hb__<env>`` naming and
    ``_sanitize_docker_image_name`` (``docker/docker.py``).
  - resource limits map exactly as harbor's ``write_resources_compose_file``
    (``environments/docker/__init__.py``): ONLY cpus + memory are enforced on
    the Docker backend (``resource_capabilities`` = cpu_limit/memory_limit).
    ``storage_mb`` is captured but not enforced; ``gpus > 0`` is unsupported on
    the Docker backend (harbor raises ``RuntimeError`` at ``base.py``).
  - ``allow_internet`` maps to the no-network override
    (``docker-compose-no-network.yaml`` → ``network_mode: none``).
  - the agent workspace is staged to ``/workspace/agent`` exactly as the harbor
    runner does today (``runner.py`` mounts target ``/workspace/agent``).
  - a bad/non-existent base image fails with a typed error carrying a known
    reason code — no hang.

Unit tests (pure mapping) run everywhere. Build/stage tests run against a real
Docker daemon and skip when Docker or the base image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from agent_challenge.evaluation.own_runner.container_builder import (
    AGENT_WORKSPACE_TARGET,
    BUILD_FAILED_REASON_CODE,
    BUILD_TIMEOUT_REASON_CODE,
    MAIN_IMAGE_PREFIX,
    TASK_WORKDIR,
    BuiltTaskContainer,
    ContainerBuildError,
    TaskContainerBuilder,
    dind_bringup_script,
    image_tag_for,
    network_arg,
    resource_run_args,
    sanitize_image_name,
    should_use_prebuilt,
    validate_resources,
)
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
from agent_challenge.evaluation.own_runner.reason_codes import REASON_CODES
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
    reason=f"docker + {_IMAGE} image required for container-builder build tests",
)


# --------------------------------------------------------------------------- #
# helpers: synthesize a real on-disk task tree and parse it (uses the real
# Task-5 parser so we consume a genuine ParsedTask).
# --------------------------------------------------------------------------- #
def _write_task(
    root: Path,
    *,
    dockerfile: str,
    docker_image: str | None = None,
    cpus: float | None = None,
    memory_mb: int | None = None,
    storage_mb: int | None = None,
    gpus: int | None = None,
    allow_internet: bool | None = None,
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
    if cpus is not None:
        env_lines.append(f"cpus = {cpus}")
    if memory_mb is not None:
        env_lines.append(f"memory_mb = {memory_mb}")
    if storage_mb is not None:
        env_lines.append(f"storage_mb = {storage_mb}")
    if gpus is not None:
        env_lines.append(f"gpus = {gpus}")
    if allow_internet is not None:
        env_lines.append(f"allow_internet = {str(allow_internet).lower()}")
    if build_timeout_sec is not None:
        env_lines.append(f"build_timeout_sec = {build_timeout_sec}")
    env_block = "\n".join(env_lines)
    (root / "task.toml").write_text(f'[task]\nname = "t/sample"\n\n[environment]\n{env_block}\n')
    return parse_task(root, task_id="sample-task")


# --------------------------------------------------------------------------- #
# constants / seam reuse
# --------------------------------------------------------------------------- #
def test_workspace_target_matches_harbor() -> None:
    # harbor stages the agent workspace at /workspace/agent (runner.py mounts).
    assert AGENT_WORKSPACE_TARGET == "/workspace/agent"


def test_task_workdir_is_app() -> None:
    assert TASK_WORKDIR == "/app"


def test_image_prefix_matches_harbor() -> None:
    assert MAIN_IMAGE_PREFIX == "hb__"


def test_dind_bringup_script_reuses_runner_seam() -> None:
    # Must REUSE the existing bring-up seam (runner._terminal_bench_dockerd_block),
    # not reimplement it. Identity check against the source of truth.
    from agent_challenge.evaluation.runner import _terminal_bench_dockerd_block

    script = dind_bringup_script()
    assert script == _terminal_bench_dockerd_block()
    assert "BASE_DOCKERD" in script


# --------------------------------------------------------------------------- #
# image-name sanitization (parity with harbor _sanitize_docker_image_name)
# --------------------------------------------------------------------------- #
def test_sanitize_lowercases_and_replaces() -> None:
    assert sanitize_image_name("Foo/Bar Baz") == "foo-bar-baz"


def test_sanitize_prepends_zero_for_nonalnum_start() -> None:
    assert sanitize_image_name("_weird").startswith("0")


def test_sanitize_keeps_allowed_punctuation() -> None:
    assert sanitize_image_name("a.b_c-d") == "a.b_c-d"


# --------------------------------------------------------------------------- #
# image selection (parity with should_use_prebuilt_docker_image)
# --------------------------------------------------------------------------- #
def test_should_use_prebuilt_true_when_docker_image_set(tmp_path: Path) -> None:
    task = _write_task(tmp_path / "t", dockerfile="FROM python:3.12-slim\n", docker_image="org/x:1")
    assert should_use_prebuilt(task) is True
    assert image_tag_for(task) == "org/x:1"


def test_should_use_prebuilt_false_without_docker_image(tmp_path: Path) -> None:
    task = _write_task(tmp_path / "t", dockerfile="FROM python:3.12-slim\n")
    assert should_use_prebuilt(task) is False
    assert image_tag_for(task) == f"{MAIN_IMAGE_PREFIX}sample-task"


def test_force_build_ignores_prebuilt_when_dockerfile_present(tmp_path: Path) -> None:
    task = _write_task(tmp_path / "t", dockerfile="FROM python:3.12-slim\n", docker_image="org/x:1")
    # force_build=True + Dockerfile present -> build (mirror harbor definition.py).
    assert should_use_prebuilt(task, force_build=True) is False


# --------------------------------------------------------------------------- #
# resource mapping (parity with write_resources_compose_file)
# --------------------------------------------------------------------------- #
def test_resource_args_cpus_and_memory() -> None:
    args = resource_run_args(ResourceLimits(cpus=1, memory_mb=2048))
    assert "--cpus" in args
    assert args[args.index("--cpus") + 1] == "1"
    assert "--memory" in args
    assert args[args.index("--memory") + 1] == "2048M"
    # harbor writes reservations == limits in AUTO mode.
    assert "--memory-reservation" in args
    assert args[args.index("--memory-reservation") + 1] == "2048M"


def test_resource_args_storage_not_enforced_on_docker() -> None:
    # harbor Docker resource_capabilities = cpu_limit/memory_limit only;
    # storage_mb is NOT mapped to a docker run flag.
    args = resource_run_args(ResourceLimits(cpus=1, memory_mb=512, storage_mb=10240))
    assert "--storage-opt" not in args
    assert not any("10240" in a for a in args)


def test_resource_args_empty_when_unset() -> None:
    assert resource_run_args(ResourceLimits()) == []


def test_validate_resources_rejects_gpus() -> None:
    # harbor Docker env has no GPU capability: gpus > 0 raises (base.py).
    with pytest.raises(ContainerBuildError) as exc:
        validate_resources(ResourceLimits(gpus=1))
    assert exc.value.reason_code in REASON_CODES


def test_validate_resources_allows_zero_gpus() -> None:
    validate_resources(ResourceLimits(gpus=0))  # no raise


# --------------------------------------------------------------------------- #
# network mapping (allow_internet -> no-network override)
# --------------------------------------------------------------------------- #
def test_network_none_when_internet_disallowed() -> None:
    assert network_arg(ResourceLimits(allow_internet=False)) == "none"


def test_network_none_when_unset_default_isolated() -> None:
    # Default (harbor's no_network policy + exec_bridge default) is isolated.
    assert network_arg(ResourceLimits()) == "none"


def test_network_default_when_internet_allowed() -> None:
    # allow_internet=True -> default bridge network (no --network none).
    assert network_arg(ResourceLimits(allow_internet=True)) is None


# --------------------------------------------------------------------------- #
# typed error contract
# --------------------------------------------------------------------------- #
def test_container_build_error_carries_known_reason_code() -> None:
    err = ContainerBuildError("boom", reason_code=BUILD_FAILED_REASON_CODE)
    assert err.reason_code in REASON_CODES
    assert BUILD_FAILED_REASON_CODE in REASON_CODES
    assert BUILD_TIMEOUT_REASON_CODE in REASON_CODES


# --------------------------------------------------------------------------- #
# real docker build + stage (S1)
# --------------------------------------------------------------------------- #
@_docker
def test_build_and_stage_representative_task(tmp_path: Path) -> None:
    # prepare() now bakes tmux at build time via a derived image. Ship a stub
    # ``tmux`` in the base so that layer's ``command -v tmux`` guard short-circuits
    # (no build-time network needed here); the real package-manager install path
    # is covered by test_own_runner_container_builder_tmux.py.
    dockerfile = textwrap.dedent(
        f"""\
        FROM {_IMAGE}
        WORKDIR /app
        RUN echo built > /built.marker
        RUN printf '#!/bin/sh\\necho tmux 3.5\\n' > /usr/local/bin/tmux \\
            && chmod +x /usr/local/bin/tmux
        """
    )
    task = _write_task(
        tmp_path / "task",
        dockerfile=dockerfile,
        cpus=1,
        memory_mb=512,
        storage_mb=10240,
        gpus=0,
        allow_internet=False,
        build_timeout_sec=600.0,
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "agent.py").write_text("print('hi')\n")

    builder = TaskContainerBuilder(image_prefix=f"hb__test{uuid.uuid4().hex[:8]}-")
    built = builder.prepare(task, workspace)
    try:
        assert isinstance(built, BuiltTaskContainer)
        assert built.workspace_target == "/workspace/agent"
        # honored setup step (RUN) baked into the image.
        marker = subprocess.run(
            ["docker", "exec", built.env.container_name, "cat", "/built.marker"],
            capture_output=True,
            text=True,
        )
        assert marker.returncode == 0
        assert "built" in marker.stdout
        # workspace staged at /workspace/agent.
        listing = subprocess.run(
            ["docker", "exec", built.env.container_name, "ls", "/workspace/agent"],
            capture_output=True,
            text=True,
        )
        assert listing.returncode == 0
        assert "agent.py" in listing.stdout
        # resource limits applied (cpus=1 -> 1e9 nanocpus; mem=512M).
        nanocpus = subprocess.run(
            ["docker", "inspect", "-f", "{{.HostConfig.NanoCpus}}", built.env.container_name],
            capture_output=True,
            text=True,
        )
        assert nanocpus.stdout.strip() == "1000000000"
        memory = subprocess.run(
            ["docker", "inspect", "-f", "{{.HostConfig.Memory}}", built.env.container_name],
            capture_output=True,
            text=True,
        )
        assert memory.stdout.strip() == str(512 * 1024 * 1024)
        # network isolated (allow_internet=False).
        netmode = subprocess.run(
            ["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", built.env.container_name],
            capture_output=True,
            text=True,
        )
        assert netmode.stdout.strip() == "none"
    finally:
        built.remove()
        subprocess.run(["docker", "image", "rm", "-f", built.image], capture_output=True, text=True)


@_docker
def test_prepare_isolated_exec_via_bridge(tmp_path: Path) -> None:
    # The returned env is a real DockerExecEnvironment usable for exec. Ship a
    # stub ``tmux`` so prepare()'s build-time tmux layer is an offline no-op (see
    # test_build_and_stage_representative_task).
    task = _write_task(
        tmp_path / "task",
        dockerfile=(
            f"FROM {_IMAGE}\nWORKDIR /app\n"
            "RUN printf '#!/bin/sh\\necho tmux 3.5\\n' > /usr/local/bin/tmux "
            "&& chmod +x /usr/local/bin/tmux\n"
        ),
        cpus=1,
        memory_mb=512,
    )
    builder = TaskContainerBuilder(image_prefix=f"hb__test{uuid.uuid4().hex[:8]}-")
    built = builder.prepare(task)
    try:
        assert isinstance(built.env, DockerExecEnvironment)
        import asyncio

        result = asyncio.run(built.env.exec("pwd"))
        assert result.stdout is not None
        assert result.stdout.strip() == "/app"
    finally:
        built.remove()
        subprocess.run(["docker", "image", "rm", "-f", built.image], capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# bad base image -> typed error, no hang (S2)
# --------------------------------------------------------------------------- #
@_docker
def test_bad_base_image_fails_cleanly(tmp_path: Path) -> None:
    bad = f"nonexistent-registry.invalid/nope:{uuid.uuid4().hex[:8]}"
    task = _write_task(
        tmp_path / "task",
        dockerfile=f"FROM {bad}\nRUN echo nope\n",
        build_timeout_sec=120.0,
    )
    builder = TaskContainerBuilder(image_prefix=f"hb__test{uuid.uuid4().hex[:8]}-")
    start = time.monotonic()
    with pytest.raises(ContainerBuildError) as exc:
        builder.prepare(task, tmp_path / "ws_missing_ok")
    elapsed = time.monotonic() - start
    assert exc.value.reason_code in REASON_CODES
    assert elapsed < 120, f"bad-base build hung for {elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# build timeout -> typed error, no hang (S4)
# --------------------------------------------------------------------------- #
@_docker
def test_build_timeout_maps_to_reason_code(tmp_path: Path) -> None:
    task = _write_task(
        tmp_path / "task",
        dockerfile=f"FROM {_IMAGE}\nRUN sleep 60\n",
        build_timeout_sec=2.0,
    )
    builder = TaskContainerBuilder(image_prefix=f"hb__test{uuid.uuid4().hex[:8]}-")
    start = time.monotonic()
    with pytest.raises(ContainerBuildError) as exc:
        builder.build_image(task)
    elapsed = time.monotonic() - start
    assert exc.value.reason_code == BUILD_TIMEOUT_REASON_CODE
    assert elapsed < 40, f"build timeout not enforced ({elapsed:.1f}s)"
