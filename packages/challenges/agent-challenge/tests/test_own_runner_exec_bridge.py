"""Tests for the own-runner environment exec-bridge (Task 10).

These tests pin the exec-bridge contract against harbor 0.13.1's
``DockerComposeEnvironment.exec`` behavior proven in gate G3:

  - default cwd = task workdir (``/app``) when ``cwd`` is None
  - the Docker backend MERGES stderr into stdout (``stderr`` is always None;
    all output lands in ``stdout``)
  - ``return_code = process.returncode or 0`` (faithful exit-code propagation)
  - empty output decodes to None (``... if stdout_bytes else None``)
  - timeout: host-side terminate -> 5s grace -> kill, then
    ``raise RuntimeError("Command timed out after {timeout_sec} seconds")``
  - signature matches harbor EXACTLY (no ``timeout`` / ``workdir`` kwargs), so
    baseagent's adapter cascade lands on ``timeout_sec=`` just like real harbor

They run against a throwaway container (``python:3.12-slim``, which ships bash +
coreutils, same base image harbor's runner uses). The suite skips only when
Docker or the image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from agent_challenge.evaluation.own_runner.exec_bridge import (
    DockerExecEnvironment,
    ExecResult,
)

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


pytestmark = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"docker + {_IMAGE} image required for exec-bridge container tests",
)


@pytest.fixture(scope="module")
def env():
    environment = DockerExecEnvironment.launch(_IMAGE)
    try:
        yield environment
    finally:
        environment.remove()


# --- ExecResult model parity (harbor.environments.base.ExecResult) ---------


def test_exec_result_fields_match_harbor() -> None:
    result = ExecResult(stdout=None, stderr=None, return_code=0)
    assert result.stdout is None
    assert result.stderr is None
    assert result.return_code == 0
    # harbor names the exit field `return_code`, NOT `exit_code`.
    assert not hasattr(result, "exit_code")


# --- output shape: stderr->stdout merge + exit-code propagation -------------


async def test_exec_merges_stderr_into_stdout_and_propagates_exit_code(env) -> None:
    result = await env.exec("echo hi; echo err 1>&2; exit 7")
    assert isinstance(result, ExecResult)
    # Docker backend merges stderr into stdout; stderr is always None.
    assert result.stderr is None
    assert "hi" in result.stdout
    assert "err" in result.stdout  # the stderr line landed in stdout
    # exit code propagates exactly.
    assert result.return_code == 7


async def test_exec_zero_exit_clean_stdout(env) -> None:
    result = await env.exec("printf hello")
    assert result.return_code == 0
    assert result.stdout == "hello"
    assert result.stderr is None


async def test_exec_empty_output_is_none(env) -> None:
    # No output + exit 0 -> stdout is None (`... if stdout_bytes else None`).
    result = await env.exec("true")
    assert result.return_code == 0
    assert result.stdout is None
    assert result.stderr is None


# --- cwd default = /app (the task workdir) ---------------------------------


async def test_exec_cwd_defaults_to_app(env) -> None:
    result = await env.exec("pwd")
    assert result.return_code == 0
    assert result.stdout is not None
    assert result.stdout.strip() == "/app"


async def test_exec_cwd_override(env) -> None:
    result = await env.exec("pwd", cwd="/tmp")
    assert result.return_code == 0
    assert result.stdout is not None
    assert result.stdout.strip() == "/tmp"


# --- env injection parity --------------------------------------------------


async def test_exec_env_vars_injected(env) -> None:
    result = await env.exec("echo $FOO", env={"FOO": "bar"})
    assert result.return_code == 0
    assert result.stdout is not None
    assert result.stdout.strip() == "bar"


# --- timeout kill semantics (terminate -> 5s grace -> kill -> raise) -------


async def test_exec_timeout_raises_runtime_error_with_harbor_message(env) -> None:
    start = time.monotonic()
    with pytest.raises(RuntimeError) as exc_info:
        await env.exec("sleep 60", timeout_sec=2)
    elapsed = time.monotonic() - start
    # Byte-identical message to harbor's RuntimeError.
    assert str(exc_info.value) == "Command timed out after 2 seconds"
    # Killed at ~2s (+ up to 5s grace) -- NOT waited out for the full 60s sleep.
    assert elapsed < 30, f"timeout took {elapsed:.1f}s; expected host-side kill near 2s"


# --- signature parity: NO `timeout`/`workdir` kwargs (harbor fidelity) -----


async def test_exec_signature_rejects_timeout_kwarg(env) -> None:
    # Harbor's exec has NO `timeout` kwarg (only `timeout_sec`). baseagent's
    # adapter relies on `timeout=` raising TypeError so it falls through to the
    # `timeout_sec=` attempt. The bridge MUST reproduce that exactly.
    with pytest.raises(TypeError):
        await env.exec("echo hi", timeout=2)


async def test_exec_signature_rejects_workdir_kwarg(env) -> None:
    # Harbor's exec uses `cwd`, NOT `workdir`.
    with pytest.raises(TypeError):
        await env.exec("echo hi", workdir="/tmp")
