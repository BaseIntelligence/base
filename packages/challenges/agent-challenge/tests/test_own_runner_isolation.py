"""Tests for the own-runner network/secrets isolation parity probe (Task 18).

These tests pin the own-runner's reproduction of harbor 0.13.1's network
isolation posture and secret-injection posture:

* **Network** — harbor maps a task's ``allow_internet`` flag to a network mode:
  ``True`` -> ``NetworkMode.PUBLIC`` (default bridge, egress allowed),
  ``False`` / unset -> ``NetworkMode.NO_NETWORK`` (``network_mode: none``,
  egress blocked). The own-runner reproduces this via
  ``container_builder.network_arg`` (``"none"`` when internet is disallowed,
  ``None`` -- default bridge -- when allowed).
* **Secrets** — Base LLM gateway vars are **not** forwarded. Agent env
  allowlist is ``LLM_COST_LIMIT`` (+ measured ``OPENROUTER_API_KEY`` when
  product permits). No Base gateway token, no non-measured provider embeds.

The pure-logic + fake-environment tests run everywhere. The single real-docker
integration test launches a throwaway ``python:3.12-slim`` container with
``network=none`` and proves -- inside a real container -- that disallowed egress
is blocked and that only allowlisted secrets are present. It skips when docker /
the image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from agent_challenge.evaluation.own_runner.container_builder import network_arg
from agent_challenge.evaluation.own_runner.isolation import (
    AGENT_ENV_ALLOWLIST,
    HARNESS_CONTROL_ENV_KEYS,
    EgressProbeResult,
    EnvProbeResult,
    IsolationParityError,
    IsolationReport,
    assert_isolation_parity,
    disallowed_secret_keys,
    docker_network_arg,
    egress_should_be_blocked,
    filter_agent_env,
    harbor_network_mode,
    looks_like_secret,
    probe_egress,
    probe_env,
    run_isolation_probe,
)
from agent_challenge.evaluation.own_runner.taskdefs import ResourceLimits

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


# --------------------------------------------------------------------------- #
# fake environment (records exec calls, returns canned ExecResult-likes)
# --------------------------------------------------------------------------- #
class _FakeExecResult:
    def __init__(self, stdout: str | None, return_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = None
        self.return_code = return_code


class _FakeEnv:
    """Records ``exec`` calls and replays canned output per command substring."""

    def __init__(self, *, egress_stdout: str, env_stdout: str) -> None:
        self.egress_stdout = egress_stdout
        self.env_stdout = env_stdout
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> _FakeExecResult:
        self.calls.append((command, env))
        if "socket" in command or "EGRESS" in command:
            return _FakeExecResult(self.egress_stdout)
        # env-enumeration command: echo the canned env plus whatever was injected.
        injected = "".join(f"{k}={v}\n" for k, v in (env or {}).items())
        return _FakeExecResult(self.env_stdout + injected)


# --------------------------------------------------------------------------- #
# allowlist (the context.env master-gateway-only contract)
# --------------------------------------------------------------------------- #
def test_allowlist_excludes_base_gateway_vars() -> None:
    """VAL-ACAT-013: agent env allowlist must not include Base gateway secrets."""

    assert "BASE_LLM_GATEWAY_URL" not in AGENT_ENV_ALLOWLIST
    assert "BASE_GATEWAY_TOKEN" not in AGENT_ENV_ALLOWLIST
    assert "LLM_COST_LIMIT" in AGENT_ENV_ALLOWLIST
    assert AGENT_ENV_ALLOWLIST == frozenset({"LLM_COST_LIMIT", "OPENROUTER_API_KEY"})


def test_harness_control_keys_are_not_secrets() -> None:
    # Harness control vars (HOME, XDG_CACHE_HOME, BASE_*) are allowed but are
    # NOT secrets and must be disjoint from the agent env allowlist.
    assert HARNESS_CONTROL_ENV_KEYS.isdisjoint(AGENT_ENV_ALLOWLIST)
    assert "BASE_AGENT_PATH" in HARNESS_CONTROL_ENV_KEYS
    assert "HOME" in HARNESS_CONTROL_ENV_KEYS


# --------------------------------------------------------------------------- #
# network posture parity (harbor allow_internet -> NetworkMode)
# --------------------------------------------------------------------------- #
def test_harbor_network_mode_mapping() -> None:
    assert harbor_network_mode(True) == "public"
    assert harbor_network_mode(False) == "no-network"
    assert harbor_network_mode(None) == "no-network"


def test_docker_network_arg_matches_container_builder() -> None:
    # docker_network_arg must agree with container_builder.network_arg for the
    # same allow_internet flag (single source of truth for the network posture).
    assert docker_network_arg(False) == "none"
    assert docker_network_arg(True) is None
    assert docker_network_arg(None) == "none"
    assert docker_network_arg(False) == network_arg(ResourceLimits(allow_internet=False))
    assert docker_network_arg(True) == network_arg(ResourceLimits(allow_internet=True))


def test_egress_should_be_blocked() -> None:
    assert egress_should_be_blocked(False) is True
    assert egress_should_be_blocked(None) is True
    assert egress_should_be_blocked(True) is False


# --------------------------------------------------------------------------- #
# secret-shape detection + filtering
# --------------------------------------------------------------------------- #
def test_looks_like_secret_flags_credentials_not_baseline() -> None:
    assert looks_like_secret("BASE_GATEWAY_TOKEN")
    assert looks_like_secret("AWS_SECRET_ACCESS_KEY")
    assert looks_like_secret("GITHUB_TOKEN")
    assert looks_like_secret("DB_PASSWORD")
    # Non-secret config vars must NOT be flagged as secrets.
    assert not looks_like_secret("LLM_COST_LIMIT")
    assert not looks_like_secret("BASE_LLM_GATEWAY_URL")
    # baseline image vars must NOT be flagged as secrets.
    assert not looks_like_secret("GPG_KEY")
    assert not looks_like_secret("PYTHON_SHA256")
    assert not looks_like_secret("PATH")
    assert not looks_like_secret("HOME")


def test_filter_agent_env_keeps_only_allowlist() -> None:
    raw = {
        "BASE_LLM_GATEWAY_URL": "https://master-gateway.test/llm/v1",
        "BASE_GATEWAY_TOKEN": "scoped-token",
        "LLM_COST_LIMIT": "5",
        "OPENROUTER_API_KEY": "or-key",
        "AWS_SECRET_ACCESS_KEY": "leak",
        "PATH": "/usr/bin",
    }
    assert filter_agent_env(raw) == {
        "LLM_COST_LIMIT": "5",
        "OPENROUTER_API_KEY": "or-key",
    }


def test_disallowed_secret_keys_flags_extra_secrets() -> None:
    env = {
        "LLM_COST_LIMIT": "5",
        "PATH": "/usr/bin",
        "GPG_KEY": "abc",
        "AWS_SECRET_ACCESS_KEY": "leak",
        "GITHUB_TOKEN": "leak2",
        "BASE_GATEWAY_TOKEN": "residual-forbidden",
    }
    # Residual Base gateway tokens are secret-shaped and not allowlisted.
    assert disallowed_secret_keys(env) == {
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "BASE_GATEWAY_TOKEN",
    }


def test_disallowed_secret_keys_clean_when_only_allowlist() -> None:
    env = {
        "LLM_COST_LIMIT": "5",
        "OPENROUTER_API_KEY": "or-key",
        "PATH": "/usr/bin",
        "GPG_KEY": "abc",
        "HOME": "/tmp",
    }
    assert disallowed_secret_keys(env) == set()


# --------------------------------------------------------------------------- #
# egress probe (fake env)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_probe_egress_blocked() -> None:
    fake = _FakeEnv(
        egress_stdout="EGRESS_BLOCKED:OSError:[Errno 101] Network is unreachable\n",
        env_stdout="",
    )
    result = await probe_egress(fake)
    assert isinstance(result, EgressProbeResult)
    assert result.blocked is True
    assert result.reached is False


@pytest.mark.asyncio
async def test_probe_egress_allowed() -> None:
    fake = _FakeEnv(egress_stdout="EGRESS_OK\n", env_stdout="")
    result = await probe_egress(fake)
    assert result.blocked is False
    assert result.reached is True


# --------------------------------------------------------------------------- #
# env probe (fake env)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_probe_env_clean_only_allowlist() -> None:
    fake = _FakeEnv(
        egress_stdout="",
        env_stdout="PATH=/usr/bin\nHOME=/root\nGPG_KEY=abc\nPYTHON_SHA256=def\n",
    )
    agent_env = {"LLM_COST_LIMIT": "5", "OPENROUTER_API_KEY": "or-key"}
    result = await probe_env(fake, agent_env=agent_env)
    assert isinstance(result, EnvProbeResult)
    assert result.leaked_secrets == set()
    assert result.injected_present == {"LLM_COST_LIMIT", "OPENROUTER_API_KEY"}
    assert result.clean is True


@pytest.mark.asyncio
async def test_probe_env_detects_host_secret_leak() -> None:
    fake = _FakeEnv(
        egress_stdout="",
        env_stdout="PATH=/usr/bin\nAWS_SECRET_ACCESS_KEY=leak\n",
    )
    result = await probe_env(fake, agent_env={"LLM_COST_LIMIT": "5"})
    assert result.leaked_secrets == {"AWS_SECRET_ACCESS_KEY"}
    assert result.clean is False


@pytest.mark.asyncio
async def test_probe_env_injects_only_filtered_allowlist() -> None:
    # Even when handed a dirty env, the probe injects only allowlisted vars.
    fake = _FakeEnv(egress_stdout="", env_stdout="PATH=/usr/bin\n")
    await probe_env(
        fake,
        agent_env={
            "LLM_COST_LIMIT": "5",
            "BASE_GATEWAY_TOKEN": "tok",
            "AWS_SECRET_ACCESS_KEY": "leak",
        },
    )
    _, injected = fake.calls[-1]
    assert injected == {"LLM_COST_LIMIT": "5"}


# --------------------------------------------------------------------------- #
# combined parity probe + assertion (fake env)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_isolation_probe_parity_ok_no_network() -> None:
    fake = _FakeEnv(
        egress_stdout="EGRESS_BLOCKED:OSError:[Errno 101] Network is unreachable\n",
        env_stdout="PATH=/usr/bin\nGPG_KEY=abc\n",
    )
    report = await run_isolation_probe(
        fake,
        allow_internet=False,
        agent_env={"LLM_COST_LIMIT": "5"},
    )
    assert isinstance(report, IsolationReport)
    assert report.parity_ok is True
    assert report.egress.blocked is True
    assert report.env.clean is True
    assert report.expected_network_mode == "no-network"


@pytest.mark.asyncio
async def test_assert_isolation_parity_raises_on_egress_mismatch() -> None:
    # no-network expected, but egress reached -> parity failure.
    fake = _FakeEnv(egress_stdout="EGRESS_OK\n", env_stdout="PATH=/usr/bin\n")
    report = await run_isolation_probe(
        fake, allow_internet=False, agent_env={"LLM_COST_LIMIT": "5"}
    )
    assert report.parity_ok is False
    with pytest.raises(IsolationParityError):
        assert_isolation_parity(report)


@pytest.mark.asyncio
async def test_assert_isolation_parity_raises_on_secret_leak() -> None:
    fake = _FakeEnv(
        egress_stdout="EGRESS_BLOCKED:OSError\n",
        env_stdout="PATH=/usr/bin\nGITHUB_TOKEN=leak\n",
    )
    report = await run_isolation_probe(
        fake, allow_internet=False, agent_env={"LLM_COST_LIMIT": "5"}
    )
    assert report.parity_ok is False
    with pytest.raises(IsolationParityError):
        assert_isolation_parity(report)


@pytest.mark.asyncio
async def test_assert_isolation_parity_passes_clean() -> None:
    fake = _FakeEnv(
        egress_stdout="EGRESS_BLOCKED:OSError\n",
        env_stdout="PATH=/usr/bin\n",
    )
    report = await run_isolation_probe(
        fake, allow_internet=False, agent_env={"LLM_COST_LIMIT": "5"}
    )
    assert_isolation_parity(report)  # must not raise


# --------------------------------------------------------------------------- #
# REAL in-container probe — the hard evidence (skips without docker)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _docker_ready(),
    reason=f"docker + {_IMAGE} image required for the in-container isolation probe",
)
@pytest.mark.asyncio
async def test_real_container_no_network_blocks_egress_and_env_allowlist() -> None:
    from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment

    # Launch with the EXACT no-network posture harbor uses for a no-internet task
    # (allow_internet=False -> network none).
    env = DockerExecEnvironment.launch(_IMAGE, network=docker_network_arg(False))
    try:
        report = await run_isolation_probe(
            env,
            allow_internet=False,
            agent_env={
                "LLM_COST_LIMIT": "5",
            },
        )
        # Disallowed egress is blocked identically to harbor's no-network mode.
        assert report.egress.blocked is True
        assert report.egress.reached is False
        # Only allowlisted vars are present; Base gateway secrets absent.
        assert report.env.leaked_secrets == set()
        assert report.env.injected_present == {
            "LLM_COST_LIMIT",
        }
        assert report.parity_ok is True
        assert_isolation_parity(report)
    finally:
        env.remove()
