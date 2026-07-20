"""Network/secrets isolation parity + in-container probe (own-runner, Task 18).

This module reproduces -- and *proves*, inside a real container -- harbor
0.13.1's two isolation guarantees:

#. **Network posture.** Harbor maps a task's ``allow_internet`` flag to a docker
   network mode: ``True`` -> ``NetworkMode.PUBLIC`` (default bridge, egress
   allowed), ``False`` / unset -> ``NetworkMode.NO_NETWORK`` (``network_mode:
   none``, egress blocked). The own-runner reproduces this through
   :func:`agent_challenge.evaluation.own_runner.container_builder.network_arg`;
   :func:`docker_network_arg` here is the byte-identical mirror used by the probe
   so the two never drift.
#. **Secret posture.** The only variables handed to the agent are the master
   LLM gateway configuration injected through ``context.env``:
   ``BASE_LLM_GATEWAY_URL``, ``BASE_GATEWAY_TOKEN`` and ``LLM_COST_LIMIT`` (the
   variables baseagent reads). No provider key or model name is forwarded (the
   gateway injects both), no host secrets are inherited (a ``docker exec`` does
   not inherit the daemon/host environment), and no additional egress is opened.

The probe is transport-agnostic: it talks to anything implementing the
:class:`ExecEnvironment` protocol (the real
:class:`~agent_challenge.evaluation.own_runner.exec_bridge.DockerExecEnvironment`
in integration, a fake in unit tests). It launches two commands inside the
container -- an egress check and an environment dump -- and folds the results
into an :class:`IsolationReport` whose :attr:`~IsolationReport.parity_ok` flag
encodes whether the live container matches harbor's expected posture.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Protocol

from agent_challenge.evaluation.own_runner.container_builder import network_arg
from agent_challenge.evaluation.own_runner.taskdefs import ResourceLimits

__all__ = [
    "AGENT_ENV_ALLOWLIST",
    "HARNESS_CONTROL_ENV_KEYS",
    "EgressProbeResult",
    "EnvProbeResult",
    "ExecEnvironment",
    "IsolationParityError",
    "IsolationReport",
    "assert_isolation_parity",
    "disallowed_secret_keys",
    "docker_network_arg",
    "egress_should_be_blocked",
    "filter_agent_env",
    "harbor_network_mode",
    "looks_like_secret",
    "probe_egress",
    "probe_env",
    "run_isolation_probe",
]

# --------------------------------------------------------------------------- #
# allowlists
# --------------------------------------------------------------------------- #
#: Variables that may be forwarded to the agent via ``context.env``.
#: VAL-ACAT-013/014: Base LLM gateway (``BASE_LLM_GATEWAY_URL`` /
#: ``BASE_GATEWAY_TOKEN``) is **not** on this list. When agents may call models,
#: only measured OpenRouter material (miner encrypted_env inside eval CVM) or
#: tools-only (no LLM keys) is legal. Cost-limit remains for budget accounting.
AGENT_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "LLM_COST_LIMIT",
        "OPENROUTER_API_KEY",
    }
)

#: Non-secret harness/control variables the runner is allowed to set on the
#: container (paths and cache locations). These are *not* secrets and must never
#: overlap the agent env allowlist.
HARNESS_CONTROL_ENV_KEYS: frozenset[str] = frozenset(
    {
        "BASE_AGENT_PATH",
        "BASE_BENCHMARK_DATASET",
        "HOME",
        "XDG_CACHE_HOME",
    }
)

#: Substrings whose presence in an env-var *name* marks it as secret-shaped.
#: ``BASE_GATEWAY_TOKEN`` is caught by the ``TOKEN`` marker and is redacted.
_SECRET_MARKERS: tuple[str, ...] = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)

#: Baseline ``python:3.12-slim`` env-var names that look key-ish but are NOT
#: secrets and must never be flagged (avoids false positives).
_NON_SECRET_ALLOWLIST: frozenset[str] = frozenset(
    {
        "GPG_KEY",
        "PYTHON_SHA256",
        "PYTHON_GET_PIP_SHA256",
    }
)


def looks_like_secret(name: str) -> bool:
    """Return ``True`` when ``name`` is shaped like a secret env-var name.

    Detection is by name only (values are never inspected). Baseline image vars
    such as ``GPG_KEY`` / ``PYTHON_SHA256`` are explicitly excluded.
    """
    upper = name.upper()
    if upper in _NON_SECRET_ALLOWLIST:
        return False
    return any(marker in upper for marker in _SECRET_MARKERS)


def filter_agent_env(agent_env: dict[str, str]) -> dict[str, str]:
    """Keep only the allowlisted master gateway variables from ``agent_env``."""
    return {k: v for k, v in agent_env.items() if k in AGENT_ENV_ALLOWLIST}


def disallowed_secret_keys(env: dict[str, str]) -> set[str]:
    """Return secret-shaped keys in ``env`` that are not on the allowlist."""
    return {key for key in env if looks_like_secret(key) and key not in AGENT_ENV_ALLOWLIST}


# --------------------------------------------------------------------------- #
# network posture parity
# --------------------------------------------------------------------------- #
def harbor_network_mode(allow_internet: bool | None) -> str:
    """Map ``allow_internet`` to harbor's network-mode label.

    ``True`` -> ``"public"`` (default bridge); ``False`` / ``None`` ->
    ``"no-network"`` (``network_mode: none``).
    """
    return "public" if allow_internet else "no-network"


def docker_network_arg(allow_internet: bool | None) -> str | None:
    """Mirror of ``container_builder.network_arg`` keyed on ``allow_internet``.

    Returns ``"none"`` when internet is disallowed and ``None`` (docker default
    bridge) when allowed -- delegating to the single source of truth so the
    probe's posture can never drift from the builder's.
    """
    return network_arg(ResourceLimits(allow_internet=allow_internet))


def egress_should_be_blocked(allow_internet: bool | None) -> bool:
    """Return ``True`` when the container's outbound egress must be blocked."""
    return not allow_internet


# --------------------------------------------------------------------------- #
# exec transport protocol
# --------------------------------------------------------------------------- #
class _ExecResultLike(Protocol):
    stdout: str | None
    return_code: int


class ExecEnvironment(Protocol):
    """Structural type for anything that can ``exec`` a command in a container."""

    async def exec(
        self,
        command: str,
        cwd: str | None = ...,
        env: dict[str, str] | None = ...,
        timeout_sec: int | None = ...,
        user: str | int | None = ...,
    ) -> _ExecResultLike: ...


# --------------------------------------------------------------------------- #
# probe scripts (passed via base64 to dodge bash quoting)
# --------------------------------------------------------------------------- #
# Attempts a raw-IP TCP connect (no DNS) AND a DNS resolution; prints a single
# marker line. Either success => egress reachable.
_EGRESS_SCRIPT = r"""
import socket
reached = False
for target in (("1.1.1.1", 443), ("8.8.8.8", 53)):
    try:
        s = socket.create_connection(target, timeout=4)
        s.close()
        reached = True
        break
    except OSError:
        pass
if not reached:
    try:
        socket.getaddrinfo("example.com", 443)
        reached = True
    except OSError:
        pass
print("EGRESS_OK" if reached else "EGRESS_BLOCKED")
"""


def _egress_command() -> str:
    encoded = base64.b64encode(_EGRESS_SCRIPT.encode()).decode()
    # Leading marker comment keeps the command identifiable to fakes; the script
    # itself is decoded and piped to python3 inside the container.
    return f"# EGRESS_PROBE\necho {encoded} | base64 -d | python3 -"


def _env_command() -> str:
    # ``env -0`` would be NUL-delimited; plain ``env`` is line-delimited and
    # sufficient since we only read key names.
    return "env"


# --------------------------------------------------------------------------- #
# probe results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EgressProbeResult:
    """Outcome of the in-container outbound-egress check."""

    reached: bool
    blocked: bool
    raw_output: str


@dataclass(frozen=True)
class EnvProbeResult:
    """Outcome of the in-container environment inspection."""

    injected_present: set[str]
    leaked_secrets: set[str]
    observed_keys: set[str]
    raw_output: str

    @property
    def clean(self) -> bool:
        """``True`` when no disallowed secrets are present in the container."""
        return not self.leaked_secrets


@dataclass(frozen=True)
class IsolationReport:
    """Combined network + secret isolation posture of a live container."""

    allow_internet: bool | None
    expected_network_mode: str
    expected_egress_blocked: bool
    egress: EgressProbeResult
    env: EnvProbeResult
    reasons: list[str] = field(default_factory=list)

    @property
    def parity_ok(self) -> bool:
        """``True`` when the live container matches harbor's expected posture."""
        return not self.reasons


class IsolationParityError(RuntimeError):
    """Raised when a container's live isolation posture diverges from harbor."""


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _parse_env_keys(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            parsed[key] = value
    return parsed


async def probe_egress(env: ExecEnvironment) -> EgressProbeResult:
    """Run the egress check inside ``env`` and classify the result."""
    result = await env.exec(_egress_command(), timeout_sec=30)
    output = result.stdout or ""
    reached = "EGRESS_OK" in output
    blocked = ("EGRESS_BLOCKED" in output) and not reached
    return EgressProbeResult(reached=reached, blocked=blocked, raw_output=output)


async def probe_env(env: ExecEnvironment, *, agent_env: dict[str, str]) -> EnvProbeResult:
    """Inject the filtered allowlist into ``env`` and inspect the result.

    Only allowlisted master gateway vars are injected; the container's full
    environment is then dumped to detect any disallowed secret leakage.
    """
    injected = filter_agent_env(agent_env)
    result = await env.exec(_env_command(), env=injected, timeout_sec=30)
    output = result.stdout or ""
    observed = _parse_env_keys(output)
    injected_present = {k for k in injected if k in observed}
    leaked = disallowed_secret_keys(observed)
    return EnvProbeResult(
        injected_present=injected_present,
        leaked_secrets=leaked,
        observed_keys=set(observed),
        raw_output=output,
    )


async def run_isolation_probe(
    env: ExecEnvironment,
    *,
    allow_internet: bool | None,
    agent_env: dict[str, str],
) -> IsolationReport:
    """Probe ``env`` and build an :class:`IsolationReport` vs harbor's posture."""
    expected_mode = harbor_network_mode(allow_internet)
    must_block = egress_should_be_blocked(allow_internet)

    egress = await probe_egress(env)
    env_result = await probe_env(env, agent_env=agent_env)

    reasons: list[str] = []
    if must_block:
        if not egress.blocked:
            reasons.append(
                "expected no-network egress to be BLOCKED but the container "
                f"reached the internet (output={egress.raw_output!r})"
            )
    else:
        if not egress.reached:
            reasons.append(
                "expected public network egress to be REACHABLE but the "
                f"container could not reach the internet (output={egress.raw_output!r})"
            )
    if env_result.leaked_secrets:
        reasons.append(
            "disallowed secret-shaped env vars present in container: "
            f"{sorted(env_result.leaked_secrets)}"
        )

    return IsolationReport(
        allow_internet=allow_internet,
        expected_network_mode=expected_mode,
        expected_egress_blocked=must_block,
        egress=egress,
        env=env_result,
        reasons=reasons,
    )


def assert_isolation_parity(report: IsolationReport) -> None:
    """Raise :class:`IsolationParityError` when ``report`` is not parity-clean."""
    if not report.parity_ok:
        raise IsolationParityError(
            "container isolation posture diverged from harbor: " + "; ".join(report.reasons)
        )
