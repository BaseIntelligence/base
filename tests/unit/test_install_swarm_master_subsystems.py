"""M7 deploy wiring: install-swarm.sh must provision the NEW master subsystems
(validator coordination plane, LLM gateway, HF checkpoint publisher) introduced
by the decentralization mission. Covers VAL-CICD-021 and supports VAL-CICD-022.

`base master proxy` now ALWAYS builds the LLM gateway and fails fast at startup
if the gateway token secret is missing, so the installer MUST provision a
MANDATORY ``gateway_token_secret`` docker secret (mounted at
``/run/secrets/gateway_token_secret``). In ``provider_mode=real`` it must also
provision the single yunwu provider key the gateway injects server-side (the
gateway is yunwu-only + provider-agnostic; deepseek/openrouter removed from code
AND config). It must render ``gateway.public_base_url`` (the external master
gateway root advertised to validators) so eval runtimes target the gateway and
NOT the ``master.registry_url`` (chain registry) fallback, and carry the
coordination-plane config into the base-master config.

These are BEHAVIORAL tests: they execute the real installer in DEFAULT dry-run
(mutates nothing) with a stub ``docker`` on PATH whose every ``inspect`` misses,
so every resource is *planned* (printed) regardless of what the live host
already has. They respect the imperative-Swarm contract (no compose YAML).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"

# Sentinel secret VALUES that must NEVER appear in plan/log output (dry-run only
# prints the env var NAME via plan_secret_stdin; values would reach stdin only).
GATEWAY_TOKEN_SENTINEL = "gtok-SENTINEL-must-not-leak"
YUNWU_SENTINEL = "yunwu-SENTINEL-must-not-leak"
HF_SENTINEL = "hf-SENTINEL-must-not-leak"
CENTRAL_GATEWAY_TOKEN_SENTINEL = "central-gtok-SENTINEL-must-not-leak"

# Deterministic gateway root for the consumer-URL assertions (avoids depending on
# the live default advertise address). The single source-driven gateway route is
# ``/llm/v1`` (yunwu-only; the old provider-path routes are removed).
GATEWAY_PUBLIC_BASE_URL = "http://master.example:19080"
GATEWAY_V1_ROUTE = f"{GATEWAY_PUBLIC_BASE_URL}/llm/v1"

REQUIRED_SECRET_ENV = {
    "GHCR_USER": "ci-user",
    "GHCR_TOKEN": "ci-token",
    "BASE_ADMIN_TOKEN": "x",
    "MASTER_DATABASE_URL": "postgresql+asyncpg://base@base-master-postgres:5432/base",
    "MASTER_PG_PASSWORD": "x",
    "AGENT_CHALLENGE_CHALLENGE_TOKEN": "x",
    "AGENT_CHALLENGE_DOCKER_BROKER_TOKEN": "x",
    "AGENT_CHALLENGE_SUBMISSION_ENV_KEY": "x",
    "AGENT_CHALLENGE_DATABASE_URL": "postgresql+asyncpg://challenge@h:5432/challenge",
    "AGENT_CHALLENGE_PG_PASSWORD": "x",
    "PRISM_CHALLENGE_TOKEN": "x",
    "PRISM_DOCKER_BROKER_TOKEN": "x",
    "PRISM_DATABASE_URL": "postgresql+asyncpg://challenge@h:5432/challenge",
    "PRISM_PG_PASSWORD": "x",
    "GATEWAY_TOKEN": GATEWAY_TOKEN_SENTINEL,
    "CENTRAL_GATEWAY_TOKEN": CENTRAL_GATEWAY_TOKEN_SENTINEL,
    "YUNWU_API_KEY": YUNWU_SENTINEL,
}


def _docker_stub(bin_dir: Path) -> None:
    """`docker` stub: recent engine, INACTIVE swarm, every `inspect` MISSES.

    A missed inspect makes the installer *plan* (print) the create, so the
    full plan is exercised even though the live host already has the services.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  version) echo '29.1.3' ;;\n"
        "  info) echo 'inactive' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *,
    provider_mode: str | None = None,
    hf_token: str | None = HF_SENTINEL,
    drop_central_gateway_token: bool = False,
    eval_ram_total_gb: str | None = "62",
    eval_task_concurrency: str | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir)
    env = dict(os.environ)
    env.update(REQUIRED_SECRET_ENV)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    env["GATEWAY_PUBLIC_BASE_URL"] = GATEWAY_PUBLIC_BASE_URL
    # Pin the RAM->concurrency inputs so the RAM-derived EVAL_TASK_CONCURRENCY is
    # deterministic regardless of the CI host's real memory (62 GiB, default
    # reserve 10, @4GB/task -> 13). A direct EVAL_TASK_CONCURRENCY override wins.
    if eval_ram_total_gb is not None:
        env["EVAL_RAM_TOTAL_GB"] = eval_ram_total_gb
    else:
        env.pop("EVAL_RAM_TOTAL_GB", None)
    if eval_task_concurrency is not None:
        env["EVAL_TASK_CONCURRENCY"] = eval_task_concurrency
    else:
        env.pop("EVAL_TASK_CONCURRENCY", None)
    if provider_mode is not None:
        env["GATEWAY_PROVIDER_MODE"] = provider_mode
    if hf_token is not None:
        env["HF_TOKEN"] = hf_token
    else:
        env.pop("HF_TOKEN", None)
    if drop_central_gateway_token:
        env.pop("CENTRAL_GATEWAY_TOKEN", None)
    return subprocess.run(
        [
            "bash",
            str(SWARM_INSTALLER),
            "--backup-dir",
            str(tmp_path / "missing"),
            "--greenfield",
            "--static-challenges",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _service_block(plan_lines: list[str], name: str) -> str:
    """Return the single planned `docker service create --name NAME ` line.

    Plan lines are shell-quoted via ``printf '%q'`` (commas escaped as ``\\,``),
    so backslashes are stripped to compare against the unescaped ``source=...,
    target=...`` secret specs.
    """
    needle = f"--name {name} "
    for line in plan_lines:
        if "docker service create" in line and needle in line:
            return line.replace("\\", "")
    raise AssertionError(f"no `docker service create --name {name}` line planned")


def test_mandatory_gateway_token_secret_created_and_mounted_on_proxy(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # The token-signing secret is provisioned (value via stdin, never argv).
    assert "docker secret create base_gateway_token_secret" in out

    # ...and mounted into the proxy at the exact path the gateway reads.
    proxy = _service_block(out.splitlines(), "base-master-proxy")
    assert "source=base_gateway_token_secret,target=gateway_token_secret" in proxy


def test_real_mode_provisions_provider_keys_on_the_gateway(tmp_path: Path) -> None:
    result = _run(tmp_path, provider_mode="real")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # The single yunwu provider key is created and mounted on the proxy at the
    # exact path the gateway reads (GatewaySettings.providers.yunwu.api_key_file);
    # deepseek/openrouter are gone (yunwu-only gateway).
    assert "docker secret create base_gateway_yunwu_api_key" in out
    assert "docker secret create base_gateway_deepseek_api_key" not in out
    assert "docker secret create base_openrouter_api_key" not in out

    proxy = _service_block(out.splitlines(), "base-master-proxy")
    assert "source=base_gateway_yunwu_api_key,target=yunwu_api_key" in proxy
    assert "target=deepseek_api_key" not in proxy
    assert "target=openrouter_api_key" not in proxy


def test_mock_mode_keeps_token_secret_but_omits_provider_keys(tmp_path: Path) -> None:
    result = _run(tmp_path, provider_mode="mock")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # Token secret is mandatory in every mode.
    assert "docker secret create base_gateway_token_secret" in out
    proxy = _service_block(out.splitlines(), "base-master-proxy")
    assert "source=base_gateway_token_secret,target=gateway_token_secret" in proxy

    # Mock provider needs no provider key: none created/mounted.
    assert "docker secret create base_gateway_yunwu_api_key" not in out
    assert "target=yunwu_api_key" not in proxy
    assert "provider_mode: mock" in out


def test_rendered_master_config_wires_gateway_and_coordination(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # LLM gateway block (architecture sec 5/sec 10; yunwu-only, config-driven).
    assert "provider_mode: real" in out
    assert "public_base_url:" in out
    assert "token_secret_file: /run/secrets/gateway_token_secret" in out
    # Provider registry + source map: the single yunwu provider (key injected from
    # the mounted secret) + default_model + per-source routes. No deepseek/openrouter.
    assert "api_key_file: /run/secrets/yunwu_api_key" in out
    assert "default_provider: yunwu" in out
    assert "default_model: claude-opus-4-8" in out
    assert "base_url: https://yunwu.ai/v1" in out
    assert "deepseek_api_key_file" not in out
    assert "openrouter_api_key_file" not in out

    # public_base_url must NOT be the chain registry fallback.
    assert "public_base_url: https://chain.joinbase.ai" not in out

    # Validator coordination plane (architecture sec 4).
    assert "validator_heartbeat_interval_seconds:" in out
    assert "validator_heartbeat_timeout_seconds:" in out
    assert "validator_health_interval_seconds:" in out
    assert "assignment_lease_seconds:" in out
    assert "orchestration_interval_seconds:" in out


def test_rendered_master_config_carries_global_broker_cap(tmp_path: Path) -> None:
    """The rendered master.yaml docker block sets the server-wide broker cap.

    ``docker.broker_max_concurrent_global`` bounds TOTAL concurrent broker eval
    jobs across ALL challenge slugs on the single manager node. It is RAM-derived
    at install time (@4GB/task); with the pinned 62 GiB / reserve-10 inputs the
    rendered value is 13. ``broker_log_limit_bytes`` raises the returned-log bound
    so challenges get effectively-full eval output.
    """
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    assert "broker_max_concurrent_global: 13" in out
    assert "broker_log_limit_bytes: 5000000" in out


def test_agent_challenge_services_carry_evaluation_concurrency(
    tmp_path: Path,
) -> None:
    """agent-challenge api + worker carry the RAM-derived eval concurrency.

    The durable eval-loop concurrency is RAM-derived at install time (@4GB/task);
    with the pinned 62 GiB / reserve-10 inputs it renders 13. It is applied to
    BOTH the api and the worker (via the static ``ac_eval_env`` array), mirroring
    the dynamic path's ``cli_app._agent_challenge_own_runner_env``. The same array
    carries the durable own-runner (DooD client) memory ceiling
    ``CHALLENGE_DOCKER_MEMORY=2g``.
    """
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    for service in ("challenge-agent-challenge", "challenge-agent-challenge-worker"):
        block = _service_block(lines, service)
        assert "CHALLENGE_EVALUATION_CONCURRENCY=13" in block
        assert "CHALLENGE_DOCKER_MEMORY=2g" in block


def test_eval_task_concurrency_is_ram_derived(tmp_path: Path) -> None:
    """EVAL_TASK_CONCURRENCY is RAM-derived at a 4 GB/task budget, override-able.

    With ``EVAL_RAM_TOTAL_GB=62`` and the default reserve (10 GiB):
    ``max(4, floor((62 - 10) / 4)) = 13`` is rendered into BOTH the master.yaml
    ``broker_max_concurrent_global`` and the ``ac_eval_env``
    ``CHALLENGE_EVALUATION_CONCURRENCY`` (api + worker). A direct
    ``EVAL_TASK_CONCURRENCY`` override wins over the RAM computation verbatim.
    """
    services = ("challenge-agent-challenge", "challenge-agent-challenge-worker")

    # RAM-derived: 62 GiB, reserve 10, @4GB/task -> 13.
    result = _run(tmp_path, eval_ram_total_gb="62")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert "broker_max_concurrent_global: 13" in out
    for service in services:
        block = _service_block(out.splitlines(), service)
        assert "CHALLENGE_EVALUATION_CONCURRENCY=13" in block

    # Direct override wins regardless of the RAM total.
    override = _run(tmp_path, eval_ram_total_gb="62", eval_task_concurrency="7")
    assert override.returncode == 0, f"stderr={override.stderr!r}"
    out2 = override.stdout
    assert "broker_max_concurrent_global: 7" in out2
    for service in services:
        block = _service_block(out2.splitlines(), service)
        assert "CHALLENGE_EVALUATION_CONCURRENCY=7" in block


def test_agent_challenge_services_carry_log_streaming_env(
    tmp_path: Path,
) -> None:
    """agent-challenge api + worker carry the live running-log streaming env.

    The worker points the terminal-bench runner's log producer at the api
    (CHALLENGE_TERMINAL_BENCH_LOG_STREAM_URL) and pins the runner JOB onto the
    isolated eval overlay (CHALLENGE_DOCKER_BROKER_NETWORK=base_jobs_internal) so
    it resolves challenge-agent-challenge by name to POST task.log events; the
    api ingests them and serves the SSE feed. Applied to BOTH services via the
    static ``ac_eval_env`` array, mirroring the dynamic path's
    ``cli_app._agent_challenge_own_runner_env``. Without these the runner lands
    on the default bridge network and log streaming silently no-ops.
    """
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    for service in ("challenge-agent-challenge", "challenge-agent-challenge-worker"):
        block = _service_block(lines, service)
        assert (
            "CHALLENGE_TERMINAL_BENCH_LOG_STREAM_URL="
            "http://challenge-agent-challenge:8000" in block
        )
        assert "CHALLENGE_DOCKER_BROKER_NETWORK=base_jobs_internal" in block


def test_hf_publisher_token_mounted_on_prism_when_present(tmp_path: Path) -> None:
    result = _run(tmp_path, hf_token=HF_SENTINEL)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    assert "docker secret create base_hf_token" in out
    prism = _service_block(out.splitlines(), "challenge-prism")
    assert "source=base_hf_token,target=hf_token" in prism


def test_hf_publisher_token_skipped_when_absent(tmp_path: Path) -> None:
    result = _run(tmp_path, hf_token=None)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # Optional secret: skipped (not created) and not mounted onto prism.
    assert "optional secret base_hf_token skipped" in out
    prism = _service_block(out.splitlines(), "challenge-prism")
    assert "base_hf_token" not in prism


def test_central_gate_token_secret_created(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    # Scoped central-gate token secret provisioned (value via stdin, never argv).
    # The trailing ``-`` disambiguates from the ``base_gateway_token_secret`` HMAC
    # secret.
    assert "docker secret create base_gateway_token -" in result.stdout


def test_central_gate_token_required_hard_fails_when_unset(tmp_path: Path) -> None:
    """The central-gate token is REQUIRED: an unset value hard-fails the installer.

    The master gateway is the sole LLM path for the central gates (no direct-key
    fallback), so ``_ensure_secret`` dies when ``CENTRAL_GATEWAY_TOKEN`` is unset.
    """
    result = _run(tmp_path, drop_central_gateway_token=True)
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "required secret env var $CENTRAL_GATEWAY_TOKEN is empty" in result.stderr
    assert "docker secret create base_gateway_token -" not in result.stdout


def test_central_gateway_routes_agent_challenge_consumer(tmp_path: Path) -> None:
    """agent-challenge api+worker get the gateway ROOT URL + scoped token mount.

    The analyzer appends ``/llm/v1`` to the base URL itself, so the installer
    renders the gateway ROOT — via the INTERNAL overlay service name
    (``http://base-master-proxy:${MASTER_PROXY_PORT}``), NOT the public IP: the
    agent-challenge eval JOB + analyzer run on the ``--internal``
    base_jobs_internal overlay (no egress) where the public IP is unreachable. The
    scoped token mounts at ``/run/secrets/base_gateway_token`` and NO direct
    provider key is rendered.
    """
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    for service in ("challenge-agent-challenge", "challenge-agent-challenge-worker"):
        block = _service_block(lines, service)
        assert "CHALLENGE_LLM_GATEWAY_BASE_URL=http://base-master-proxy:19080" in block
        token_file = "CHALLENGE_LLM_GATEWAY_TOKEN_FILE=/run/secrets/base_gateway_token"
        assert token_file in block
        assert "source=base_gateway_token,target=base_gateway_token" in block
        # No direct provider key on the challenge service: the gateway is the sole
        # LLM path.
        assert "target=openrouter_api_key" not in block
        assert "CHALLENGE_OPENROUTER_API_KEY_FILE" not in block


def test_central_gateway_routes_prism_consumer(tmp_path: Path) -> None:
    """prism api+worker get the full ``/llm/v1`` route + scoped token mount.

    prism uses ``PRISM_LLM_GATEWAY_URL`` directly as the chat base_url, so the
    installer renders the FULL single source-driven gateway route +
    ``BASE_GATEWAY_TOKEN_FILE``. The scoped token mounts at
    ``/run/secrets/base_gateway_token``, the LLM-review max tokens are raised to
    4096, and NO direct provider key is rendered (the gateway injects it).
    """
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    for service in ("challenge-prism", "challenge-prism-worker"):
        block = _service_block(lines, service)
        assert f"PRISM_LLM_GATEWAY_URL={GATEWAY_V1_ROUTE}" in block
        assert "BASE_GATEWAY_TOKEN_FILE=/run/secrets/base_gateway_token" in block
        assert "PRISM_LLM_REVIEW_ENABLED=true" in block
        assert "PRISM_LLM_REVIEW_MAX_TOKENS=4096" in block
        assert "source=base_gateway_token,target=base_gateway_token" in block
        # No direct provider key on the challenge service: the gateway is the sole
        # LLM path.
        assert "target=openrouter_api_key" not in block
        assert "target=yunwu_api_key" not in block


def test_master_and_challenge_services_carry_update_rollback_and_health_policy(
    tmp_path: Path,
) -> None:
    """The master services + the challenge API are created with the Swarm
    self-healing update/rollback policy and an HTTP /health container healthcheck
    (belt-and-suspenders auto-rollback for a broken image roll)."""
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    policy = (
        "--update-failure-action rollback",
        "--update-monitor 45s",
        "--update-max-failure-ratio 0",
        "--update-order stop-first",
        "--rollback-failure-action pause",
        "--rollback-monitor 45s",
        "--rollback-max-failure-ratio 1",
    )
    for service in (
        "base-docker-broker",
        "base-master-proxy",
        "challenge-agent-challenge",
    ):
        block = _service_block(lines, service)
        for flag in policy:
            assert flag in block, f"{service} missing {flag!r}"

    # Health flags on the HTTP services (proxy + broker + challenge api), on port.
    for service, port in (
        ("base-master-proxy", "19080"),
        ("base-docker-broker", "8082"),
        ("challenge-agent-challenge", "8000"),
    ):
        block = _service_block(lines, service)
        assert "--health-cmd" in block, service
        assert f"localhost:{port}/health" in block, service
        assert "--health-start-period 40s" in block, service


def test_secret_values_never_leak_in_plan_output(tmp_path: Path) -> None:
    result = _run(tmp_path)
    combined = result.stdout + result.stderr
    assert GATEWAY_TOKEN_SENTINEL not in combined
    assert YUNWU_SENTINEL not in combined
    assert HF_SENTINEL not in combined
    assert CENTRAL_GATEWAY_TOKEN_SENTINEL not in combined
