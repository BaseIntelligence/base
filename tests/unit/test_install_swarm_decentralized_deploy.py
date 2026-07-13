"""G3 decentralized no-chain deploy support for ``install-swarm.sh``.

Encodes VAL-CODE-DEPLOY-001..005: the installer must (1) honor ``IMAGE_*``
overrides so a dry-run renders the provided ``:sha-*`` images, (2) render the
mock-metagraph validator set + coordination/driver intervals + LLM gateway
provider config into the master config, (3) manager-pin the proxy (no hard
``node.role==worker`` pin) with configurable published ports for the 2-node
no-chain deploy, (4) ship a ``validator.yaml`` template + a documented N-validator
run path, and (5) keep dry-run the DEFAULT, deterministic/idempotent, and
``bash -n`` clean.

These are BEHAVIORAL tests: they run the real installer in DEFAULT dry-run
(mutates nothing) with a stub ``docker`` whose every ``inspect`` misses, so every
resource is *planned* (printed) regardless of host state. No compose YAML.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from base.config.loader import load_settings

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"
VALIDATOR_TEMPLATE = ROOT / "deploy" / "swarm" / "validator.yaml"
SWARM_README = ROOT / "deploy" / "swarm" / "README.md"

# Sentinel image tags proving IMAGE_* overrides flow into the dry-run plan.
IMAGE_MASTER_SHA = "ghcr.io/baseintelligence/base-master:sha-DEPLOY01M"
IMAGE_AGENT_CHALLENGE_SHA = "ghcr.io/baseintelligence/agent-challenge:sha-DEPLOY01A"
IMAGE_PRISM_SHA = "ghcr.io/baseintelligence/prism:sha-DEPLOY01P"
IMAGE_PRISM_EVALUATOR_SHA = "ghcr.io/baseintelligence/prism-evaluator:sha-DEPLOY01E"

# Distinctive mock-metagraph validator hotkeys (not in the upload allowlist).
MMG_VAL_HOTKEY_1 = "5ValMMGdeploy0000000000000000000000000000000001"
MMG_VAL_HOTKEY_2 = "5ValMMGdeploy0000000000000000000000000000000002"
MOCK_METAGRAPH_JSON = (
    f'[{{"hotkey":"{MMG_VAL_HOTKEY_1}","validator_permit":true,"stake":1000}},'
    f'{{"hotkey":"{MMG_VAL_HOTKEY_2}","validator_permit":true,"stake":1000}}]'
)

GATEWAY_PUBLIC_BASE_URL = "http://master.example:19080"

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
    "GATEWAY_TOKEN": "x",
    "CENTRAL_GATEWAY_TOKEN": "x",
    "YUNWU_API_KEY": "x",
}


def _docker_stub(bin_dir: Path) -> None:
    """`docker` stub: recent engine, INACTIVE swarm, every `inspect` MISSES."""
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
    extra_env: dict[str, str] | None = None,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir)
    env = dict(os.environ)
    env.update(REQUIRED_SECRET_ENV)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    env["GATEWAY_PUBLIC_BASE_URL"] = GATEWAY_PUBLIC_BASE_URL
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(SWARM_INSTALLER),
            "--backup-dir",
            str(tmp_path / "missing"),
            "--greenfield",
            "--static-challenges",
            *extra_args,
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
    so backslashes are stripped to compare against unescaped specs.
    """
    needle = f"--name {name} "
    for line in plan_lines:
        if "docker service create" in line and needle in line:
            return line.replace("\\", "")
    raise AssertionError(f"no `docker service create --name {name}` line planned")


# ---------------------------------------------------------------------------
# VAL-CODE-DEPLOY-001: IMAGE_* overrides flow into the dry-run plan
# ---------------------------------------------------------------------------


def test_image_overrides_flow_into_dry_run_plan(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        extra_env={
            "IMAGE_MASTER": IMAGE_MASTER_SHA,
            "IMAGE_AGENT_CHALLENGE": IMAGE_AGENT_CHALLENGE_SHA,
            "IMAGE_PRISM": IMAGE_PRISM_SHA,
            "IMAGE_PRISM_EVALUATOR": IMAGE_PRISM_EVALUATOR_SHA,
        },
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    # base-master image used by BOTH the broker + proxy services.
    for service in ("base-docker-broker", "base-master-proxy"):
        block = _service_block(lines, service)
        assert IMAGE_MASTER_SHA in block

    # agent-challenge image on api + worker.
    for service in ("challenge-agent-challenge", "challenge-agent-challenge-worker"):
        assert IMAGE_AGENT_CHALLENGE_SHA in _service_block(lines, service)

    # prism image on api + worker; the evaluator image is the eval job image.
    for service in ("challenge-prism", "challenge-prism-worker"):
        block = _service_block(lines, service)
        assert IMAGE_PRISM_SHA in block
        assert f"PRISM_BASE_EVAL_IMAGE={IMAGE_PRISM_EVALUATOR_SHA}" in block


# ---------------------------------------------------------------------------
# VAL-CODE-DEPLOY-002: mock-metagraph + coordination + gateway rendered
# ---------------------------------------------------------------------------


def test_mock_metagraph_validator_set_rendered_into_master_config(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, extra_env={"MOCK_METAGRAPH": MOCK_METAGRAPH_JSON})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # network.mock_metagraph carries both permitted validator hotkeys.
    assert "mock_metagraph:" in out
    assert MMG_VAL_HOTKEY_1 in out
    assert MMG_VAL_HOTKEY_2 in out
    assert "validator_permit" in out


def test_default_mock_metagraph_is_empty_and_off(tmp_path: Path) -> None:
    # Unset MOCK_METAGRAPH => the seam is rendered OFF (empty list), so the
    # live-metagraph path is unchanged (production-safe default).
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "mock_metagraph: []" in result.stdout


def test_coordination_intervals_and_gateway_config_rendered(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # Coordination / driver intervals (architecture sec 4).
    assert "validator_heartbeat_interval_seconds:" in out
    assert "validator_heartbeat_timeout_seconds:" in out
    assert "validator_health_interval_seconds:" in out
    assert "assignment_lease_seconds:" in out
    assert "orchestration_interval_seconds:" in out

    # LLM gateway provider config (architecture sec 5/sec 10; yunwu-only,
    # config-driven provider registry + source map; deepseek/openrouter removed).
    assert "provider_mode: real" in out
    assert "public_base_url:" in out
    assert "token_secret_file: /run/secrets/gateway_token_secret" in out
    assert "api_key_file: /run/secrets/yunwu_api_key" in out
    assert "default_provider: yunwu" in out
    assert "default_model: claude-opus-4-8" in out
    assert "deepseek_api_key_file" not in out
    assert "openrouter_api_key_file" not in out


# ---------------------------------------------------------------------------
# VAL-CODE-DEPLOY-003: manager-pinned proxy + configurable ports
# ---------------------------------------------------------------------------


def test_proxy_is_manager_pinned_with_no_worker_pin(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    proxy = _service_block(lines, "base-master-proxy")
    assert "--constraint node.role==manager" in proxy
    # The old hard worker pin must be gone (no-chain deploy: no chain to reach).
    assert "node.role==worker" not in proxy

    # The broker stays manager-pinned too (control plane co-located on manager).
    broker = _service_block(lines, "base-docker-broker")
    assert "--constraint node.role==manager" in broker


def test_proxy_constraint_is_configurable_and_droppable(tmp_path: Path) -> None:
    # Empty MASTER_PROXY_CONSTRAINT drops the pin entirely.
    dropped = _run(tmp_path, extra_env={"MASTER_PROXY_CONSTRAINT": ""})
    assert dropped.returncode == 0, f"stderr={dropped.stderr!r}"
    proxy = _service_block(dropped.stdout.splitlines(), "base-master-proxy")
    assert "node.role==" not in proxy

    # A custom constraint is honored verbatim.
    custom = _run(
        tmp_path,
        extra_env={"MASTER_PROXY_CONSTRAINT": "node.labels.base.control==true"},
    )
    assert custom.returncode == 0, f"stderr={custom.stderr!r}"
    proxy = _service_block(custom.stdout.splitlines(), "base-master-proxy")
    assert "--constraint node.labels.base.control==true" in proxy


def test_published_ports_are_configurable(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        extra_env={"MASTER_PROXY_PORT": "28080", "MASTER_BROKER_PORT": "28082"},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    lines = out.splitlines()

    proxy = _service_block(lines, "base-master-proxy")
    assert "published=28080,target=28080,mode=host" in proxy

    broker = _service_block(lines, "base-docker-broker")
    assert "published=28082,target=28082,mode=host" in broker

    # Ports also flow into the rendered master config (proxy_port + broker_url).
    assert "proxy_port: 28080" in out
    assert "broker_port: 28082" in out
    assert "broker_url: http://base-docker-broker:28082" in out


# ---------------------------------------------------------------------------
# swarm_init advertise-addr: the operational default auto-detects THIS host's
# primary IPv4 (source IP of the default route) so a blank non-master box
# advertises its OWN address, while an explicit override is honored verbatim and
# the old hardcoded master IP is never advertised on a non-master box.
# ---------------------------------------------------------------------------

# The exact detection the installer runs at ADVERTISE_ADDR default resolution.
_DETECT_ADVERTISE_ADDR_CMD = (
    "ip -4 route get 1.1.1.1 2>/dev/null | "
    "awk '{for(i=1;i<=NF;i++) if($i==\"src\"){print $(i+1); exit}}'"
)


def _detect_primary_ipv4() -> str:
    """Detect this host's primary IPv4 exactly like the installer's default."""
    proc = subprocess.run(
        ["bash", "-c", _DETECT_ADVERTISE_ADDR_CMD],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def test_advertise_addr_unset_auto_detects_host_primary_ip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With ADVERTISE_ADDR unset the installer auto-detects the source IP of the
    # default route; the swarm_init plan + STEP log must carry that same IP.
    monkeypatch.delenv("ADVERTISE_ADDR", raising=False)
    detected = _detect_primary_ipv4()
    if not detected:
        pytest.skip("no default-route source IPv4 detectable on this host")

    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert f"docker swarm init --advertise-addr {detected}" in result.stdout
    assert f"swarm_init (advertise-addr={detected})" in result.stdout


def test_advertise_addr_override_is_honored_and_hardcoded_master_ip_gone(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, extra_env={"ADVERTISE_ADDR": "203.0.113.7"})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # The explicit override flows straight into `docker swarm init`.
    assert "docker swarm init --advertise-addr 203.0.113.7" in out
    # The old hardcoded master IP must never be the advertise addr on a
    # non-master box (regression on the 86.38.238.235 default).
    assert "86.38.238.235" not in out


# ---------------------------------------------------------------------------
# Eval job network isolation (base_jobs_internal): the proxy + agent-challenge
# api/worker are multi-homed onto the isolated internal eval overlay so the eval
# JOB reaches ONLY the gateway + API by name, never postgres; the broker + prism
# services stay off it.
# ---------------------------------------------------------------------------


def test_proxy_is_attached_to_internal_eval_overlay(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    proxy = _service_block(lines, "base-master-proxy")
    assert "--network base_jobs_internal" in proxy
    assert "--network base_challenges" in proxy

    # The broker dispatches eval jobs but does NOT serve the gateway, so it is
    # deliberately kept OFF the eval overlay.
    broker = _service_block(lines, "base-docker-broker")
    assert "--network base_jobs_internal" not in broker


def test_agent_challenge_services_attached_to_internal_eval_overlay(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    for service in ("challenge-agent-challenge", "challenge-agent-challenge-worker"):
        block = _service_block(lines, service)
        assert "--network base_jobs_internal" in block
        assert "--network base_challenges" in block

    # prism services are not multi-homed: the prism eval JOB is egress-locked by
    # the broker (pinned to the internal overlay), not via service multi-homing.
    for service in ("challenge-prism", "challenge-prism-worker"):
        block = _service_block(lines, service)
        assert "--network base_jobs_internal" not in block


# ---------------------------------------------------------------------------
# VAL-CODE-AUTO-007: the master proxy now hosts the challenge-image-updater +
# registry reconciler, so the base-master-proxy service must mount the manager
# docker socket (+ GHCR read creds RO) so those in-process loops can roll
# challenge services via `docker service create/update`.
# ---------------------------------------------------------------------------


def test_proxy_mounts_docker_socket_and_ghcr_creds(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    proxy = _service_block(lines, "base-master-proxy")
    # The docker socket is bind-mounted so the in-proxy reconciler + challenge
    # image updater can `docker service create/update`.
    assert (
        "type=bind,source=/var/run/docker.sock,destination=/var/run/docker.sock"
        in proxy
    )
    # GHCR read creds mounted read-only (+ DOCKER_CONFIG) so private
    # ghcr.io/baseintelligence digests resolve/pull — mirrors the broker branch.
    assert "target=/root/.docker,readonly" in proxy
    assert "DOCKER_CONFIG=/root/.docker" in proxy
    # Runs as root: the host docker.sock is root-owned (mirrors the broker).
    assert "--user root" in proxy


def test_broker_still_mounts_docker_socket(tmp_path: Path) -> None:
    # Regression guard: the broker keeps its own docker.sock mount (both the
    # broker and the proxy roll services).
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    broker = _service_block(lines, "base-docker-broker")
    assert "source=/var/run/docker.sock" in broker


# ---------------------------------------------------------------------------
# Swarm self-healing update/rollback policy on every long-lived service
# (crash-detection + auto-rollback), plus an HTTP /health container healthcheck
# on the services that serve one. Kept in sync with swarm_backend.py
# SERVICE_UPDATE_ROLLBACK_POLICY (the dynamic orchestrator path).
# ---------------------------------------------------------------------------

_SWARM_UPDATE_POLICY_FLAGS = (
    "--update-failure-action rollback",
    "--update-monitor 45s",
    "--update-max-failure-ratio 0",
    "--update-order stop-first",
    "--rollback-failure-action pause",
    "--rollback-monitor 45s",
    "--rollback-max-failure-ratio 1",
)


def test_all_long_lived_services_carry_update_rollback_policy(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()
    for service in (
        "base-master-postgres",
        "base-docker-broker",
        "base-master-proxy",
        "challenge-agent-challenge",
        "challenge-agent-challenge-worker",
        "challenge-prism",
        "challenge-prism-worker",
    ):
        block = _service_block(lines, service)
        for flag in _SWARM_UPDATE_POLICY_FLAGS:
            assert flag in block, f"{service} missing {flag!r}"
        # stop-first must appear exactly once (no leftover duplicate flag).
        assert block.count("--update-order stop-first") == 1, service


def test_http_services_get_health_probe_and_others_do_not(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()
    # Services with a real HTTP /health endpoint get a container healthcheck on
    # their own listen port with conservative timings.
    for service, port in (
        ("base-docker-broker", "8082"),
        ("base-master-proxy", "19080"),
        ("challenge-agent-challenge", "8000"),
        ("challenge-prism", "8080"),
    ):
        block = _service_block(lines, service)
        assert "--health-cmd" in block, service
        assert f"localhost:{port}/health" in block, service
        assert "--health-interval 10s" in block, service
        assert "--health-timeout 5s" in block, service
        assert "--health-retries 3" in block, service
        assert "--health-start-period 40s" in block, service
    # postgres (not HTTP) and the worker sidecars (no /health server) rely on
    # crash-detection only — NO container healthcheck.
    for service in (
        "base-master-postgres",
        "challenge-agent-challenge-worker",
        "challenge-prism-worker",
    ):
        assert "--health-cmd" not in _service_block(lines, service), service


# ---------------------------------------------------------------------------
# VAL-CODE-DEPLOY-004: validator.yaml template + N-validator run path
# ---------------------------------------------------------------------------


def test_validator_template_exists_and_loads_with_agent_block() -> None:
    assert VALIDATOR_TEMPLATE.is_file()
    settings = load_settings(VALIDATOR_TEMPLATE)

    agent = settings.validator.agent
    assert agent.master_url
    assert agent.broker_url
    assert agent.capabilities  # cpu (and gpu for the GPU validator)
    # A distinct hotkey wallet identity the agent signs coordination calls with.
    assert settings.network.wallet_name
    assert settings.network.wallet_path


def test_validator_template_documents_capabilities_and_own_broker() -> None:
    text = VALIDATOR_TEMPLATE.read_text(encoding="utf-8")
    assert "base validator agent --config" in text
    # capabilities cpu|gpu guidance present.
    assert 'capabilities: ["cpu"]' in text
    assert "gpu" in text
    # the validator's OWN broker (not the master's).
    assert "broker_url" in text


def test_readme_documents_n_validator_run_path() -> None:
    readme = SWARM_README.read_text(encoding="utf-8")
    assert "base validator agent --config" in readme
    assert "validator.yaml" in readme
    # Ties the validator hotkeys to the master's no-chain mock metagraph.
    assert "mock_metagraph" in readme
    assert "MOCK_METAGRAPH" in readme
    assert "capabilities" in readme


# ---------------------------------------------------------------------------
# VAL-VDIR-DEPLOY-001: self-declared validator identity (display_name/logo_url)
# rendered per MOCK_METAGRAPH entry + carried by the validator.yaml template, so
# the live test validators show a real subnet identity + logo on the no-chain
# deploy. Dry-run only; existing guard tests stay green; bash -n clean.
# ---------------------------------------------------------------------------

# Distinctive per-validator self-declared identities (not collidable with any
# other token in the plan output).
MMG_DISPLAY_NAME_1 = "Acme Subnet Validator"
MMG_LOGO_URL_1 = "https://logos.example/acme-validator.png"
MMG_DISPLAY_NAME_2 = "Beta Subnet Validator"
MMG_LOGO_URL_2 = "https://logos.example/beta-validator.png"
MOCK_METAGRAPH_IDENTITY_JSON = (
    f'[{{"hotkey":"{MMG_VAL_HOTKEY_1}","validator_permit":true,"stake":1000,'
    f'"display_name":"{MMG_DISPLAY_NAME_1}","logo_url":"{MMG_LOGO_URL_1}"}},'
    f'{{"hotkey":"{MMG_VAL_HOTKEY_2}","validator_permit":true,"stake":1000,'
    f'"display_name":"{MMG_DISPLAY_NAME_2}","logo_url":"{MMG_LOGO_URL_2}"}}]'
)


def test_mock_metagraph_identity_fields_rendered_into_master_config(
    tmp_path: Path,
) -> None:
    # Each MOCK_METAGRAPH entry's optional self-declared identity rides through
    # the verbatim render into network.mock_metagraph, so the dry-run plan carries
    # per-validator display_name + logo_url (the no-chain identity fallback).
    result = _run(tmp_path, extra_env={"MOCK_METAGRAPH": MOCK_METAGRAPH_IDENTITY_JSON})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    assert "mock_metagraph:" in out
    assert "display_name" in out
    assert "logo_url" in out
    # Both per-validator identities are present, tied to their hotkeys.
    assert MMG_VAL_HOTKEY_1 in out
    assert MMG_DISPLAY_NAME_1 in out
    assert MMG_LOGO_URL_1 in out
    assert MMG_VAL_HOTKEY_2 in out
    assert MMG_DISPLAY_NAME_2 in out
    assert MMG_LOGO_URL_2 in out


def test_mock_metagraph_without_identity_omits_fields(tmp_path: Path) -> None:
    # Identity is OPTIONAL: a MOCK_METAGRAPH with no display_name/logo_url renders
    # neither key (identicon fallback), so the seam stays minimal when unset.
    result = _run(tmp_path, extra_env={"MOCK_METAGRAPH": MOCK_METAGRAPH_JSON})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    config_lines = [
        line for line in result.stdout.splitlines() if "mock_metagraph:" in line
    ]
    assert config_lines, "mock_metagraph not rendered"
    rendered = config_lines[0]
    assert "display_name" not in rendered
    assert "logo_url" not in rendered


def test_validator_template_carries_self_declared_identity() -> None:
    # The validator.yaml template documents + sets the validator.agent
    # self-declared identity so a copied-per-validator config surfaces a real
    # name + logo on the no-chain deploy.
    text = VALIDATOR_TEMPLATE.read_text(encoding="utf-8")
    assert "display_name:" in text
    assert "logo_url:" in text

    settings = load_settings(VALIDATOR_TEMPLATE)
    agent = settings.validator.agent
    assert agent.display_name
    assert agent.logo_url


# ---------------------------------------------------------------------------
# VAL-CODE-DEPLOY-005: dry-run default, deterministic/idempotent, bash -n clean
# ---------------------------------------------------------------------------


def test_installer_is_bash_n_clean() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SWARM_INSTALLER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr!r}"


def test_dry_run_is_the_default_and_mutates_nothing(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    # Default mode announces dry-run and only PRINTS planned commands (prefixed
    # by `  + `); the docker stub would exit 1 on any mutating subcommand, so a
    # clean exit proves nothing mutating was executed.
    assert "DRY-RUN (default)" in result.stdout
    assert "  + docker service create" in result.stdout


def test_dry_run_plan_is_deterministic(tmp_path: Path) -> None:
    # Idempotency proxy: with fixed inputs the planned output is byte-identical
    # across runs (no nondeterministic ordering / churn in the plan).
    env = {"MOCK_METAGRAPH": MOCK_METAGRAPH_JSON}
    first = _run(tmp_path, extra_env=env)
    second = _run(tmp_path, extra_env=env)
    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout == second.stdout


# ---------------------------------------------------------------------------
# VAL-CODE-REG-003 (m9 rollout-prep): challenge services are named
# `challenge-<slug>` so the default-on m7 registry reconciler ADOPTS the live
# services instead of creating `base-challenge-<slug>` duplicates. Only CHALLENGE
# services drop the `base-` prefix; master/broker keep their `base-` names.
# ---------------------------------------------------------------------------


def test_challenge_services_named_challenge_slug_not_base_prefixed(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = result.stdout.splitlines()

    # Every challenge swarm service renders as `challenge-<slug>` (the exact name
    # the master orchestrator/reconciler expects), with a matching --hostname so
    # its Docker-network DNS matches the registry internal_base_url.
    for service in (
        "challenge-agent-challenge",
        "challenge-agent-challenge-worker",
        "challenge-prism",
        "challenge-prism-worker",
    ):
        block = _service_block(lines, service)
        assert f"--name {service} " in block
        assert f"--hostname {service}" in block

    # NO challenge service/volume/secret/URL uses the legacy `base-challenge-*`
    # name anywhere in the rendered plan; a regression would make the default-on
    # m7 reconciler create live duplicates alongside base-challenge-agent-challenge
    # / base-challenge-prism.
    plan = result.stdout.replace("\\", "")
    assert "--name base-challenge-" not in plan
    assert "--hostname base-challenge-" not in plan
    assert "http://base-challenge-" not in plan

    # Standardization is SCOPED to challenge services: master/broker keep `base-`.
    assert _service_block(lines, "base-master-proxy")
    assert _service_block(lines, "base-docker-broker")


# ---------------------------------------------------------------------------
# Phase H (turnkey installer): --skip-ghcr-login makes ghcr_login a no-op AND
# ensures ${DOCKER_CONFIG:-~/.docker} exists with a `{}` config so the broker /
# proxy /root/.docker read-only binds resolve (the public-image footgun fix).
# ---------------------------------------------------------------------------


def test_skip_ghcr_login_is_noop_and_ensures_docker_config_dir(
    tmp_path: Path,
) -> None:
    docker_cfg = tmp_path / "docker-config"  # non-existent => `{}` config planned
    result = _run(
        tmp_path,
        extra_args=("--skip-ghcr-login",),
        extra_env={"DOCKER_CONFIG": str(docker_cfg)},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    # ghcr login is skipped (no `docker login ghcr.io` planned).
    assert "skipping ghcr login" in out
    assert "docker login ghcr.io" not in out
    # The docker config dir + an empty `{}` config are planned so the RO binds
    # resolve. Plan lines are printf '%q'-escaped, so strip backslashes to compare.
    plan = out.replace("\\", "")
    assert f"install -d -m 0700 {docker_cfg}" in plan
    assert "printf '{}'" in plan
    assert f"{docker_cfg}/config.json" in plan


# ---------------------------------------------------------------------------
# Phase H: the --validator-node bring-up also runs ensure_docker (STEP 0) BEFORE
# preflight (STEP 1), so a blank validator host gets an engine first.
# ---------------------------------------------------------------------------


def test_validator_node_ensure_docker_before_preflight(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        extra_args=("--validator-node",),
        extra_env={"VALIDATOR_MASTER_URL": "http://master.example:19080"},
    )
    step0 = "STEP 0 ensure_docker"
    step1 = "STEP 1/12 preflight"
    assert step0 in result.stdout
    assert step1 in result.stdout
    assert result.stdout.index(step0) < result.stdout.index(step1)
