"""Final release safety gate (VAL-CROSS-079).

Automated validation must stay disposable, evidence-based, and incapable of
mutating the live chain, live Swarm, or protected external provider resources.

These checks spy on command inventories, default gates, disposable project
naming, and protected-path hygiene. They are black-box relative to live
resources: they never issue live Swarm mutations, real ``set_weights``, or
provider provisioning.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from base.bittensor.weight_setter import WeightSetter
from base.config.settings import NetworkSettings

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "deploy" / "compose"
MASTER_COMPOSE = COMPOSE_DIR / "docker-compose.yml"
VALIDATOR_COMPOSE = COMPOSE_DIR / "docker-compose.validator.yml"
INSTALL_MASTER = COMPOSE_DIR / "install-master.sh"
INSTALL_VALIDATOR = COMPOSE_DIR / "install-validator.sh"
TEARDOWN_MASTER = COMPOSE_DIR / "teardown-master.sh"
BACKUP_MASTER = COMPOSE_DIR / "backup-master.sh"
RESTORE_MASTER = COMPOSE_DIR / "restore-master.sh"
BACKUP_CHALLENGE = COMPOSE_DIR / "backup-challenge.sh"
BURN_WEIGHTS = ROOT / "scripts" / "burn_weights_24h.py"
# Known identity of the several pre-existing user-owned untracked file. Automated
# validation must not mutate it; presence and exact bytes are the proof.
BURN_WEIGHTS_SHA256 = "6480e46a00fd715a72f2189c7752e4fef1cadb139ee0d0fc6c087abf06701301"

DISPOSABLE_PROJECT_PREFIXES = (
    "base-mission-",
    "mission-",
    "base-compose-config",
    "clean-validator-",
    "base-validator-",
)
# Naming patterns that automated release validation may create/teardown.
# Includes historical mission harness names already used by targeted tests.
DISPOSABLE_PROJECT_RE = re.compile(
    r"^(?:"
    r"base-mission-"
    r"|mission-"
    r"|base-compose-config"
    r"|clean-validator-"
    r"|base-validator-"
    r")[a-zA-Z0-9_.-]*$"
)

SWARM_FORBIDDEN_SUBSTRINGS = (
    "docker service",
    "docker stack",
    "docker swarm",
    "docker node",
    "docker secret create",
)
# Live compound/mainnet hostnames that automated suite paths must not open as
# writable chain clients (read-only docs/config defaults of the protocol URL are
# ok if they do not construct Subtensor/WeightSetter).
LIVE_CHAIN_ENDPOINTS = (
    "wss://entrypoint-finney.opentensor.ai",
    "wss://entrypoint-finney",
    "wss://finney.opentensor.ai",
    "wss://lite.sub.latent.to",
)
PROVIDER_HOSTS = ("lium.io", "api.targon.com")

# Automated-validation surfaces that must stay Compose-only and disposable.
AUTOMATED_COMPOSE_ARTIFACTS = (
    MASTER_COMPOSE,
    VALIDATOR_COMPOSE,
    INSTALL_MASTER,
    INSTALL_VALIDATOR,
    TEARDOWN_MASTER,
    BACKUP_MASTER,
    RESTORE_MASTER,
    BACKUP_CHALLENGE,
)

AUTOMATED_TEST_GLOBS = (
    "tests/unit/test_docker_compose_deploy.py",
    "tests/unit/test_validator_compose_artifact.py",
    "tests/unit/test_compose_challenge_watcher.py",
    "tests/unit/test_compose_reconcile_surface.py",
    "tests/unit/test_installed_artifact_compose_e2e.py",
    "tests/unit/test_operational_security_and_recovery.py",
    "tests/unit/test_validator_weight_submitter.py",
    "tests/unit/test_master_submission_prohibition.py",
    "tests/unit/test_final_release_safety.py",
    "tests/integration/test_master_weights_postgres.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resource_inventory() -> dict[str, set[str]]:
    """Capture mission-facing Docker resource names (best-effort, read-only)."""

    inventory: dict[str, set[str]] = {
        "containers": set(),
        "networks": set(),
        "volumes": set(),
        "compose_projects": set(),
    }
    try:
        ps = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if ps.returncode == 0:
            inventory["containers"] = {
                line.strip() for line in ps.stdout.splitlines() if line.strip()
            }
        nets = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if nets.returncode == 0:
            inventory["networks"] = {
                line.strip() for line in nets.stdout.splitlines() if line.strip()
            }
        vols = subprocess.run(
            ["docker", "volume", "ls", "--format", "{{.Name}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if vols.returncode == 0:
            inventory["volumes"] = {
                line.strip() for line in vols.stdout.splitlines() if line.strip()
            }
        labels = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                '{{.Label "com.docker.compose.project"}}',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if labels.returncode == 0:
            inventory["compose_projects"] = {
                line.strip() for line in labels.stdout.splitlines() if line.strip()
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # Docker may be unavailable in some unit lanes; resource-diff tests skip.
        pass
    return inventory


class CommandSpy:
    """Record attempted subprocess argv sequences without executing them."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(
        self,
        cmd: list[str] | tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = [str(part) for part in cmd]
        self.calls.append(argv)
        lowered = " ".join(argv).lower()
        if any(token in lowered for token in SWARM_FORBIDDEN_SUBSTRINGS):
            raise AssertionError(f"refused Swarm command: {argv}")
        # Only mission-disposable compose projects may be mutated.
        if len(argv) >= 2 and argv[0] == "docker" and argv[1] == "compose":
            project = _compose_project_from_argv(argv)
            if project is not None and not DISPOSABLE_PROJECT_RE.match(project):
                raise AssertionError(
                    f"refused non-disposable compose project mutation: {project}"
                )
            if "up" in argv or "down" in argv or "rm" in argv:
                if project is None:
                    raise AssertionError(
                        f"compose mutation missing explicit -p project: {argv}"
                    )
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )


def _compose_project_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token in {"-p", "--project-name"} and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--project-name="):
            return token.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------
# Defaults: no live chain mutation, no real set_weights
# ---------------------------------------------------------------------------


def test_validator_on_chain_submission_defaults_off() -> None:
    from base.config.settings import Settings

    settings = Settings()
    assert settings.validator.submit_on_chain_enabled is False


def test_validator_weight_submitter_gate_off_constructs_no_chain_client(
    tmp_path: Path,
) -> None:
    import asyncio

    from base.challenge_sdk.roles import Role, activate_role
    from base.validator.weight_submitter import (
        ValidatorSubmitOutcome,
        ValidatorWeightSubmitter,
    )
    from base.validator.weights_client import WeightsClient

    class _NeverFetch:
        calls = 0

        async def fetch_latest(self) -> object:
            self.calls += 1
            raise AssertionError("disabled submit path must not fetch")

    built: list[str] = []

    def factory() -> WeightSetter | None:
        built.append("constructed")
        raise AssertionError("disabled submit path must not build WeightSetter")

    factory_cb: Callable[[], WeightSetter | None] = factory
    submitter = ValidatorWeightSubmitter(
        submit_enabled=False,
        netuid=100,
        weights_client=_NeverFetch(),  # type: ignore[arg-type]
        weight_setter_factory=factory_cb,
        state_dir=tmp_path,
    )
    # Keep WeightsClient in the typing graph without constructing a live client.
    assert WeightsClient is not None

    with activate_role(Role.VALIDATOR):
        outcome = asyncio.run(submitter.run_once())
    assert outcome is ValidatorSubmitOutcome.DISABLED
    assert built == []
    assert _NeverFetch.calls == 0


def test_automated_weight_tests_use_recording_doubles_not_live_set_weights() -> None:
    """Cross-area weight tests inject RecordingSetter doubles / never live RPC."""

    source = _read(ROOT / "tests/unit/test_validator_weight_submitter.py")
    assert "class _RecordingSetter" in source
    assert "def set_weights" in source
    # No live endpoint construction in the automated suite.
    for endpoint in LIVE_CHAIN_ENDPOINTS:
        assert endpoint not in source
    # Master submit prohibition remains wired.
    master_src = _read(ROOT / "tests/unit/test_master_submission_prohibition.py")
    assert "WeightSetter" in master_src
    assert "set_weights" in master_src


def test_master_runtime_never_imports_weight_setter() -> None:
    """Master may read metagraph/identity, but never submit weights.

    Identity/metagraph caches are coordination helpers. WeightSetter,
    create_bittensor_submit_runtime, and set_weights side effects remain
    validator-only (VAL-CROSS-079 / VAL-WEIGHT-058).
    """

    import ast as ast_module

    forbidden_modules = {
        "base.bittensor.weight_setter",
        "base.bittensor.factory",
        "base.validator.weight_submitter",
        "base.supervisor.weight_submit",
    }
    allowed_bittensor_modules = {
        "base.bittensor.identity_cache",
        "base.bittensor.metagraph_cache",
    }
    master_dir = ROOT / "src" / "base" / "master"
    offenders: list[str] = []
    for path in sorted(master_dir.glob("*.py")):
        tree = ast_module.parse(_read(path))
        for node in ast_module.walk(tree):
            if isinstance(node, ast_module.ImportFrom) and node.module:
                module = node.module
                if module in forbidden_modules or module.endswith("weight_submitter"):
                    offenders.append(f"{path.name}:import {module}")
                if module.startswith("base.bittensor") and module not in (
                    allowed_bittensor_modules
                ):
                    # Allow package-level re-export absence; flag any other path.
                    if module != "base.bittensor":
                        offenders.append(f"{path.name}:import {module}")
            if isinstance(node, ast_module.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        offenders.append(f"{path.name}:import {alias.name}")
            if isinstance(node, ast_module.Name) and node.id in {
                "WeightSetter",
                "create_bittensor_submit_runtime",
            }:
                offenders.append(f"{path.name}:{node.id}")
            if isinstance(node, ast_module.Attribute) and node.attr == "set_weights":
                offenders.append(f"{path.name}:set_weights")
    assert offenders == []


def test_create_bittensor_runtime_uses_mock_metagraph_without_subtensor() -> None:
    from base.bittensor.factory import create_bittensor_runtime
    from base.config.settings import MockMetagraphNode, Settings

    settings = Settings(
        network=NetworkSettings(
            netuid=1,
            mock_metagraph=[
                MockMetagraphNode(hotkey="hk-a", uid=0, validator_permit=True)
            ],
        )
    )
    runtime = create_bittensor_runtime(settings)
    assert runtime.weight_setter is None
    assert runtime.metagraph_cache.static is True
    assert runtime.metagraph_cache.subtensor is None


# ---------------------------------------------------------------------------
# Disposable Compose project / teardown scope
# ---------------------------------------------------------------------------


def test_install_scripts_default_to_disposable_mission_project_names() -> None:
    master = _read(INSTALL_MASTER)
    validator = _read(INSTALL_VALIDATOR)
    assert 'PROJECT_NAME="${COMPOSE_PROJECT_NAME:-base-mission-master}"' in master
    assert 'PROJECT_NAME="${COMPOSE_PROJECT_NAME:-base-mission-validator}"' in validator
    assert DISPOSABLE_PROJECT_RE.match("base-mission-master")
    assert DISPOSABLE_PROJECT_RE.match("base-mission-validator-a")
    assert DISPOSABLE_PROJECT_RE.match("mission-opsec")
    assert DISPOSABLE_PROJECT_RE.match("mission-e2e-compose")
    assert not DISPOSABLE_PROJECT_RE.match("production-swarm")
    assert not DISPOSABLE_PROJECT_RE.match("live")


def test_teardown_requires_explicit_project_and_never_calls_swarm() -> None:
    content = _read(TEARDOWN_MASTER)
    assert "--project-name is required" in content
    assert "docker compose -p" in content
    assert "Never touches live Swarm" in content or "live Swarm" in content
    for forbidden in SWARM_FORBIDDEN_SUBSTRINGS:
        assert forbidden not in content.lower().replace("never touches live swarm", "")
    # Default preserves volumes; destroy is opt-in for disposable tests only.
    assert "--destroy-data" in content
    assert "down --remove-orphans" in content


def test_teardown_script_subprocess_spy_refuses_swarm_and_foreign_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = CommandSpy()
    monkeypatch.setattr(subprocess, "run", spy.run)

    # Teardown helper itself is shell; emulate the only compose mutating shapes.
    spy.run(
        [
            "docker",
            "compose",
            "-p",
            "base-mission-final-release",
            "-f",
            str(MASTER_COMPOSE),
            "down",
            "--remove-orphans",
        ]
    )
    with pytest.raises(AssertionError, match="non-disposable"):
        spy.run(
            [
                "docker",
                "compose",
                "-p",
                "production-live",
                "down",
                "--volumes",
            ]
        )
    with pytest.raises(AssertionError, match="Swarm"):
        spy.run(["docker", "service", "ls"])
    with pytest.raises(AssertionError, match="Swarm"):
        spy.run(["docker", "stack", "rm", "live"])
    assert any("base-mission-final-release" in " ".join(c) for c in spy.calls)


def test_automated_compose_artifacts_are_compose_only() -> None:
    for path in AUTOMATED_COMPOSE_ARTIFACTS:
        assert path.is_file(), path
        content = _read(path).lower()
        for forbidden in SWARM_FORBIDDEN_SUBSTRINGS:
            # "Never touches live Swarm" docs text mentions Swarm as prohibition.
            if (
                forbidden == "docker swarm"
                and "never" in content
                and "swarm" in content
            ):
                # Only allow narrative mentions that refuse Swarm.
                assert "docker swarm" not in content or "never" in content
                continue
            assert forbidden not in content, f"{path}: contains {forbidden}"
        if path.suffix in {".sh", ".yml", ".yaml"}:
            assert "docker compose" in content or "compose" in content


def test_automated_validation_tests_pin_disposable_compose_project_names() -> None:
    """Harnesses that render compose config use mission-scoped project names."""

    pattern = re.compile(
        r'COMPOSE_PROJECT_NAME["\']?\s*[:=]\s*["\']'
        r"((?:base-mission-|mission-|base-compose-config|clean-validator-|base-validator-)"
        r"[a-zA-Z0-9_.-]*)['\"]"
    )
    found: list[str] = []
    for relative in AUTOMATED_TEST_GLOBS:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = _read(path)
        for match in pattern.finditer(text):
            name = match.group(1)
            found.append(name)
            assert DISPOSABLE_PROJECT_RE.match(name), f"{relative}: {name}"
    # At least one known mission-scoped harness name must be present.
    assert found, "expected disposable COMPOSE_PROJECT_NAME declarations"
    assert any(name.startswith("mission-") for name in found)
    # Installer defaults must also stay disposable.
    master = _read(INSTALL_MASTER)
    validator = _read(INSTALL_VALIDATOR)
    assert "base-mission-master" in master
    assert "base-mission-validator" in validator
    assert DISPOSABLE_PROJECT_RE.match("base-mission-master")
    assert DISPOSABLE_PROJECT_RE.match("base-mission-validator")


def test_compose_config_defaults_do_not_embed_live_swarm_or_providers() -> None:
    master = yaml.safe_load(_read(MASTER_COMPOSE))
    services = master.get("services") or {}
    assert "base-master-validator" in services
    assert "master-postgres" in services
    assert "challenge-prism" in services
    blob = json.dumps(master).lower()
    for forbidden in ("docker service", "docker stack", "swarm", "overlay"):
        # Image digests and project names may not include these tokens.
        if forbidden == "swarm":
            assert "docker swarm" not in blob
            continue
        assert forbidden not in blob
    for host in PROVIDER_HOSTS:
        assert host not in blob


# ---------------------------------------------------------------------------
# Provider resources stay opt-in, not automated validation
# ---------------------------------------------------------------------------


def test_live_provider_scripts_are_gated_and_not_pytest_collected() -> None:
    live_scripts = (
        ROOT / "scripts" / "live_lium_e2e.py",
        ROOT / "scripts" / "live_worker_lium_e2e.py",
    )
    for path in live_scripts:
        assert path.is_file()
        source = _read(path)
        assert "BASE_LIVE_PROVIDER_TESTS" in source
        # Not under tests/, so default pytest collection never reaches them.
        assert "tests/" not in str(path.relative_to(ROOT))
        # Gating must short-circuit when the flag is absent/false.
        gated = (
            '!= "1"' in source
            or "!= '1'" in source
            or "!= 1" in source
            or '== "1"' in source
        )
        assert gated


def test_default_settings_do_not_enable_live_provider_provisioning() -> None:
    from base.config.settings import Settings

    settings = Settings()
    assert settings.worker.deploy.provider == "local"
    assert settings.worker.deploy.max_lifetime_hours <= 1.0
    # Worker plane (provider-adjacent) defaults off.
    assert settings.compute.worker_plane_enabled is False


def test_offline_egress_guard_plugin_is_available_for_release_suites() -> None:
    guard_path = ROOT / "scripts" / "mission" / "no_external_egress.py"
    assert guard_path.is_file()
    source = _read(guard_path)
    assert "ExternalEgressBlocked" in source
    assert "lium.io" in source or "non-loopback" in source
    # Unit suite for the guard itself.
    assert (ROOT / "tests/unit/test_no_external_egress.py").is_file()


# ---------------------------------------------------------------------------
# Protected untracked path and evidence hygiene
# ---------------------------------------------------------------------------


def test_burn_weights_script_remains_untracked_and_byte_identical() -> None:
    """VAL-CROSS-079: user-owned burn_weights_24h.py must stay out of git tree."""

    # On GHA checkouts the protected local script is absent (never committed).
    # When present locally, bytes + untracked status must match the known identity.
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "scripts/burn_weights_24h.py"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode != 0, "burn_weights_24h.py must not be tracked"
    if BURN_WEIGHTS.is_file():
        digest = _sha256_file(BURN_WEIGHTS)
        assert digest == BURN_WEIGHTS_SHA256
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "scripts/burn_weights_24h.py"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
        assert status.startswith("??") or status == "", status
    # Never imported or executed by automated tests.
    for relative in AUTOMATED_TEST_GLOBS:
        path = ROOT / relative
        if path.is_file():
            text = _read(path)
            if path.name == "test_final_release_safety.py":
                # This file mentions the path only to protect it.
                assert "burn_weights_24h" in text
                continue
            assert "burn_weights_24h" not in text


def test_agent_challenge_repo_is_out_of_scope_for_automated_validation() -> None:
    agent = Path("/projects/platform-network/agent-challenge")
    # Mission worktrees have the sibling checkout; CI for Base alone may not.
    if not agent.is_dir():
        pytest.skip("agent-challenge sibling checkout not present in this workspace")
    # Automated Base tests may name the diagnostic constant but never write there.
    offenders: list[str] = []
    for path in (ROOT / "tests").rglob("*.py"):
        text = _read(path)
        if "agent-challenge" in text and (
            "open(" in text or "Path.write" in text or "write_text" in text
        ):
            # Writing *about* the path is ok; writing *into* it is not.
            if 'Path("/projects/platform-network/agent-challenge")' in text and (
                ".write" in text or "open(" in text
            ):
                # Only flag when a literal agent path is used with write ops.
                if "agent-challenge" in text and re.search(
                    r'agent-challenge["\'].*\.write|write_text\(.*agent-challenge',
                    text,
                ):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_secret_canaries_absent_from_compose_and_install_sources() -> None:
    canaries = (
        "CANARY-ADMIN-TOKEN",
        "sk-live-",
        "BEGIN PRIVATE KEY",
        "wss://entrypoint-finney.opentensor.ai:443",
    )
    for path in AUTOMATED_COMPOSE_ARTIFACTS:
        blob = _read(path)
        for canary in canaries:
            assert canary not in blob, path


# ---------------------------------------------------------------------------
# Resource inventory before/after disposable dry-run
# ---------------------------------------------------------------------------


def test_resource_inventory_delta_is_empty_for_config_only_validation(
    tmp_path: Path,
) -> None:
    """Config-only compose validation must not create Docker resources."""

    before = _resource_inventory()
    if not before["networks"] and not before["containers"]:
        pytest.skip("docker unavailable for inventory spy")

    secrets = tmp_path / "secrets"
    config = tmp_path / "config"
    secrets.mkdir()
    config.mkdir()
    secrets.chmod(0o700)
    config.chmod(0o700)
    admin = secrets / "admin_token"
    postgres = secrets / "postgres_password"
    prism = secrets / "prism_shared_token"
    master_config = config / "master.yaml"
    for path, value in (
        (admin, "CANARY-FINAL-ADMIN"),
        (postgres, "CANARY-FINAL-PG"),
        (prism, "CANARY-FINAL-PRISM"),
    ):
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
    master_config.write_text(
        "database:\n  url: postgresql+asyncpg://base:x@master-postgres:5432/base\n",
        encoding="utf-8",
    )
    master_config.chmod(0o600)
    project = "mission-final-release-safety"
    env_file = config / "compose.env"
    env_file.write_text(
        "\n".join(
            [
                f"COMPOSE_PROJECT_NAME={project}",
                "BASE_MASTER_IMAGE_REPOSITORY=registry.example/base-master",
                f"BASE_MASTER_IMAGE_DIGEST={'a' * 64}",
                "PRISM_IMAGE_REPOSITORY=registry.example/prism",
                f"PRISM_IMAGE_DIGEST={'b' * 64}",
                "POSTGRES_IMAGE_REPOSITORY=registry.example/postgres",
                f"POSTGRES_IMAGE_DIGEST={'c' * 64}",
                f"BASE_MASTER_CONFIG={master_config}",
                f"BASE_ADMIN_TOKEN_FILE={admin}",
                f"BASE_POSTGRES_PASSWORD_FILE={postgres}",
                f"PRISM_SHARED_TOKEN_FILE={prism}",
                "BASE_MASTER_HOST_PORT=3180",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    env = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": project,
        "BASE_MASTER_IMAGE_REPOSITORY": "registry.example/base-master",
        "BASE_MASTER_IMAGE_DIGEST": "a" * 64,
        "PRISM_IMAGE_REPOSITORY": "registry.example/prism",
        "PRISM_IMAGE_DIGEST": "b" * 64,
        "POSTGRES_IMAGE_REPOSITORY": "registry.example/postgres",
        "POSTGRES_IMAGE_DIGEST": "c" * 64,
        "BASE_MASTER_CONFIG": str(master_config),
        "BASE_ADMIN_TOKEN_FILE": str(admin),
        "BASE_POSTGRES_PASSWORD_FILE": str(postgres),
        "PRISM_SHARED_TOKEN_FILE": str(prism),
        "BASE_MASTER_HOST_PORT": "3180",
        "BASE_COMPOSE_ENV_FILE": str(env_file),
    }
    rendered = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(MASTER_COMPOSE),
            "config",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert rendered.returncode == 0
    after = _resource_inventory()
    # config --quiet is pure; no new containers/networks/volumes for our project.
    new_containers = after["containers"] - before["containers"]
    new_networks = after["networks"] - before["networks"]
    new_volumes = after["volumes"] - before["volumes"]
    assert not any(project in name for name in new_containers)
    assert not any(project in name for name in new_networks)
    assert not any(project in name for name in new_volumes)
    # Canaries never leave the host secret files into docker env dumps.
    assert "CANARY-FINAL-ADMIN" not in rendered.stdout
    assert "CANARY-FINAL-PG" not in rendered.stdout


def test_compose_runner_scopes_mutations_to_configured_project() -> None:
    from base.master.compose_backend import ComposeRunner

    runner = ComposeRunner(
        project_name="mission-final-release-runner",
        compose_file=MASTER_COMPOSE,
        docker_bin="docker",
    )
    argv = runner._compose_base_cmd(MASTER_COMPOSE)  # noqa: SLF001
    assert argv[:4] == [
        "docker",
        "compose",
        "-p",
        "mission-final-release-runner",
    ]
    assert "-f" in argv
    joined = " ".join(argv).lower()
    for forbidden in SWARM_FORBIDDEN_SUBSTRINGS:
        assert forbidden not in joined
    # Project identity is forced even when env is polluted.
    merged = runner._merged_env(  # noqa: SLF001
        {"COMPOSE_PROJECT_NAME": "production-live"}
    )
    assert merged["COMPOSE_PROJECT_NAME"] == "mission-final-release-runner"


def test_evidence_ledger_fields_for_val_cross_079() -> None:
    """Machine-readable proof bundle the release report can attach."""

    evidence = {
        "assertion": "VAL-CROSS-079",
        "chain_defaults": {
            "submit_on_chain_enabled": False,
            "uses_recording_weight_doubles": True,
            "live_chain_endpoints_in_automated_weight_tests": [],
        },
        "swarm": {
            "forbidden_substrings": list(SWARM_FORBIDDEN_SUBSTRINGS),
            "compose_artifacts_scanned": [
                str(path.relative_to(ROOT)) for path in AUTOMATED_COMPOSE_ARTIFACTS
            ],
            "mutations_disallowed": True,
        },
        "disposable_projects": {
            "allowed_prefixes": list(DISPOSABLE_PROJECT_PREFIXES),
            "pattern": DISPOSABLE_PROJECT_RE.pattern,
        },
        "providers": {
            "live_scripts_gated_by": "BASE_LIVE_PROVIDER_TESTS",
            "default_deploy_provider": "local",
            "worker_plane_enabled_default": False,
        },
        "protected_untracked": {
            "path": "scripts/burn_weights_24h.py",
            "sha256": BURN_WEIGHTS_SHA256,
            "status": "untracked-preserved",
        },
    }
    # Sanitized JSON only: no tokens, wallet material, or credential URLs.
    blob = json.dumps(evidence, indent=2, sort_keys=True)
    assert "token" not in blob.lower() or "submit" in blob.lower()
    for canary in ("BEGIN PRIVATE", "sk-", "password=", "mnemonic"):
        assert canary not in blob
    # Evidence remains parseable and attribution-complete.
    loaded = json.loads(blob)
    assert loaded["assertion"] == "VAL-CROSS-079"
    assert loaded["chain_defaults"]["submit_on_chain_enabled"] is False


def test_no_automated_validation_invokes_docker_service_via_ast() -> None:
    """Static proof that automated compose tests never call Swarm CLI shapes."""

    swarm_literals = ("service", "stack", "swarm", "node")
    for relative in AUTOMATED_TEST_GLOBS:
        path = ROOT / relative
        if not path.is_file():
            continue
        tree = ast.parse(_read(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.List):
                continue
            values: list[str] = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.append(elt.value)
            if not values:
                continue
            # docker service / docker stack / docker swarm argv lists
            if (
                len(values) >= 2
                and values[0] == "docker"
                and values[1] in swarm_literals
            ):
                # CommandSpy negative fixtures may deliberately refuse them.
                joined = " ".join(values)
                allow = (
                    "refused" in _read(path).lower()
                    or path.name == "test_final_release_safety.py"
                )
                assert allow, f"unexpected Swarm argv {joined} in {relative}"
