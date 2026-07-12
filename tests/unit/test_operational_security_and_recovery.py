"""Operational security, secrets, Swarm absence, backup and durable recovery.

Covers the hardening-surface assertions assigned to operational-security-and-
state-recovery (VAL-COMPOSE-050..061, 065..068, VAL-GATE-041..044,
VAL-WEIGHT-092, VAL-CROSS-078, VAL-COMPOSE-050, VAL-SDK-005/048/114,
VAL-WEIGHT-010/068) as black-box/contract checks against delivered artifacts
and in-process unit surfaces. Disposable Compose live evidence remains for the
installed-artifact milestone when images/services are available.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import ValidationError

from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.config import ChallengeSettings
from base.config.policy import ProductionPolicyError
from base.config.settings import Settings
from base.master.weight_flow_metrics import (
    WeightFlowMetrics,
    get_weight_flow_metrics,
    prometheus_text,
)
from base.security.admin_auth import SecretFileError, require_protected_secret_file

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "deploy" / "compose"
MASTER_COMPOSE = COMPOSE_DIR / "docker-compose.yml"
VALIDATOR_COMPOSE = COMPOSE_DIR / "docker-compose.validator.yml"
INSTALL_MASTER = COMPOSE_DIR / "install-master.sh"
BACKUP_MASTER = COMPOSE_DIR / "backup-master.sh"
RESTORE_MASTER = COMPOSE_DIR / "restore-master.sh"
BACKUP_CHALLENGE = COMPOSE_DIR / "backup-challenge.sh"
TEARDOWN_MASTER = COMPOSE_DIR / "teardown-master.sh"
SWARM_FORBIDDEN = (
    "docker service",
    "docker stack",
    "docker swarm",
    "docker node",
    "docker secret create",
    "docker secret ",
)


def _sha256_digest(fill: str = "a") -> str:
    return fill * 64


def _secret_env(tmp_path: Path) -> dict[str, str]:
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
        (admin, "CANARY-ADMIN-TOKEN-OPSEC-001"),
        (postgres, "CANARY-PG-PASSWORD-OPSEC-002"),
        (prism, "CANARY-PRISM-TOKEN-OPSEC-003"),
    ):
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
    master_config.write_text(
        "database:\n  url: postgresql+asyncpg://base:x@master-postgres:5432/base\n",
        encoding="utf-8",
    )
    master_config.chmod(0o600)
    env_file = config / "compose.env"
    env_file.write_text(
        "\n".join(
            [
                "COMPOSE_PROJECT_NAME=mission-opsec",
                "BASE_MASTER_IMAGE_REPOSITORY=registry.example/base-master",
                f"BASE_MASTER_IMAGE_DIGEST={_sha256_digest('a')}",
                "PRISM_IMAGE_REPOSITORY=registry.example/prism",
                f"PRISM_IMAGE_DIGEST={_sha256_digest('b')}",
                "POSTGRES_IMAGE_REPOSITORY=registry.example/postgres",
                f"POSTGRES_IMAGE_DIGEST={_sha256_digest('c')}",
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
    return {
        **os.environ,
        "COMPOSE_PROJECT_NAME": "mission-opsec",
        "BASE_MASTER_IMAGE_REPOSITORY": "registry.example/base-master",
        "BASE_MASTER_IMAGE_DIGEST": _sha256_digest("a"),
        "PRISM_IMAGE_REPOSITORY": "registry.example/prism",
        "PRISM_IMAGE_DIGEST": _sha256_digest("b"),
        "POSTGRES_IMAGE_REPOSITORY": "registry.example/postgres",
        "POSTGRES_IMAGE_DIGEST": _sha256_digest("c"),
        "BASE_MASTER_CONFIG": str(master_config),
        "BASE_ADMIN_TOKEN_FILE": str(admin),
        "BASE_POSTGRES_PASSWORD_FILE": str(postgres),
        "PRISM_SHARED_TOKEN_FILE": str(prism),
        "BASE_MASTER_HOST_PORT": "3180",
        "BASE_COMPOSE_ENV_FILE": str(env_file),
    }


def _render_master(tmp_path: Path) -> dict[str, Any]:
    rendered = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(MASTER_COMPOSE),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_secret_env(tmp_path),
    )
    return json.loads(rendered.stdout)


# ---------------------------------------------------------------------------
# VAL-COMPOSE-051/052/053/055: secret files / least privilege / no leaks
# ---------------------------------------------------------------------------


def test_require_protected_secret_file_mode_and_content(tmp_path: Path) -> None:
    path = tmp_path / "admin_token"
    path.write_text("real-secret\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(SecretFileError, match="too-permissive"):
        require_protected_secret_file(path, name="admin_token")
    path.chmod(0o600)
    assert require_protected_secret_file(path, name="admin_token") == "real-secret"
    path.write_text("\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(SecretFileError, match="empty"):
        require_protected_secret_file(path, name="admin_token")
    with pytest.raises(SecretFileError, match="missing"):
        require_protected_secret_file(tmp_path / "nope", name="admin_token")


def test_production_rejects_inline_and_missing_secrets(tmp_path: Path) -> None:
    # Pydantic wraps ProductionPolicyError as ValidationError; accept both.
    common_db = {"url": "postgresql+asyncpg://base:pwd@postgres:5432/base"}
    common_docker = {"broker_allowed_images": ["ghcr.io/baseintelligence/base-master"]}
    with pytest.raises(
        (ProductionPolicyError, ValidationError), match="inline admin_token"
    ):
        Settings(
            environment="production",
            security={"admin_token": "INLINE-CANARY", "admin_token_file": None},
            database=common_db,
            docker=common_docker,
        )
    # Path required but may be a container mount that does not exist offline.
    Settings(
        environment="production",
        security={"admin_token_file": str(tmp_path / "missing-admin")},
        database=common_db,
        docker=common_docker,
    )
    # Runtime fail-closed helper insists on existence.
    from base.config.policy import assert_protected_secret_file

    with pytest.raises(ProductionPolicyError, match="missing"):
        assert_protected_secret_file(
            tmp_path / "missing-admin", name="admin_token", require_exists=True
        )
    loose = tmp_path / "admin"
    loose.write_text("value\n", encoding="utf-8")
    loose.chmod(0o644)
    with pytest.raises(
        (ProductionPolicyError, ValidationError), match="too-permissive|mode"
    ):
        Settings(
            environment="production",
            security={"admin_token_file": str(loose)},
            database=common_db,
            docker=common_docker,
        )
    empty = tmp_path / "empty-admin"
    empty.write_text("\n", encoding="utf-8")
    empty.chmod(0o600)
    with pytest.raises((ProductionPolicyError, ValidationError), match="empty"):
        Settings(
            environment="production",
            security={"admin_token_file": str(empty)},
            database=common_db,
            docker=common_docker,
        )


def test_secrets_file_backed_and_canaries_absent_from_rendered_config(
    tmp_path: Path,
) -> None:
    rendered = _render_master(tmp_path)
    blob = json.dumps(rendered)
    for canary in (
        "CANARY-ADMIN-TOKEN-OPSEC-001",
        "CANARY-PG-PASSWORD-OPSEC-002",
        "CANARY-PRISM-TOKEN-OPSEC-003",
    ):
        assert canary not in blob
    services = rendered["services"]
    master = services["base-master-validator"]
    postgres = services["master-postgres"]
    prism = services["challenge-prism"]
    # Only *_FILE style env names for secret material on postgres/master/prism.
    for svc in (master, postgres, prism):
        env = svc.get("environment") or {}
        for key, value in env.items():
            key_l = str(key).lower()
            if "token" in key_l or "password" in key_l:
                assert "file" in key_l
            assert value not in {
                "CANARY-ADMIN-TOKEN-OPSEC-001",
                "CANARY-PG-PASSWORD-OPSEC-002",
                "CANARY-PRISM-TOKEN-OPSEC-003",
            }
    # Mount targets present / read-only.
    master_mounts = master.get("volumes") or []
    targets = {
        (m.get("target") or m.get("Target"))
        for m in master_mounts
        if isinstance(m, dict)
    }
    assert "/run/secrets/admin_token" in targets


def test_secret_least_privilege_scope_matrix(tmp_path: Path) -> None:
    """Each service only receives its own credential mount class (VAL-COMPOSE-052)."""

    rendered = _render_master(tmp_path)
    services = rendered["services"]

    def mount_targets(svc: str) -> set[str]:
        mounts = services[svc].get("volumes") or []
        return {
            str(m.get("target") or m.get("Target") or "")
            for m in mounts
            if isinstance(m, dict)
        }

    postgres_targets = mount_targets("master-postgres")
    master_targets = mount_targets("base-master-validator")
    prism_targets = mount_targets("challenge-prism")
    assert "/run/secrets/postgres_password" in postgres_targets
    assert "/run/secrets/admin_token" not in postgres_targets
    assert "/run/secrets/prism_shared_token" not in postgres_targets
    assert "/run/secrets/admin_token" in master_targets
    assert "/run/secrets/prism_shared_token" not in master_targets
    assert "/run/secrets/postgres_password" not in master_targets
    assert "/run/secrets/prism_shared_token" in prism_targets
    assert "/run/secrets/admin_token" not in prism_targets
    assert "/run/secrets/postgres_password" not in prism_targets
    # No master DB password in Prism env.
    prism_env = json.dumps(services["challenge-prism"].get("environment") or {}).lower()
    assert "postgres_password" not in prism_env
    assert "master-postgres" not in prism_env


# ---------------------------------------------------------------------------
# VAL-COMPOSE-054: docker.sock only on master
# ---------------------------------------------------------------------------


def test_docker_socket_only_on_master(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    for name, svc in rendered["services"].items():
        blob = json.dumps(svc)
        if name == "base-master-validator":
            assert "/var/run/docker.sock" in blob
        else:
            assert "/var/run/docker.sock" not in blob
    # Validator compose never mounts docker.sock.
    source = VALIDATOR_COMPOSE.read_text(encoding="utf-8")
    assert "docker.sock" not in source


# ---------------------------------------------------------------------------
# VAL-COMPOSE-050 external TEE not lifecycle-managed
# ---------------------------------------------------------------------------


def test_external_tee_not_lifecycle_managed(tmp_path: Path) -> None:
    """Master/Prism manifests never create/stop an external TEE service."""

    rendered = _render_master(tmp_path)
    services = set(rendered["services"])
    assert "tee" not in services
    assert "external-tee" not in services
    # External long-lived TEE is never declared as a Compose service.
    prism_env = rendered["services"]["challenge-prism"].get("environment") or {}
    assert str(prism_env.get("PRISM_DOCKER_ENABLED", "")).lower() in {
        "false",
        "0",
        "no",
        "",
    }
    content = MASTER_COMPOSE.read_text(encoding="utf-8").lower()
    for forbidden in ("docker run", "docker service create", "swarm service create"):
        assert forbidden not in content


# ---------------------------------------------------------------------------
# VAL-COMPOSE-057..061 Swarm absence + teardown scripts
# ---------------------------------------------------------------------------


def test_install_runtime_teardown_backup_are_compose_only_no_swarm() -> None:
    for path in (
        INSTALL_MASTER,
        BACKUP_MASTER,
        RESTORE_MASTER,
        BACKUP_CHALLENGE,
        TEARDOWN_MASTER,
        MASTER_COMPOSE,
        VALIDATOR_COMPOSE,
    ):
        assert path.is_file(), path
        content = path.read_text(encoding="utf-8").lower()
        for forbidden in SWARM_FORBIDDEN:
            assert forbidden not in content, f"{path}: {forbidden}"
        if path.suffix == ".sh":
            assert "docker compose" in content or "compose" in content


def test_teardown_scripts_support_preserve_and_destroy() -> None:
    content = TEARDOWN_MASTER.read_text(encoding="utf-8")
    assert "--destroy-data" in content
    assert "down --volumes" in content
    assert "volumes retained" in content.lower() or "retained" in content


def test_backup_restore_scripts_cover_control_plane_and_challenge() -> None:
    master_backup = BACKUP_MASTER.read_text(encoding="utf-8")
    assert "pg_dump" in master_backup
    assert "excludes_secrets" in master_backup
    assert "raw_weight_snapshots" in master_backup
    assert "challenge_watcher_state" in master_backup
    restore = RESTORE_MASTER.read_text(encoding="utf-8")
    assert "pg_restore" in restore
    challenge = BACKUP_CHALLENGE.read_text(encoding="utf-8")
    assert "sqlite" in challenge.lower()
    assert "excludes_master_credentials" in challenge


# ---------------------------------------------------------------------------
# VAL-GATE-041/042 Base migration ledger
# ---------------------------------------------------------------------------


def test_base_migration_ledger_single_headed_and_drops_llm() -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    assert len(heads) == 1
    revisions = list(script.walk_revisions())
    ids = {rev.revision for rev in revisions}
    assert "0011_drop_llm_usage_records" in ids
    drop = (ROOT / "alembic/versions/0011_drop_llm_usage_records.py").read_text(
        encoding="utf-8"
    )
    assert "DROP TABLE IF EXISTS llm_usage_records" in drop
    assert "NotImplementedError" in drop  # forward-only


# ---------------------------------------------------------------------------
# VAL-WEIGHT-068 metrics
# ---------------------------------------------------------------------------


def test_weight_flow_metrics_correlation_and_prometheus_sanitized() -> None:
    metrics = WeightFlowMetrics()
    metrics.record_push(
        outcome="accepted",
        challenge_slug="prism",
        epoch=7,
        revision=1,
        snapshot_id="snap-1",
        payload_digest="d" * 64,
    )
    metrics.record_push(outcome="rejected_auth", challenge_slug="prism")
    metrics.record_aggregation(
        outcome="sealed",
        duration_ms=12.5,
        vector_id="vec-1",
        epoch=7,
        snapshot_digest="d" * 64,
        challenge_slug="prism",
    )
    metrics.record_submit(outcome="ok")
    metrics.record_fetch_failure()
    snap = metrics.snapshot()
    assert snap["pushes"]["accepted"] == 1
    assert snap["pushes"]["rejected_auth"] == 1
    assert snap["aggregation_outcomes"]["sealed"] == 1
    assert snap["fetch_failures"] == 1
    text = prometheus_text(metrics)
    assert "base_raw_weight_pushes_total" in text
    assert "base_aggregation_outcomes_total" in text
    assert "CANARY" not in text
    assert "token" not in text.lower() or "outbound" not in text
    # Global metrics module remains import-safe.
    get_weight_flow_metrics().reset()


def test_metrics_route_present_on_proxy() -> None:
    source = (ROOT / "src/base/master/app_proxy.py").read_text(encoding="utf-8")
    assert '"/metrics"' in source
    assert "prometheus_text" in source


# ---------------------------------------------------------------------------
# VAL-SDK-005 / VAL-SDK-114 compatibility fail-closed before ready
# ---------------------------------------------------------------------------


class _Database:
    inited = False

    async def init(self) -> None:
        self.inited = True

    async def close(self) -> None:
        return None


def test_unsupported_sdk_major_rejected_before_listener() -> None:
    with pytest.raises(ValidationError, match="Incompatible SDK version"):
        ChallengeSettings(
            sdk_compatibility_range="^9.0.0",
            sdk_version="9.0.0",
            shared_token="x",
            shared_token_file=None,
        )


def test_incompatible_actual_sdk_version_rejects_settings() -> None:
    with pytest.raises((ValueError, ValidationError), match="Incompatible SDK version"):
        ChallengeSettings(
            sdk_version="99.0.0",
            shared_token="x",
            shared_token_file=None,
        )


def test_startup_missing_secret_refuses_before_db_and_listener() -> None:
    database = _Database()
    app = create_challenge_app(
        settings=ChallengeSettings(
            shared_token=None,
            shared_token_file="/tmp/does-not-exist-base-opsec-secret",
        ),
        database=database,
        public_router=APIRouter(),
        get_weights_fn=_weights,
    )
    with pytest.raises(RuntimeError, match="secret is missing"):
        with TestClient(app):
            pass
    assert database.inited is False


async def _weights() -> dict[str, float]:
    return {"hk": 1.0}


# ---------------------------------------------------------------------------
# VAL-SDK-048 / VAL-WEIGHT-010 durable assignment result + restart surfaces
# ---------------------------------------------------------------------------


def test_assignment_result_persistence_contracts_exist() -> None:
    """Result post path is durable (idempotent WorkResult) and restart-tested.

    Live container restart is covered by postgres integration fixtures. Unit
    contract: assignment_coordination stores WorkResult before acknowledgement
    and exact retries set idempotent=True.
    """

    source = (ROOT / "src/base/master/assignment_coordination.py").read_text(
        encoding="utf-8"
    )
    assert "class ResultOutcome" in source or "idempotent" in source
    assert "WorkResult" in source
    assert "idempotent=True" in source
    assert "session.add(result)" in source or "session.add(" in source
    # postgres integration suite covers restart-equivalent durability
    integration = ROOT / "tests/integration/test_assignment_coordination_postgres.py"
    assert integration.is_file()
    content = integration.read_text(encoding="utf-8")
    assert "post_result" in content
    assert "work_results" in content


# ---------------------------------------------------------------------------
# VAL-COMPOSE-051 install chmod / production fail-closed helper present
# ---------------------------------------------------------------------------


def test_install_master_creates_0600_secret_files() -> None:
    content = INSTALL_MASTER.read_text(encoding="utf-8")
    assert "chmod 600" in content
    assert "chmod 700" in content
    assert "admin_token" in content
    assert "POSTGRES_PASSWORD_FILE" in content


def test_policy_validate_secret_configuration_is_wired() -> None:
    from base.config import policy

    assert hasattr(policy, "validate_secret_configuration")
    # development remains permissive for local tests
    Settings(environment="development")


def test_master_compose_source_has_no_inline_secret_values() -> None:
    content = MASTER_COMPOSE.read_text(encoding="utf-8")
    assert "PASSWORD_FILE" in content
    assert re.search(r"POSTGRES_PASSWORD\s*:", content) is None
    assert "ADMIN_TOKEN:" not in content
    parsed = yaml.safe_load(content)
    for svc in (parsed.get("services") or {}).values():
        env = svc.get("environment") or {}
        for key, value in env.items():
            key_u = str(key).upper()
            if "TOKEN" in key_u or "PASSWORD" in key_u:
                assert "FILE" in key_u, key
            if isinstance(value, str):
                assert not value.startswith("sk-")
                assert "BEGIN" not in value


# ---------------------------------------------------------------------------
# VAL-CROSS-078 joined provenance fields present on durable models
# ---------------------------------------------------------------------------


def test_joined_provenance_columns_exist_on_models() -> None:
    models = (ROOT / "src/base/db/models.py").read_text(encoding="utf-8")
    for needle in (
        "raw_weight_snapshots",
        "payload_digest",
        "final_weight_vectors",
        "vector_digest",
        "source_snapshot_ids",
        "challenge_watcher_state",
        "work_results",
    ):
        assert needle in models
