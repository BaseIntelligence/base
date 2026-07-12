"""Installed artifact + Compose-only release contracts (VAL-CROSS-075, 077).

Black-box checks against delivered compose artifacts, immutable image pins,
challenge adoption activation, one-time credential disclosure, and operator
navigation surfaces. Live long-lived ``docker compose up`` remains optional when
mission images are unavailable; the asserted contracts are exerciseable without
mutating live Swarm or chain.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from base.master.app_admin import create_admin_app
from base.master.challenge_adoption import (
    ChallengeAdoptionError,
    require_digest_pinned_image,
    validate_payload_for_registration,
    validate_record_for_activation,
)
from base.master.compose_backend import ComposeChallengeOrchestrator
from base.master.docker_orchestrator import DockerOrchestrationError
from base.master.registry import ChallengeRegistry, record_to_admin_view
from base.schemas.challenge import ChallengeCreate, ChallengeStatus, ChallengeUpdate

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "deploy" / "compose"
MASTER_COMPOSE = COMPOSE_DIR / "docker-compose.yml"
VALIDATOR_COMPOSE = COMPOSE_DIR / "docker-compose.validator.yml"
INSTALL_MASTER = COMPOSE_DIR / "install-master.sh"
INSTALL_VALIDATOR = COMPOSE_DIR / "install-validator.sh"
BACKUP_MASTER = COMPOSE_DIR / "backup-master.sh"
RESTORE_MASTER = COMPOSE_DIR / "restore-master.sh"
TEARDOWN_MASTER = COMPOSE_DIR / "teardown-master.sh"
BACKUP_CHALLENGE = COMPOSE_DIR / "backup-challenge.sh"
COMPOSE_DOCS = ROOT / "docs" / "compose.md"
DEPLOY_DOCS = ROOT / "docs" / "deploy.md"
README = ROOT / "README.md"
DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[a-f0-9]{64}$")
SWARM_FORBIDDEN = (
    "docker service",
    "docker stack",
    "docker swarm",
    "docker node",
    "docker secret create",
)


def _sha256(fill: str = "a") -> str:
    return fill * 64


def _pinned(
    repo: str = "ghcr.io/baseintelligence/demo",
    tag: str = "1.2.3",
    fill: str = "a",
) -> str:
    return f"{repo}:{tag}@sha256:{_sha256(fill)}"


def _secret_env(tmp_path: Path, project: str = "mission-e2e-compose") -> dict[str, str]:
    secrets = tmp_path / "secrets"
    config = tmp_path / "config"
    secrets.mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)
    secrets.chmod(0o700)
    config.chmod(0o700)
    admin = secrets / "admin_token"
    postgres = secrets / "postgres_password"
    prism = secrets / "prism_shared_token"
    master_config = config / "master.yaml"
    for path, value in (
        (admin, "CANARY-E2E-ADMIN"),
        (postgres, "CANARY-E2E-PG"),
        (prism, "CANARY-E2E-PRISM"),
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
                f"COMPOSE_PROJECT_NAME={project}",
                "BASE_MASTER_IMAGE_REPOSITORY=registry.example/base-master",
                f"BASE_MASTER_IMAGE_DIGEST={_sha256('1')}",
                "PRISM_IMAGE_REPOSITORY=registry.example/prism",
                f"PRISM_IMAGE_DIGEST={_sha256('2')}",
                "POSTGRES_IMAGE_REPOSITORY=registry.example/postgres",
                f"POSTGRES_IMAGE_DIGEST={_sha256('3')}",
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
        "COMPOSE_PROJECT_NAME": project,
        "BASE_MASTER_IMAGE_REPOSITORY": "registry.example/base-master",
        "BASE_MASTER_IMAGE_DIGEST": _sha256("1"),
        "PRISM_IMAGE_REPOSITORY": "registry.example/prism",
        "PRISM_IMAGE_DIGEST": _sha256("2"),
        "POSTGRES_IMAGE_REPOSITORY": "registry.example/postgres",
        "POSTGRES_IMAGE_DIGEST": _sha256("3"),
        "BASE_MASTER_CONFIG": str(master_config),
        "BASE_ADMIN_TOKEN_FILE": str(admin),
        "BASE_POSTGRES_PASSWORD_FILE": str(postgres),
        "PRISM_SHARED_TOKEN_FILE": str(prism),
        "BASE_MASTER_HOST_PORT": "3180",
        "BASE_COMPOSE_ENV_FILE": str(env_file),
    }


def _render(compose_file: Path, env: dict[str, str]) -> dict[str, Any]:
    rendered = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "config", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(rendered.stdout)


def _admin_client(registry: ChallengeRegistry, token: str = "admin-e2e") -> TestClient:
    class Controller:
        async def pull(self, slug: str):  # noqa: ANN201
            from base.schemas.challenge import RuntimeOperationResponse

            return RuntimeOperationResponse(slug=slug, operation="pull", status="ok")

        async def restart(self, slug: str):  # noqa: ANN201
            from base.schemas.challenge import RuntimeOperationResponse

            return RuntimeOperationResponse(slug=slug, operation="restart", status="ok")

        async def status(self, slug: str):  # noqa: ANN201
            from base.schemas.challenge import RuntimeOperationResponse

            return RuntimeOperationResponse(slug=slug, operation="status", status="ok")

    app = create_admin_app(
        registry=registry,
        runtime_controller=Controller(),
        admin_token_provider=lambda: token,
        enforce_production_policy=True,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Delivered Compose/install artifacts (Compose-only operator path)
# ---------------------------------------------------------------------------


def test_operator_navigation_documents_compose_only_install_paths() -> None:
    """VAL-CROSS-077: fresh operators discover Compose install surfaces."""

    for path in (COMPOSE_DOCS, DEPLOY_DOCS, README, INSTALL_MASTER, INSTALL_VALIDATOR):
        assert path.is_file(), path
    compose_docs = COMPOSE_DOCS.read_text(encoding="utf-8")
    deploy_docs = DEPLOY_DOCS.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for blob in (compose_docs, deploy_docs, readme):
        assert "install-master.sh" in blob
        assert "install-validator.sh" in blob
        assert "Docker Compose" in blob or "docker compose" in blob
    # Historical Swarm remains labeled non-target; no ship-of-record path.
    assert "not a supported" in deploy_docs.lower() or "HISTORICAL" in (
        ROOT / "deploy" / "swarm" / "README.md"
    ).read_text(encoding="utf-8")
    # Removed gateway/swarm operator verbs are absent from install helpers.
    install_master = INSTALL_MASTER.read_text(encoding="utf-8")
    install_validator = INSTALL_VALIDATOR.read_text(encoding="utf-8")
    for script in (install_master, install_validator):
        for forbidden in SWARM_FORBIDDEN:
            assert forbidden not in script
        assert "gateway" not in script.lower()


def test_master_and_validator_compose_are_digest_pinned_and_isolated(
    tmp_path: Path,
) -> None:
    """Compose-only boundaries: master+postgres+prism vs independent validator."""

    master_env = _secret_env(tmp_path / "master", "mission-e2e-master")
    quiet = subprocess.run(
        ["docker", "compose", "-f", str(MASTER_COMPOSE), "config", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        env=master_env,
    )
    assert quiet.returncode == 0, quiet.stderr
    master = _render(MASTER_COMPOSE, master_env)
    assert set(master["services"]) == {
        "base-master-validator",
        "master-postgres",
        "challenge-prism",
    }
    for name, svc in master["services"].items():
        assert DIGEST_IMAGE_RE.match(svc["image"]), (name, svc["image"])
        assert "build" not in svc
    # App+db isolation: postgres has no host ports; prism not on public/db.
    postgres = master["services"]["master-postgres"]
    assert not postgres.get("ports")
    blob = json.dumps(master).lower()
    assert "docker.sock" in blob  # master watches compose; sock is expected
    assert "gateway" not in blob
    assert "docker service" not in blob

    # Validator project is fully independent.
    vcfg = tmp_path / "validator.yaml"
    identity = tmp_path / "identity"
    broker = tmp_path / "broker"
    vcfg.write_text("validator:\n  agent:\n    master_url: http://127.0.0.1:3180\n")
    identity.mkdir()
    broker.write_text("tok\n")
    validator_env = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": "mission-e2e-validator",
        "BASE_VALIDATOR_IMAGE_REPOSITORY": "registry.example/base-validator-runtime",
        "BASE_VALIDATOR_IMAGE_DIGEST": _sha256("4"),
        "BASE_VALIDATOR_CONFIG": str(vcfg),
        "BASE_VALIDATOR_PROTOCOL_IDENTITY": str(identity),
        "BASE_VALIDATOR_BROKER_TOKEN": str(broker),
    }
    validator = _render(VALIDATOR_COMPOSE, validator_env)
    assert set(validator["services"]) == {"validator"}
    vsvc = validator["services"]["validator"]
    assert DIGEST_IMAGE_RE.match(vsvc["image"])
    vblob = json.dumps(validator).lower()
    assert "docker.sock" not in vblob
    assert "postgres" not in vblob
    assert "challenge" not in vblob
    assert "gateway" not in vblob
    for forbidden in SWARM_FORBIDDEN:
        assert forbidden not in vblob


def test_ops_scripts_are_compose_only_and_executable() -> None:
    for path in (
        INSTALL_MASTER,
        INSTALL_VALIDATOR,
        BACKUP_MASTER,
        RESTORE_MASTER,
        TEARDOWN_MASTER,
        BACKUP_CHALLENGE,
    ):
        assert path.is_file()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & stat.S_IXUSR, path
        text = path.read_text(encoding="utf-8")
        for forbidden in SWARM_FORBIDDEN:
            assert forbidden not in text
        assert "set_weights" not in text


# ---------------------------------------------------------------------------
# VAL-CROSS-075: immutable challenge adoption contract
# ---------------------------------------------------------------------------


def test_adoption_rejects_mutable_image_and_unsafe_policy() -> None:
    with pytest.raises(ChallengeAdoptionError, match="digest-pinned"):
        require_digest_pinned_image("ghcr.io/baseintelligence/demo:latest")
    with pytest.raises(ChallengeAdoptionError, match="digest"):
        require_digest_pinned_image("ghcr.io/baseintelligence/demo:1.2.3")
    # Missing tag before digest is also rejected.
    with pytest.raises(ChallengeAdoptionError):
        require_digest_pinned_image(f"ghcr.io/baseintelligence/demo@sha256:{_sha256()}")

    ok = require_digest_pinned_image(_pinned())
    assert ok.endswith(_sha256())

    with pytest.raises(ChallengeAdoptionError, match="capability"):
        validate_payload_for_registration(
            ChallengeCreate(
                slug="bad-cap",
                name="Bad",
                image=_pinned(),
                version="1.0.0",
                required_capabilities=["validator.own_set_weights"],
            ),
            production_policy=True,
        )
    with pytest.raises(ChallengeAdoptionError, match="volume|bind|socket"):
        validate_payload_for_registration(
            ChallengeCreate(
                slug="bad-vol",
                name="Bad",
                image=_pinned(),
                version="1.0.0",
                volumes={"data": "/var/run/docker.sock"},
            ),
            production_policy=True,
        )


def test_production_registry_activation_requires_digest_and_returns_token_once() -> (
    None
):
    registry = ChallengeRegistry(production_policy=True)
    # Production create requires digests.
    with pytest.raises((ChallengeAdoptionError, ValueError), match="digest|tag"):
        registry.create(
            ChallengeCreate(
                slug="mutable-draft",
                name="Mutable",
                image="ghcr.io/baseintelligence/demo:latest",
                version="1.0.0",
            )
        )

    record, token = registry.create(
        ChallengeCreate(
            slug="prism-e2e",
            name="Prism E2E",
            image=_pinned("ghcr.io/baseintelligence/prism", "3.1.2", "e"),
            version="3.1.2",
            emission_percent=Decimal("25"),
            required_capabilities=["get_weights", "proxy_routes", "challenge.scoring"],
            volumes={"sqlite": "base_prism_e2e_sqlite"},
            secrets=["challenge_token"],
            env={"PRISM_COMBINED_MODE": "true"},
            metadata={"expected_role": "challenge", "expected_health_status": "ok"},
            internal_base_url="http://challenge-prism-e2e:8080",
        )
    )
    assert record.status == ChallengeStatus.DRAFT
    assert token
    assert "…" in record.token_hint
    assert token not in record.token_hint
    admin = record_to_admin_view(record).model_dump()
    assert "challenge_token" not in admin
    assert admin["token_hint"] == record.token_hint
    # Token remains available once via get_token internal wiring.
    assert registry.get_token("prism-e2e") == token

    activated = registry.set_status("prism-e2e", ChallengeStatus.ACTIVE)
    assert activated.status == ChallengeStatus.ACTIVE
    # Admin views still never re-surface the clear credential.
    again = record_to_admin_view(registry.get("prism-e2e")).model_dump(mode="json")
    assert token not in json.dumps(again)
    assert "challenge_token" not in again


def test_admin_api_registration_matrix_activate_rejects_mutable(tmp_path: Path) -> None:
    """Admin CLI/API black-box: DRAFT ok under policy, activate needs pin."""

    # enforce_production_policy=True makes create require digests too.
    registry = ChallengeRegistry(production_policy=True)
    client = _admin_client(registry)
    # Mutable rejected at create under production.
    bad = client.post(
        "/v1/admin/challenges",
        json={
            "slug": "mutable-agent",
            "name": "Mutable",
            "image": "ghcr.io/baseintelligence/demo:latest",
            "version": "1.0.0",
            "required_capabilities": ["get_weights", "proxy_routes"],
            "emission_percent": "10",
        },
        headers={"X-Admin-Token": "admin-e2e"},
    )
    assert bad.status_code == 400
    detail = bad.json()["detail"].lower()
    assert "digest" in detail or "tag" in detail

    good_body = {
        "slug": "immutable-agent",
        "name": "Immutable",
        "image": _pinned(fill="f"),
        "version": "1.0.0",
        "required_capabilities": ["get_weights", "proxy_routes"],
        "emission_percent": "10",
        "volumes": {"sqlite": "base_immutable_agent_sqlite"},
        "internal_base_url": "http://challenge-immutable-agent:8000",
        "metadata": {"expected_role": "challenge"},
    }
    created = client.post(
        "/v1/admin/challenges",
        json=good_body,
        headers={"X-Admin-Token": "admin-e2e"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["challenge"]["status"] == "draft"
    assert body["challenge_token"]
    assert body["docker_broker_token"]
    clear_token = body["challenge_token"]
    # Later GET never reissues clear token.
    fetched = client.get(
        "/v1/admin/challenges/immutable-agent",
        headers={"X-Admin-Token": "admin-e2e"},
    )
    assert fetched.status_code == 200
    assert "challenge_token" not in fetched.json()
    assert clear_token not in json.dumps(fetched.json())

    activated = client.post(
        "/v1/admin/challenges/immutable-agent/activate",
        headers={"X-Admin-Token": "admin-e2e"},
    )
    assert activated.status_code == 200
    assert activated.json()["status"] == "active"
    assert clear_token not in json.dumps(activated.json())

    # Registry public surface shows ACTIVE, never secrets.
    public = client.get("/v1/registry")
    assert public.status_code == 200
    challenges = public.json()["challenges"]
    assert any(c["slug"] == "immutable-agent" for c in challenges)
    public_blob = json.dumps(public.json())
    assert clear_token not in public_blob
    assert "token_hash" not in public_blob

    # Patch ACTIVE to mutable image is rejected and leaves status intact
    # (fail closed before mutation).
    patched = client.patch(
        "/v1/admin/challenges/immutable-agent",
        json={"image": "ghcr.io/baseintelligence/demo:latest"},
        headers={"X-Admin-Token": "admin-e2e"},
    )
    assert patched.status_code == 400
    assert registry.get("immutable-agent").image == good_body["image"]
    assert registry.get("immutable-agent").status == ChallengeStatus.ACTIVE


def test_dev_draft_can_exist_but_activate_requires_pin() -> None:
    """Non-production drafts may use local tags; activation cannot."""

    registry = ChallengeRegistry(production_policy=False)
    record, token = registry.create(
        ChallengeCreate(
            slug="local-draft",
            name="Local",
            image="localhost:5000/platform/demo:latest",
            version="dev",
        )
    )
    assert record.status == ChallengeStatus.DRAFT
    assert token
    with pytest.raises(ChallengeAdoptionError, match="digest-pinned|digest"):
        registry.set_status("local-draft", ChallengeStatus.ACTIVE)
    assert registry.get("local-draft").status == ChallengeStatus.DRAFT

    # ACTIVE-at-create with a mutable tag is also rejected (no bypass).
    with pytest.raises(ChallengeAdoptionError, match="digest-pinned|digest"):
        registry.create(
            ChallengeCreate(
                slug="local-active-mutable",
                name="Local Active",
                image="localhost:5000/platform/demo:latest",
                version="dev",
                status=ChallengeStatus.ACTIVE,
            )
        )

    registry.update(
        "local-draft",
        ChallengeUpdate(image=_pinned("localhost:5000/platform/demo", "1.0.0", "9")),
    )
    activated = registry.set_status("local-draft", ChallengeStatus.ACTIVE)
    assert activated.status == ChallengeStatus.ACTIVE


def test_compose_orchestrator_refuses_non_immutable_image(tmp_path: Path) -> None:
    del tmp_path
    orchestrator = ComposeChallengeOrchestrator(
        project_name="mission-e2e-orch",
        compose_file=MASTER_COMPOSE,
    )
    with pytest.raises(DockerOrchestrationError, match="immutable|unpinned|sha256"):
        orchestrator._require_pinned_image("ghcr.io/baseintelligence/demo:latest")  # noqa: SLF001
    orchestrator._require_pinned_image(_pinned())  # noqa: SLF001


def test_validate_record_for_activation_is_idempotent_on_compliant_record() -> None:
    registry = ChallengeRegistry()
    record, _ = registry.create(
        ChallengeCreate(
            slug="ok-activate",
            name="OK",
            image=_pinned(fill="7"),
            version="1.0.0",
            emission_percent=Decimal("5"),
            required_capabilities=["get_weights", "proxy_routes"],
            volumes={"sqlite": "base_ok_activate_sqlite"},
            internal_base_url="http://challenge-ok-activate:8000",
        )
    )
    validate_record_for_activation(record)
    validate_record_for_activation(record)  # pure, no mutation
    registry.set_status("ok-activate", ChallengeStatus.ACTIVE)
    assert registry.get("ok-activate").status == ChallengeStatus.ACTIVE


def test_master_compose_yaml_source_has_no_mutable_image_defaults() -> None:
    text = MASTER_COMPOSE.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    services = data["services"]
    for name, svc in services.items():
        image = str(svc.get("image", ""))
        # Source form uses ${REPO}@sha256:${DIGEST}; must not ship :latest.
        assert ":latest" not in image, name
        assert "sha256" in image.lower() or "@sha256:" in text
    validator_text = VALIDATOR_COMPOSE.read_text(encoding="utf-8")
    assert ":latest" not in validator_text
    assert "sha256" in validator_text.lower()
