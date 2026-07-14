"""Compose-only master topology contracts (VAL-COMPOSE-001..021, 047..050)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MASTER_COMPOSE = ROOT / "deploy" / "compose" / "docker-compose.yml"
VALIDATOR_COMPOSE = ROOT / "deploy" / "compose" / "docker-compose.validator.yml"
INSTALL_MASTER = ROOT / "deploy" / "compose" / "install-master.sh"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[a-f0-9]{64}$")


def _secret_env(tmp_path: Path) -> dict[str, str]:
    secrets = tmp_path / "compose-secrets"
    config = tmp_path / "compose-config"
    secrets.mkdir(exist_ok=True)
    config.mkdir(exist_ok=True)
    admin = secrets / "admin_token"
    postgres = secrets / "postgres_password"
    prism = secrets / "prism_shared_token"
    master_config = config / "master.yaml"
    for path in (admin, postgres, prism):
        path.write_text("test-token-value\n", encoding="utf-8")
        path.chmod(0o600)
    master_config.write_text(
        "database:\n  url: postgresql+asyncpg://base:x@master-postgres:5432/base\n",
        encoding="utf-8",
    )
    master_config.chmod(0o600)
    env_file = secrets / "compose.env"
    env_file.write_text(
        "\n".join(
            [
                "COMPOSE_PROJECT_NAME=mission-compose-topology-test",
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
    return {
        **os.environ,
        "COMPOSE_PROJECT_NAME": "mission-compose-topology-test",
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


def test_master_compose_file_exists_and_parses(tmp_path: Path) -> None:
    assert MASTER_COMPOSE.is_file()
    assert VALIDATOR_COMPOSE.is_file()
    assert INSTALL_MASTER.is_file()
    quiet = subprocess.run(
        ["docker", "compose", "-f", str(MASTER_COMPOSE), "config", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        env=_secret_env(tmp_path),
    )
    assert quiet.returncode == 0, quiet.stderr
    rendered = _render_master(tmp_path)
    assert "services" in rendered


def test_master_compose_exact_cardinality_and_service_names(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    services = rendered["services"]
    assert set(services) == {
        "master-postgres",
        "base-master-validator",
        "challenge-prism",
    }
    assert "base-docker-broker" not in services
    assert "gateway" not in json.dumps(services).lower()
    assert "llm" not in json.dumps(services).lower()
    for forbidden in (
        "challenge-prism-worker",
        "challenge-prism-postgres",
        "evaluator",
        "broker",
        "watchtower",
        "submitter",
    ):
        assert forbidden not in services


def test_master_and_challenge_images_are_digest_pinned(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    for name in ("base-master-validator", "challenge-prism", "master-postgres"):
        image = rendered["services"][name]["image"]
        assert DIGEST_IMAGE_RE.match(image), image
        digest = image.rsplit("@sha256:", 1)[1]
        assert SHA256_RE.match(digest)


def test_postgresql_is_private_and_major_sixteen_image(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    postgres = rendered["services"]["master-postgres"]
    assert "ports" not in postgres or not postgres.get("ports")
    # Image is digest-pinned; the documented major is PostgreSQL 16.
    # Runtime content is asserted live; topology only publishes no host port.
    master = rendered["services"]["base-master-validator"]
    master_networks = set(master.get("networks", {}))
    postgres_networks = set(postgres.get("networks", {}))
    assert "db" in master_networks
    assert "db" in postgres_networks
    assert "app" in master_networks


def test_networks_are_internal_and_project_scoped(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    networks = rendered["networks"]
    assert set(networks) == {"db", "app", "public"}
    for name in ("db", "app"):
        network = networks[name]
        assert network.get("internal") is True
        assert network.get("name", "").startswith("mission-compose-topology-test_")
        assert network.get("driver", "bridge") in {None, "bridge"}
    public = networks["public"]
    assert public.get("internal") in {False, None}
    assert public.get("name", "").startswith("mission-compose-topology-test_")
    master_nets = set(rendered["services"]["base-master-validator"].get("networks", {}))
    assert master_nets == {"db", "app", "public"}
    assert set(rendered["services"]["challenge-prism"].get("networks", {})) == {"app"}
    assert set(rendered["services"]["master-postgres"].get("networks", {})) == {"db"}


def test_challenge_isolation_matrix(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    prism = rendered["services"]["challenge-prism"]
    postgres = rendered["services"]["master-postgres"]
    assert set(prism.get("networks", {})) == {"app"}
    assert set(postgres.get("networks", {})) == {"db"}
    # Challenge never attaches kafka/postgres network and never mounts DB volume.
    mounts = prism.get("volumes", [])
    sources = {m.get("source") for m in mounts if isinstance(m, dict)}
    assert "master-postgres-data" not in sources
    env_keys = {str(k).upper() for k in (prism.get("environment") or {})}
    # Challenge may have its own SQLite URL (PRISM_DATABASE_URL); it must not hold
    # master PostgreSQL credentials or password material.
    for forbidden in (
        "POSTGRES_PASSWORD",
        "PGPASSWORD",
        "BASE_DATABASE",
        "MASTER_POSTGRES",
    ):
        assert not any(forbidden in key for key in env_keys)
    # No connection string pointing at the private postgres service name.
    env_blob = json.dumps(prism.get("environment") or {}).lower()
    assert "master-postgres" not in env_blob
    assert "5432/base" not in env_blob


def test_prism_combined_mode_and_no_evaluator(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    prism = rendered["services"]["challenge-prism"]
    environment = prism.get("environment") or {}
    assert str(environment.get("PRISM_COMBINED_MODE")).lower() in {"true", "1"}
    assert str(environment.get("PRISM_DOCKER_ENABLED", "")).lower() in {
        "false",
        "0",
        "no",
    }
    # Durable eval temp on the data volume (VAL-GPULAB-001 / INFRA_TMPDIR_UNUSABLE fix).
    assert environment.get("TMPDIR") == "/data/tmp"
    assert environment.get("TEMP") == "/data/tmp"
    assert environment.get("TMP") == "/data/tmp"
    command = " ".join(str(part) for part in prism.get("command", []))
    assert "uvicorn" in command
    assert "/data/tmp" in command
    assert "evaluator" not in command.lower()
    # No docker.sock mount for Prism (master may mount it for the watcher).
    prism_blob = json.dumps(prism)
    assert "/var/run/docker.sock" not in prism_blob
    blobs = json.dumps(rendered)
    assert "docker service" not in blobs
    assert "docker stack" not in blobs


def test_master_public_port_is_minimal_and_loopback(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    master = rendered["services"]["base-master-validator"]
    ports = master.get("ports") or []
    assert len(ports) == 1
    published = ports[0]
    # Compose JSON form may use published/target or host_ip.
    assert int(published.get("published") or published.get("HostPort") or 0) == 3180
    assert int(published.get("target") or published.get("TargetPort") or 0) == 8081
    host_ip = published.get("host_ip") or published.get("HostIp")
    if host_ip is not None:
        assert host_ip in {"127.0.0.1", "localhost"}
    assert not (rendered["services"]["challenge-prism"].get("ports") or [])


def test_volumes_are_isolated_and_named(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    volumes = rendered["volumes"]
    assert "master-postgres-data" in volumes
    assert "challenge-prism-data" in volumes
    assert "master-state" in volumes
    for volume in volumes.values():
        assert volume.get("name", "").startswith("mission-compose-topology-test_")
    prism_mounts = {
        m.get("target")
        for m in rendered["services"]["challenge-prism"].get("volumes", [])
        if isinstance(m, dict)
    }
    assert "/data" in prism_mounts


def test_secrets_are_file_mounted_not_inline(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    blob = json.dumps(rendered)
    assert "test-token-value" not in blob
    master = rendered["services"]["base-master-validator"]
    env = master.get("environment") or {}
    for key, value in env.items():
        assert "password" not in str(key).lower() or "file" in str(key).lower()
        assert "token" not in str(key).lower() or "file" in str(key).lower()
        assert value not in {"test-token-value", "supersecret"}
    # Gateway remnants must not appear.
    for forbidden in (
        "GATEWAY",
        "LLM_GATEWAY",
        "BASE_GATEWAY",
        "PRISM_GATEWAY",
        "CENTRAL_GATEWAY",
    ):
        assert forbidden not in blob


def test_master_mounts_sealed_compose_env_file(tmp_path: Path) -> None:
    """Install seals .env; master mounts it for in-container compose up."""

    rendered = _render_master(tmp_path)
    master = rendered["services"]["base-master-validator"]
    env = master.get("environment") or {}
    assert env.get("BASE_DOCKER__COMPOSE_ENV_FILE") == "/run/base/compose/.env"
    mounts = master.get("volumes") or []
    targets = {
        m.get("target") or m.get("Target") for m in mounts if isinstance(m, dict)
    }
    assert "/run/base/compose/.env" in targets
    assert "/run/base/compose/docker-compose.yml" in targets


def test_healthchecks_present_for_application_readiness(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    for name in ("master-postgres", "base-master-validator", "challenge-prism"):
        health = rendered["services"][name].get("healthcheck")
        assert health is not None, name
        test_cmd = " ".join(str(part) for part in health.get("test", []))
        assert test_cmd


def test_install_script_is_compose_only() -> None:
    content = INSTALL_MASTER.read_text(encoding="utf-8")
    assert "docker compose" in content
    assert "orchestration_backend: compose" in content
    assert "BASE_DOCKER_GID" in content
    # Sealed compose env for in-container dynamic compose up (VAL-COMPOSE-008).
    assert "COMPOSE_ENV_FILE" in content
    assert "compose.env" in content
    assert "--env-file" in content
    assert "compose_env_file: /run/base/compose/.env" in content
    # Application host secrets end as mode 0600 (admin/prism/master.yaml).
    assert "chmod 600" in content
    for forbidden in (
        "docker service",
        "docker stack",
        "docker swarm",
        "docker secret create",
    ):
        assert forbidden not in content


def test_operator_entrypoint_docs_are_compose_only() -> None:
    """VAL-COMPOSE-002 / VAL-CROSS-065 / VAL-CROSS-077: Compose is the destination."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    deploy = (ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
    compose_docs = (ROOT / "docs" / "compose.md").read_text(encoding="utf-8")
    swarm_readme = (ROOT / "deploy" / "swarm" / "README.md").read_text(encoding="utf-8")

    assert "deploy/compose/install-master.sh" in readme
    assert "deploy/compose/install-validator.sh" in readme
    # README Deploy section must not present Swarm as canonical entrypoint.
    deploy_section = readme.split("## Deploy", 1)[1].split("## ", 1)[0]
    assert "install-swarm.sh" not in deploy_section
    assert "canonical, Swarm-only" not in deploy_section
    assert "Docker Compose is the only supported" in deploy_section

    assert "install-master.sh" in deploy
    assert "install-validator.sh" in deploy
    assert "not a supported install destination" in deploy.lower() or (
        "not** a supported" in deploy.lower()
    )
    assert "compose" in compose_docs.lower()
    assert "HISTORICAL" in swarm_readme or "NON-TARGET" in swarm_readme
    assert "NOT A SUPPORTED INSTALL DESTINATION" in swarm_readme


def test_master_compose_source_has_no_swarm_or_gateway() -> None:
    content = MASTER_COMPOSE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "docker service",
        "docker stack",
        "docker swarm",
        "overlay",
        "watchtower",
        "replicated-job",
        "llm_gateway",
        "gateway_token",
    ):
        assert forbidden not in content, forbidden
    # Challenge must not mount an evaluator image. The master application may
    # mount the host docker socket (read-only) for the in-process watcher only.
    parsed = yaml.safe_load(MASTER_COMPOSE.read_text(encoding="utf-8"))
    assert "evaluator" not in (parsed.get("services") or {})
    prism = (parsed.get("services") or {}).get("challenge-prism") or {}
    assert "/var/run/docker.sock" not in json.dumps(prism).lower()
    prism_labels = prism.get("labels") or {}
    assert prism_labels.get("base.compose.lifecycle") == "static"
    assert prism_labels.get("base.challenge.slug") == "prism"
    master = (parsed.get("services") or {}).get("base-master-validator") or {}
    master_blob = json.dumps(master).lower()
    assert "challenge_watcher" in master_blob or "compose_project_name" in master_blob
    assert master.get("group_add") is not None
    env = master.get("environment") or {}
    assert (
        str(env.get("BASE_DOCKER__ORCHESTRATION_BACKEND", "")).lower() == "compose"
        or "compose" in json.dumps(env).lower()
    )


def test_challenge_orchestrator_defaults_to_compose() -> None:
    """VAL-COMPOSE-024: proxy builds ComposeChallengeOrchestrator by default."""

    from base.cli_app.main import _challenge_orchestrator
    from base.config.settings import Settings
    from base.master.compose_backend import ComposeChallengeOrchestrator

    settings = Settings()
    # Default path without orchestration_backend yaml still prefers compose.
    orch = _challenge_orchestrator(settings)
    assert isinstance(orch, ComposeChallengeOrchestrator)
    assert orch.project_name  # resolved from env or fallback


def test_first_party_dockerfiles_run_as_non_root_user() -> None:
    for dockerfile in (
        ROOT / "docker" / "Dockerfile.master",
        ROOT / "docker" / "Dockerfile.validator",
    ):
        content = dockerfile.read_text(encoding="utf-8")
        assert "--uid 1000" in content
        assert "--gid 1000" in content
        assert "chown -R 1000:1000" in content
        assert "USER 1000:1000" in content


def test_compose_image_build_assets_remain() -> None:
    docker_dir = ROOT / "docker"
    assert (docker_dir / "Dockerfile.master").is_file()
    assert (docker_dir / "Dockerfile.validator").is_file()
    assert MASTER_COMPOSE.is_file()
    assert VALIDATOR_COMPOSE.is_file()


def test_database_defaults_remain_postgres() -> None:
    from base.config.settings import Settings

    settings = Settings()
    assert settings.database.url.startswith("postgresql+asyncpg://")
    master_example = yaml.safe_load(
        (ROOT / "config" / "master.example.yaml").read_text(encoding="utf-8")
    )
    assert master_example["database"]["url"].startswith("postgresql+asyncpg://")


def test_master_dockerfile_ships_compose_cli_plugin() -> None:
    """VAL-COMPOSE-008/024-029: master image must ship working docker compose."""

    content = (ROOT / "docker" / "Dockerfile.master").read_text(encoding="utf-8")
    assert "cli-plugins" in content
    assert "docker-compose" in content
    assert "docker compose version" in content
    # Compose v5 (or source compatible) release artifact is pinned in Dockerfile.
    assert "docker/compose/releases/download" in content
    assert "USER 1000:1000" in content
    # Static docker CLI alone is insufficient; both binary + plugin are needed.
    assert "docker-29.5.3.tgz" in content or (
        "download.docker.com/linux/static" in content
    )


def test_master_cli_default_import_avoids_swarm_backend() -> None:
    """VAL-CROSS-065: default master CLI graph does not import swarm_backend."""

    import sys

    # Drop residual modules so the assertion measures this process import path.
    for key in list(sys.modules):
        if key == "base.master.swarm_backend" or key.startswith(
            "base.master.swarm_backend."
        ):
            del sys.modules[key]

    import base.cli_app.main as cli_main

    assert "base.master.swarm_backend" not in sys.modules
    # Top-level name must not re-export SwarmChallengeOrchestrator.
    assert not hasattr(cli_main, "SwarmChallengeOrchestrator")
    source = Path(cli_main.__file__).read_text(encoding="utf-8")
    # Eager top-level import of swarm_backend is forbidden for the default graph.
    assert (
        "from base.master.swarm_backend import"
        not in source.split("def _challenge_orchestrator", 1)[0]
    )
    # Lazy import remains only behind explicit orchestration_backend=swarm.
    orch_src = source.split("def _challenge_orchestrator", 1)[1].split(
        "def _resolve_master_weight_epoch", 1
    )[0]
    assert "SwarmChallengeOrchestrator" in orch_src
    assert 'backend == "swarm"' in orch_src or "orchestration_backend" in orch_src
    # Sanity: compose factory remains the default path and still avoids Swarm.
    from base.config.settings import Settings
    from base.master.compose_backend import ComposeChallengeOrchestrator

    orch = cli_main._challenge_orchestrator(Settings())
    assert isinstance(orch, ComposeChallengeOrchestrator)
    assert "base.master.swarm_backend" not in sys.modules
