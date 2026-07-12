from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.validator.yml"
INSTALL_VALIDATOR = ROOT / "deploy" / "compose" / "install-validator.sh"
COMPOSE_DOCS = ROOT / "docs" / "compose.md"
DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[a-f0-9]{64}$")


def _render(tmp_path: Path, project_name: str) -> dict[str, Any]:
    config = tmp_path / f"{project_name}.yaml"
    identity = tmp_path / f"{project_name}-identity"
    broker_token = tmp_path / f"{project_name}-broker-token"
    config.write_text("{}\n", encoding="utf-8")
    identity.mkdir()
    broker_token.write_text("test-token\n", encoding="utf-8")
    environment = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": project_name,
        "BASE_VALIDATOR_IMAGE_REPOSITORY": "registry.example/base-validator-runtime",
        "BASE_VALIDATOR_IMAGE_DIGEST": "a" * 64,
        "BASE_VALIDATOR_CONFIG": str(config),
        "BASE_VALIDATOR_PROTOCOL_IDENTITY": str(identity),
        "BASE_VALIDATOR_BROKER_TOKEN": str(broker_token),
    }
    rendered = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(rendered.stdout)


def test_validator_compose_is_source_free_and_digest_pinned(tmp_path: Path) -> None:
    rendered = _render(tmp_path, "validator-a")
    service = rendered["services"]["validator"]

    assert DIGEST_IMAGE_RE.match(service["image"])
    assert service["image"] == (
        "registry.example/base-validator-runtime@sha256:" + "a" * 64
    )
    assert "build" not in service
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert set(rendered["services"]) == {"validator"}

    mounts = service["volumes"]
    targets = {mount["target"] for mount in mounts}
    assert targets == {
        "/run/base/validator.yaml",
        "/run/secrets/base_broker_token",
        "/var/lib/base/identity",
        "/var/lib/base/state",
    }
    blob = json.dumps(rendered).lower()
    assert "/var/run/docker.sock" not in blob
    assert "postgres" not in blob
    assert "challenge" not in blob
    assert "gateway" not in blob
    assert "docker service" not in blob
    assert "docker stack" not in blob
    assert str(ROOT) not in json.dumps(rendered)


def test_validator_projects_render_distinct_network_and_state(tmp_path: Path) -> None:
    first = _render(tmp_path, "validator-a")
    second = _render(tmp_path, "validator-b")

    assert first["networks"]["validator"]["name"] == "validator-a_network"
    assert second["networks"]["validator"]["name"] == "validator-b_network"
    assert first["volumes"]["validator-state"]["name"] == "validator-a_state"
    assert second["volumes"]["validator-state"]["name"] == "validator-b_state"
    # Distinct project names => distinct external resource names.
    first_net = first["networks"]["validator"]["name"]
    second_net = second["networks"]["validator"]["name"]
    assert first_net != second_net
    assert (
        first["volumes"]["validator-state"]["name"]
        != second["volumes"]["validator-state"]["name"]
    )


def test_validator_compose_has_no_master_or_challenge_services(tmp_path: Path) -> None:
    rendered = _render(tmp_path, "validator-isolation")
    services = set(rendered["services"])
    assert services == {"validator"}
    for forbidden in (
        "base-master-validator",
        "master-postgres",
        "challenge-prism",
        "evaluator",
        "broker",
        "submitter",
        "watcher",
    ):
        assert forbidden not in services


def test_validator_compose_config_quiet_from_clean_paths(tmp_path: Path) -> None:
    config = tmp_path / "validator.yaml"
    identity = tmp_path / "identity"
    token = tmp_path / "broker"
    config.write_text("role: validator\n", encoding="utf-8")
    identity.mkdir()
    token.write_text("tok\n", encoding="utf-8")
    env = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": "clean-validator-config",
        "BASE_VALIDATOR_IMAGE_REPOSITORY": "registry.example/base-validator-runtime",
        "BASE_VALIDATOR_IMAGE_DIGEST": "b" * 64,
        "BASE_VALIDATOR_CONFIG": str(config),
        "BASE_VALIDATOR_PROTOCOL_IDENTITY": str(identity),
        "BASE_VALIDATOR_BROKER_TOKEN": str(token),
    }
    quiet = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),  # run away from source tree root
    )
    assert quiet.returncode == 0, quiet.stderr


def test_install_validator_script_is_compose_only_and_executable() -> None:
    assert INSTALL_VALIDATOR.is_file()
    mode = INSTALL_VALIDATOR.stat().st_mode
    assert mode & stat.S_IXUSR
    content = INSTALL_VALIDATOR.read_text(encoding="utf-8")
    assert "docker compose" in content
    assert "--master-url" in content
    assert "docker-compose.validator.yml" in content
    for forbidden in (
        "docker service",
        "docker stack",
        "docker swarm",
        "docker secret create",
        "kubectl",
        "helm",
    ):
        assert forbidden not in content


def test_install_validator_requires_master_url() -> None:
    result = subprocess.run(
        ["bash", str(INSTALL_VALIDATOR)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATOR_MASTER_URL": ""},
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "master-url" in combined or "master_url" in combined


def test_compose_docs_document_independent_validator_install() -> None:
    docs = COMPOSE_DOCS.read_text(encoding="utf-8")
    assert "install-validator.sh" in docs
    assert "docker-compose.validator.yml" in docs
    assert "independent" in docs.lower()
    assert "docker compose" in docs.lower()
    assert "docker.sock" in docs.lower() or "docker socket" in docs.lower()
