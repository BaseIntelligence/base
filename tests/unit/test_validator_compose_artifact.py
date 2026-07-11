from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.validator.yml"


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

    assert service["image"] == (
        "registry.example/base-validator-runtime@sha256:" + "a" * 64
    )
    assert "build" not in service
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]

    mounts = service["volumes"]
    targets = {mount["target"] for mount in mounts}
    assert targets == {
        "/run/base/validator.yaml",
        "/run/secrets/base_broker_token",
        "/var/lib/base/identity",
        "/var/lib/base/state",
    }
    assert "/var/run/docker.sock" not in json.dumps(rendered)
    assert "postgres" not in json.dumps(rendered).lower()
    assert str(ROOT) not in json.dumps(rendered)


def test_validator_projects_render_distinct_network_and_state(tmp_path: Path) -> None:
    first = _render(tmp_path, "validator-a")
    second = _render(tmp_path, "validator-b")

    assert first["networks"]["validator"]["name"] == "validator-a_network"
    assert second["networks"]["validator"]["name"] == "validator-b_network"
    assert first["volumes"]["validator-state"]["name"] == "validator-a_state"
    assert second["volumes"]["validator-state"]["name"] == "validator-b_state"
