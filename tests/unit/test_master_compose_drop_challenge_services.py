"""Master compose drop of challenge-* services (VAL-MEMB-003..006).

After embed scaffold, the shipping Compose path has only master + postgres.
Challenges bind loopback inside the master container; registry seed uses
localhost internal_base_url; watcher/reconcile stay off so missing challenge-*
services never break master health; PRISM_IMAGE is not required.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_YML = REPO_ROOT / "deploy/compose/docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / "deploy/compose/.env.example"
INSTALL_MASTER = REPO_ROOT / "deploy/compose/install-master.sh"
MASTER_COMPOSE_CFG = REPO_ROOT / "deploy/compose/config/master.compose.yaml"
DOCS_COMPOSE = REPO_ROOT / "docs/compose.md"
DOCS_MASTER = REPO_ROOT / "docs/master/README.md"
REGISTRY = REPO_ROOT / "src/base/master/registry.py"
CLI_MAIN = REPO_ROOT / "src/base/cli_app/main.py"
WATCHER = REPO_ROOT / "src/base/supervisor/challenge_watcher.py"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


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
                "COMPOSE_PROJECT_NAME=mission-compose-drop-test",
                "BASE_MASTER_IMAGE_REPOSITORY=registry.example/base-master",
                f"BASE_MASTER_IMAGE_DIGEST={'a' * 64}",
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
        "COMPOSE_PROJECT_NAME": "mission-compose-drop-test",
        "BASE_MASTER_IMAGE_REPOSITORY": "registry.example/base-master",
        "BASE_MASTER_IMAGE_DIGEST": "a" * 64,
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
            str(COMPOSE_YML),
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


# --- VAL-MEMB-003: no challenge-* services ---


def test_compose_has_no_challenge_services(tmp_path: Path) -> None:
    quiet = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_YML), "config", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        env=_secret_env(tmp_path),
    )
    assert quiet.returncode == 0, quiet.stderr
    rendered = _render_master(tmp_path)
    services = rendered["services"]
    assert set(services) == {"master-postgres", "base-master-validator"}
    for name in services:
        assert not str(name).startswith("challenge-"), name
    source = COMPOSE_YML.read_text(encoding="utf-8")
    assert "challenge-prism" not in source
    assert "challenge-agent-challenge" not in source
    assert "PRISM_IMAGE_REPOSITORY" not in source
    assert "challenge-prism-data" not in source


def test_compose_without_prism_image_env_still_renders(tmp_path: Path) -> None:
    """PRISM_IMAGE is not required for static master topology (VAL-MEMB-006)."""

    env = _secret_env(tmp_path)
    env.pop("PRISM_IMAGE_REPOSITORY", None)
    env.pop("PRISM_IMAGE_DIGEST", None)
    quiet = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_YML), "config", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert quiet.returncode == 0, quiet.stderr
    rendered = json.loads(
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_YML),
                "config",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout
    )
    assert "challenge-prism" not in rendered["services"]
    for name in ("base-master-validator", "master-postgres"):
        image = rendered["services"][name]["image"]
        assert DIGEST_IMAGE_RE.match(image), image
        digest = image.rsplit("@sha256:", 1)[1]
        assert SHA256_RE.match(digest)


# --- VAL-MEMB-004: registry seed localhost internal_base_url ---


def test_install_master_seeds_localhost_internal_base_url() -> None:
    text = INSTALL_MASTER.read_text(encoding="utf-8")
    assert '"internal_base_url": "http://127.0.0.1:18080"' in text
    assert "http://challenge-prism:8080" not in text
    assert "http://challenge-prism:8000" not in text
    assert "18081" in text  # documented AC default
    assert "default_internal_base_url" in text or "127.0.0.1:18080" in text


def test_default_internal_base_url_uses_embed_ports() -> None:
    from base.master.registry import default_internal_base_url

    assert default_internal_base_url("prism") == "http://127.0.0.1:18080"
    assert default_internal_base_url("agent-challenge") == "http://127.0.0.1:18081"
    # Unknown slug keeps historical docker DNS form for emergency dual-run.
    assert default_internal_base_url("custom-lab") == "http://challenge-custom-lab:8000"


def test_prism_seed_create_uses_localhost() -> None:
    from base.cli_app.main import prism_challenge_create

    payload = prism_challenge_create()
    assert payload.internal_base_url == "http://127.0.0.1:18080"


def test_registry_module_documents_embed_defaults() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    assert "127.0.0.1:18080" in text
    assert "127.0.0.1:18081" in text
    assert "_EMBEDDED_INTERNAL_BASE_URLS" in text


# --- VAL-MEMB-005: watcher safe without challenge-* services ---


def test_watcher_interval_default_zero_in_compose_and_install() -> None:
    compose = COMPOSE_YML.read_text(encoding="utf-8")
    install = INSTALL_MASTER.read_text(encoding="utf-8")
    cfg = MASTER_COMPOSE_CFG.read_text(encoding="utf-8")
    assert "BASE_MASTER__CHALLENGE_WATCHER_INTERVAL_SECONDS" in compose
    assert "${BASE_MASTER_CHALLENGE_WATCHER_INTERVAL_SECONDS:-0}" in compose
    assert "${BASE_MASTER_REGISTRY_RECONCILE_INTERVAL_SECONDS:-0}" in compose
    assert "challenge_watcher_interval_seconds: 0" in install
    assert "registry_reconcile_interval_seconds: 0" in install
    assert "challenge_watcher_interval_seconds: 0" in cfg


def test_watcher_lifespan_disabled_when_interval_non_positive() -> None:
    from base.config.settings import Settings
    from base.supervisor.challenge_watcher import build_challenge_watcher_lifespan

    settings = Settings()
    assert build_challenge_watcher_lifespan(settings, 0) is None
    assert build_challenge_watcher_lifespan(settings, -1) is None
    assert build_challenge_watcher_lifespan(None, 60) is None
    # Positive interval still constructs a lifespan (emergency dual-run).
    assert build_challenge_watcher_lifespan(settings, 60) is not None


def test_watcher_source_skips_inspect_errors() -> None:
    """Missing challenge container must not crash master (skip path present)."""

    text = WATCHER.read_text(encoding="utf-8")
    assert "skipped-inspect-error" in text
    assert "interval_seconds <= 0" in text


# --- VAL-MEMB-006: docs + env example; no required PRISM_IMAGE ---


def test_env_example_has_no_required_prism_image() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "BASE_MASTER_IMAGE_REPOSITORY" in text
    assert "PRISM_SHARED_TOKEN_FILE" in text
    # Not a required active pin: only commented optional historical note.
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    active = "\n".join(lines)
    assert "PRISM_IMAGE_REPOSITORY=" not in active
    assert "PRISM_IMAGE_DIGEST=" not in active
    assert "127.0.0.1:18080" in text
    assert "127.0.0.1:18081" in text
    assert "challenge-prism" in text  # mentioned as removed/absent


def test_install_does_not_require_prism_image() -> None:
    text = INSTALL_MASTER.read_text(encoding="utf-8")
    missing_msg = "PRISM_IMAGE_REPOSITORY/DIGEST unset and no local prism image found"
    assert missing_msg not in text
    assert "exit 1" in text  # still fails on missing master image
    assert "PRISM_IMAGE_* is unused" in text
    # Sealed env omits PRISM_IMAGE keys.
    assert "PRISM_IMAGE_REPOSITORY=${PRISM_IMAGE_REPOSITORY}" not in text
    assert "PRISM_SHARED_TOKEN_FILE=${PRISM_TOKEN_FILE}" in text


def test_docs_document_embed_tokens_and_no_prism_image_service() -> None:
    compose = DOCS_COMPOSE.read_text(encoding="utf-8")
    master = DOCS_MASTER.read_text(encoding="utf-8")
    blob = "\n".join((compose, master))
    assert "127.0.0.1:18080" in blob
    assert "127.0.0.1:18081" in blob
    assert "/var/lib/base/challenges" in blob
    assert "PRISM_IMAGE" in blob  # explained as not required
    assert "not required" in blob.lower() or "not required" in compose.lower()
    assert "challenge_watcher_interval_seconds" in blob or "WATCHER_INTERVAL" in blob
    # Shipping cardinality no longer lists challenge-prism as a service.
    assert "Exact cardinality" in compose or "exact cardinality" in compose.lower()
    assert "no" in compose.lower() and "challenge-prism" in compose.lower()


def test_master_mounts_shared_token_for_embedded_challenges(tmp_path: Path) -> None:
    rendered = _render_master(tmp_path)
    master = rendered["services"]["base-master-validator"]
    env = master.get("environment") or {}
    assert env.get("PRISM_SHARED_TOKEN_FILE") == "/run/secrets/prism_shared_token"
    mounts = {
        m.get("target") for m in (master.get("volumes") or []) if isinstance(m, dict)
    }
    assert "/run/secrets/prism_shared_token" in mounts
    assert "/var/lib/base" in mounts


def test_yaml_source_has_exactly_two_services() -> None:
    parsed = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    services = parsed.get("services") or {}
    assert set(services) == {"master-postgres", "base-master-validator"}
    cli = CLI_MAIN.read_text(encoding="utf-8")
    assert 'internal_base_url="http://127.0.0.1:18080"' in cli
