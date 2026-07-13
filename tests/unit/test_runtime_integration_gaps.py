from __future__ import annotations

from pathlib import Path

from base.master.docker_orchestrator import (
    ChallengeSpec,
    DockerOrchestrator,
)
from base.master.registry import FileChallengeRegistry
from base.schemas.challenge import ChallengeCreate, ChallengeStatus
from base.security.admin_auth import read_secret


def test_file_registry_reloads_between_instances(tmp_path: Path) -> None:
    state_file = tmp_path / "registry.json"
    secret_dir = tmp_path / "secrets"

    writer = FileChallengeRegistry(state_file, secret_dir=secret_dir)
    writer.create(
        ChallengeCreate(
            slug="demo",
            name="Demo",
            image=f"ghcr.io/baseintelligence/demo:1.0.0@sha256:{'d' * 64}",
            version="1.0.0",
        )
    )

    reader = FileChallengeRegistry(state_file, secret_dir=secret_dir)
    assert reader.list() == [reader.get("demo")]

    writer.set_status("demo", ChallengeStatus.ACTIVE)
    assert reader.registry_response().challenges[0].slug == "demo"


def test_file_registry_persists_challenge_token_file(tmp_path: Path) -> None:
    registry = FileChallengeRegistry(
        tmp_path / "registry.json",
        secret_dir=tmp_path / "secrets",
    )
    _record, token = registry.create(
        ChallengeCreate(
            slug="demo",
            name="Demo",
            image=f"ghcr.io/baseintelligence/demo:1.0.0@sha256:{'d' * 64}",
            version="1.0.0",
        )
    )

    restarted = FileChallengeRegistry(
        tmp_path / "registry.json",
        secret_dir=tmp_path / "secrets",
    )
    assert restarted.get_token("demo") == token


def test_orchestrator_sets_challenge_shared_token_file(tmp_path: Path) -> None:
    orchestrator = DockerOrchestrator(secret_dir=tmp_path)
    env = orchestrator._build_environment(  # noqa: SLF001
        ChallengeSpec(
            slug="demo",
            image="ghcr.io/baseintelligence/demo:1.0.0@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            challenge_token="secret",
        )
    )
    assert env["CHALLENGE_TOKEN_FILE"] == "/run/secrets/base/challenge_token"
    assert env["CHALLENGE_SHARED_TOKEN_FILE"] == "/run/secrets/base/challenge_token"


def test_read_secret_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_secret(file_path=str(tmp_path / "missing")) == ""
