"""Validator operator docs after Compose cutover (minimal docs set)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_GUIDE = ROOT / "docs" / "validator.md"
COMPOSE_GUIDE = ROOT / "docs" / "compose.md"
COMPOSE_INSTALLER = ROOT / "deploy" / "compose" / "install-validator.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_validator_docs_are_compose_only_and_k8s_free() -> None:
    for path in (VALIDATOR_GUIDE, COMPOSE_GUIDE):
        text = _read(path)
        lowered = text.lower()
        assert "docker compose" in lowered or "compose" in lowered
        assert "kubectl" not in lowered
        assert "helm" not in lowered


def test_validator_docs_document_own_wallet_submission() -> None:
    guide = _read(VALIDATOR_GUIDE)
    compose = _read(COMPOSE_GUIDE)
    blob = guide + "\n" + compose
    assert "set_weights" in blob or "weights" in blob.lower()
    assert "wallet" in blob.lower()
    assert "docker service ls" not in compose
    assert "docker service create" not in guide


def test_validator_docs_document_weight_only_default() -> None:
    guide = _read(VALIDATOR_GUIDE)
    compose = _read(COMPOSE_GUIDE)
    blob = guide + "\n" + compose
    assert "weight-only" in blob.lower()
    assert "https://chain.joinbase.ai" in blob
    assert "challenge_execution_enabled" in blob
    assert "/v1/weights/latest" in blob


def test_compose_validator_installer_exists() -> None:
    assert COMPOSE_INSTALLER.is_file()
    script = _read(COMPOSE_INSTALLER)
    assert "docker compose" in script or "compose" in script.lower()
    assert "challenge_execution_enabled: false" in script
    assert "https://chain.joinbase.ai" in script
