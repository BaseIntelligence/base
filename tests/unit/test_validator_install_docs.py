"""Validator operator docs after Compose cutover."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_GUIDE = ROOT / "docs" / "validator" / "README.md"
VALIDATOR_OPS = ROOT / "docs" / "operations" / "validator.md"
COMPOSE_INSTALLER = ROOT / "deploy" / "compose" / "install-validator.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_validator_docs_are_compose_only_and_k8s_free() -> None:
    for path in (VALIDATOR_GUIDE, VALIDATOR_OPS):
        text = _read(path)
        lowered = text.lower()
        assert "docker compose" in lowered or "compose" in lowered
        assert "kubectl" not in lowered
        assert "helm" not in lowered


def test_validator_docs_document_own_wallet_submission() -> None:
    guide = _read(VALIDATOR_GUIDE)
    ops = _read(VALIDATOR_OPS)
    blob = guide + "\n" + ops
    assert "set_weights" in blob or "weights" in blob.lower()
    assert "wallet" in blob.lower()
    # Swarm service CLI is not the required path
    assert "docker service ls" not in ops
    assert "docker service create" not in guide


def test_compose_validator_installer_exists() -> None:
    assert COMPOSE_INSTALLER.is_file()
    script = _read(COMPOSE_INSTALLER)
    assert "docker compose" in script or "compose" in script.lower()
