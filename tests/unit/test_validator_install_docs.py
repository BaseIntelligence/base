from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_DOCS = [
    ROOT / "docs" / "validator.md",
    ROOT / "docs" / "validator" / "README.md",
    ROOT / "docs" / "operations" / "validator.md",
]
SUBMITTER_DIR = ROOT / "deploy" / "swarm" / "submitter"
OPERATIONS_DOC = ROOT / "docs" / "operations" / "validator.md"
RECOMMENDED_HOTKEY = "5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_kubernetes_validator_installer_is_removed() -> None:
    # The Kubernetes/Helm validator installer was removed in the Swarm migration.
    assert not (ROOT / "scripts" / "install-validator.sh").exists()
    assert not (ROOT / "deploy" / "helm").exists()


def test_validator_submitter_assets_are_committed() -> None:
    assert (SUBMITTER_DIR / "run_submitter.py").is_file()
    assert (SUBMITTER_DIR / "submitter.yaml").is_file()
    assert (SUBMITTER_DIR / "platform-submitter.service").is_file()


def test_validator_docs_are_submitter_only_and_kubernetes_tooling_free() -> None:
    for path in VALIDATOR_DOCS:
        content = _read(path)
        lowered = content.lower()
        assert "kubectl" not in lowered, f"kubectl found in {path}"
        assert "helm" not in lowered, f"helm found in {path}"
        assert "docker compose" not in lowered, f"docker compose found in {path}"
        assert "install-validator.sh" not in content, f"k8s installer in {path}"
        assert "deploy/swarm/submitter" in content
        assert "platform-submitter.service" in content


def test_validator_docs_keep_operators_on_the_submitter() -> None:
    docs = "\n".join(_read(path) for path in VALIDATOR_DOCS)

    assert "https://chain.platform.network" in docs
    assert "/v1/weights/latest" in docs
    assert "run_submitter.py" in docs
    assert RECOMMENDED_HOTKEY in docs
    # The submit-only validator never needs coldkey material.
    assert "coldkey" in docs.lower()
    assert "There is no Kubernetes" in docs
    assert "no challenge orchestration" in docs.lower()


def test_validator_operations_use_swarm_cli() -> None:
    operations = _read(OPERATIONS_DOC)

    assert "docker service ls" in operations
    assert "platform master worker" in operations
    assert "platform-submitter.service" in operations


def test_validator_extra_is_bittensor_only_after_swarm_migration() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    validator_extra = pyproject["project"]["optional-dependencies"]["validator"]

    assert any(item.startswith("bittensor") for item in validator_extra)
    # Swarm needs no python kubernetes client; the runtime dep was dropped.
    assert not any(item.startswith("kubernetes") for item in validator_extra)
