from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"
MASTER_GUIDE = ROOT / "docs" / "master" / "README.md"
README = ROOT / "README.md"

FOUNDATION_WARNING = (
    "Foundation-only installer for Cortex Foundation master infrastructure. "
    "Do not run this for validators or third-party operators."
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_master_deploy_is_docker_swarm_not_kubernetes() -> None:
    # The Kubernetes/Helm master installer was removed in the Swarm migration.
    assert not (ROOT / "scripts" / "install-master.sh").exists()
    assert not (ROOT / "deploy" / "helm").exists()
    assert SWARM_INSTALLER.is_file()


def test_master_installer_is_swarm_only_and_dry_run_by_default() -> None:
    script = _read(SWARM_INSTALLER)

    assert "kubectl" not in script
    assert "helm" not in script
    assert "docker swarm" in script
    assert "docker service create" in script
    assert "docker secret" in script
    # Dry-run by default; mutations require an explicit --apply opt-in.
    assert "APPLY=false" in script
    assert "--apply" in script
    assert "base master worker" in script


def test_master_installer_worker_enrollment_uses_swarm_join_tokens() -> None:
    guide = _read(MASTER_GUIDE)

    assert "base master worker token" in guide
    assert "base master worker label" in guide
    assert "docker swarm join --token" in guide
    assert "node.labels.base.workload" in guide


def test_master_guide_is_foundation_only_and_swarm() -> None:
    guide = _read(MASTER_GUIDE)

    assert FOUNDATION_WARNING in guide
    assert "Docker Swarm" in guide
    assert "install-swarm.sh" in guide
    assert "node.role==manager" in guide
    assert "base master worker" in guide
    # Kubernetes/Helm tooling is gone from the foundation guide.
    assert "kubectl" not in guide
    assert "helm" not in guide
    # The foundation bring-up never asks for or stores key material.
    assert "mnemonic" not in guide.lower()
    assert "coldkey" not in guide.lower()
    assert "hotkey" not in guide.lower()


def test_master_guide_does_not_target_validators_or_operators() -> None:
    guide = _read(MASTER_GUIDE)

    assert "validators" in guide
    assert "third-party operators" in guide
    assert "install-validator" not in guide


def test_readme_documents_swarm_master_deployment() -> None:
    readme = _read(README)

    assert "deploy/swarm/install-swarm.sh" in readme
    assert "base master worker" in readme
    assert "node.role==manager" in readme
    assert "node.labels.base.workload==cpu" in readme
    assert "node.labels.base.workload==gpu" in readme
    # Swarm is the only backend; there is no Kubernetes/Helm install path.
    assert "There is no Kubernetes" in readme
