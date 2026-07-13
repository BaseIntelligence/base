"""Master install docs after Compose cutover (Swarm is historical only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_INSTALLER = ROOT / "deploy" / "compose" / "install-master.sh"
COMPOSE_VALIDATOR = ROOT / "deploy" / "compose" / "install-validator.sh"
MASTER_GUIDE = ROOT / "docs" / "master" / "README.md"
README = ROOT / "README.md"
SWARM_README = ROOT / "deploy" / "swarm" / "README.md"
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_compose_master_installer_is_present_and_entrypoint() -> None:
    assert COMPOSE_INSTALLER.is_file()
    assert COMPOSE_VALIDATOR.is_file()
    script = _read(COMPOSE_INSTALLER)
    assert "docker compose" in script or "docker-compose" in script
    # Compose installer never installs live Swarm services.
    assert "docker service create" not in script


def test_master_guide_documents_compose_not_swarm_as_required_path() -> None:
    guide = _read(MASTER_GUIDE)

    assert "Docker Compose" in guide
    assert "install-master" in guide or "compose" in guide.lower()
    # Swarm is not documented as required install path.
    assert "docker service create" not in guide
    # Guide must not re-introduce operator chain submit for master.
    assert "set_weights" not in guide or "never" in guide.lower()


def test_master_guide_is_compose_operator_surface() -> None:
    guide = _read(MASTER_GUIDE)

    assert "Compose" in guide
    # Historical Swarm installer may exist; docs must not position it as required.
    assert "required Swarm" not in guide
    assert "node.role==manager" not in guide


def test_readme_documents_compose_master_deployment() -> None:
    readme = _read(README)

    assert "Docker Compose" in readme
    assert "deploy/compose" in readme or "install-master" in readme
    # Swarm is not the required path.
    assert "There is no Kubernetes" in readme or "Compose" in readme
    # Does not force install-swarm as the operator bring-up.
    # Historical script may still be named for archival reading.


def test_swarm_tree_is_historical_when_present() -> None:
    """If historical Swarm artifacts remain, they stay out of the required path."""

    if SWARM_INSTALLER.is_file() and SWARM_README.is_file():
        readme = _read(SWARM_README)
        lowered = readme.lower()
        assert (
            "unsupported" in lowered
            or "historical" in lowered
            or "not the supported" in lowered
            or "compose" in lowered
        )
