from __future__ import annotations

from pathlib import Path

import yaml

from base.config.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


def test_compose_deployment_files_are_removed_but_image_build_assets_remain() -> None:
    docker_dir = ROOT / "docker"

    assert not list(docker_dir.glob("compose*.yml"))
    assert not (docker_dir / "compose.yml").exists()
    assert not (docker_dir / "compose.dev.yml").exists()
    assert not (docker_dir / "compose.watchtower.yml").exists()

    assert (docker_dir / "Dockerfile.master").is_file()
    assert (docker_dir / "Dockerfile.validator").is_file()
    # Helm is removed in the Swarm migration; the first-party deployment is the
    # imperative Docker Swarm installer.
    assert not (ROOT / "deploy" / "helm").exists()
    assert (ROOT / "deploy" / "swarm" / "install-swarm.sh").is_file()


def test_base_dockerfiles_run_as_non_root_user() -> None:
    for dockerfile in (
        ROOT / "docker" / "Dockerfile.master",
        ROOT / "docker" / "Dockerfile.validator",
    ):
        content = dockerfile.read_text(encoding="utf-8")

        assert "--uid 1000" in content
        assert "--gid 1000" in content
        assert "chown -R 1000:1000" in content
        assert "USER 1000:1000" in content


def test_first_party_defaults_are_docker_swarm() -> None:
    settings = Settings()
    master_example = yaml.safe_load(
        (ROOT / "config" / "master.example.yaml").read_text(encoding="utf-8")
    )
    validator_example = yaml.safe_load(
        (ROOT / "config" / "validator.example.yaml").read_text(encoding="utf-8")
    )

    # The Kubernetes/runtime backend selector is gone: Swarm is the only backend.
    assert not hasattr(settings, "runtime")
    assert not hasattr(settings, "kubernetes")
    assert settings.database.url.startswith("postgresql+asyncpg://")
    assert settings.docker.broker_allowed_images == ["ghcr.io/baseintelligence/"]
    # Swarm placement defaults: challenge services on the manager, broker jobs on
    # CPU/GPU-labeled workers.
    assert settings.docker.challenge_placement_constraint == "node.role==manager"
    assert settings.docker.cpu_job_constraint == "node.labels.base.workload==cpu"
    assert settings.docker.gpu_job_constraint == "node.labels.base.workload==gpu"

    for example in (master_example, validator_example):
        assert "runtime" not in example
        assert "kubernetes" not in example
        assert example["database"]["url"].startswith("postgresql+asyncpg://")
        assert example["docker"]["broker_allowed_images"] == [
            "ghcr.io/baseintelligence/"
        ]


def test_first_party_docs_and_ci_do_not_advertise_compose_or_watchtower() -> None:
    forbidden = [
        "docker compose",
        "compose.yml",
        "compose.dev.yml",
        "compose.watchtower.yml",
        "compose-validation",
        "watchtower",
        "com.centurylinklabs.watchtower.enable",
    ]
    # README, security docs, and CI carry zero compose/watchtower mentions.
    strict_paths = [
        ROOT / "README.md",
        ROOT / "docs" / "security.md",
        ROOT / ".github" / "workflows" / "ci.yml",
    ]
    for path in strict_paths:
        content = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in content, f"{token!r} found in {path}"

    # architecture.md mentions Docker Compose only to disavow it, so it must not
    # advertise any compose FILE artifact or watchtower, and must describe the
    # Swarm-only deployment.
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    lowered = architecture.lower()
    for token in forbidden:
        if token == "docker compose":
            continue
        assert token not in lowered, f"{token!r} found in architecture.md"
    assert "docker swarm" in lowered
    assert "no kubernetes manifests" in lowered
