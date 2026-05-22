from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _step_uses(job: dict) -> set[str]:
    return {step.get("uses", "") for step in job["steps"] if "uses" in step}


def test_ci_workflow_builds_platform_images_without_publishing_on_prs() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    docker_build = jobs["docker-build"]

    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request:" in CI_WORKFLOW.read_text(encoding="utf-8")
    assert "packages" not in docker_build.get("permissions", {})
    assert docker_build["needs"] == [
        "ruff",
        "format",
        "mypy",
        "coverage",
        "compose-validation",
        "helm-kubeconform",
        "production-policy",
    ]
    assert {
        item["image"] for item in docker_build["strategy"]["matrix"]["include"]
    } == {
        "platform",
        "platform-master",
    }
    assert _step_uses(docker_build) >= {
        "actions/checkout@v4",
        "docker/setup-buildx-action@v3",
        "docker/build-push-action@v6",
    }
    build_step = next(
        step
        for step in docker_build["steps"]
        if step.get("uses") == "docker/build-push-action@v6"
    )
    assert build_step["with"]["push"] is False


def test_ci_workflow_publishes_platform_images_to_ghcr_on_trusted_events() -> None:
    workflow_text = CI_WORKFLOW.read_text(encoding="utf-8")
    workflow = _workflow()
    docker_publish = workflow["jobs"]["docker-publish"]

    assert "workflow_dispatch" in workflow_text
    assert "confirm_publish" in workflow_text
    assert "refs/heads/main" in docker_publish["if"]
    assert "refs/tags/v" in docker_publish["if"]
    assert "pull_request" in docker_publish["if"]
    assert docker_publish["permissions"] == {
        "contents": "read",
        "packages": "write",
    }
    assert _step_uses(docker_publish) >= {
        "actions/checkout@v4",
        "docker/setup-buildx-action@v3",
        "docker/login-action@v3",
        "docker/metadata-action@v5",
        "docker/build-push-action@v6",
    }

    metadata = next(
        step
        for step in docker_publish["steps"]
        if step.get("uses") == "docker/metadata-action@v5"
    )
    assert metadata["with"]["images"] == "ghcr.io/platformnetwork/${{ matrix.image }}"
    assert "type=sha,prefix=sha-" in metadata["with"]["tags"]
    assert "type=semver,pattern={{version}}" in metadata["with"]["tags"]
    assert "type=semver,pattern={{raw}}" in metadata["with"]["tags"]
    assert "type=raw,value=latest" in metadata["with"]["tags"]

    publish_step = next(
        step
        for step in docker_publish["steps"]
        if step.get("uses") == "docker/build-push-action@v6"
    )
    assert publish_step["with"]["push"] is True
    assert publish_step["with"]["tags"] == "${{ steps.meta.outputs.tags }}"
    assert publish_step["with"]["labels"] == "${{ steps.meta.outputs.labels }}"
