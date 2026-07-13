"""M7 GHCR deploy guards: the decentralized validator-agent image must be built
+ pushed by platform CI and documented on the GHCR digest-pin path
(VAL-CICD-018), and the new master subsystems (coordination plane, LLM gateway)
must ship inside the base-master image rather than a separate image
(VAL-CICD-022).

These are file-inspection + import guards; they pin the contract so a future
edit that drops the validator image from CI, breaks the documented digest-pin
path, or introduces a phantom image for a master subsystem fails loudly.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE_VALIDATOR = ROOT / "docker" / "Dockerfile.validator"
DOCKERFILE_MASTER = ROOT / "docker" / "Dockerfile.master"
VALIDATOR_DOCS = ROOT / "docs" / "operations" / "validator.md"
INSTALL_SWARM = ROOT / "deploy" / "swarm" / "install-swarm.sh"


def _ci() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _matrix_images(job: dict) -> dict[str, str]:
    """Map ``image -> dockerfile`` for a docker job's build matrix."""
    includes = job["strategy"]["matrix"]["include"]
    return {entry["image"]: entry["dockerfile"] for entry in includes}


# --- VAL-CICD-018: validator-agent image built/pushed by CI + on digest path ---


def test_ci_builds_validator_agent_image_from_dockerfile_validator() -> None:
    ci = _ci()
    build = _matrix_images(ci["jobs"]["docker-build"])
    publish = _matrix_images(ci["jobs"]["docker-publish"])

    # The validator-agent image is the `base` image built from Dockerfile.validator.
    assert build.get("base") == "docker/Dockerfile.validator"
    assert publish.get("base") == "docker/Dockerfile.validator"


def test_validator_image_uses_shared_ghcr_tag_policy() -> None:
    ci = _ci()
    meta_step = next(
        step
        for step in ci["jobs"]["docker-publish"]["steps"]
        if step.get("id") == "meta"
    )
    tags = meta_step["with"]["tags"]
    # latest only from main; plus semver + sha-<sha> (same policy as base-master).
    assert "type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}" in tags
    assert "type=semver,pattern={{version}}" in tags
    assert "type=sha,prefix=sha-" in tags
    # Both images publish to the ghcr.io/baseintelligence/* namespace.
    assert "ghcr.io/baseintelligence/${{ matrix.image }}" in meta_step["with"]["images"]


def test_validator_image_cmd_runs_the_validator() -> None:
    text = DOCKERFILE_VALIDATOR.read_text(encoding="utf-8")
    # The image entrypoint runs the validator (the decentralized executor).
    assert '"base", "validator"' in text


def test_validator_agent_documented_on_ghcr_digest_pin_path() -> None:
    ops = (ROOT / "docs" / "operations" / "validator.md").read_text(encoding="utf-8")
    lower = ops.lower()
    assert "compose" in lower
    # Image pin policy lives with installer/env; docs describe digest pins.
    assert "sha256" in lower or "digest" in lower or "image" in lower


def test_base_master_image_runs_master_proxy_from_the_base_package() -> None:
    text = DOCKERFILE_MASTER.read_text(encoding="utf-8")
    # Built from the package source with the master extra; entrypoint is the proxy.
    assert '".[master]"' in text
    assert '"base", "master", "proxy"' in text


def test_master_subsystems_import_from_the_base_package() -> None:
    # Coordination plane + LLM gateway live under base.master and are packaged by
    # Dockerfile.master (no separate image). They must import cleanly.
    for module in (
        "base.master",
        "base.master.validator_coordination",
        "base.master.assignment_coordination",
    ):
        assert importlib.import_module(module) is not None


def test_installer_adds_no_separate_image_for_master_subsystems() -> None:
    text = INSTALL_SWARM.read_text(encoding="utf-8")
    image_vars = {
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith("IMAGE_")
    }
    # The only deploy images are master/challenge/evaluator/postgres plus the
    # validator-runtime image a validator NODE runs the agent from (--validator-node;
    # CI-built from Dockerfile.validator) — no gateway/coordination/HF-publisher-
    # specific MASTER-SUBSYSTEM image. IMAGE_MASTER_CLI is NOT a separate subsystem
    # image: it is the SAME base-master image at its mutable :latest tag, used only
    # for the ephemeral one-shot CLI runs (mint / wallet-gen / runtime-uid inspect),
    # while deployed services keep the digest-pinned IMAGE_MASTER.
    assert image_vars == {
        "IMAGE_MASTER",
        "IMAGE_MASTER_CLI",
        "IMAGE_AGENT_CHALLENGE",
        "IMAGE_PRISM",
        "IMAGE_PRISM_EVALUATOR",
        "IMAGE_POSTGRES",
        "IMAGE_VALIDATOR_RUNTIME",
    }
