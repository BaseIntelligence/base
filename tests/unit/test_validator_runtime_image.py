"""Guard tests for the combined validator runtime image (m5-validator-runtime-image).

A decentralized validator must be able to DISPATCH both challenges from a single
image. That requires two things to stay in lockstep:

1. the in-process dispatch registry exposes exactly the two known slugs
   (``agent-challenge`` and ``prism``) so the per-slug adapters resolve, and
2. platform CI builds + publishes a ``base-validator-runtime`` image (from
   ``docker/Dockerfile.validator-runtime``) that installs ``base`` PLUS both
   challenge dispatch packages and proves both ``validator_dispatch`` modules are
   importable via a build-time smoke check.

These are file-inspection + import guards: a future edit that drops a dispatch
slug, removes the runtime image from CI, breaks the no-deps challenge install
(which would clobber the local ``base``), or deletes the import smoke check fails
loudly here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from base.validator.agent.challenge_dispatch import (
    DEFAULT_CHALLENGE_EXECUTOR_FACTORIES,
)

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE_RUNTIME = ROOT / "docker" / "Dockerfile.validator-runtime"

RUNTIME_IMAGE = "base-validator-runtime"
RUNTIME_DOCKERFILE = "docker/Dockerfile.validator-runtime"
AGENT_CHALLENGE_REF = "d02f7329b17dbc3b663bcd518c746022bbc0afe8"
PRISM_REF = "680440d59411fa578ba564b0b04bf437a78c7f66"


def _ci() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _matrix_images(job: dict) -> dict[str, str]:
    """Map ``image -> dockerfile`` for a docker job's build matrix."""
    includes = job["strategy"]["matrix"]["include"]
    return {entry["image"]: entry["dockerfile"] for entry in includes}


# --- Dispatch registry slugs -------------------------------------------------


def test_dispatch_registry_exposes_both_challenge_slugs() -> None:
    # The runtime image only matters because these are the slugs a validator can
    # dispatch in-process; keep the registry and the image in lockstep.
    assert set(DEFAULT_CHALLENGE_EXECUTOR_FACTORIES) == {"agent-challenge", "prism"}


# --- Runtime image build target ---------------------------------------------


def test_ci_builds_and_publishes_validator_runtime_image() -> None:
    ci = _ci()
    build = _matrix_images(ci["jobs"]["docker-build"])
    publish = _matrix_images(ci["jobs"]["docker-publish"])

    assert build.get(RUNTIME_IMAGE) == RUNTIME_DOCKERFILE
    assert publish.get(RUNTIME_IMAGE) == RUNTIME_DOCKERFILE


def test_runtime_image_uses_shared_ghcr_tag_policy() -> None:
    ci = _ci()
    meta_step = next(
        step
        for step in ci["jobs"]["docker-publish"]["steps"]
        if step.get("id") == "meta"
    )
    tags = meta_step["with"]["tags"]
    # Same policy as base/base-master: latest only from main + semver + sha-<sha>.
    assert "type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}" in tags
    assert "type=semver,pattern={{version}}" in tags
    assert "type=sha,prefix=sha-" in tags
    assert "ghcr.io/baseintelligence/${{ matrix.image }}" in meta_step["with"]["images"]


def test_runtime_image_publish_gate_is_pr_safe_and_main_only_latest() -> None:
    ci = _ci()
    publish_if = ci["jobs"]["docker-publish"]["if"]
    # Publishing (incl. the runtime image) never happens on PRs; latest only main.
    assert "github.event_name != 'pull_request'" in publish_if
    assert "refs/heads/main" in publish_if


# --- Dockerfile recipe -------------------------------------------------------


def test_runtime_dockerfile_exists() -> None:
    assert DOCKERFILE_RUNTIME.is_file()


def test_runtime_dockerfile_installs_base_then_both_dispatch_packages_no_deps() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")

    # Base is built as a wheel and installed non-editably before challenge
    # adapters. Only the completed virtual environment crosses into runtime.
    assert "uv build --wheel" in text
    assert '"base @ file://${wheel}"' in text
    assert "--require-hashes --requirements /tmp/base-requirements.txt" in text
    assert " -e " not in text
    assert "COPY --from=builder /opt/validator /opt/validator" in text

    # Both challenge packages are checked out at required commit arguments,
    # built as wheels, and installed without replacing canonical Base.
    assert "--no-deps" in text
    assert "https://github.com/BaseIntelligence/agent-challenge.git" in text
    assert "https://github.com/BaseIntelligence/prism.git" in text
    assert "--no-emit-package base" in text

    base_at = text.index('"base @ file://${wheel}"')
    challenge_at = text.index("git clone --filter=blob:none")
    assert base_at < challenge_at


def test_runtime_dockerfile_requires_immutable_challenge_commits() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")

    assert "ARG AGENT_CHALLENGE_REF\n" in text
    assert "ARG PRISM_REF\n" in text
    assert "AGENT_CHALLENGE_REF=main" not in text
    assert "PRISM_REF=main" not in text
    assert text.count("grep -Eq '^[0-9a-f]{40}$'") == 2


def test_ci_passes_immutable_challenge_commits_to_runtime_builds() -> None:
    ci = _ci()

    for job_name in ("docker-build", "docker-publish"):
        entry = next(
            item
            for item in ci["jobs"][job_name]["strategy"]["matrix"]["include"]
            if item["image"] == RUNTIME_IMAGE
        )
        assert entry["build_args"].splitlines() == [
            f"AGENT_CHALLENGE_REF={AGENT_CHALLENGE_REF}",
            f"PRISM_REF={PRISM_REF}",
        ]
        build_step = next(
            step
            for step in ci["jobs"][job_name]["steps"]
            if step.get("uses") == "docker/build-push-action@v6"
        )
        assert build_step["with"]["build-args"] == "${{ matrix.build_args }}"


def test_runtime_stage_contains_no_source_checkout_or_docker_socket_client() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    runtime = text.split("FROM python:3.12-slim AS runtime", maxsplit=1)[1]

    assert "COPY . ." not in runtime
    assert "git" not in runtime
    assert "uv pip install" not in runtime
    assert "docker.tgz" not in runtime


def test_runtime_dockerfile_installs_locked_challenge_dependency_graphs() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")

    assert "uv export --project /build/agent-challenge --frozen" in text
    assert "uv export --project /build/prism --frozen" in text
    assert text.count("--require-hashes --requirements") == 3
    assert "torch>=2.3" not in text


def test_runtime_dockerfile_is_bittensor_only_no_legacy_substrate_stack() -> None:
    # VAL-CODE-VRT-001: the runtime image must NOT install the legacy
    # substrate-interface/scalecodec stack. Both challenges sign/verify sr25519
    # via bittensor.Keypair (bittensor comes from base[validator]); bittensor 10
    # bundles async-substrate-interface, and the legacy stack conflicts with it
    # (that conflict is what forced the :hotfix-scalecodec pin).
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    # Only the executable (non-comment) lines matter: the recipe documents WHY
    # the legacy stack is absent, so comments may name it.
    instructions = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "substrate-interface" not in instructions
    assert "substrateinterface" not in instructions
    assert "scalecodec" not in instructions


def test_runtime_dockerfile_has_import_smoke_check() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    assert (
        "import base, agent_challenge.validator_dispatch, "
        "prism_challenge.validator_dispatch" in text
    )


def test_runtime_dockerfile_cmd_runs_the_validator_agent() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    assert '"base", "validator", "agent"' in text
