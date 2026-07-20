"""Guard tests for the combined validator runtime image (monorepo, VAL-MONO-009/010).

A decentralized validator must be able to DISPATCH both challenges from a single
image. That requires two things to stay in lockstep:

1. the in-process dispatch registry exposes exactly the two known slugs
   (``agent-challenge`` and ``prism``) so the per-slug adapters resolve, and
2. platform CI builds + publishes a ``base-validator-runtime`` image (from
   ``docker/Dockerfile.validator-runtime``) that installs ``base`` PLUS both
   monorepo challenge packages (COPY, not git clone) and proves both
   ``validator_dispatch`` modules are importable via a build-time smoke check.

These are file-inspection + import guards: a future edit that drops a dispatch
slug, removes the runtime image from CI, reintroduces external challenge clones,
or deletes the import smoke check fails loudly here.
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

# Forbidden external clone surfaces (pre-monorepo).
FORBIDDEN_CLONE_MARKERS = (
    "AGENT_CHALLENGE_REF",
    "PRISM_REF",
    "https://github.com/BaseIntelligence/agent-challenge.git",
    "https://github.com/BaseIntelligence/prism.git",
    "git clone",
)


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


def test_runtime_dockerfile_installs_base_then_monorepo_challenge_wheels() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")

    # Base is built as a wheel and installed non-editably before challenge
    # packages. Only the completed virtual environment crosses into runtime.
    assert "uv build --wheel" in text
    assert '"base @ file://${wheel}"' in text
    assert "--require-hashes --requirements /tmp/base-requirements.txt" in text
    assert " -e " not in text
    assert "COPY --from=builder /opt/validator /opt/validator" in text

    # Challenges come from monorepo package paths (COPY context), not external git.
    assert "packages/challenges/agent-challenge" in text
    assert "packages/challenges/prism" in text
    assert "uv build --package agent-challenge" in text
    assert "uv build --package prism-challenge" in text
    assert "--no-deps" in text
    assert "--no-emit-package base" in text

    base_at = text.index('"base @ file://${wheel}"')
    challenge_export_at = text.index("uv export --package agent-challenge")
    assert base_at < challenge_export_at


def test_runtime_dockerfile_has_no_external_challenge_clone() -> None:
    """VAL-MONO-009: no AGENT_CHALLENGE_REF/PRISM_REF clone of external repos."""
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    # Ignore comment-only lines that document the removal.
    instructions = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    for marker in FORBIDDEN_CLONE_MARKERS:
        assert marker not in instructions, (
            f"forbidden clone marker still present: {marker}"
        )
    assert "ARG AGENT_CHALLENGE_REF" not in instructions
    assert "ARG PRISM_REF" not in instructions


def test_ci_runtime_build_has_no_external_challenge_ref_build_args() -> None:
    """CI must not pass legacy external SHA pins for monorepo runtime builds."""
    ci = _ci()

    for job_name in ("docker-build", "docker-publish"):
        entry = next(
            item
            for item in ci["jobs"][job_name]["strategy"]["matrix"]["include"]
            if item["image"] == RUNTIME_IMAGE
        )
        build_args = (entry.get("build_args") or "").strip()
        assert "AGENT_CHALLENGE_REF" not in build_args
        assert "PRISM_REF" not in build_args
        # Empty string or omitted is fine; no clone pins.
        assert build_args in ("",)


def test_runtime_stage_contains_no_source_checkout_or_docker_socket_client() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    runtime = text.split("FROM python:3.12-slim AS runtime", maxsplit=1)[1]

    assert "COPY . ." not in runtime
    assert "git clone" not in runtime
    assert "uv pip install" not in runtime
    assert "docker.tgz" not in runtime


def test_runtime_dockerfile_installs_locked_challenge_dependency_graphs() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")

    assert "uv export --package agent-challenge --frozen" in text
    assert "uv export --package prism-challenge --frozen" in text
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
    """VAL-MONO-010: build-time import of both validator_dispatch modules."""
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    assert (
        "import base, agent_challenge.validator_dispatch, "
        "prism_challenge.validator_dispatch" in text
    )
    # PrismSettings defaults docker_backend=broker, which requires a token at
    # module import. The smoke check pins CLI + dummy token so install-time
    # imports stay deploy-agnostic (runtime still gets real deploy env).
    assert "PRISM_DOCKER_BACKEND=cli" in text
    assert "PRISM_SHARED_TOKEN=ci-import-smoke" in text


def test_runtime_dockerfile_cmd_runs_the_validator_agent() -> None:
    text = DOCKERFILE_RUNTIME.read_text(encoding="utf-8")
    assert '"base", "validator", "agent"' in text


def test_monorepo_packages_ship_validator_dispatch_modules() -> None:
    """In-tree packages must expose the dispatch entrypoints the image smokes."""
    ac_dispatch = (
        ROOT
        / "packages"
        / "challenges"
        / "agent-challenge"
        / "src"
        / "agent_challenge"
        / "validator_dispatch.py"
    )
    prism_dispatch = (
        ROOT
        / "packages"
        / "challenges"
        / "prism"
        / "src"
        / "prism_challenge"
        / "validator_dispatch.py"
    )
    assert ac_dispatch.is_file()
    assert prism_dispatch.is_file()
