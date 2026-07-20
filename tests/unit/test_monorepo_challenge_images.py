"""Monorepo challenge Docker/CI contracts (VAL-MONO-007, VAL-MONO-008)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Production GHCR names — NEVER rename (mission invariant).
INVARIANT_GHCR_NAMES: frozenset[str] = frozenset(
    {
        "ghcr.io/baseintelligence/prism",
        "ghcr.io/baseintelligence/prism-evaluator",
        "ghcr.io/baseintelligence/agent-challenge",
        "ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner",
    }
)

PRISM_DOCKERFILE = REPO_ROOT / "packages/challenges/prism/Dockerfile"
AC_DOCKERFILE = REPO_ROOT / "packages/challenges/agent-challenge/Dockerfile"
CHALLENGE_IMAGES_WORKFLOW = REPO_ROOT / ".github/workflows/challenge-images.yml"
OCI_SOURCE = (
    'org.opencontainers.image.source="https://github.com/BaseIntelligence/base"'
)


def test_challenge_dockerfiles_exist_under_monorepo_packages() -> None:
    assert PRISM_DOCKERFILE.is_file()
    assert AC_DOCKERFILE.is_file()
    assert (REPO_ROOT / "packages/challenges/prism/src/prism_challenge").is_dir()
    assert (
        REPO_ROOT / "packages/challenges/agent-challenge/src/agent_challenge"
    ).is_dir()


def test_challenge_dockerfiles_use_monorepo_named_context() -> None:
    """Build context copies workspace base via BuildKit named context `monorepo`."""
    for path in (PRISM_DOCKERFILE, AC_DOCKERFILE):
        text = path.read_text(encoding="utf-8")
        assert "COPY --from=monorepo" in text, path
        assert "src/base" in text, path
        assert OCI_SOURCE in text, path
        # Must not reintroduce floating external git pins in Docker install path.
        assert "git+https://github.com/BaseIntelligence/base" not in text
        assert "releases/download/v" not in text


def test_prism_dockerfile_exposes_service_and_evaluator_targets() -> None:
    text = PRISM_DOCKERFILE.read_text(encoding="utf-8")
    assert "AS service" in text
    assert "AS evaluator" in text
    assert "prism_challenge.app:app" in text


def test_agent_challenge_dockerfile_exposes_runtime_and_runner_targets() -> None:
    text = AC_DOCKERFILE.read_text(encoding="utf-8")
    assert "AS runtime" in text
    assert "AS terminal-bench-runner" in text
    assert "agent_challenge.app:app" in text


def test_challenge_images_workflow_uses_identical_ghcr_names() -> None:
    """CI publish tags must be the exact four invariant GHCR names (VAL-MONO-007)."""
    text = CHALLENGE_IMAGES_WORKFLOW.read_text(encoding="utf-8")
    assert CHALLENGE_IMAGES_WORKFLOW.is_file()
    for name in INVARIANT_GHCR_NAMES:
        assert name in text, f"missing GHCR name {name}"

    data = yaml.safe_load(text)
    assert data["name"] == "Challenge images"
    # PyYAML 1.1 treats bare key `on` as boolean True.
    on = data.get("on", data.get(True))
    assert on is not None
    # Path filters present on push/PR (workflow_dispatch may omit paths).
    push_paths = on["push"].get("paths") or []
    pr_paths = on["pull_request"].get("paths") or []
    for required in (
        "packages/challenges/prism/**",
        "packages/challenges/agent-challenge/**",
        ".github/workflows/challenge-images.yml",
    ):
        assert required in push_paths, required
        assert required in pr_paths, required

    build_job = data["jobs"]["docker-build"]
    matrix_include = build_job["strategy"]["matrix"]["include"]
    ghcr_names = {row["ghcr"] for row in matrix_include}
    assert ghcr_names == set(INVARIANT_GHCR_NAMES)

    for row in matrix_include:
        assert row["package_dir"].startswith("packages/challenges/")
        assert row["dockerfile"].startswith("packages/challenges/")
        # Primary context is package path (monorepo layout).
        assert Path(row["package_dir"]).name in {
            "prism",
            "agent-challenge",
        }


def test_challenge_images_workflow_build_uses_monorepo_build_context() -> None:
    text = CHALLENGE_IMAGES_WORKFLOW.read_text(encoding="utf-8")
    assert "build-contexts:" in text
    assert "monorepo=." in text
    assert (
        "org.opencontainers.image.source=https://github.com/BaseIntelligence/base"
        in text
    )
    # Push is gated; PR path never force-publishes.
    assert "confirm_publish" in text
    publish = yaml.safe_load(text)["jobs"]["docker-publish"]
    assert "packages: write" in yaml.dump(publish.get("permissions", {})) or (
        publish.get("permissions", {}).get("packages") == "write"
    )


def test_docs_monorepo_lists_challenge_image_ghcr_names() -> None:
    text = (REPO_ROOT / "docs/monorepo.md").read_text(encoding="utf-8")
    for name in INVARIANT_GHCR_NAMES:
        # ADR may list brace-expanded form; require each short name component.
        short = name.rsplit("/", 1)[-1]
        assert short in text, short
    assert "challenge-images" in text or "mono-ci-images" in text
