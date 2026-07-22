"""Minimal shipping docs contracts (VAL-CLEAN-004..007 / VAL-MINERDY companions).

Locks banner retention, short README, thin docs tree, OpenAPI-as-API-truth,
miner day-1 via chain.joinbase.ai, and absence of essay sprawl.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Thin shipping docs set only.
ALLOWED_DOCS = {
    "docs/compose.md",
    "docs/validator.md",
    "docs/versioning.md",
    "docs/miner/getting-started.md",
}


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _shipping_docs() -> list[Path]:
    docs_root = REPO_ROOT / "docs"
    return sorted(p for p in docs_root.rglob("*") if p.is_file())


def test_readme_keeps_centered_banner() -> None:
    """Banner retained on short product README."""
    readme = _read("README.md")
    assert "assets/banner.jpg" in readme
    assert "![BASE Banner](assets/banner.jpg)" in readme
    assert '<div align="center">' in readme
    assert (REPO_ROOT / "assets/banner.jpg").is_file()


def test_readme_is_short_and_useful() -> None:
    """VAL-CLEAN-004: README ≤80 lines, no monorepo essay / SDK tables."""
    readme = _read("README.md")
    lines = readme.splitlines()
    assert len(lines) <= 80, f"README has {len(lines)} lines (max 80)"

    assert "https://chain.joinbase.ai" in readme
    assert "https://joinbase.ai" in readme
    assert "docs/miner/getting-started.md" in readme
    assert "docs/validator.md" in readme
    assert "openapi.json" in readme.lower() or "OpenAPI" in readme
    assert "Apache-2.0" in readme or "License" in readme

    # Forbidden sprawl markers.
    low = readme.lower()
    assert "sha-256" not in low
    _sdk_sha = "3a61c2d3a343ed6de55e80215486e3de0c9639276443d08f2ed316bc807f2ff0"
    assert _sdk_sha not in readme
    assert "| Audience | Guide |" not in readme
    assert "docs/monorepo.md" not in readme
    assert "SOURCE_OF_TRUTH" not in readme
    assert "reward-semantics" not in low
    assert "mission-harness" not in low
    assert "lives in its own repository" not in low
    assert "one long-lived combined service per active challenge" not in low


def test_docs_tree_is_thin() -> None:
    """VAL-CLEAN-005: only the thin shipping docs set remains under docs/."""
    rels = {
        p.relative_to(REPO_ROOT).as_posix()
        for p in _shipping_docs()
    }
    assert rels == ALLOWED_DOCS, f"unexpected docs tree: {sorted(rels)}"

    # Explicitly deleted essays / hubs must stay gone.
    for banned in (
        "docs/monorepo.md",
        "docs/SOURCE_OF_TRUTH.md",
        "docs/architecture.md",
        "docs/deploy.md",
        "docs/security.md",
        "docs/challenges.md",
        "docs/reward-semantics.md",
        "docs/challenge-integration.md",
        "docs/master/README.md",
        "docs/operations/mission-harness.md",
        "docs/operations/validator.md",
        "docs/miner/README.md",
        "docs/miner/how-to.md",
        "docs/miner/prism/README.md",
        "docs/miner/agent-challenge/README.md",
        "docs/validator/README.md",
    ):
        assert not (REPO_ROOT / banned).exists(), banned


def test_api_truth_is_openapi_not_markdown_dumps() -> None:
    """VAL-CLEAN-007: shipping docs point at OpenAPI, not long API tables."""
    blob = "\n".join(_read(rel) for rel in sorted(ALLOWED_DOCS))
    blob += "\n" + _read("README.md")
    low = blob.lower()
    assert "openapi" in low
    assert "openapi.json" in low
    # No giant endpoint inventory novels in shipping docs.
    assert "GET /internal/v1/get_weights" not in blob
    assert "POST /owner/submissions/{submission_id}/revalidate" not in blob


def test_miner_getting_started_joinbase_and_packages() -> None:
    """Day-1 getting started points at joinbase + monorepo packages."""
    gs = _read("docs/miner/getting-started.md")
    assert "https://chain.joinbase.ai" in gs
    assert "https://joinbase.ai" in gs
    assert "packages/challenges/prism" in gs
    assert "packages/challenges/agent-challenge" in gs
    assert "/challenges/prism" in gs
    assert "/challenges/agent-challenge" in gs
    assert "openapi.json" in gs
    assert "Prism** repository" not in gs
    assert "agent-challenge repo" not in gs
    assert "lives in its own repository" not in gs.lower()


def test_mission_harness_absent_from_shipping_docs() -> None:
    """Mission harness is not shipping day-1 material."""
    shipping = (
        "README.md",
        "docs/miner/getting-started.md",
        "docs/validator.md",
        "docs/compose.md",
    )
    for rel in shipping:
        text = _read(rel)
        assert "mission-harness.md" not in text
        assert "operations/mission-harness" not in text


def test_shipping_docs_no_required_separate_challenge_services_lead() -> None:
    """Deny required challenge-* service cardinality in remaining docs."""
    shipping = (
        "docs/compose.md",
        "docs/validator.md",
        "README.md",
        "docs/miner/getting-started.md",
    )
    for rel in shipping:
        text = _read(rel)
        assert "one `challenge-<slug>`" not in text, rel
        assert "one long-lived `challenge-<slug>`" not in text, rel
        assert "| one `challenge-<slug>` |" not in text, rel
