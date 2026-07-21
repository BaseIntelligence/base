"""Shipping docs simplify contracts (VAL-MINERDY-008..010).

Locks banner retention, monorepo + master-embed challenges narrative, miner
day-1 via chain.joinbase.ai / weight-only validators, and demotion of mission
harness / obsolete multi-repo lead language.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_readme_keeps_centered_banner() -> None:
    """VAL-MINERDY-008: README still centers assets/banner.jpg."""
    readme = _read("README.md")
    assert "assets/banner.jpg" in readme
    assert "![BASE Banner](assets/banner.jpg)" in readme
    assert '<div align="center">' in readme
    assert (REPO_ROOT / "assets/banner.jpg").is_file()


def test_challenges_docs_describe_master_embed_monorepo() -> None:
    """VAL-MINERDY-009: challenges docs are miner+ops embed monorepo entry."""
    challenges = _read("docs/challenges.md")
    challenges_index = _read("docs/challenges/README.md")
    miner_hub = _read("docs/miner/README.md")
    blob = "\n".join((challenges, challenges_index, miner_hub))
    low = blob.lower()

    assert "packages/challenges/prism" in blob
    assert "packages/challenges/agent-challenge" in blob
    assert "https://chain.joinbase.ai" in blob
    assert "weight-only" in low or "weight only" in low
    assert "127.0.0.1:18080" in challenges or "18080" in challenges
    assert "127.0.0.1:18081" in challenges or "18081" in challenges
    assert "/challenges/prism" in blob
    assert "/challenges/agent-challenge" in blob
    assert "master-embed" in low or "embeds" in low or "embedded" in low
    # Must not require separate multi-repo containers as shipping topology.
    assert "not" in challenges.lower() and "challenge-*" in challenges
    assert "https://joinbase.ai" in blob


def test_readme_overview_leads_monorepo_master_embed() -> None:
    """README Overview leads with monorepo packages + master-embed."""
    readme = _read("README.md")
    overview = readme.split("## Overview", 1)[1].split("## Architecture", 1)[0]
    low = overview.lower()
    assert "packages/challenges" in overview
    assert "master-embed" in low or "embedded" in low or "embeds" in overview
    assert "challenge-*" in overview
    # Obsolete multi-repo lead must not be the Overview story.
    assert "lives in its own repository" not in overview
    assert "each challenge lives in its own" not in low
    assert "one long-lived container per" not in low
    assert "one long-lived combined service per active challenge" not in low


def test_mission_harness_not_day1() -> None:
    """VAL-MINERDY-010: mission harness is not miner day-1."""
    harness = _read("docs/operations/mission-harness.md")
    miner_hub = _read("docs/miner/README.md")
    readme = _read("README.md")

    harness_low = harness.lower()
    assert "not miner day-1" in harness_low
    assert "not a production path" in harness_low
    # Miner hub must not link mission-harness as a table/doc path.
    hub_table = miner_hub.split("## Canonical public URLs", 1)[0]
    assert "mission-harness.md" not in hub_table
    assert "operations/mission-harness" not in hub_table
    # If mentioned at all, it must be explicitly demoted (not day-1).
    if "mission harness" in hub_table.lower():
        hub_low = hub_table.lower()
        assert "not" in hub_low and "day-1" in hub_low
    # Root README must not list mission-harness as primary miner entry.
    assert "mission-harness.md" not in readme


def test_shipping_docs_no_required_separate_challenge_services_lead() -> None:
    """VAL-MINERDY-010: deny required challenge-* service cardinality."""
    for rel in (
        "docs/architecture.md",
        "docs/compose.md",
        "docs/deploy.md",
        "docs/challenges.md",
        "README.md",
    ):
        text = _read(rel)
        assert "one `challenge-<slug>`" not in text, rel
        assert "one long-lived `challenge-<slug>`" not in text, rel
        assert "| one `challenge-<slug>` |" not in text, rel


def test_miner_getting_started_points_monorepo_not_external_repos() -> None:
    """Day-1 getting started points at monorepo miner hubs."""
    gs = _read("docs/miner/getting-started.md")
    assert "https://chain.joinbase.ai" in gs
    assert "prism/README.md" in gs or "docs/miner/prism" in gs
    assert "agent-challenge/README.md" in gs or "docs/miner/agent-challenge" in gs
    assert "Prism** repository" not in gs
    assert "agent-challenge repo" not in gs
