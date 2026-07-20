"""Monorepo deploy/miner docs + public slug contracts (VAL-MONO-011..014)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_SLUGS = (
    "/challenges/prism",
    "/challenges/agent-challenge",
)

INVARIANT_GHCR_NAMES = (
    "ghcr.io/baseintelligence/prism",
    "ghcr.io/baseintelligence/prism-evaluator",
    "ghcr.io/baseintelligence/agent-challenge",
    "ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner",
    "ghcr.io/baseintelligence/base-master",
    "ghcr.io/baseintelligence/base-validator-runtime",
)


def test_deploy_docs_reference_monorepo_local_build_paths() -> None:
    """VAL-MONO-011: compose/deploy docs know monorepo paths + public GHCR names."""
    deploy = (REPO_ROOT / "docs/deploy.md").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docs/compose.md").read_text(encoding="utf-8")
    install = (REPO_ROOT / "deploy/compose/install-master.sh").read_text(
        encoding="utf-8"
    )
    swarm = (REPO_ROOT / "deploy/swarm/README.md").read_text(encoding="utf-8")

    for text in (deploy, compose, install):
        assert "packages/challenges/prism" in text
        assert "packages/challenges/agent-challenge" in text or "PRISM" in text
        assert "ghcr.io/baseintelligence/prism" in text
        assert "--build-context monorepo=." in text or "monorepo=." in text

    # Swarm historical note must mention monorepo SoT without becoming the path.
    assert "packages/challenges" in swarm
    assert "NOT A SUPPORTED INSTALL DESTINATION" in swarm or "HISTORICAL" in swarm

    for name in (
        "ghcr.io/baseintelligence/prism",
        "ghcr.io/baseintelligence/base-master",
        "ghcr.io/baseintelligence/base-validator-runtime",
    ):
        assert name in deploy
        assert name in compose


def test_miner_docs_unified_under_base() -> None:
    """VAL-MONO-012: docs/miner/{prism,agent-challenge}/ hubs exist in base."""
    prism_hub = REPO_ROOT / "docs/miner/prism/README.md"
    ac_hub = REPO_ROOT / "docs/miner/agent-challenge/README.md"
    how_to = REPO_ROOT / "docs/miner/how-to.md"
    miner_index = REPO_ROOT / "docs/miner/README.md"

    assert prism_hub.is_file()
    assert ac_hub.is_file()
    assert (REPO_ROOT / "docs/miner/prism/getting-started.md").is_file()
    assert (REPO_ROOT / "docs/miner/agent-challenge/getting-started.md").is_file()
    assert (REPO_ROOT / "docs/miner/agent-challenge/submit-agent.md").is_file()

    prism_text = prism_hub.read_text(encoding="utf-8")
    ac_text = ac_hub.read_text(encoding="utf-8")
    how_text = how_to.read_text(encoding="utf-8")
    index_text = miner_index.read_text(encoding="utf-8")

    assert "Monorepo hub" in prism_text
    assert "Monorepo hub" in ac_text
    assert "/challenges/prism" in prism_text
    assert "/challenges/agent-challenge" in ac_text
    assert "prism/getting-started.md" in how_text
    assert "agent-challenge/getting-started.md" in how_text
    assert "prism/README.md" in index_text
    assert "agent-challenge/README.md" in index_text


def test_source_of_truth_transition_note() -> None:
    """VAL-MONO-013: monorepo SoT note for old remotes / transition."""
    sot = (REPO_ROOT / "docs/SOURCE_OF_TRUTH.md").read_text(encoding="utf-8")
    mono = (REPO_ROOT / "docs/monorepo.md").read_text(encoding="utf-8")
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pkg_prism = (REPO_ROOT / "packages/challenges/prism/README.md").read_text(
        encoding="utf-8"
    )
    pkg_ac = (REPO_ROOT / "packages/challenges/agent-challenge/README.md").read_text(
        encoding="utf-8"
    )

    assert "BaseIntelligence/base" in sot
    assert "packages/challenges/prism" in sot
    assert "packages/challenges/agent-challenge" in sot
    assert "BaseIntelligence/prism" in sot
    assert "BaseIntelligence/agent-challenge" in sot
    for slug in PUBLIC_SLUGS:
        assert slug in sot
    assert "SOURCE_OF_TRUTH" in mono or "Source of truth" in mono
    assert "SOURCE_OF_TRUTH" in root_readme or "Source of truth" in root_readme
    assert "Source of truth" in pkg_prism
    assert "Source of truth" in pkg_ac


def test_public_api_slugs_unchanged_in_master_proxy() -> None:
    """VAL-MONO-014: master proxy still routes /challenges/{prism,agent-challenge}."""
    proxy = (REPO_ROOT / "src/base/master/app_proxy.py").read_text(encoding="utf-8")
    cli = (REPO_ROOT / "src/base/cli_app/main.py").read_text(encoding="utf-8")

    # Generic proxy mount stays slug-parameterized (not renamed away).
    assert '"/challenges/{slug}"' in proxy or "'/challenges/{slug}'" in proxy
    assert (
        '"/challenges/{slug}/{path:path}"' in proxy
        or "'/challenges/{slug}/{path:path}'" in proxy
    )
    # Bridge + registry seeds still use the production slug strings.
    assert 'PRISM_SLUG = "prism"' in cli
    assert 'AGENT_CHALLENGE_SLUG = "agent-challenge"' in cli
    assert 'f"/challenges/{slug}"' in proxy or 'f"/challenges/{slug}' in proxy

    # Docs must still advertise the production public prefixes.
    both_slugs_docs = (
        "docs/miner/README.md",
        "docs/miner/how-to.md",
        "docs/SOURCE_OF_TRUTH.md",
        "docs/deploy.md",
    )
    for rel in both_slugs_docs:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for slug in PUBLIC_SLUGS:
            assert slug in text, f"{rel} missing {slug}"

    prism_hub = (REPO_ROOT / "docs/miner/prism/README.md").read_text(encoding="utf-8")
    ac_hub = (REPO_ROOT / "docs/miner/agent-challenge/README.md").read_text(
        encoding="utf-8"
    )
    assert "/challenges/prism" in prism_hub
    assert "/challenges/agent-challenge" in ac_hub


def test_docs_do_not_rename_ghcr_image_names() -> None:
    """Safety companion: docs still document the invariant GHCR names."""
    blobs = "\n".join(
        (REPO_ROOT / rel).read_text(encoding="utf-8")
        for rel in (
            "docs/deploy.md",
            "docs/compose.md",
            "docs/monorepo.md",
            "docs/SOURCE_OF_TRUTH.md",
        )
    )
    for name in INVARIANT_GHCR_NAMES:
        assert name in blobs

    # Must not invent alternate org/name renames in monorepo docs.
    banned = r"ghcr\.io/baseintelligence/(prism-challenge|ac-challenge)\b"
    assert not re.search(banned, blobs)
