"""Monorepo deploy/miner docs + public slug contracts (VAL-MONO-011..014).

Updated for minimal shipping docs: compose + validator + miner getting-started
carry layout/slug/GHCR facts; monorepo essays and miner hub trees are gone.
"""

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
    """VAL-MONO-011: compose docs know monorepo paths + public GHCR names."""
    compose = (REPO_ROOT / "docs/compose.md").read_text(encoding="utf-8")
    install = (REPO_ROOT / "deploy/compose/install-master.sh").read_text(
        encoding="utf-8"
    )
    swarm = (REPO_ROOT / "deploy/swarm/README.md").read_text(encoding="utf-8")

    for text in (compose, install):
        assert "packages/challenges/prism" in text or "PRISM" in text
        assert (
            "packages/challenges/agent-challenge" in text
            or "agent-challenge" in text
            or "PRISM" in text
        )

    assert "packages/challenges/prism" in compose
    assert "packages/challenges/agent-challenge" in compose
    assert "ghcr.io/baseintelligence/prism" in compose
    assert "--build-context monorepo=." in compose or "monorepo=." in compose

    # Swarm historical note must mention monorepo SoT without becoming the path.
    assert "packages/challenges" in swarm
    assert "NOT A SUPPORTED INSTALL DESTINATION" in swarm or "HISTORICAL" in swarm

    for name in (
        "ghcr.io/baseintelligence/prism",
        "ghcr.io/baseintelligence/base-master",
        "ghcr.io/baseintelligence/base-validator-runtime",
    ):
        assert name in compose


def test_miner_docs_unified_under_base() -> None:
    """VAL-MONO-012: thin miner day-1 doc points at monorepo packages + public slugs."""
    gs_path = REPO_ROOT / "docs/miner/getting-started.md"
    assert gs_path.is_file()
    # Hub trees collapsed; package sources remain the product home.
    assert not (REPO_ROOT / "docs/miner/prism").exists()
    assert not (REPO_ROOT / "docs/miner/agent-challenge").exists()
    assert not (REPO_ROOT / "docs/miner/how-to.md").exists()
    assert not (REPO_ROOT / "docs/miner/README.md").exists()

    gs = gs_path.read_text(encoding="utf-8")
    assert "packages/challenges/prism" in gs
    assert "packages/challenges/agent-challenge" in gs
    assert "/challenges/prism" in gs
    assert "/challenges/agent-challenge" in gs
    assert "https://chain.joinbase.ai" in gs
    assert "openapi.json" in gs

    prism_pkg = (REPO_ROOT / "packages/challenges/prism/README.md").read_text(
        encoding="utf-8"
    )
    ac_pkg = (REPO_ROOT / "packages/challenges/agent-challenge/README.md").read_text(
        encoding="utf-8"
    )
    assert "prism_challenge" in prism_pkg
    assert "agent_challenge" in ac_pkg
    assert "/challenges/prism" in prism_pkg
    assert "/challenges/agent-challenge" in ac_pkg


def test_source_of_truth_is_package_and_agents_not_essay() -> None:
    """VAL-MONO-013: SoT is packages + AGENTS; no SOURCE_OF_TRUTH essay."""
    assert not (REPO_ROOT / "docs/SOURCE_OF_TRUTH.md").exists()
    assert not (REPO_ROOT / "docs/monorepo.md").exists()

    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pkg_prism = (REPO_ROOT / "packages/challenges/prism/README.md").read_text(
        encoding="utf-8"
    )
    pkg_ac = (REPO_ROOT / "packages/challenges/agent-challenge/README.md").read_text(
        encoding="utf-8"
    )

    assert "packages/challenges" in agents
    assert "BaseIntelligence/base" in agents or "base monorepo" in agents.lower()
    assert "packages/challenges/prism" in pkg_prism or "prism_challenge" in pkg_prism
    assert (
        "packages/challenges/agent-challenge" in pkg_ac
        or "agent_challenge" in pkg_ac
    )
    for slug in PUBLIC_SLUGS:
        assert slug in pkg_prism or slug in pkg_ac or slug in root_readme
    # Root README must not reintroduce the essay filenames.
    assert "SOURCE_OF_TRUTH" not in root_readme
    assert "docs/monorepo.md" not in root_readme


def test_public_api_slugs_unchanged_in_master_proxy() -> None:
    """VAL-MONO-014: master proxy still routes /challenges/{prism,agent-challenge}."""
    proxy = (REPO_ROOT / "src/base/master/app_proxy.py").read_text(encoding="utf-8")
    cli = (REPO_ROOT / "src/base/cli_app/main.py").read_text(encoding="utf-8")

    assert '"/challenges/{slug}"' in proxy or "'/challenges/{slug}'" in proxy
    assert (
        '"/challenges/{slug}/{path:path}"' in proxy
        or "'/challenges/{slug}/{path:path}'" in proxy
    )
    assert 'PRISM_SLUG = "prism"' in cli
    assert 'AGENT_CHALLENGE_SLUG = "agent-challenge"' in cli
    assert 'f"/challenges/{slug}"' in proxy or 'f"/challenges/{slug}' in proxy

    # Remaining shipping docs advertise production public prefixes.
    both_slugs_docs = (
        "docs/miner/getting-started.md",
        "docs/compose.md",
        "README.md",
    )
    for rel in both_slugs_docs:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for slug in PUBLIC_SLUGS:
            assert slug in text, f"{rel} missing {slug}"

    prism_pkg = (REPO_ROOT / "packages/challenges/prism/README.md").read_text(
        encoding="utf-8"
    )
    ac_pkg = (REPO_ROOT / "packages/challenges/agent-challenge/README.md").read_text(
        encoding="utf-8"
    )
    assert "/challenges/prism" in prism_pkg
    assert "/challenges/agent-challenge" in ac_pkg


def test_docs_do_not_rename_ghcr_image_names() -> None:
    """Safety companion: remaining docs still document the invariant GHCR names."""
    blobs = "\n".join(
        (REPO_ROOT / rel).read_text(encoding="utf-8")
        for rel in (
            "docs/compose.md",
            "packages/challenges/prism/README.md",
            "packages/challenges/agent-challenge/README.md",
            "packages/challenges/README.md",
        )
    )
    for name in INVARIANT_GHCR_NAMES:
        assert name in blobs, name

    banned = r"ghcr\.io/baseintelligence/(prism-challenge|ac-challenge)\b"
    assert not re.search(banned, blobs)
