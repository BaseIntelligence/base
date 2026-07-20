"""Monorepo uv workspace skeleton contracts (VAL-MONO-001 / VAL-MONO-002)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import base.challenge_sdk as challenge_sdk

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_uv_workspace_lists_challenge_members() -> None:
    """Root pyproject declares packages/challenges/{prism,agent-challenge}."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    members = data["tool"]["uv"]["workspace"]["members"]
    assert "packages/challenges/prism" in members
    assert "packages/challenges/agent-challenge" in members
    assert data["tool"]["uv"]["sources"]["base"] == {"workspace": True}


def test_challenge_member_stubs_exist_with_import_packages() -> None:
    """Stub member trees are valid packages (subtree product code lands later)."""
    prism = REPO_ROOT / "packages/challenges/prism"
    agent = REPO_ROOT / "packages/challenges/agent-challenge"

    assert (prism / "pyproject.toml").is_file()
    assert (agent / "pyproject.toml").is_file()
    assert (prism / "src/prism_challenge/__init__.py").is_file()
    assert (agent / "src/agent_challenge/__init__.py").is_file()

    prism_proj = tomllib.loads((prism / "pyproject.toml").read_text(encoding="utf-8"))
    agent_proj = tomllib.loads((agent / "pyproject.toml").read_text(encoding="utf-8"))
    assert prism_proj["project"]["name"] == "prism-challenge"
    assert agent_proj["project"]["name"] == "agent-challenge"


def test_base_package_stays_at_src_base() -> None:
    """ADR choice: keep base installable from root src/base (minimal churn)."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "base"
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/base"
    ]
    assert (REPO_ROOT / "src/base").is_dir()
    assert challenge_sdk.__file__ is not None
    assert "src/base/challenge_sdk" in Path(challenge_sdk.__file__).as_posix()


def test_monorepo_adr_documents_layout_choice() -> None:
    text = (REPO_ROOT / "docs/monorepo.md").read_text(encoding="utf-8")
    assert "src/base" in text
    assert "packages/challenges/prism" in text
    assert "packages/challenges/agent-challenge" in text
    assert "workspace" in text.lower()
