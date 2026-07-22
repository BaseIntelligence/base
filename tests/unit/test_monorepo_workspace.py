"""Monorepo uv workspace contracts (VAL-MONO-001..006)."""

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


def test_challenge_members_contain_product_packages() -> None:
    """Imported member trees carry real packages (not empty stubs)."""
    prism = REPO_ROOT / "packages/challenges/prism"
    agent = REPO_ROOT / "packages/challenges/agent-challenge"

    assert (prism / "pyproject.toml").is_file()
    assert (agent / "pyproject.toml").is_file()
    assert (prism / "src/prism_challenge/__init__.py").is_file()
    assert (agent / "src/agent_challenge/__init__.py").is_file()
    assert (agent / "src/agent_challenge_runner/__init__.py").is_file()
    # Product surface beyond package shell
    assert (prism / "src/prism_challenge/app.py").is_file()
    assert (agent / "src/agent_challenge/app.py").is_file()

    prism_proj = tomllib.loads((prism / "pyproject.toml").read_text(encoding="utf-8"))
    agent_proj = tomllib.loads((agent / "pyproject.toml").read_text(encoding="utf-8"))
    assert prism_proj["project"]["name"] == "prism-challenge"
    assert agent_proj["project"]["name"] == "agent-challenge"


def test_challenges_path_depend_on_workspace_base() -> None:
    """Challenge deps use workspace base, not release wheel or floating git+base."""
    prism_proj = tomllib.loads(
        (REPO_ROOT / "packages/challenges/prism/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    agent_proj = tomllib.loads(
        (REPO_ROOT / "packages/challenges/agent-challenge/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert root["tool"]["uv"]["sources"]["base"] == {"workspace": True}

    def _dep_names(proj: dict) -> list[str]:
        deps = proj["project"]["dependencies"]
        names: list[str] = []
        for raw in deps:
            name = raw.split("@", 1)[0].split(">", 1)[0].split("<", 1)[0]
            name = name.split("=", 1)[0].split("[", 1)[0].strip().lower()
            names.append(name)
        return names

    prism_deps = prism_proj["project"]["dependencies"]
    agent_deps = agent_proj["project"]["dependencies"]
    assert "base" in _dep_names(prism_proj)
    assert "base" in _dep_names(agent_proj)
    joined = "\n".join(prism_deps + agent_deps)
    assert "releases/download" not in joined
    assert "git+https://github.com/BaseIntelligence/base" not in joined
    assert "base @ " not in joined
    assert all(
        d.strip() == "base" or not d.strip().startswith("base ")
        for d in prism_deps + agent_deps
        if "base" in d
    )


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


def test_monorepo_adr_documents_layout_and_sdk_sharing() -> None:
    """Layout/SDK SoT is AGENTS + package READMEs (monorepo essay removed)."""
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    pkg_index = (REPO_ROOT / "packages/challenges/README.md").read_text(
        encoding="utf-8"
    )
    prism = (REPO_ROOT / "packages/challenges/prism/README.md").read_text(
        encoding="utf-8"
    )
    blob = "\n".join((agents, pkg_index, prism))
    assert "src/base" in blob
    assert "packages/challenges/prism" in blob
    assert "packages/challenges/agent-challenge" in blob
    assert "workspace" in blob.lower()
    assert "challenge_sdk" in blob
    assert "base.challenge_sdk" in blob
    assert not (REPO_ROOT / "docs/monorepo.md").exists()
