from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_base_release_version_sources_match() -> None:
    pyproject = _pyproject()
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    version = pyproject["project"]["version"]

    assert f'name = "base"\nversion = "{version}"' in lock


def test_versioning_policy_documents_release_contract() -> None:
    policy = (ROOT / "docs" / "versioning.md").read_text(encoding="utf-8")

    required = [
        "Semantic Versioning",
        "pyproject.toml",
        "GHCR",
        "GitHub Release",
        "generate",
        "base-master",
        "main",
        "type=semver,pattern={{version}}",
        "sha256",
        "latest",
        "Production",
    ]
    for token in required:
        assert token in policy


def test_github_workflow_publishes_canonical_semver_tags() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "v*.*.*" in workflow
    assert "type=semver,pattern={{version}}" in workflow
    assert "type=semver,pattern={{raw}}" in workflow
    assert "type=ref,event=tag" not in workflow
