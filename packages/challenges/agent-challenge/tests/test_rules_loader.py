from __future__ import annotations

import hashlib

import pytest

from agent_challenge.rules import RulesLoadError, load_rules


def test_default_rules_bundle_loads_visible_policy_files():
    bundle = load_rules()

    assert bundle.files == [
        ".rules/acceptance.md",
        ".rules/anti-cheat.md",
        ".rules/hardcoding.md",
        ".rules/security.md",
    ]
    assert len(bundle.rules_version) == 64
    assert "# Acceptance Policy" in bundle.policy_text
    assert "# Anti-Cheat Policy" in bundle.policy_text
    assert "# Hardcoding Policy" in bundle.policy_text
    assert "# Security Policy" in bundle.policy_text


def test_rules_version_hashes_sorted_relative_paths_and_contents(tmp_path):
    rules_dir = tmp_path / ".rules"
    rules_dir.mkdir()
    (rules_dir / "z.md").write_text("last\n", encoding="utf-8")
    (rules_dir / "a.md").write_text("first\n", encoding="utf-8")

    bundle = load_rules(tmp_path)

    expected_digest = hashlib.sha256()
    for relative_path, contents in (
        (".rules/a.md", b"first\n"),
        (".rules/z.md", b"last\n"),
    ):
        expected_digest.update(relative_path.encode("utf-8"))
        expected_digest.update(b"\0")
        expected_digest.update(contents)
        expected_digest.update(b"\0")

    assert bundle.files == [".rules/a.md", ".rules/z.md"]
    assert bundle.policy_text == "first\n\nlast\n"
    assert bundle.rules_version == expected_digest.hexdigest()


def test_missing_rules_directory_raises_controlled_error(tmp_path):
    with pytest.raises(RulesLoadError, match="rules directory not found"):
        load_rules(tmp_path)


def test_empty_rules_directory_raises_controlled_error(tmp_path):
    (tmp_path / ".rules").mkdir()

    with pytest.raises(RulesLoadError, match="no Markdown rules"):
        load_rules(tmp_path)
