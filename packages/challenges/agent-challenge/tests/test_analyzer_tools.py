from __future__ import annotations

from pathlib import Path

import pytest

from agent_challenge.analyzer.tools import AnalyzerTools, PathEscapeError, WorkspaceToolError


def test_list_files_returns_capped_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("a\n", encoding="utf-8")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b\n", encoding="utf-8")

    tools = AnalyzerTools(workspace, list_max_entries=2)

    assert tools.list_files() == ["a.txt", "nested/"]


def test_read_file_with_lines_prefixes_and_caps_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools = AnalyzerTools(workspace, read_max_lines=2)

    assert tools.read_file_with_lines("notes.txt") == "1: alpha\n2: beta\n[truncated]"


def test_read_file_denies_parent_absolute_and_symlink_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    tools = AnalyzerTools(workspace)

    with pytest.raises(PathEscapeError):
        tools.read_file_with_lines("../secret.txt")
    with pytest.raises(PathEscapeError):
        tools.read_file_with_lines(str(outside))
    with pytest.raises(PathEscapeError):
        tools.read_file_with_lines("link.txt")


def test_grep_repo_returns_line_matches_and_respects_caps(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "one.txt").write_text("needle\nnope\nneedle again\n", encoding="utf-8")
    (workspace / "two.txt").write_text("needle two\n", encoding="utf-8")

    tools = AnalyzerTools(workspace, grep_max_matches=2)

    assert tools.grep_repo("needle") == "one.txt:1: needle\none.txt:3: needle again\n[truncated]"


def test_restricted_bash_runs_in_workspace_with_sanitized_env(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "hello.txt").write_text("hello\n", encoding="utf-8")
    tools = AnalyzerTools(workspace)

    pwd = tools.run_restricted_bash("pwd")
    read_file = tools.run_restricted_bash("cat hello.txt")
    assert pwd.returncode == 0
    assert pwd.stdout.strip() == str(workspace)
    assert read_file.returncode == 0
    assert read_file.stdout == "hello\n"


def test_restricted_bash_denies_unsafe_commands_and_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    tools = AnalyzerTools(workspace)

    with pytest.raises(WorkspaceToolError):
        tools.run_restricted_bash("python -c 'print(1)'")
    with pytest.raises(PathEscapeError):
        tools.run_restricted_bash("cat /etc/passwd")
    with pytest.raises(PathEscapeError):
        tools.run_restricted_bash("cat ../secret.txt")
    with pytest.raises(PathEscapeError):
        tools.run_restricted_bash("cat link.txt")


def test_restricted_bash_caps_output_and_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "big.txt").write_text("x" * 200, encoding="utf-8")
    tools = AnalyzerTools(workspace, output_max_chars=40, bash_timeout_seconds=0.1)

    capped = tools.run_restricted_bash("cat big.txt")
    timed_out = tools.run_restricted_bash("sleep 1")

    assert len(capped.stdout) == 40
    assert capped.stdout.endswith("[truncated]")
    assert timed_out.timed_out is True
