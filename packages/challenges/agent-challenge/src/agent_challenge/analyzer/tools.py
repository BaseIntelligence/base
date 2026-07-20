from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LIST_MAX_ENTRIES = 1_000
DEFAULT_READ_MAX_BYTES = 64_000
DEFAULT_READ_MAX_LINES = 1_000
DEFAULT_GREP_MAX_MATCHES = 200
DEFAULT_OUTPUT_MAX_CHARS = 64_000
DEFAULT_BASH_TIMEOUT_SECONDS = 5.0
MAX_BASH_TIMEOUT_SECONDS = 10.0
MAX_COMMAND_CHARS = 4_096

_TRUNCATED_MARKER = "\n[truncated]"
_SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
_ALLOWED_COMMANDS = frozenset(
    {
        "cat",
        "find",
        "grep",
        "head",
        "ls",
        "pwd",
        "sleep",
        "sort",
        "tail",
        "uniq",
        "wc",
    }
)
_SHELL_OPERATOR_CHARS = frozenset(";&|<>$`(){}[]\n\r")


class WorkspaceToolError(ValueError):
    pass


class PathEscapeError(WorkspaceToolError):
    pass


@dataclass(frozen=True)
class RestrictedBashResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class AnalyzerTools:
    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        list_max_entries: int = DEFAULT_LIST_MAX_ENTRIES,
        read_max_bytes: int = DEFAULT_READ_MAX_BYTES,
        read_max_lines: int = DEFAULT_READ_MAX_LINES,
        grep_max_matches: int = DEFAULT_GREP_MAX_MATCHES,
        output_max_chars: int = DEFAULT_OUTPUT_MAX_CHARS,
        bash_timeout_seconds: float = DEFAULT_BASH_TIMEOUT_SECONDS,
    ) -> None:
        root = Path(workspace_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise WorkspaceToolError("workspace_root must be an existing directory")
        self.workspace_root = root
        self.list_max_entries = max(list_max_entries, 1)
        self.read_max_bytes = max(read_max_bytes, 1)
        self.read_max_lines = max(read_max_lines, 1)
        self.grep_max_matches = max(grep_max_matches, 1)
        self.output_max_chars = max(output_max_chars, len(_TRUNCATED_MARKER))
        self.bash_timeout_seconds = min(
            max(float(bash_timeout_seconds), 0.1),
            MAX_BASH_TIMEOUT_SECONDS,
        )

    def list_files(self, path: str | os.PathLike[str] = ".") -> list[str]:
        target = self._resolve_path(path)
        if not target.exists():
            raise WorkspaceToolError(f"path does not exist: {path}")
        if target.is_file():
            return [self._relative_name(target)]
        if not target.is_dir():
            raise WorkspaceToolError(f"path is not a file or directory: {path}")

        entries: list[str] = []
        for child in sorted(target.rglob("*")):
            resolved_child = child.resolve(strict=False)
            if not self._is_under_workspace(resolved_child):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{self._relative_name(resolved_child)}{suffix}")
            if len(entries) >= self.list_max_entries:
                break
        return entries

    def read_file_with_lines(self, path: str | os.PathLike[str]) -> str:
        target = self._resolve_existing_file(path)
        data = target.read_bytes()[: self.read_max_bytes + 1]
        source_truncated = len(data) > self.read_max_bytes
        if source_truncated:
            data = data[: self.read_max_bytes]
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        line_truncated = len(lines) > self.read_max_lines
        numbered = [
            f"{index}: {line}" for index, line in enumerate(lines[: self.read_max_lines], 1)
        ]
        if source_truncated or line_truncated:
            numbered.append("[truncated]")
        return _cap_text("\n".join(numbered), self.output_max_chars)

    def grep_repo(
        self,
        pattern: str,
        path: str | os.PathLike[str] = ".",
        *,
        case_sensitive: bool = True,
    ) -> str:
        target = self._resolve_path(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            expression = re.compile(pattern, flags)
        except re.error as exc:
            raise WorkspaceToolError(f"invalid grep pattern: {exc}") from exc

        matches: list[str] = []
        for file_path in self._iter_files(target):
            text = _read_limited_text(file_path, self.read_max_bytes)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    matches.append(f"{self._relative_name(file_path)}:{line_number}: {line}")
                    if len(matches) >= self.grep_max_matches:
                        return _cap_text(
                            "\n".join(matches + ["[truncated]"]),
                            self.output_max_chars,
                        )
        return _cap_text("\n".join(matches), self.output_max_chars)

    def run_restricted_bash(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> RestrictedBashResult:
        timeout = self._effective_timeout(timeout_seconds)
        self._validate_restricted_command(command)
        env = dict(_SAFE_ENV)
        env["HOME"] = str(self.workspace_root)
        try:
            completed = subprocess.run(
                ["bash", "--noprofile", "--norc", "-c", command],
                cwd=self.workspace_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_timeout_output(exc.stdout)
            stderr = _coerce_timeout_output(exc.stderr)
            return RestrictedBashResult(
                returncode=-1,
                stdout=_cap_text(stdout, self.output_max_chars),
                stderr=_cap_text(stderr, self.output_max_chars),
                timed_out=True,
            )
        return RestrictedBashResult(
            returncode=completed.returncode,
            stdout=_cap_text(completed.stdout, self.output_max_chars),
            stderr=_cap_text(completed.stderr, self.output_max_chars),
            timed_out=False,
        )

    def _resolve_existing_file(self, path: str | os.PathLike[str]) -> Path:
        target = self._resolve_path(path)
        if not target.exists():
            raise WorkspaceToolError(f"file does not exist: {path}")
        if not target.is_file():
            raise WorkspaceToolError(f"path is not a file: {path}")
        return target

    def _resolve_path(self, path: str | os.PathLike[str]) -> Path:
        candidate_path = Path(path)
        if candidate_path.is_absolute():
            raise PathEscapeError("absolute paths are not allowed")
        if ".." in candidate_path.parts:
            raise PathEscapeError("parent path segments are not allowed")
        resolved = (self.workspace_root / candidate_path).resolve(strict=False)
        if not self._is_under_workspace(resolved):
            raise PathEscapeError("path escapes workspace")
        return resolved

    def _is_under_workspace(self, path: Path) -> bool:
        return path == self.workspace_root or self.workspace_root in path.parents

    def _relative_name(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()

    def _iter_files(self, target: Path) -> Iterable[Path]:
        if not target.exists():
            raise WorkspaceToolError(f"path does not exist: {self._relative_name(target)}")
        if target.is_file():
            yield target
            return
        if not target.is_dir():
            relative_target = self._relative_name(target)
            raise WorkspaceToolError(f"path is not a file or directory: {relative_target}")
        for child in sorted(target.rglob("*")):
            resolved_child = child.resolve(strict=False)
            if resolved_child.is_file() and self._is_under_workspace(resolved_child):
                yield resolved_child

    def _effective_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is None:
            return self.bash_timeout_seconds
        return min(max(float(timeout_seconds), 0.1), MAX_BASH_TIMEOUT_SECONDS)

    def _validate_restricted_command(self, command: str) -> None:
        if not command or len(command) > MAX_COMMAND_CHARS:
            raise WorkspaceToolError("command must be non-empty and bounded")
        if any(character in _SHELL_OPERATOR_CHARS for character in command):
            raise WorkspaceToolError("shell operators and expansions are not allowed")
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise WorkspaceToolError(f"invalid command: {exc}") from exc
        if not tokens:
            raise WorkspaceToolError("command must be non-empty")
        executable = Path(tokens[0]).name
        if executable not in _ALLOWED_COMMANDS or tokens[0] != executable:
            raise WorkspaceToolError(f"command is not allowed: {tokens[0]}")
        for token in tokens[1:]:
            self._validate_command_token(token)

    def _validate_command_token(self, token: str) -> None:
        if not token or token.startswith("-"):
            return
        token_path = Path(token)
        if token_path.is_absolute():
            raise PathEscapeError("absolute command paths are not allowed")
        if ".." in token_path.parts:
            raise PathEscapeError("parent path segments are not allowed in commands")
        candidate = (self.workspace_root / token_path).resolve(strict=False)
        if candidate.exists() and not self._is_under_workspace(candidate):
            raise PathEscapeError("command path escapes workspace")


def list_files(
    workspace_root: str | os.PathLike[str],
    path: str | os.PathLike[str] = ".",
    **kwargs: object,
) -> list[str]:
    return AnalyzerTools(workspace_root, **kwargs).list_files(path)


def read_file_with_lines(
    workspace_root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    **kwargs: object,
) -> str:
    return AnalyzerTools(workspace_root, **kwargs).read_file_with_lines(path)


def grep_repo(
    workspace_root: str | os.PathLike[str],
    pattern: str,
    path: str | os.PathLike[str] = ".",
    **kwargs: object,
) -> str:
    return AnalyzerTools(workspace_root, **kwargs).grep_repo(pattern, path)


def run_restricted_bash(
    workspace_root: str | os.PathLike[str],
    command: str,
    **kwargs: object,
) -> RestrictedBashResult:
    return AnalyzerTools(workspace_root, **kwargs).run_restricted_bash(command)


def _read_limited_text(path: Path, max_bytes: int) -> str | None:
    data = path.read_bytes()[: max_bytes + 1]
    if b"\0" in data:
        return None
    return data[:max_bytes].decode("utf-8", errors="replace")


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(_TRUNCATED_MARKER)
    return f"{text[:keep]}{_TRUNCATED_MARKER}"


def _coerce_timeout_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output
