"""Atomic, content-addressed capture of validator-owned dynamic rule files."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .schemas import (
    MAX_RULES_BYTES,
    MAX_RULES_FILES,
    RulesSchemaError,
    build_rules_bundle,
    rules_snapshot_sha256,
)

MAX_RULE_SNAPSHOT_ATTEMPTS = 3


class RulesSnapshotCaptureError(RuntimeError):
    """The active rules changed during capture or are not safe to snapshot."""


def capture_rules_bundle(repository_root: Path | str | None = None) -> dict[str, object]:
    """Read a complete old-or-new `.rules` revision, never a mixed file set.

    A bounded before/after content fingerprint (path identity plus SHA-256 of
    exact file bytes) rejects concurrent same-size rewrites instead of
    producing a partial mixed revision. Publishers should atomically replace
    rule files; this guard also protects direct writes. Aggregate size and
    file-count bounds are applied before the snapshot is returned.
    """

    root = Path(repository_root) if repository_root is not None else Path(__file__).parents[3]
    rules_dir = root / ".rules"
    if not rules_dir.is_dir():
        raise RulesSnapshotCaptureError(f"rules directory not found: {rules_dir}")
    for _ in range(MAX_RULE_SNAPSHOT_ATTEMPTS):
        before = _rule_paths_with_content_fingerprint(root, rules_dir)
        if not before:
            raise RulesSnapshotCaptureError(f"rules directory has no Markdown rules: {rules_dir}")
        if len(before) > MAX_RULES_FILES:
            raise RulesSnapshotCaptureError(f"rules directory exceeds {MAX_RULES_FILES} file bound")
        files = {relative: content for relative, content, _ in before}
        aggregate = sum(len(content) for content in files.values())
        if aggregate > MAX_RULES_BYTES:
            raise RulesSnapshotCaptureError(f"rules aggregate bytes exceed {MAX_RULES_BYTES} bound")
        after = _rule_paths_with_content_fingerprint(root, rules_dir)
        if [(relative, digest) for relative, _, digest in before] != [
            (relative, digest) for relative, _, digest in after
        ]:
            continue
        try:
            provisional = build_rules_bundle(revision_id="pending", files=files)
        except RulesSchemaError as exc:
            raise RulesSnapshotCaptureError(str(exc)) from exc
        revision_id = rules_snapshot_sha256(provisional)
        return build_rules_bundle(revision_id=revision_id, files=files)
    raise RulesSnapshotCaptureError("rules changed during atomic snapshot capture")


def _rule_paths_with_content_fingerprint(
    root: Path,
    rules_dir: Path,
) -> list[tuple[str, bytes, str]]:
    """Return sorted ``(relative, content, sha256)`` fingerprints for all rules."""

    items: list[tuple[str, bytes, str]] = []
    for path in sorted(candidate for candidate in rules_dir.glob("*.md") if candidate.is_file()):
        content = path.read_bytes()
        items.append(
            (
                path.relative_to(root).as_posix(),
                content,
                sha256(content).hexdigest(),
            )
        )
    return items
