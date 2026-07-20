from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_ARTIFACT_MAX_BYTES = 10_000_000_000
CHECKPOINTS_DIRNAME = "checkpoints"
CURRENT_CHECKPOINT_DIRNAME = "current"
CHECKPOINT_METADATA_API_VERSION = 1
CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.v1.json"
CHECKPOINT_METADATA_FIELDS = frozenset(
    {
        "checkpoint_api_version",
        "submission_id",
        "attempt",
        "code_hash",
        "arch_hash",
        "recipe_fingerprint",
        "created_at",
        "checkpoint_path",
        "hook_return",
        "world_size",
        "rank_writer",
        "checkpoint_dir",
        "bytes_total",
    }
)


class CheckpointWorkspaceError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointWorkspace:
    artifact_output: Path
    submission_id: str
    attempt: int
    current_dir: Path
    resume_dir: Path | None


def checkpoint_workspace(
    artifact_output: Path, *, submission_id: str, attempt: int
) -> CheckpointWorkspace:
    if attempt < 1:
        raise CheckpointWorkspaceError("checkpoint attempt must be >= 1")
    _validate_submission_id(submission_id)

    artifact_output_path = Path(artifact_output)
    attempt_base = artifact_output_path / CHECKPOINTS_DIRNAME / submission_id
    current_dir = attempt_base / f"attempt-{attempt}" / CURRENT_CHECKPOINT_DIRNAME
    resume_dir = (
        attempt_base / f"attempt-{attempt - 1}" / CURRENT_CHECKPOINT_DIRNAME
        if attempt > 1
        else None
    )
    return CheckpointWorkspace(
        artifact_output=artifact_output_path,
        submission_id=submission_id,
        attempt=attempt,
        current_dir=current_dir,
        resume_dir=resume_dir,
    )


def resolve_checkpoint_artifact_path(checkpoint_dir: Path, artifact_path: str | Path) -> Path:
    relative_path = _validate_relative_artifact_path(artifact_path)
    root = Path(checkpoint_dir)
    target = root / relative_path
    _reject_symlink_path_components(root, target)
    _require_within_checkpoint_dir(root, target)
    return target


def checkpoint_artifact_logical_size(
    checkpoint_dir: Path,
    artifact_paths: Iterable[str | Path] | None = None,
    *,
    max_bytes: int = CHECKPOINT_ARTIFACT_MAX_BYTES,
) -> int:
    root = Path(checkpoint_dir)
    if artifact_paths is None:
        paths = _walk_checkpoint_files(root)
    else:
        paths = tuple(resolve_checkpoint_artifact_path(root, item) for item in artifact_paths)

    total_bytes = 0
    for path in paths:
        if path.is_symlink():
            raise CheckpointWorkspaceError(f"checkpoint artifact path contains symlink: {path}")
        if not path.is_file():
            raise CheckpointWorkspaceError(f"checkpoint artifact is not a regular file: {path}")
        total_bytes += path.stat(follow_symlinks=False).st_size
        if total_bytes > max_bytes:
            raise CheckpointWorkspaceError(
                f"checkpoint artifacts exceed {max_bytes} bytes: {total_bytes}"
            )
    return total_bytes


def metadata_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).parent / CHECKPOINT_METADATA_FILENAME


def persist_checkpoint(
    workspace: CheckpointWorkspace,
    *,
    state_files: Mapping[str, bytes],
    code_hash: str,
    arch_hash: str,
    recipe_fingerprint: str,
    created_at: str,
    hook_return: Mapping[str, Any] | None = None,
    world_size: int = 1,
) -> Path:
    """Persist a crash-recovery checkpoint into the workspace ``current_dir`` with a v1 sidecar.

    Writes each ``state_files`` entry (relative name -> bytes) under the per-submission/per-attempt
    ``current_dir`` through the path-safe resolver (no traversal/symlink escape), bounds the total
    logical size, and writes the schema-valid ``checkpoint_metadata.v1.json`` sidecar
    (``rank_writer == 0``, ``world_size >= 1``, bounded ``bytes_total``). Returns ``current_dir``.
    """
    if not state_files:
        raise CheckpointWorkspaceError("checkpoint must contain at least one state file")
    if world_size < 1:
        raise CheckpointWorkspaceError("checkpoint world_size must be >= 1")
    current = workspace.current_dir
    current.mkdir(parents=True, exist_ok=True)
    for name, data in state_files.items():
        target = resolve_checkpoint_artifact_path(current, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))
    bytes_total = checkpoint_artifact_logical_size(current)
    primary_name = next(iter(state_files))
    metadata = {
        "checkpoint_api_version": CHECKPOINT_METADATA_API_VERSION,
        "submission_id": workspace.submission_id,
        "attempt": workspace.attempt,
        "code_hash": code_hash,
        "arch_hash": arch_hash,
        "recipe_fingerprint": recipe_fingerprint,
        "created_at": created_at,
        "checkpoint_path": primary_name,
        "hook_return": dict(hook_return) if hook_return is not None else None,
        "world_size": world_size,
        "rank_writer": 0,
        "checkpoint_dir": str(current),
        "bytes_total": bytes_total,
    }
    write_checkpoint_metadata(metadata_path_for_checkpoint(current / primary_name), metadata)
    return current


def load_checkpoint_metadata(metadata_path: Path) -> dict[str, Any]:
    path = Path(metadata_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointWorkspaceError(f"checkpoint metadata is malformed JSON: {path}") from exc
    except OSError as exc:
        raise CheckpointWorkspaceError(f"checkpoint metadata cannot be read: {path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointWorkspaceError("checkpoint metadata must be a JSON object")
    validate_checkpoint_metadata_schema(payload)
    return payload


def write_checkpoint_metadata(metadata_path: Path, metadata: Mapping[str, Any]) -> None:
    validate_checkpoint_metadata_schema(metadata)
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")


def validate_checkpoint_metadata_schema(metadata: Mapping[str, Any]) -> None:
    keys = set(metadata)
    if keys != CHECKPOINT_METADATA_FIELDS:
        missing = CHECKPOINT_METADATA_FIELDS - keys
        extra = keys - CHECKPOINT_METADATA_FIELDS
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"extra: {', '.join(sorted(extra))}")
        raise CheckpointWorkspaceError(
            f"checkpoint metadata fields do not match v1 schema ({'; '.join(details)})"
        )
    if metadata["checkpoint_api_version"] != CHECKPOINT_METADATA_API_VERSION:
        raise CheckpointWorkspaceError("checkpoint metadata version is not supported")
    if not isinstance(metadata["submission_id"], str) or not metadata["submission_id"]:
        raise CheckpointWorkspaceError(
            "checkpoint metadata submission_id must be a non-empty string"
        )
    if not isinstance(metadata["attempt"], int) or metadata["attempt"] < 1:
        raise CheckpointWorkspaceError("checkpoint metadata attempt must be >= 1")
    for field in ("code_hash", "arch_hash", "recipe_fingerprint", "created_at"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise CheckpointWorkspaceError(
                f"checkpoint metadata {field} must be a non-empty string"
            )
    for field in ("checkpoint_path", "checkpoint_dir"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise CheckpointWorkspaceError(
                f"checkpoint metadata {field} must be a non-empty string"
            )
    if metadata["hook_return"] is not None and not isinstance(metadata["hook_return"], dict):
        raise CheckpointWorkspaceError("checkpoint metadata hook_return must be an object or null")
    if not isinstance(metadata["world_size"], int) or metadata["world_size"] < 1:
        raise CheckpointWorkspaceError("checkpoint metadata world_size must be >= 1")
    if metadata["rank_writer"] != 0:
        raise CheckpointWorkspaceError("checkpoint metadata rank_writer must be 0")
    if not isinstance(metadata["bytes_total"], int) or metadata["bytes_total"] < 0:
        raise CheckpointWorkspaceError(
            "checkpoint metadata bytes_total must be a non-negative integer"
        )
    if metadata["bytes_total"] > CHECKPOINT_ARTIFACT_MAX_BYTES:
        raise CheckpointWorkspaceError(
            f"checkpoint metadata bytes_total exceeds {CHECKPOINT_ARTIFACT_MAX_BYTES}"
        )
    try:
        json.dumps(metadata, sort_keys=True)
    except TypeError as exc:
        raise CheckpointWorkspaceError("checkpoint metadata must be JSON serializable") from exc


def _validate_submission_id(submission_id: str) -> None:
    if not submission_id:
        raise CheckpointWorkspaceError("checkpoint submission_id must not be empty")
    path = Path(submission_id)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise CheckpointWorkspaceError("checkpoint submission_id must be a single path segment")


def _validate_relative_artifact_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        raise CheckpointWorkspaceError("checkpoint artifact path must be relative")
    if path == Path("."):
        raise CheckpointWorkspaceError("checkpoint artifact path must name a file")
    if not path.parts:
        raise CheckpointWorkspaceError("checkpoint artifact path must not be empty")
    if any(part == ".." for part in path.parts):
        raise CheckpointWorkspaceError("checkpoint artifact path must not contain '..'")
    return path


def _walk_checkpoint_files(root: Path) -> tuple[Path, ...]:
    if root.is_symlink():
        raise CheckpointWorkspaceError(f"checkpoint artifact path contains symlink: {root}")
    if not root.exists():
        raise CheckpointWorkspaceError(f"checkpoint directory does not exist: {root}")
    if not root.is_dir():
        raise CheckpointWorkspaceError(f"checkpoint path is not a directory: {root}")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CheckpointWorkspaceError(f"checkpoint artifact path contains symlink: {path}")
        _require_within_checkpoint_dir(root, path)
        if path.is_file():
            files.append(path)
    return tuple(files)


def _reject_symlink_path_components(root: Path, target: Path) -> None:
    current = root
    if current.is_symlink():
        raise CheckpointWorkspaceError(f"checkpoint artifact path contains symlink: {current}")
    for part in target.relative_to(root).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise CheckpointWorkspaceError(f"checkpoint artifact path contains symlink: {current}")


def _require_within_checkpoint_dir(root: Path, target: Path) -> None:
    root_resolved = root.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise CheckpointWorkspaceError(
            f"checkpoint artifact path escapes checkpoint directory: {target}"
        ) from exc
