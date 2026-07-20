from __future__ import annotations

from pathlib import Path

import pytest

from prism_challenge.evaluator.checkpoints import (
    CHECKPOINT_ARTIFACT_MAX_BYTES,
    CheckpointWorkspaceError,
    checkpoint_artifact_logical_size,
    checkpoint_workspace,
    resolve_checkpoint_artifact_path,
)


def test_checkpoint_workspace_uses_plan_mapping() -> None:
    workspace = checkpoint_workspace(Path("/artifacts"), submission_id="submission-123", attempt=3)

    assert workspace.current_dir == Path("/artifacts/checkpoints/submission-123/attempt-3/current")
    assert workspace.resume_dir == Path("/artifacts/checkpoints/submission-123/attempt-2/current")


def test_first_attempt_has_no_resume_checkpoint_dir() -> None:
    workspace = checkpoint_workspace(Path("/artifacts"), submission_id="sub", attempt=1)

    assert workspace.current_dir == Path("/artifacts/checkpoints/sub/attempt-1/current")
    assert workspace.resume_dir is None


@pytest.mark.parametrize(
    "artifact_path, match",
    [
        ("/tmp/model.pt", "relative"),
        ("../model.pt", "'..'"),
        ("nested/../../model.pt", "'..'"),
        (Path("."), "name a file"),
    ],
)
def test_checkpoint_artifact_path_rejects_untrusted_paths(
    tmp_path: Path, artifact_path: str | Path, match: str
) -> None:
    checkpoint_dir = tmp_path / "current"
    checkpoint_dir.mkdir()

    with pytest.raises(CheckpointWorkspaceError, match=match):
        resolve_checkpoint_artifact_path(checkpoint_dir, artifact_path)


def test_checkpoint_artifact_path_rejects_symlink_escape(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "current"
    outside_dir = tmp_path / "outside"
    checkpoint_dir.mkdir()
    outside_dir.mkdir()
    (outside_dir / "model.pt").write_text("outside", encoding="utf-8")
    (checkpoint_dir / "linked").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(CheckpointWorkspaceError, match="symlink"):
        resolve_checkpoint_artifact_path(checkpoint_dir, "linked/model.pt")


def test_checkpoint_size_walk_accepts_exact_decimal_10g_sparse_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "current"
    checkpoint_dir.mkdir()
    model = checkpoint_dir / "model.pt"
    with model.open("wb") as file:
        file.truncate(CHECKPOINT_ARTIFACT_MAX_BYTES)

    assert checkpoint_artifact_logical_size(checkpoint_dir) == CHECKPOINT_ARTIFACT_MAX_BYTES


def test_checkpoint_size_walk_rejects_one_byte_over_decimal_10g_sparse_file(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "current"
    checkpoint_dir.mkdir()
    model = checkpoint_dir / "model.pt"
    with model.open("wb") as file:
        file.truncate(CHECKPOINT_ARTIFACT_MAX_BYTES + 1)

    with pytest.raises(CheckpointWorkspaceError, match="exceed"):
        checkpoint_artifact_logical_size(checkpoint_dir)


def test_checkpoint_size_walk_rejects_symlink_artifact(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "current"
    outside = tmp_path / "outside.pt"
    checkpoint_dir.mkdir()
    outside.write_text("outside", encoding="utf-8")
    (checkpoint_dir / "model.pt").symlink_to(outside)

    with pytest.raises(CheckpointWorkspaceError, match="symlink"):
        checkpoint_artifact_logical_size(checkpoint_dir)


def test_checkpoint_size_for_manifest_paths_rejects_absolute_path(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "current"
    checkpoint_dir.mkdir()

    with pytest.raises(CheckpointWorkspaceError, match="relative"):
        checkpoint_artifact_logical_size(checkpoint_dir, [tmp_path / "outside.pt"])
