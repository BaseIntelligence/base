from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.checkpoint_publisher import (
    DEFAULT_CHECKPOINT_REPO_ID,
    CheckpointUpload,
    HuggingFaceCheckpointPublisher,
    MockCheckpointPublisher,
    revision_for,
)
from prism_challenge.evaluator.checkpoints import (
    CheckpointWorkspaceError,
    checkpoint_workspace,
    load_checkpoint_metadata,
    metadata_path_for_checkpoint,
    persist_checkpoint,
)

# --- VAL-PRISM-017: validator persists checkpoints into a path-safe workspace ---------------------


def test_persist_checkpoint_writes_workspace_and_v1_metadata(tmp_path):
    workspace = checkpoint_workspace(tmp_path / "artifacts", submission_id="sub-1", attempt=2)
    current = persist_checkpoint(
        workspace,
        state_files={"model.pt": b"trained-weights-bytes"},
        code_hash="codehash",
        arch_hash="archhash",
        recipe_fingerprint="recipe",
        created_at="2026-06-27T00:00:00Z",
        world_size=1,
    )
    assert current == workspace.current_dir
    assert (current / "model.pt").read_bytes() == b"trained-weights-bytes"

    metadata = load_checkpoint_metadata(metadata_path_for_checkpoint(current / "model.pt"))
    assert metadata["rank_writer"] == 0
    assert metadata["world_size"] == 1
    assert metadata["submission_id"] == "sub-1"
    assert metadata["attempt"] == 2
    assert metadata["bytes_total"] == len(b"trained-weights-bytes")
    assert metadata["checkpoint_path"] == "model.pt"


def test_persist_checkpoint_rejects_traversal(tmp_path):
    workspace = checkpoint_workspace(tmp_path / "artifacts", submission_id="sub-1", attempt=1)
    with pytest.raises(CheckpointWorkspaceError, match="'..'"):
        persist_checkpoint(
            workspace,
            state_files={"../escape.pt": b"x"},
            code_hash="c",
            arch_hash="a",
            recipe_fingerprint="r",
            created_at="2026-06-27T00:00:00Z",
        )


def test_persist_checkpoint_requires_a_file(tmp_path):
    workspace = checkpoint_workspace(tmp_path / "artifacts", submission_id="sub-1", attempt=1)
    with pytest.raises(CheckpointWorkspaceError, match="at least one"):
        persist_checkpoint(
            workspace,
            state_files={},
            code_hash="c",
            arch_hash="a",
            recipe_fingerprint="r",
            created_at="2026-06-27T00:00:00Z",
        )


# --- VAL-PRISM-018: checkpoints publish via a MOCKED publisher interface (no real network) --------


def _persisted_upload(tmp_path) -> tuple[CheckpointUpload, Path]:
    workspace = checkpoint_workspace(tmp_path / "artifacts", submission_id="sub-x", attempt=1)
    current = persist_checkpoint(
        workspace,
        state_files={"model.pt": b"weights"},
        code_hash="c",
        arch_hash="a",
        recipe_fingerprint="r",
        created_at="2026-06-27T00:00:00Z",
    )
    files = ("model.pt",)
    upload = CheckpointUpload(
        submission_id="sub-x",
        attempt=1,
        checkpoint_dir=current,
        files=files,
        revision=revision_for("sub-x", 1, files),
    )
    return upload, current


def test_mock_publisher_records_upload_and_returns_ref(tmp_path):
    upload, _ = _persisted_upload(tmp_path)
    publisher = MockCheckpointPublisher()
    published = publisher.publish(upload)

    assert publisher.call_count == 1
    assert publisher.uploads[0].files == ("model.pt",)
    assert published.repo_id == DEFAULT_CHECKPOINT_REPO_ID
    assert published.checkpoint_ref.startswith(DEFAULT_CHECKPOINT_REPO_ID + "@")
    assert published.files == ("model.pt",)


def test_mock_publisher_roundtrips_checkpoint_bytes(tmp_path):
    upload, _ = _persisted_upload(tmp_path)
    publisher = MockCheckpointPublisher()
    published = publisher.publish(upload)

    dest = tmp_path / "resume"
    publisher.download(published.checkpoint_ref, dest)
    assert (dest / "model.pt").read_bytes() == b"weights"


def test_mock_publisher_does_not_import_huggingface_hub(tmp_path, monkeypatch):
    # The MOCK path must never touch huggingface_hub (no real network); simulate it being absent.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    upload, _ = _persisted_upload(tmp_path)
    publisher = MockCheckpointPublisher()
    published = publisher.publish(upload)
    assert published.checkpoint_ref


def test_hf_publisher_construction_is_lazy(monkeypatch):
    # Constructing the real publisher must not import huggingface_hub (deploy-time only).
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    publisher = HuggingFaceCheckpointPublisher(repo_id="org/repo", token=None)
    assert publisher.repo_id == "org/repo"


def test_hf_publisher_publish_uses_injected_api(tmp_path):
    upload, _ = _persisted_upload(tmp_path)

    class _FakeHfApi:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.uploaded: list[str] = []

        def create_repo(self, **kwargs):
            self.created.append(kwargs["repo_id"])

        def upload_file(self, **kwargs):
            self.uploaded.append(kwargs["path_in_repo"])

    api = _FakeHfApi()
    publisher = HuggingFaceCheckpointPublisher(repo_id="org/repo", api=api)
    published = publisher.publish(upload)
    assert api.created == ["org/repo"]
    assert api.uploaded == ["sub-x/attempt-1/model.pt"]
    assert published.checkpoint_ref == "org/repo@" + upload.revision


# --- VAL-PRISM-019: checkpoint cadence is configurable, hourly by default -------------------------


def test_checkpoint_cadence_defaults_to_hourly():
    assert PrismSettings().checkpoint_cadence_seconds == 3600


def test_checkpoint_cadence_is_configurable(monkeypatch):
    monkeypatch.setenv("PRISM_CHECKPOINT_CADENCE_SECONDS", "120")
    assert PrismSettings().checkpoint_cadence_seconds == 120
    assert PrismSettings(checkpoint_cadence_seconds=300).checkpoint_cadence_seconds == 300


# --- VAL-PRISM-020: huggingface_hub is a declared prism dependency --------------------------------


def test_huggingface_hub_is_declared_dependency():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert any(
        dep.split(">=")[0].split("==")[0].strip().replace("-", "_") == "huggingface_hub"
        for dep in deps
    )
