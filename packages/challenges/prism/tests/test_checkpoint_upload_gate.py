"""PRISM_CHECKPOINT_UPLOAD_ENABLED gate (host landmine on checkpoint_publisher).

Default OFF: publisher_from_env returns DisabledCheckpointPublisher; HF publish
refuses unless explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prism_challenge.evaluator.checkpoint_publisher import (
    DEFAULT_CHECKPOINT_REPO_ID,
    CheckpointUpload,
    DisabledCheckpointPublisher,
    HuggingFaceCheckpointPublisher,
    publisher_from_env,
    revision_for,
)
from prism_challenge.evaluator.checkpoints import checkpoint_workspace, persist_checkpoint


def _upload(tmp_path: Path) -> CheckpointUpload:
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
    return CheckpointUpload(
        submission_id="sub-x",
        attempt=1,
        checkpoint_dir=current,
        files=files,
        revision=revision_for("sub-x", 1, files),
    )


def test_publisher_from_env_defaults_to_disabled(monkeypatch, tmp_path: Path) -> None:
    """Given upload env unset, When publisher_from_env, Then DisabledCheckpointPublisher."""
    monkeypatch.delenv("PRISM_CHECKPOINT_UPLOAD_ENABLED", raising=False)
    pub = publisher_from_env()
    assert isinstance(pub, DisabledCheckpointPublisher)
    upload = _upload(tmp_path)
    published = pub.publish(upload)
    assert published.repo_id == DEFAULT_CHECKPOINT_REPO_ID
    assert published.revision == upload.revision


def test_publisher_from_env_enabled_returns_hf(monkeypatch) -> None:
    """Given PRISM_CHECKPOINT_UPLOAD_ENABLED=true, When factory, Then HF publisher."""
    monkeypatch.setenv("PRISM_CHECKPOINT_UPLOAD_ENABLED", "true")
    pub = publisher_from_env()
    assert isinstance(pub, HuggingFaceCheckpointPublisher)


def test_disabled_publisher_download_raises(tmp_path: Path) -> None:
    """Given disabled publisher, When download, Then RuntimeError upload_disabled."""
    pub = DisabledCheckpointPublisher()
    with pytest.raises(RuntimeError, match="prism_checkpoint_upload_disabled"):
        pub.download("BaseIntelligence/top-prism-architecture@rev", tmp_path / "out")


def test_hf_publish_refuses_when_upload_disabled(monkeypatch, tmp_path: Path) -> None:
    """Given upload env false, When HF.publish, Then RuntimeError without contacting API."""
    monkeypatch.setenv("PRISM_CHECKPOINT_UPLOAD_ENABLED", "false")
    upload = _upload(tmp_path)

    class _BoomApi:
        def create_repo(self, **kwargs):  # noqa: ANN003
            raise AssertionError("must not call HF when upload disabled")

        def upload_file(self, **kwargs):  # noqa: ANN003
            raise AssertionError("must not call HF when upload disabled")

    publisher = HuggingFaceCheckpointPublisher(repo_id=DEFAULT_CHECKPOINT_REPO_ID, api=_BoomApi())
    with pytest.raises(RuntimeError, match="prism_checkpoint_upload_disabled"):
        publisher.publish(upload)
