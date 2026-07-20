"""Checkpoint publisher interface for prism crash-recovery checkpoints (architecture.md section 7).

The validator persists periodic training checkpoints (see :func:`persist_checkpoint`) and pushes
them to the master, which publishes them to HuggingFace so a crashed/reassigned run can resume from
the last PUBLIC checkpoint. The publish/download path is an INTERFACE so tests use an in-memory mock
(no ``huggingface_hub`` network) while deploy wires the real Hub-backed implementation; the real
``huggingface_hub`` import is lazy so this module stays importable offline.

This module owns ONLY the publisher seam + mock + the deploy-time real client. The master-side
HTTP endpoint, on-assignment ``checkpoint_ref`` persistence, and resume wiring are layered on top by
the coordination-plane features.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .checkpoints import resolve_checkpoint_artifact_path

DEFAULT_CHECKPOINT_REPO_ID = "baseintelligence/prism-checkpoints"


@dataclass(frozen=True)
class CheckpointUpload:
    """A persisted checkpoint ready to publish (relative ``files`` under ``checkpoint_dir``)."""

    submission_id: str
    attempt: int
    checkpoint_dir: Path
    files: tuple[str, ...]
    revision: str


@dataclass(frozen=True)
class PublishedCheckpoint:
    """The public reference a reassigned run resumes from."""

    checkpoint_ref: str
    repo_id: str
    revision: str
    files: tuple[str, ...]


@runtime_checkable
class CheckpointPublisher(Protocol):
    """Publish a persisted checkpoint and restore it on resume."""

    def publish(self, upload: CheckpointUpload) -> PublishedCheckpoint: ...

    def download(self, checkpoint_ref: str, dest_dir: Path) -> Path: ...


def _read_checkpoint_files(upload: CheckpointUpload) -> dict[str, bytes]:
    if not upload.files:
        raise ValueError("checkpoint upload must list at least one file")
    contents: dict[str, bytes] = {}
    for name in upload.files:
        # Path-safe: reject traversal/symlink escape out of the checkpoint dir before reading.
        source = resolve_checkpoint_artifact_path(upload.checkpoint_dir, name)
        if not source.is_file():
            raise ValueError(f"checkpoint file is missing: {name}")
        contents[name] = source.read_bytes()
    return contents


def checkpoint_ref_for(repo_id: str, revision: str) -> str:
    """Canonical public checkpoint reference (``{repo_id}@{revision}``)."""
    return f"{repo_id}@{revision}"


@dataclass
class MockCheckpointPublisher:
    """In-memory checkpoint publisher for tests (no ``huggingface_hub`` / no network).

    Records every publish call (repo/ref/files) and keeps the uploaded bytes so ``download`` can
    restore them, exactly mirroring the real publisher's observable contract without any network.
    """

    repo_id: str = DEFAULT_CHECKPOINT_REPO_ID
    uploads: list[CheckpointUpload] = field(default_factory=list)
    published: list[PublishedCheckpoint] = field(default_factory=list)
    _store: dict[str, dict[str, bytes]] = field(default_factory=dict)

    def publish(self, upload: CheckpointUpload) -> PublishedCheckpoint:
        contents = _read_checkpoint_files(upload)
        ref = checkpoint_ref_for(self.repo_id, upload.revision)
        self._store[ref] = contents
        self.uploads.append(upload)
        result = PublishedCheckpoint(
            checkpoint_ref=ref,
            repo_id=self.repo_id,
            revision=upload.revision,
            files=tuple(upload.files),
        )
        self.published.append(result)
        return result

    def download(self, checkpoint_ref: str, dest_dir: Path) -> Path:
        contents = self._store.get(checkpoint_ref)
        if contents is None:
            raise KeyError(f"unknown checkpoint ref: {checkpoint_ref}")
        dest = Path(dest_dir)
        for name, data in contents.items():
            target = resolve_checkpoint_artifact_path(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return dest

    @property
    def call_count(self) -> int:
        return len(self.uploads)


class HuggingFaceCheckpointPublisher:
    """Deploy-time publisher backed by ``huggingface_hub`` (imported lazily; mocked in tests).

    The Hub client and token are resolved only when ``publish``/``download`` is actually called, so
    importing this module never requires ``huggingface_hub`` or a network/token. Tests use
    :class:`MockCheckpointPublisher`; this class is exercised at deploy with a real ``HF_TOKEN``.
    """

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_CHECKPOINT_REPO_ID,
        token: str | None = None,
        api: Any | None = None,
    ) -> None:
        self.repo_id = repo_id
        self._token = token
        self._api = api

    def _hf_api(self) -> Any:
        if self._api is not None:
            return self._api
        from huggingface_hub import HfApi  # lazy: real network client wired only at deploy

        self._api = HfApi(token=self._token)
        return self._api

    def publish(self, upload: CheckpointUpload) -> PublishedCheckpoint:
        files = _read_checkpoint_files(upload)
        api = self._hf_api()
        api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True, private=True)
        for name in files:
            source = resolve_checkpoint_artifact_path(upload.checkpoint_dir, name)
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=f"{upload.submission_id}/attempt-{upload.attempt}/{name}",
                repo_id=self.repo_id,
                repo_type="model",
                revision=upload.revision,
            )
        return PublishedCheckpoint(
            checkpoint_ref=checkpoint_ref_for(self.repo_id, upload.revision),
            repo_id=self.repo_id,
            revision=upload.revision,
            files=tuple(upload.files),
        )

    def download(self, checkpoint_ref: str, dest_dir: Path) -> Path:
        from huggingface_hub import snapshot_download  # lazy

        repo_id, _, revision = checkpoint_ref.partition("@")
        local = snapshot_download(
            repo_id=repo_id or self.repo_id,
            repo_type="model",
            revision=revision or None,
            local_dir=str(dest_dir),
            token=self._token,
        )
        return Path(local)


def revision_for(submission_id: str, attempt: int, files: Sequence[str]) -> str:
    """Deterministic checkpoint revision derived from submission/attempt + file names."""
    digest = hashlib.sha256()
    digest.update(submission_id.encode("utf-8"))
    digest.update(f":{attempt}:".encode())
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return f"sub-{submission_id}-attempt-{attempt}-{digest.hexdigest()[:12]}"
