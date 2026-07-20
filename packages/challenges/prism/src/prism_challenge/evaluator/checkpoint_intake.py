"""Master-side checkpoint intake + HF publication (architecture.md sections 2.1, 7).

A validator persists a crash-recovery checkpoint and PUSHES it to the master; the master
republishes it to HuggingFace through the :class:`CheckpointPublisher` interface (a mock in tests,
the real ``huggingface_hub`` client at deploy) and records the returned ``checkpoint_ref`` on the
submission's assignment so a later reassignment resumes from the last PUBLIC checkpoint.

This module owns ONLY the master-side intake/publish/record step. The hotkey-signed, permit-gated
HTTP endpoint is wired in :mod:`prism_challenge.app`; the validator-side cadence + push client lives
in :mod:`prism_challenge.evaluator.checkpoint_push`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from .checkpoint_publisher import (
    CheckpointPublisher,
    CheckpointUpload,
    PublishedCheckpoint,
    revision_for,
)
from .checkpoints import resolve_checkpoint_artifact_path


class CheckpointIntakeError(ValueError):
    """Raised when an uploaded checkpoint payload is malformed (no files / unsafe path)."""


class SupportsRecordCheckpoint(Protocol):
    """The slice of :class:`~prism_challenge.repository.PrismRepository` this service needs."""

    async def record_published_checkpoint(
        self,
        *,
        submission_id: str,
        attempt: int,
        validator_hotkey: str,
        checkpoint_ref: str,
        arch_hash: str = "",
    ) -> None: ...


@dataclass
class CheckpointIntakeService:
    """Receive a pushed checkpoint, publish it via the publisher, and record the public ref."""

    publisher: CheckpointPublisher
    repository: SupportsRecordCheckpoint

    async def publish(
        self,
        *,
        submission_id: str,
        attempt: int,
        validator_hotkey: str,
        files: Mapping[str, bytes],
        revision: str | None = None,
        arch_hash: str = "",
    ) -> PublishedCheckpoint:
        """Publish the uploaded ``files`` and persist the resulting ``checkpoint_ref``.

        The (mock) publisher upload runs off the event loop. Only AFTER a successful publish is the
        ``checkpoint_ref`` recorded on the assignment, so a failed publish records nothing.
        """
        if not files:
            raise CheckpointIntakeError("checkpoint upload must contain at least one file")
        names = tuple(sorted(files))
        resolved_revision = revision or revision_for(submission_id, attempt, names)
        published = await asyncio.to_thread(
            self._publish_files,
            submission_id=submission_id,
            attempt=attempt,
            names=names,
            files=files,
            revision=resolved_revision,
        )
        await self.repository.record_published_checkpoint(
            submission_id=submission_id,
            attempt=attempt,
            validator_hotkey=validator_hotkey,
            checkpoint_ref=published.checkpoint_ref,
            arch_hash=arch_hash,
        )
        return published

    def _publish_files(
        self,
        *,
        submission_id: str,
        attempt: int,
        names: tuple[str, ...],
        files: Mapping[str, bytes],
        revision: str,
    ) -> PublishedCheckpoint:
        with TemporaryDirectory(prefix="prism-ckpt-intake-") as tmp:
            checkpoint_dir = Path(tmp)
            for name in names:
                # Path-safe: reject traversal/symlink escape before writing the uploaded bytes.
                target = resolve_checkpoint_artifact_path(checkpoint_dir, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(files[name])
            upload = CheckpointUpload(
                submission_id=submission_id,
                attempt=attempt,
                checkpoint_dir=checkpoint_dir,
                files=names,
                revision=revision,
            )
            return self.publisher.publish(upload)
