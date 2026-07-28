"""Master-side checkpoint intake + HF publication (architecture.md sections 2.1, 7).

A validator persists a crash-recovery checkpoint and PUSHES it to the master; the master
republishes it to HuggingFace through the :class:`CheckpointPublisher` interface (a mock in tests,
the real ``huggingface_hub`` client at deploy) and records the returned ``checkpoint_ref`` on the
submission's assignment so a later reassignment resumes from the last PUBLIC checkpoint.

This module owns ONLY the master-side intake/publish/record step. The hotkey-signed, permit-gated
HTTP endpoint is wired in :mod:`prism_challenge.app`; the validator-side cadence + push client lives
in :mod:`prism_challenge.evaluator.checkpoint_push`.

Observability: every publish attempt updates :attr:`CheckpointIntakeService.last_status` /
``last_error`` / ``last_checkpoint_ref`` and emits a structured log line with ``repo_id`` +
``submission_id`` (never tokens). Failed publishes never record a ``checkpoint_ref``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol

from .checkpoint_publisher import (
    CheckpointPublisher,
    CheckpointUpload,
    PublishedCheckpoint,
    revision_for,
)
from .checkpoints import resolve_checkpoint_artifact_path

logger = logging.getLogger(__name__)

PublishStatus = Literal["success", "failed"]


class CheckpointIntakeError(ValueError):
    """Raised when an uploaded checkpoint payload is malformed (no files / unsafe path)."""


class CheckpointPublishError(RuntimeError):
    """Raised when the publisher fails; no ``checkpoint_ref`` is recorded."""

    def __init__(self, message: str, *, submission_id: str, repo_id: str) -> None:
        super().__init__(message)
        self.submission_id = submission_id
        self.repo_id = repo_id


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


@dataclass(frozen=True)
class CheckpointIntakeResult:
    """Observable outcome of one master-side publish attempt."""

    status: PublishStatus
    submission_id: str
    repo_id: str
    checkpoint_ref: str | None
    revision: str | None
    files: tuple[str, ...]
    last_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success" and bool(self.checkpoint_ref)


def is_training_publish_complete(
    *,
    hf_token_configured: bool,
    checkpoint_ref: str | None,
) -> bool:
    """Whether training completion may be marked fully successful.

    Fail-closed when HF is configured: a published ``checkpoint_ref`` is required.
    When HF is disabled (dev / mock offline path), missing ref is allowed.
    """
    if checkpoint_ref:
        return True
    return not hf_token_configured


@dataclass
class CheckpointIntakeService:
    """Receive a pushed checkpoint, publish it via the publisher, and record the public ref."""

    publisher: CheckpointPublisher
    repository: SupportsRecordCheckpoint
    last_status: PublishStatus | None = None
    last_error: str | None = None
    last_checkpoint_ref: str | None = None
    last_result: CheckpointIntakeResult | None = None

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
        On failure, :attr:`last_error` / :attr:`last_status` are updated and
        :class:`CheckpointPublishError` is raised (HTTP layer maps it to a non-2xx response).
        """
        if not files:
            raise CheckpointIntakeError("checkpoint upload must contain at least one file")
        names = tuple(sorted(files))
        resolved_revision = revision or revision_for(submission_id, attempt, names)
        repo_id = getattr(self.publisher, "repo_id", "") or ""
        try:
            published = await asyncio.to_thread(
                self._publish_files,
                submission_id=submission_id,
                attempt=attempt,
                names=names,
                files=files,
                revision=resolved_revision,
            )
        except CheckpointIntakeError:
            raise
        except Exception as exc:
            err = _safe_error_message(exc)
            result = CheckpointIntakeResult(
                status="failed",
                submission_id=submission_id,
                repo_id=repo_id,
                checkpoint_ref=None,
                revision=resolved_revision,
                files=names,
                last_error=err,
            )
            self._record_outcome(result)
            logger.error(
                "checkpoint publish failed submission_id=%s repo_id=%s error=%s",
                submission_id,
                repo_id,
                err,
            )
            raise CheckpointPublishError(err, submission_id=submission_id, repo_id=repo_id) from exc

        await self.repository.record_published_checkpoint(
            submission_id=submission_id,
            attempt=attempt,
            validator_hotkey=validator_hotkey,
            checkpoint_ref=published.checkpoint_ref,
            arch_hash=arch_hash,
        )
        result = CheckpointIntakeResult(
            status="success",
            submission_id=submission_id,
            repo_id=published.repo_id,
            checkpoint_ref=published.checkpoint_ref,
            revision=published.revision,
            files=tuple(published.files),
            last_error=None,
        )
        self._record_outcome(result)
        logger.info(
            "checkpoint publish success submission_id=%s repo_id=%s checkpoint_ref=%s",
            submission_id,
            published.repo_id,
            published.checkpoint_ref,
        )
        return published

    def _record_outcome(self, result: CheckpointIntakeResult) -> None:
        self.last_result = result
        self.last_status = result.status
        self.last_error = result.last_error
        self.last_checkpoint_ref = result.checkpoint_ref

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


def _safe_error_message(exc: BaseException) -> str:
    """Human-readable error without secret-shaped values (tokens never logged)."""
    name = type(exc).__name__
    text = str(exc).strip() or name
    # Hard-cap length; strip common secret-bearing substrings if a caller ever embeds them.
    lowered = text.lower()
    for marker in ("hf_token", "authorization:", "bearer ", "api_key=", "token="):
        if marker in lowered:
            return f"{name}: <redacted>"
    if len(text) > 500:
        text = text[:500] + "…"
    return f"{name}: {text}" if name not in text else text
