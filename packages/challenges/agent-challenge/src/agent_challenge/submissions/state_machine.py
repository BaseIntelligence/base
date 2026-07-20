"""Focused submission status transition service."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.models import AgentSubmission, SubmissionStatusEvent
from ..core.statuses import (
    INTERNAL_SUBMISSION_STATUSES,
    LEGACY_SUBMISSION_STATUSES,
)

logger = logging.getLogger(__name__)

_USE_CURRENT = object()

MAX_SEQUENCE_ALLOCATION_RETRIES = 5

#: Canonical submission status vocabulary (single source of truth in
#: :mod:`agent_challenge.core.statuses`); the transition graph below references
#: these same literals.
INTERNAL_STATUSES = INTERNAL_SUBMISSION_STATUSES
LEGACY_STATUSES = LEGACY_SUBMISSION_STATUSES

ALLOWED_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"received"}),
    "received": frozenset({"upload_verified", "queued", "cancelled", "admin_paused"}),
    "upload_verified": frozenset({"rate_limit_reserved", "cancelled", "admin_paused"}),
    "rate_limit_reserved": frozenset(
        {"analysis_queued", "review_queued", "cancelled", "admin_paused"}
    ),
    "review_queued": frozenset(
        {
            "review_cvm_running",
            "review_cancelled",
            "review_expired",
            "review_error",
            "cancelled",
            "admin_paused",
        }
    ),
    "review_cvm_running": frozenset(
        {
            "review_provider_standby",
            "review_verifying",
            "review_cancelled",
            "review_expired",
            "review_error",
            "cancelled",
            "admin_paused",
        }
    ),
    "review_provider_standby": frozenset(
        {
            "review_verifying",
            "review_cancelled",
            "review_expired",
            "review_error",
            "cancelled",
            "admin_paused",
        }
    ),
    "review_verifying": frozenset(
        {
            "review_allowed",
            "review_rejected",
            "review_escalated",
            "review_error",
        }
    ),
    "review_allowed": frozenset({"tb_completed"}),
    "review_rejected": frozenset({"review_queued"}),
    "review_escalated": frozenset({"review_queued"}),
    "review_expired": frozenset({"review_queued"}),
    "review_cancelled": frozenset({"review_queued"}),
    "review_error": frozenset({"review_queued"}),
    "analysis_queued": frozenset(
        {"ast_running", "llm_running", "analysis_rejected", "cancelled", "admin_paused"}
    ),
    "ast_running": frozenset(
        {
            "analysis_queued",
            "llm_running",
            "analysis_allowed",
            "analysis_rejected",
            "analysis_escalated",
        }
    ),
    "llm_running": frozenset(
        {
            "llm_standby",
            "analysis_queued",
            "analysis_allowed",
            "analysis_rejected",
            "analysis_escalated",
        }
    ),
    "llm_standby": frozenset({"analysis_queued", "cancelled", "admin_paused"}),
    "analysis_allowed": frozenset({"waiting_miner_env", "cancelled", "admin_paused"}),
    "waiting_miner_env": frozenset({"tb_queued", "cancelled", "admin_paused"}),
    "analysis_rejected": frozenset({"admin_paused", "cancelled"}),
    "analysis_escalated": frozenset({"admin_paused", "analysis_allowed", "analysis_rejected"}),
    "tb_queued": frozenset({"tb_running", "cancelled", "admin_paused"}),
    "tb_running": frozenset({"tb_completed", "tb_failed_retryable", "tb_failed_final"}),
    "tb_failed_retryable": frozenset({"tb_queued", "tb_failed_final", "cancelled", "admin_paused"}),
    "tb_completed": frozenset({"tb_queued"}),
    "tb_failed_final": frozenset({"tb_queued"}),
    "cancelled": frozenset(),
    "admin_paused": frozenset(
        {
            "analysis_queued",
            "analysis_allowed",
            "waiting_miner_env",
            "analysis_rejected",
            "tb_queued",
            "cancelled",
        }
    ),
    "pending": frozenset({"queued", "received"}),
    "queued": frozenset({"evaluating", "cancelled", "admin_paused"}),
    "evaluating": frozenset({"valid", "invalid", "suspicious", "error", "queued"}),
    "valid": frozenset({"queued", "tb_completed"}),
    "invalid": frozenset({"queued"}),
    "suspicious": frozenset({"queued"}),
    "error": frozenset({"queued"}),
    "completed": frozenset({"queued"}),
    "overridden_valid": frozenset({"queued"}),
    "overridden_invalid": frozenset({"queued"}),
}

PUBLIC_STATUS_BY_RAW_STATUS: dict[str, str] = {
    "received": "received",
    "upload_verified": "queued",
    "rate_limit_reserved": "queued",
    "review_queued": "queued",
    "review_cvm_running": "LLM review",
    "review_provider_standby": "LLM standby",
    "review_verifying": "LLM review",
    "review_allowed": "queued",
    "review_rejected": "invalid",
    "review_escalated": "suspicious",
    "review_expired": "error",
    "review_cancelled": "cancelled",
    "review_error": "error",
    "analysis_queued": "queued",
    "ast_running": "AST review",
    "llm_running": "LLM review",
    "llm_standby": "LLM standby",
    "analysis_allowed": "queued",
    "waiting_miner_env": "Waiting environments",
    "analysis_rejected": "invalid",
    "analysis_escalated": "suspicious",
    "tb_queued": "evaluation queued",
    "tb_running": "evaluating",
    "tb_completed": "valid",
    "tb_failed_retryable": "evaluating",
    "tb_failed_final": "error",
    "cancelled": "cancelled",
    "admin_paused": "admin_paused",
    "pending": "pending",
    "queued": "queued",
    "evaluating": "evaluating",
    "valid": "valid",
    "invalid": "invalid",
    "suspicious": "suspicious",
    "error": "error",
    "completed": "completed",
    "overridden_valid": "overridden_valid",
    "overridden_invalid": "overridden_invalid",
}


@dataclass(eq=False)
class InvalidSubmissionStatusTransition(ValueError):
    """Raised when a submission status transition is not allowed.

    Not frozen: a frozen dataclass blocks ``__setattr__``, so Python cannot set
    ``__traceback__`` while the exception propagates (it raises FrozenInstanceError
    that masks the real error). ``eq=False`` keeps identity hashing/equality.
    """

    from_status: str | None
    to_status: str

    def __str__(self) -> str:
        return f"invalid submission status transition: {self.from_status!r} -> {self.to_status!r}"


def public_status_for(raw_status: str) -> str:
    """Return the stable public label for a raw status."""

    return PUBLIC_STATUS_BY_RAW_STATUS.get(raw_status, raw_status)


async def record_initial_status(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    actor: str | None,
    reason: str = "",
    metadata: Mapping[str, object] | None = None,
) -> SubmissionStatusEvent:
    """Record the initial received status event for a persisted submission."""

    return await transition_submission_status(
        session,
        submission,
        "received",
        actor=actor,
        reason=reason,
        metadata=metadata,
        from_status=None,
    )


async def transition_submission_status(
    session: AsyncSession,
    submission: AgentSubmission,
    to_status: str,
    *,
    actor: str | None,
    reason: str = "",
    metadata: Mapping[str, object] | None = None,
    from_status: str | None | object = _USE_CURRENT,
) -> SubmissionStatusEvent:
    """Validate a transition, update submission status fields, and append one event."""

    current_status = submission.raw_status if from_status is _USE_CURRENT else from_status
    _validate_transition(current_status, to_status)
    event = await _append_status_event(
        session,
        submission_id=submission.id,
        from_status=current_status,
        to_status=to_status,
        reason=reason,
        actor=actor,
        metadata=metadata,
    )
    submission.raw_status = to_status
    submission.status = public_status_for(to_status)
    submission.effective_status = public_status_for(to_status)
    await session.flush()
    return event


async def _append_status_event(
    session: AsyncSession,
    *,
    submission_id: int,
    from_status: str | None,
    to_status: str,
    reason: str,
    actor: str | None,
    metadata: Mapping[str, object] | None,
) -> SubmissionStatusEvent:
    await session.flush()
    last_collision: IntegrityError | None = None
    for _ in range(MAX_SEQUENCE_ALLOCATION_RETRIES):
        try:
            async with session.begin_nested():
                event = SubmissionStatusEvent(
                    submission_id=submission_id,
                    sequence=await _next_sequence(session, submission_id),
                    from_status=from_status,
                    to_status=to_status,
                    reason=reason,
                    actor=actor,
                    metadata_json=_metadata_json(metadata),
                )
                session.add(event)
                await session.flush()
            return event
        except IntegrityError as exc:
            if not _is_sequence_collision(exc):
                raise
            last_collision = exc
    if last_collision is not None:
        logger.error(
            "submission status sequence allocation exhausted after %d retries "
            "(submission_id=%s, to_status=%s)",
            MAX_SEQUENCE_ALLOCATION_RETRIES,
            submission_id,
            to_status,
        )
        raise last_collision
    raise RuntimeError("MAX_SEQUENCE_ALLOCATION_RETRIES must be greater than zero")


def _is_sequence_collision(exc: IntegrityError) -> bool:
    message = str(exc.orig)
    return (
        "uq_submission_status_events_submission_sequence" in message
        or "submission_status_events.submission_id, submission_status_events.sequence" in message
    )


async def ensure_submission_status(
    session: AsyncSession,
    submission: AgentSubmission,
    to_status: str,
    *,
    actor: str | None,
    reason: str = "",
    metadata: Mapping[str, object] | None = None,
) -> SubmissionStatusEvent | None:
    """Transition unless the submission already has the requested raw status."""

    if submission.raw_status == to_status:
        public_status = public_status_for(to_status)
        submission.status = public_status
        submission.effective_status = public_status
        await session.flush()
        return None
    return await transition_submission_status(
        session,
        submission,
        to_status,
        actor=actor,
        reason=reason,
        metadata=metadata,
    )


def _validate_transition(from_status: str | None, to_status: str) -> None:
    if to_status not in INTERNAL_STATUSES and to_status not in LEGACY_STATUSES:
        raise InvalidSubmissionStatusTransition(from_status, to_status)
    if to_status not in ALLOWED_TRANSITIONS.get(from_status, frozenset()):
        raise InvalidSubmissionStatusTransition(from_status, to_status)


async def _next_sequence(session: AsyncSession, submission_id: int) -> int:
    current = await session.scalar(
        select(func.max(SubmissionStatusEvent.sequence)).where(
            SubmissionStatusEvent.submission_id == submission_id
        )
    )
    return int(current or 0) + 1


def _metadata_json(metadata: Mapping[str, object] | None) -> str:
    return json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":"))
