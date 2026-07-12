"""Hotkey-signed assignment coordination endpoints on the master app.

Implements the pull/progress/result routes of the coordination plane
(architecture.md sec 4). Online validators pull their assigned work units,
heartbeat progress on a running unit (prism may report a checkpoint ref), and
post results back; the master persists results for weight computation. The
master only coordinates; validators execute the work on their own brokers.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.challenge_sdk.version import API_VERSION
from base.db.models import (
    Validator,
    WorkAssignment,
    WorkAssignmentStatus,
    WorkResult,
)
from base.db.session import session_scope
from base.master.assignment import capability_matches
from base.schemas.assignment import (
    AssignmentProgressRequest,
    AssignmentProgressResponse,
    AssignmentPullResponse,
    AssignmentResultRequest,
    AssignmentResultResponse,
    AssignmentView,
    compute_payload_digest,
    compute_result_digest,
)
from base.security.validator_auth import ValidatorIdentity

DEFAULT_LEASE_SECONDS = 900

_PULLABLE_STATUSES = (WorkAssignmentStatus.ASSIGNED, WorkAssignmentStatus.RUNNING)
_TERMINAL_STATUSES = (
    WorkAssignmentStatus.COMPLETED,
    WorkAssignmentStatus.FAILED,
    WorkAssignmentStatus.DISPUTED,
)

# Legacy gateway payload field names. Preferences reject/drop these rather than
# re-issue tokens; the LLM gateway has been removed from Base.
_LEGACY_GATEWAY_PAYLOAD_KEYS = frozenset(
    {
        "gateway_token",
        "gateway_url",
        "BASE_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN_FILE",
        "BASE_LLM_GATEWAY_URL",
        "PRISM_GATEWAY_TOKEN",
        "PRISM_GATEWAY_TOKEN_FILE",
        "PRISM_LLM_GATEWAY_URL",
    }
)


class AssignmentNotFoundError(LookupError):
    """No ``work_assignments`` row with the given id (HTTP 404)."""


class AssignmentOwnershipError(PermissionError):
    """The assignment is not owned by the calling validator (HTTP 403)."""


class AssignmentStateError(ValueError):
    """The assignment is not in a state that permits the operation (HTTP 409)."""


class AssignmentResultConflictError(ValueError):
    """A result post conflicts with an already-committed terminal result."""


@dataclass(frozen=True)
class ResultOutcome:
    """Outcome of a result post (idempotent when the unit was already terminal)."""

    status: str
    result_ref: str | None
    idempotent: bool


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def _sanitize_assignment_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return assignment payload without legacy LLM-gateway fields."""
    cleaned = dict(payload or {})
    for key in list(cleaned):
        legacy = _LEGACY_GATEWAY_PAYLOAD_KEYS
        if key in legacy or key.upper() in legacy:
            cleaned.pop(key, None)
    return cleaned


def _assignment_payload_digest(assignment: WorkAssignment) -> str:
    if assignment.payload_digest:
        return assignment.payload_digest
    return compute_payload_digest(_sanitize_assignment_payload(assignment.payload))


def assignment_to_view(assignment: WorkAssignment) -> AssignmentView:
    """Convert a persisted assignment row to its public view."""

    payload = _sanitize_assignment_payload(assignment.payload)
    digest = assignment.payload_digest or compute_payload_digest(payload)
    # attempt_count starts at 0 before first assignment; wire contract requires >=1.
    attempt = max(int(assignment.attempt_count or 0), 1)
    revision = max(int(getattr(assignment, "revision", None) or 1), 1)
    return AssignmentView(
        api_version=API_VERSION,
        assignment_id=str(assignment.id),
        challenge_slug=assignment.challenge_slug,
        work_unit_id=assignment.work_unit_id,
        submission_ref=assignment.submission_ref,
        payload=payload,
        payload_digest=digest,
        required_capability=assignment.required_capability,
        status=WorkAssignmentStatus(assignment.status).value,
        revision=revision,
        attempt=attempt,
        max_attempts=assignment.max_attempts,
        lease_deadline=assignment.deadline_at,
        last_progress_at=assignment.last_progress_at,
        checkpoint_ref=assignment.checkpoint_ref,
    )


class AssignmentCoordinationService:
    """Serve pull/progress/result for coordinated work-unit assignments."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        gpu_serves_cpu: bool = True,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._lease_seconds = lease_seconds
        self._gpu_serves_cpu = gpu_serves_cpu
        self._now_fn = now_fn

    async def pull(self, *, hotkey: str) -> list[WorkAssignment]:
        """Return the caller's assigned/running, capability-matched work units.

        Pulling an ``assigned`` unit transitions it to ``running`` and sets a
        future lease ``deadline_at`` plus ``last_progress_at``. Completed/failed
        units and other validators' units are never returned.
        """

        now = self._now_fn()
        lease = timedelta(seconds=self._lease_seconds)
        async with session_scope(self._session_factory) as session:
            validator = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one_or_none()
            capabilities = (
                set(validator.capabilities) if validator is not None else set()
            )

            rows = (
                (
                    await session.execute(
                        select(WorkAssignment)
                        .where(
                            WorkAssignment.assigned_validator_hotkey == hotkey,
                            WorkAssignment.status.in_(_PULLABLE_STATUSES),
                        )
                        .order_by(
                            WorkAssignment.created_at, WorkAssignment.work_unit_id
                        )
                    )
                )
                .scalars()
                .all()
            )

            pulled: list[WorkAssignment] = []
            for unit in rows:
                if not capability_matches(
                    unit.required_capability,
                    capabilities,
                    gpu_serves_cpu=self._gpu_serves_cpu,
                ):
                    continue
                if not unit.payload_digest:
                    unit.payload_digest = compute_payload_digest(
                        _sanitize_assignment_payload(unit.payload)
                    )
                if unit.status == WorkAssignmentStatus.ASSIGNED:
                    unit.status = WorkAssignmentStatus.RUNNING
                    unit.deadline_at = now + lease
                    unit.last_progress_at = now
                pulled.append(unit)
            return pulled

    async def progress(
        self,
        *,
        assignment_id: str,
        hotkey: str,
        checkpoint_ref: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> WorkAssignment:
        """Refresh a running assignment's lease; persist any checkpoint ref.

        Raises :class:`AssignmentNotFoundError` (unknown id),
        :class:`AssignmentOwnershipError` (not the caller's), or
        :class:`AssignmentStateError` (not ``running``) without mutating the row.
        """

        now = self._now_fn()
        lease = timedelta(seconds=self._lease_seconds)
        parsed = _parse_uuid(assignment_id)
        if parsed is None:
            raise AssignmentNotFoundError(assignment_id)
        async with session_scope(self._session_factory) as session:
            unit = await self._load(session, parsed)
            if unit is None:
                raise AssignmentNotFoundError(assignment_id)
            if unit.assigned_validator_hotkey != hotkey:
                raise AssignmentOwnershipError(assignment_id)
            if unit.status != WorkAssignmentStatus.RUNNING:
                raise AssignmentStateError(assignment_id)
            unit.last_progress_at = now
            unit.deadline_at = now + lease
            if checkpoint_ref is not None:
                unit.checkpoint_ref = checkpoint_ref
            return unit

    async def post_result(
        self,
        *,
        assignment_id: str,
        hotkey: str,
        success: bool,
        payload: Mapping[str, Any] | None = None,
        checkpoint_ref: str | None = None,
        proof: Mapping[str, Any] | None = None,
    ) -> ResultOutcome:
        """Persist a reported result and transition the unit to terminal state.

        Exact retries of an already-committed result are idempotent and return the
        historic reference. Conflicting payload / success / proof / checkpoint_ref
        against a terminal or committed result return a conflict without mutation.
        Rejects a post for an assignment not owned by the caller without
        persisting anything.
        """

        now = self._now_fn()
        parsed = _parse_uuid(assignment_id)
        if parsed is None:
            raise AssignmentNotFoundError(assignment_id)
        result_payload = dict(payload or {})
        proof_payload = dict(proof) if proof is not None else None
        incoming_digest = compute_result_digest(
            success=success,
            payload=result_payload,
            checkpoint_ref=checkpoint_ref,
            proof=proof_payload,
        )
        try:
            async with session_scope(self._session_factory) as session:
                return await self._post_result_in_session(
                    session,
                    now=now,
                    assignment_id=assignment_id,
                    parsed=parsed,
                    hotkey=hotkey,
                    success=success,
                    result_payload=result_payload,
                    proof_payload=proof_payload,
                    checkpoint_ref=checkpoint_ref,
                    incoming_digest=incoming_digest,
                )
        except IntegrityError:
            # Concurrent identical deliveries: unique assignment_id wins; retry
            # as exact/conflicting read against the committed winner.
            async with session_scope(self._session_factory) as session:
                return await self._post_result_in_session(
                    session,
                    now=now,
                    assignment_id=assignment_id,
                    parsed=parsed,
                    hotkey=hotkey,
                    success=success,
                    result_payload=result_payload,
                    proof_payload=proof_payload,
                    checkpoint_ref=checkpoint_ref,
                    incoming_digest=incoming_digest,
                )

    async def _post_result_in_session(
        self,
        session: AsyncSession,
        *,
        now: datetime,
        assignment_id: str,
        parsed: uuid.UUID,
        hotkey: str,
        success: bool,
        result_payload: dict[str, Any],
        proof_payload: dict[str, Any] | None,
        checkpoint_ref: str | None,
        incoming_digest: str,
    ) -> ResultOutcome:
        unit = await self._load(session, parsed)
        if unit is None:
            raise AssignmentNotFoundError(assignment_id)
        if unit.assigned_validator_hotkey != hotkey:
            raise AssignmentOwnershipError(assignment_id)

        existing = (
            await session.execute(
                select(WorkResult).where(WorkResult.assignment_id == unit.id)
            )
        ).scalar_one_or_none()
        if existing is not None or unit.status in _TERMINAL_STATUSES:
            existing_digest = None
            if existing is not None:
                existing_digest = existing.result_digest or compute_result_digest(
                    success=bool(existing.success),
                    payload=existing.payload,
                    checkpoint_ref=existing.checkpoint_ref,
                    proof=existing.proof or None,
                )
            if (
                existing is not None
                and existing_digest is not None
                and existing_digest == incoming_digest
            ):
                return ResultOutcome(
                    status=WorkAssignmentStatus(unit.status).value
                    if unit.status in _TERMINAL_STATUSES
                    else (
                        WorkAssignmentStatus.COMPLETED.value
                        if existing.success
                        else WorkAssignmentStatus.FAILED.value
                    ),
                    result_ref=unit.result_ref or str(existing.id),
                    idempotent=True,
                )
            # Non-exact terminal retries (including assigned/expired/disputed
            # without exact historical bytes) are conflicts, not writes.
            raise AssignmentResultConflictError(assignment_id)

        if unit.status != WorkAssignmentStatus.RUNNING:
            raise AssignmentStateError(assignment_id)

        result = WorkResult(
            id=uuid.uuid4(),
            assignment_id=unit.id,
            challenge_slug=unit.challenge_slug,
            work_unit_id=unit.work_unit_id,
            submission_ref=unit.submission_ref,
            validator_hotkey=hotkey,
            success=success,
            payload=result_payload,
            result_digest=incoming_digest,
            checkpoint_ref=checkpoint_ref,
            proof=dict(proof_payload or {}),
            created_at=now,
        )
        session.add(result)
        await session.flush()

        unit.status = (
            WorkAssignmentStatus.COMPLETED if success else WorkAssignmentStatus.FAILED
        )
        unit.result_ref = str(result.id)
        unit.last_progress_at = now
        if checkpoint_ref is not None:
            unit.checkpoint_ref = checkpoint_ref
        if not unit.payload_digest:
            unit.payload_digest = _assignment_payload_digest(unit)
        return ResultOutcome(
            status=WorkAssignmentStatus(unit.status).value,
            result_ref=unit.result_ref,
            idempotent=False,
        )

    @staticmethod
    async def _load(
        session: AsyncSession, assignment_id: uuid.UUID
    ) -> WorkAssignment | None:
        return (
            await session.execute(
                select(WorkAssignment).where(WorkAssignment.id == assignment_id)
            )
        ).scalar_one_or_none()


def build_assignment_coordination_router(
    *,
    service: AssignmentCoordinationService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    """Build the assignment coordination router (pull + progress + result).

    ``auth_dependency`` is the validator signed-request dependency from
    :func:`base.security.validator_auth.build_validator_auth_dependency`; every
    route is therefore hotkey-signed and metagraph-permit gated.
    """

    router = APIRouter()

    @router.post("/v1/assignments/pull", response_model=AssignmentPullResponse)
    async def pull_assignments(
        identity: ValidatorIdentity = Depends(auth_dependency),
    ) -> AssignmentPullResponse:
        units = await service.pull(hotkey=identity.hotkey)
        return AssignmentPullResponse(
            api_version=API_VERSION,
            assignments=[assignment_to_view(unit) for unit in units],
        )

    @router.post(
        "/v1/assignments/{assignment_id}/progress",
        response_model=AssignmentProgressResponse,
    )
    async def assignment_progress(
        assignment_id: str,
        payload: AssignmentProgressRequest,
        identity: ValidatorIdentity = Depends(auth_dependency),
    ) -> AssignmentProgressResponse:
        try:
            unit = await service.progress(
                assignment_id=assignment_id,
                hotkey=identity.hotkey,
                checkpoint_ref=payload.checkpoint_ref,
                meta=payload.meta,
            )
        except AssignmentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="assignment not found"
            ) from exc
        except AssignmentOwnershipError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="assignment not owned by caller",
            ) from exc
        except AssignmentStateError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="assignment is not running",
            ) from exc
        progress_at = unit.last_progress_at
        if progress_at is None:
            # Contract requires a concrete progress timestamp on success path.
            progress_at = datetime.now(UTC)
        return AssignmentProgressResponse(
            api_version=API_VERSION,
            status=WorkAssignmentStatus(unit.status).value,
            lease_deadline=unit.deadline_at,
            last_progress_at=progress_at,
            checkpoint_ref=unit.checkpoint_ref,
        )

    @router.post(
        "/v1/assignments/{assignment_id}/result",
        response_model=AssignmentResultResponse,
    )
    async def assignment_result(
        assignment_id: str,
        payload: AssignmentResultRequest,
        identity: ValidatorIdentity = Depends(auth_dependency),
    ) -> AssignmentResultResponse:
        try:
            outcome = await service.post_result(
                assignment_id=assignment_id,
                hotkey=identity.hotkey,
                success=payload.success,
                payload=payload.payload,
                checkpoint_ref=payload.checkpoint_ref,
                proof=payload.proof,
            )
        except AssignmentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="assignment not found"
            ) from exc
        except AssignmentOwnershipError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="assignment not owned by caller",
            ) from exc
        except AssignmentStateError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="assignment is not running",
            ) from exc
        except AssignmentResultConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="conflicting result for terminal assignment",
            ) from exc
        return AssignmentResultResponse(
            api_version=API_VERSION,
            status=outcome.status,
            result_ref=outcome.result_ref,
            idempotent=outcome.idempotent,
        )

    return router
