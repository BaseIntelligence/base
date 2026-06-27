"""Hotkey-signed assignment coordination endpoints on the master app.

Implements the pull/progress/result routes of the coordination plane
(architecture.md sec 4). Online validators pull their assigned work units,
heartbeat progress on a running unit (prism may report a checkpoint ref), and
post results back; the master persists results for weight computation. The
master only coordinates; validators execute the work on their own brokers.

This module also provides :class:`WorkAssignmentLifecycleResolver`, the
production source of truth for the LLM gateway's token-lifecycle binding: a
scoped gateway token is rejected once its assignment is completed, failed, or
reassigned away (architecture.md sec 5).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import (
    Validator,
    WorkAssignment,
    WorkAssignmentStatus,
    WorkResult,
)
from base.db.session import session_scope
from base.master.assignment import capability_matches
from base.master.llm_gateway import (
    DEEPSEEK_BASE_URL_ENV,
    OPENROUTER_BASE_URL_ENV,
)
from base.schemas.assignment import (
    AssignmentProgressRequest,
    AssignmentProgressResponse,
    AssignmentPullResponse,
    AssignmentResultRequest,
    AssignmentResultResponse,
    AssignmentView,
)
from base.security.validator_auth import ValidatorIdentity

DEFAULT_LEASE_SECONDS = 900

_PULLABLE_STATUSES = (WorkAssignmentStatus.ASSIGNED, WorkAssignmentStatus.RUNNING)
_TERMINAL_STATUSES = (WorkAssignmentStatus.COMPLETED, WorkAssignmentStatus.FAILED)

#: Assignment-payload key carrying the scoped gateway token issued at pull time.
GATEWAY_TOKEN_PAYLOAD_KEY = "gateway_token"
#: Assignment-payload key carrying the master gateway root base URL.
GATEWAY_BASE_URL_PAYLOAD_KEY = "gateway_url"
_DEEPSEEK_GATEWAY_PATH = "/llm/deepseek"
_OPENROUTER_GATEWAY_PATH = "/llm/openrouter"


@runtime_checkable
class GatewayTokenIssuer(Protocol):
    """Mints a scoped gateway token (satisfied by ``LLMGatewayService``)."""

    def issue_token(
        self,
        *,
        validator_hotkey: str,
        assignment_id: str,
        ttl_seconds: int | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class GatewayPayloadIssuer:
    """Builds the per-assignment gateway fields stamped into a pull payload.

    Issues a fresh scoped token for ``(validator_hotkey, assignment_id)`` via the
    master gateway token authority and advertises the master gateway base URLs so
    the eval runtime points ``DEEPSEEK_BASE_URL``/``OPENROUTER_BASE_URL`` at the
    master gateway. A raw provider key is NEVER part of this payload
    (architecture.md sec 5).
    """

    issuer: GatewayTokenIssuer
    gateway_base_url: str

    def build(
        self,
        *,
        validator_hotkey: str,
        assignment_id: str,
        ttl_seconds: int | None,
    ) -> dict[str, str]:
        base = self.gateway_base_url.rstrip("/")
        token = self.issuer.issue_token(
            validator_hotkey=validator_hotkey,
            assignment_id=assignment_id,
            ttl_seconds=ttl_seconds,
        )
        return {
            GATEWAY_TOKEN_PAYLOAD_KEY: token,
            GATEWAY_BASE_URL_PAYLOAD_KEY: base,
            DEEPSEEK_BASE_URL_ENV: f"{base}{_DEEPSEEK_GATEWAY_PATH}",
            OPENROUTER_BASE_URL_ENV: f"{base}{_OPENROUTER_GATEWAY_PATH}",
        }


class AssignmentNotFoundError(LookupError):
    """No ``work_assignments`` row with the given id (HTTP 404)."""


class AssignmentOwnershipError(PermissionError):
    """The assignment is not owned by the calling validator (HTTP 403)."""


class AssignmentStateError(ValueError):
    """The assignment is not in a state that permits the operation (HTTP 409)."""


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


def assignment_to_view(
    assignment: WorkAssignment,
    *,
    gateway_payload: Mapping[str, str] | None = None,
) -> AssignmentView:
    """Convert a persisted assignment row to its public view.

    ``gateway_payload`` (when provided) is merged into the view payload only; it
    is the ephemeral per-pull scoped gateway token + base URLs and is never
    persisted to the ``work_assignments`` row.
    """

    payload = dict(assignment.payload or {})
    if gateway_payload:
        payload.update(gateway_payload)
    return AssignmentView(
        id=str(assignment.id),
        challenge_slug=assignment.challenge_slug,
        work_unit_id=assignment.work_unit_id,
        submission_ref=assignment.submission_ref,
        payload=payload,
        required_capability=assignment.required_capability,
        status=WorkAssignmentStatus(assignment.status).value,
        attempt_count=assignment.attempt_count,
        max_attempts=assignment.max_attempts,
        deadline_at=assignment.deadline_at,
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
        gateway_payload_issuer: GatewayPayloadIssuer | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._lease_seconds = lease_seconds
        self._gpu_serves_cpu = gpu_serves_cpu
        self._gateway_payload_issuer = gateway_payload_issuer
        self._now_fn = now_fn

    def gateway_payload(
        self, unit: WorkAssignment, *, hotkey: str
    ) -> dict[str, str] | None:
        """Issue an ephemeral scoped gateway token + base URLs for a pulled unit.

        Returns ``None`` when no gateway issuer is configured. The token is
        minted fresh at pull time and never persisted, so its lifecycle binding
        (VAL-LLM-023) follows the live assignment state and its expiry is bounded
        by the unit's lease deadline.
        """

        if self._gateway_payload_issuer is None:
            return None
        return self._gateway_payload_issuer.build(
            validator_hotkey=hotkey,
            assignment_id=str(unit.id),
            ttl_seconds=self._token_ttl_seconds(unit),
        )

    def _token_ttl_seconds(self, unit: WorkAssignment) -> int:
        deadline = unit.deadline_at
        if deadline is not None:
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            remaining = int((deadline - self._now_fn()).total_seconds())
            if remaining > 0:
                return remaining
        return self._lease_seconds

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
    ) -> ResultOutcome:
        """Persist a reported result and transition the unit to terminal state.

        Idempotent: posting for an already-completed/failed unit is a safe no-op
        that leaves the stored result/status/``result_ref`` unchanged. Rejects a
        post for an assignment not owned by the caller without persisting
        anything.
        """

        now = self._now_fn()
        parsed = _parse_uuid(assignment_id)
        if parsed is None:
            raise AssignmentNotFoundError(assignment_id)
        async with session_scope(self._session_factory) as session:
            unit = await self._load(session, parsed)
            if unit is None:
                raise AssignmentNotFoundError(assignment_id)
            if unit.assigned_validator_hotkey != hotkey:
                raise AssignmentOwnershipError(assignment_id)
            if unit.status in _TERMINAL_STATUSES:
                return ResultOutcome(
                    status=WorkAssignmentStatus(unit.status).value,
                    result_ref=unit.result_ref,
                    idempotent=True,
                )

            result = WorkResult(
                id=uuid.uuid4(),
                assignment_id=unit.id,
                challenge_slug=unit.challenge_slug,
                work_unit_id=unit.work_unit_id,
                submission_ref=unit.submission_ref,
                validator_hotkey=hotkey,
                success=success,
                payload=dict(payload or {}),
                created_at=now,
            )
            session.add(result)
            await session.flush()

            unit.status = (
                WorkAssignmentStatus.COMPLETED
                if success
                else WorkAssignmentStatus.FAILED
            )
            unit.result_ref = str(result.id)
            unit.last_progress_at = now
            if checkpoint_ref is not None:
                unit.checkpoint_ref = checkpoint_ref
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


class WorkAssignmentLifecycleResolver:
    """Production ``AssignmentLifecycleResolver`` backed by ``work_assignments``.

    A scoped gateway token is active only while its assignment row is still
    ``assigned``/``running`` AND owned by the same validator. Once the assignment
    is completed, failed, or reassigned to a different validator, the resolver
    reports inactive so the gateway rejects the token (architecture.md sec 5).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def is_active(self, *, validator_hotkey: str, assignment_id: str) -> bool:
        parsed = _parse_uuid(assignment_id)
        if parsed is None:
            return False
        async with self._session_factory() as session:
            unit = (
                await session.execute(
                    select(WorkAssignment).where(WorkAssignment.id == parsed)
                )
            ).scalar_one_or_none()
        if unit is None:
            return False
        if unit.assigned_validator_hotkey != validator_hotkey:
            return False
        return WorkAssignmentStatus(unit.status) in _PULLABLE_STATUSES


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
            assignments=[
                assignment_to_view(
                    unit,
                    gateway_payload=service.gateway_payload(
                        unit, hotkey=identity.hotkey
                    ),
                )
                for unit in units
            ]
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
        return AssignmentProgressResponse(
            status=WorkAssignmentStatus(unit.status).value,
            deadline_at=unit.deadline_at,
            last_progress_at=unit.last_progress_at,
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
        return AssignmentResultResponse(
            status=outcome.status,
            result_ref=outcome.result_ref,
            idempotent=outcome.idempotent,
        )

    return router
