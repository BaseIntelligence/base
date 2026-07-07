"""Master worker assignment plane: worker-authenticated pull + result.

The worker-facing counterpart to :mod:`base.master.assignment_coordination`.
Where the validator plane pulls from ``work_assignments`` (one row per unit), the
worker plane pulls from ``worker_assignments`` (one replica row per (unit,
worker)) so a gpu unit can be replicated across distinct-owner workers. Every
route authenticates as the WORKER keypair via the worker signed-request
dependency and gates on registration/liveness -- never on a metagraph validator
permit (VAL-AGENT-018):

* ``POST /v1/workers/assignments/pull`` returns the ACTIVE worker's assigned/
  running gpu replicas (transitioning ``assigned`` -> ``running`` with a lease),
  and only gpu-capability units (VAL-AGENT-007). A stale/retired worker receives
  no new units.
* ``POST /v1/workers/assignments/{id}/result`` persists the reported result and
  its ``ExecutionProof`` (VAL-AGENT-008), gated on replica ownership so a late or
  foreign post cannot corrupt another worker's replica.

The engine that CREATES worker replica rows (gpu routing, self-evaluation
exclusion, R=2, reassignment) is layered on top of :meth:`create_worker_assignment`
by the assignment-worker-plane feature.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import (
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerRegistration,
    WorkerStatus,
)
from base.db.session import session_scope
from base.master.assignment import CAPABILITY_GPU
from base.master.worker_coordination import WorkerCoordinationService
from base.schemas.assignment import (
    AssignmentPullResponse,
    AssignmentResultRequest,
    AssignmentResultResponse,
    AssignmentView,
)
from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY, PROOF_PAYLOAD_KEY

DEFAULT_LEASE_SECONDS = 900

_PULLABLE_STATUSES = (WorkAssignmentStatus.ASSIGNED, WorkAssignmentStatus.RUNNING)
_TERMINAL_STATUSES = (WorkAssignmentStatus.COMPLETED, WorkAssignmentStatus.FAILED)


class WorkerAssignmentNotFoundError(LookupError):
    """No ``worker_assignments`` row with the given id (HTTP 404)."""


class WorkerAssignmentOwnershipError(PermissionError):
    """The replica is not owned by the calling worker (HTTP 403)."""


@dataclass(frozen=True)
class WorkerResultOutcome:
    """Outcome of a worker result post (idempotent when already terminal)."""

    status: str
    result_ref: str | None
    idempotent: bool


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def _extract_manifest_sha256(payload: Mapping[str, Any]) -> str | None:
    """Read the manifest hash from a result payload's ExecutionProof envelope."""

    proof = payload.get(PROOF_PAYLOAD_KEY)
    if isinstance(proof, Mapping):
        manifest = proof.get(MANIFEST_SHA256_PAYLOAD_KEY)
        if isinstance(manifest, str) and manifest:
            return manifest
    manifest = payload.get(MANIFEST_SHA256_PAYLOAD_KEY)
    return manifest if isinstance(manifest, str) and manifest else None


def worker_assignment_to_view(assignment: WorkerAssignment) -> AssignmentView:
    """Convert a persisted worker replica row to the shared assignment view."""

    return AssignmentView(
        id=str(assignment.id),
        challenge_slug=assignment.challenge_slug,
        work_unit_id=assignment.work_unit_id,
        submission_ref=assignment.submission_ref,
        payload=dict(assignment.payload or {}),
        required_capability=assignment.required_capability,
        status=WorkAssignmentStatus(assignment.status).value,
        attempt_count=assignment.attempt_count,
        max_attempts=assignment.max_attempts,
        deadline_at=assignment.deadline_at,
        last_progress_at=assignment.last_progress_at,
        checkpoint_ref=assignment.checkpoint_ref,
    )


class WorkerAssignmentService:
    """Serve pull/result for gpu work-unit replicas coordinated to workers."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        worker_service: WorkerCoordinationService,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._worker_service = worker_service
        self._lease_seconds = lease_seconds
        self._now_fn = now_fn

    async def create_worker_assignment(
        self,
        *,
        work_unit_id: str,
        challenge_slug: str,
        submission_ref: str,
        worker_id: str,
        worker_pubkey: str,
        miner_hotkey: str,
        payload: Mapping[str, Any] | None = None,
        required_capability: str = CAPABILITY_GPU,
        status: WorkAssignmentStatus = WorkAssignmentStatus.ASSIGNED,
        attempt_count: int = 1,
        max_attempts: int = 3,
        checkpoint_ref: str | None = None,
        session: AsyncSession | None = None,
    ) -> WorkerAssignment:
        """Create one replica row binding ``work_unit_id`` to a worker.

        Low-level primitive shared by tests and the assignment engine (which adds
        gpu routing, self-evaluation exclusion, and R=2 on top). The engine is
        responsible for eligibility; this only persists the row. When ``session``
        is provided the row is added to the caller's transaction (the caller
        commits) so a whole replica-creation pass stays atomic; otherwise a fresh
        transaction is opened and committed here.
        """

        if session is not None:
            return await self._create_in_session(
                session,
                work_unit_id=work_unit_id,
                challenge_slug=challenge_slug,
                submission_ref=submission_ref,
                worker_id=worker_id,
                worker_pubkey=worker_pubkey,
                miner_hotkey=miner_hotkey,
                payload=payload,
                required_capability=required_capability,
                status=status,
                attempt_count=attempt_count,
                max_attempts=max_attempts,
                checkpoint_ref=checkpoint_ref,
            )
        async with session_scope(self._session_factory) as own_session:
            return await self._create_in_session(
                own_session,
                work_unit_id=work_unit_id,
                challenge_slug=challenge_slug,
                submission_ref=submission_ref,
                worker_id=worker_id,
                worker_pubkey=worker_pubkey,
                miner_hotkey=miner_hotkey,
                payload=payload,
                required_capability=required_capability,
                status=status,
                attempt_count=attempt_count,
                max_attempts=max_attempts,
                checkpoint_ref=checkpoint_ref,
            )

    async def _create_in_session(
        self,
        session: AsyncSession,
        *,
        work_unit_id: str,
        challenge_slug: str,
        submission_ref: str,
        worker_id: str,
        worker_pubkey: str,
        miner_hotkey: str,
        payload: Mapping[str, Any] | None,
        required_capability: str,
        status: WorkAssignmentStatus,
        attempt_count: int,
        max_attempts: int,
        checkpoint_ref: str | None,
    ) -> WorkerAssignment:
        now = self._now_fn()
        row = WorkerAssignment(
            challenge_slug=challenge_slug,
            work_unit_id=work_unit_id,
            submission_ref=submission_ref,
            worker_id=worker_id,
            worker_pubkey=worker_pubkey,
            miner_hotkey=miner_hotkey,
            payload=dict(payload or {}),
            required_capability=required_capability,
            status=status,
            attempt_count=attempt_count,
            max_attempts=max_attempts,
            checkpoint_ref=checkpoint_ref,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        await session.flush()
        return row

    async def pull(self, *, worker_pubkey: str) -> list[WorkerAssignment]:
        """Return the ACTIVE worker's assigned/running gpu replicas.

        Transitions ``assigned`` replicas to ``running`` with a fresh lease. A
        worker that is not currently ``active`` (pending/stale/retired) receives
        no units, and only ``gpu``-capability replicas are ever returned.
        """

        now = self._now_fn()
        lease = timedelta(seconds=self._lease_seconds)
        async with session_scope(self._session_factory) as session:
            worker = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_pubkey == worker_pubkey
                    )
                )
            ).scalar_one_or_none()
            if worker is None:
                return []
            if (
                self._worker_service.effective_status(worker, now)
                != WorkerStatus.ACTIVE
            ):
                return []

            rows = (
                (
                    await session.execute(
                        select(WorkerAssignment)
                        .where(
                            WorkerAssignment.worker_pubkey == worker_pubkey,
                            WorkerAssignment.status.in_(_PULLABLE_STATUSES),
                        )
                        .order_by(
                            WorkerAssignment.created_at,
                            WorkerAssignment.work_unit_id,
                        )
                    )
                )
                .scalars()
                .all()
            )

            pulled: list[WorkerAssignment] = []
            for unit in rows:
                if unit.required_capability != CAPABILITY_GPU:
                    continue
                if unit.status == WorkAssignmentStatus.ASSIGNED:
                    unit.status = WorkAssignmentStatus.RUNNING
                    unit.deadline_at = now + lease
                    unit.last_progress_at = now
                pulled.append(unit)
            return pulled

    async def post_result(
        self,
        *,
        assignment_id: str,
        worker_pubkey: str,
        success: bool,
        payload: Mapping[str, Any] | None = None,
        checkpoint_ref: str | None = None,
    ) -> WorkerResultOutcome:
        """Persist a worker replica result and transition it to terminal state.

        Ownership-gated: a post for a replica assigned to a different worker (a
        late or foreign post) is rejected without persisting anything. Idempotent
        for an already-terminal replica. The reported ``ExecutionProof``'s
        ``manifest_sha256`` is extracted onto the row for reconciliation.
        """

        now = self._now_fn()
        parsed = _parse_uuid(assignment_id)
        if parsed is None:
            raise WorkerAssignmentNotFoundError(assignment_id)
        async with session_scope(self._session_factory) as session:
            unit = (
                await session.execute(
                    select(WorkerAssignment).where(WorkerAssignment.id == parsed)
                )
            ).scalar_one_or_none()
            if unit is None:
                raise WorkerAssignmentNotFoundError(assignment_id)
            if unit.worker_pubkey != worker_pubkey:
                raise WorkerAssignmentOwnershipError(assignment_id)
            if WorkAssignmentStatus(unit.status) in _TERMINAL_STATUSES:
                return WorkerResultOutcome(
                    status=WorkAssignmentStatus(unit.status).value,
                    result_ref=str(unit.id),
                    idempotent=True,
                )

            result_payload = dict(payload or {})
            unit.result_success = success
            unit.result_payload = result_payload
            unit.manifest_sha256 = _extract_manifest_sha256(result_payload)
            unit.status = (
                WorkAssignmentStatus.COMPLETED
                if success
                else WorkAssignmentStatus.FAILED
            )
            unit.last_progress_at = now
            if checkpoint_ref is not None:
                unit.checkpoint_ref = checkpoint_ref
            return WorkerResultOutcome(
                status=WorkAssignmentStatus(unit.status).value,
                result_ref=str(unit.id),
                idempotent=False,
            )


def build_worker_assignment_router(
    *,
    service: WorkerAssignmentService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    """Build the worker assignment router (pull + result).

    ``auth_dependency`` MUST authenticate the caller as a REGISTERED worker (not
    validator): the worker plane never gates on a metagraph validator permit.
    """

    router = APIRouter()

    @router.post("/v1/workers/assignments/pull", response_model=AssignmentPullResponse)
    async def pull_worker_assignments(
        identity: Any = Depends(auth_dependency),
    ) -> AssignmentPullResponse:
        units = await service.pull(worker_pubkey=identity.hotkey)
        return AssignmentPullResponse(
            assignments=[worker_assignment_to_view(unit) for unit in units]
        )

    @router.post(
        "/v1/workers/assignments/{assignment_id}/result",
        response_model=AssignmentResultResponse,
    )
    async def post_worker_result(
        assignment_id: str,
        payload: AssignmentResultRequest,
        identity: Any = Depends(auth_dependency),
    ) -> AssignmentResultResponse:
        try:
            outcome = await service.post_result(
                assignment_id=assignment_id,
                worker_pubkey=identity.hotkey,
                success=payload.success,
                payload=payload.payload,
                checkpoint_ref=payload.checkpoint_ref,
            )
        except WorkerAssignmentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="worker assignment not found",
            ) from exc
        except WorkerAssignmentOwnershipError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="worker assignment not owned by caller",
            ) from exc
        return AssignmentResultResponse(
            status=outcome.status,
            result_ref=outcome.result_ref,
            idempotent=outcome.idempotent,
        )

    return router


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "WorkerAssignmentNotFoundError",
    "WorkerAssignmentOwnershipError",
    "WorkerAssignmentService",
    "WorkerResultOutcome",
    "build_worker_assignment_router",
    "worker_assignment_to_view",
]
