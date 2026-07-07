"""Read-only worker-plane unit status surface (architecture.md sec 3.3).

``GET /v1/workers/units`` makes the dispute -> audit -> invalidation -> fault
chain OPERATOR-DISCOVERABLE VIA API ALONE (VAL-CROSS-011): today the disputed
state and the audit unit's executor kind/outcome live only in ``worker_faults``
detail strings, so the divergence story cannot be reconstructed without reading
the database. This surface exposes, per primary gpu unit:

* the unit id + status (INCLUDING ``disputed``);
* its replicas (worker_id, owner miner hotkey, posted ``manifest_sha256``, and
  whether a proof envelope was posted);
* for a disputed unit, the linked validator AUDIT unit's id, executor kind
  (``validator``), and terminal outcome (``pending``/``passed``/
  ``mismatch-resolved``).

It reads existing control-plane tables only (``work_assignments`` primaries +
their ``worker_assignments`` replicas + the validator audit ``work_assignments``
row + ``worker_faults``); no schema change is required.

Auth is the signed-request fleet-read (``CoordinationReadEligibility`` -- a
registered worker OR an eligible validator), IDENTICAL to ``GET /v1/workers`` and
distinct from the admission fleet-read's internal bridge bearer (never accepted
here). The router is mounted only when ``compute.worker_plane_enabled`` is on;
with the flag off it is unmounted (404) and legacy behavior is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import (
    WorkAssignment,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerFault,
)
from base.master.assignment import (
    CAPABILITY_GPU,
    EXECUTOR_KIND_VALIDATOR,
    unit_executor_kind,
)
from base.master.worker_reconciliation import (
    AUDIT_OF_PAYLOAD_KEY,
    AUDIT_RESOLVED_PAYLOAD_KEY,
)
from base.schemas.worker import (
    WorkerAuditUnitView,
    WorkerReplicaView,
    WorkerUnitStatusListResponse,
    WorkerUnitStatusView,
)
from base.worker.proof import PROOF_PAYLOAD_KEY

#: Terminal audit outcomes exposed for a disputed unit's validator audit.
AUDIT_OUTCOME_PENDING = "pending"
AUDIT_OUTCOME_PASSED = "passed"
AUDIT_OUTCOME_MISMATCH_RESOLVED = "mismatch-resolved"


def _replica_has_proof(replica: WorkerAssignment) -> bool:
    """Whether the replica posted an ExecutionProof envelope in its result."""

    payload = replica.result_payload or {}
    proof = payload.get(PROOF_PAYLOAD_KEY)
    return isinstance(proof, Mapping) and bool(proof)


def _replica_to_view(replica: WorkerAssignment) -> WorkerReplicaView:
    return WorkerReplicaView(
        worker_id=replica.worker_id,
        miner_hotkey=replica.miner_hotkey,
        status=WorkAssignmentStatus(replica.status).value,
        manifest_sha256=replica.manifest_sha256,
        has_proof=_replica_has_proof(replica),
    )


def _audit_outcome(audit: WorkAssignment, *, faulted: bool) -> str:
    """Derive the audit's terminal outcome from its resolution + fault state.

    ``pending`` until the validator replay has been folded into faults
    (``AUDIT_RESOLVED_PAYLOAD_KEY``); once resolved, ``mismatch-resolved`` when a
    worker fault was attributed for the disputed unit, else ``passed``.
    """

    payload = audit.payload or {}
    if not payload.get(AUDIT_RESOLVED_PAYLOAD_KEY):
        return AUDIT_OUTCOME_PENDING
    return AUDIT_OUTCOME_MISMATCH_RESOLVED if faulted else AUDIT_OUTCOME_PASSED


class WorkerUnitStatusService:
    """Read the worker-plane unit/replica/audit/fault chain for operators."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        required_capability: str = CAPABILITY_GPU,
    ) -> None:
        self._session_factory = session_factory
        self._required_capability = required_capability

    async def list_units(self) -> list[WorkerUnitStatusView]:
        """Return each primary gpu unit with replicas and (if disputed) its audit."""

        async with self._session_factory() as session:
            return await self._list_in_session(session)

    async def _list_in_session(
        self, session: AsyncSession
    ) -> list[WorkerUnitStatusView]:
        gpu_units = (
            (
                await session.execute(
                    select(WorkAssignment).where(
                        WorkAssignment.required_capability == self._required_capability
                    )
                )
            )
            .scalars()
            .all()
        )
        primaries: list[WorkAssignment] = []
        audits_by_original: dict[tuple[str, str], WorkAssignment] = {}
        for row in gpu_units:
            payload = row.payload or {}
            if unit_executor_kind(payload) == EXECUTOR_KIND_VALIDATOR:
                original = payload.get(AUDIT_OF_PAYLOAD_KEY)
                if isinstance(original, str) and original:
                    audits_by_original[(row.challenge_slug, original)] = row
            else:
                primaries.append(row)

        replicas = (await session.execute(select(WorkerAssignment))).scalars().all()
        replicas_by_unit: dict[tuple[str, str], list[WorkerAssignment]] = {}
        for replica in replicas:
            replicas_by_unit.setdefault(
                (replica.challenge_slug, replica.work_unit_id), []
            ).append(replica)

        faults = (await session.execute(select(WorkerFault))).scalars().all()
        faulted_units = {fault.work_unit_id for fault in faults}

        views: list[WorkerUnitStatusView] = []
        for primary in sorted(
            primaries, key=lambda row: (row.challenge_slug, row.work_unit_id)
        ):
            key = (primary.challenge_slug, primary.work_unit_id)
            unit_replicas = sorted(
                replicas_by_unit.get(key, []),
                key=lambda row: (row.miner_hotkey, row.worker_id),
            )
            audit_view: WorkerAuditUnitView | None = None
            if WorkAssignmentStatus(primary.status) == WorkAssignmentStatus.DISPUTED:
                audit = audits_by_original.get(key)
                if audit is not None:
                    audit_view = WorkerAuditUnitView(
                        work_unit_id=audit.work_unit_id,
                        executor_kind=EXECUTOR_KIND_VALIDATOR,
                        outcome=_audit_outcome(
                            audit,
                            faulted=primary.work_unit_id in faulted_units,
                        ),
                    )
            views.append(
                WorkerUnitStatusView(
                    work_unit_id=primary.work_unit_id,
                    challenge_slug=primary.challenge_slug,
                    submission_ref=primary.submission_ref,
                    status=WorkAssignmentStatus(primary.status).value,
                    replicas=[_replica_to_view(row) for row in unit_replicas],
                    audit=audit_view,
                )
            )
        return views


def build_worker_unit_status_router(
    *,
    service: WorkerUnitStatusService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    """Build the read-only worker unit-status router (``GET /v1/workers/units``).

    ``auth_dependency`` MUST be the signed-request fleet-read dependency
    (``CoordinationReadEligibility``), identical to ``GET /v1/workers``; the
    internal bridge bearer is never accepted here.
    """

    router = APIRouter()

    @router.get(
        "/v1/workers/units",
        response_model=WorkerUnitStatusListResponse,
        dependencies=[Depends(auth_dependency)],
    )
    async def list_worker_units() -> WorkerUnitStatusListResponse:
        return WorkerUnitStatusListResponse(units=await service.list_units())

    return router


__all__ = [
    "AUDIT_OUTCOME_MISMATCH_RESOLVED",
    "AUDIT_OUTCOME_PASSED",
    "AUDIT_OUTCOME_PENDING",
    "WorkerUnitStatusService",
    "build_worker_unit_status_router",
]
