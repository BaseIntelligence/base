"""Worker-plane reconciliation, disputes, and audit-fault attribution.

Architecture.md sec 3.3: once the replicas of a gpu work unit report, the master
reconciles their ``ExecutionProof.manifest_sha256`` values.

* **Matching hashes** -> the unit is accepted and EXACTLY ONE result is forwarded
  to the challenge (single fold); the primary ``work_assignments`` row reaches
  ``completed``.
* **Divergent hashes** -> the unit is marked ``disputed`` (a terminal state that
  is NEVER forwarded to the challenge, before or after audit), and a validator-
  executor AUDIT unit (a fresh ``work_assignments`` row, ``required_capability=gpu``,
  ``executor_kind=validator``) is created. The validator replay is authoritative:
  when the audit result lands, every worker whose manifest diverged from it gets a
  ``worker_faults`` row (its ``worker_registrations.status`` is untouched), and the
  disputed unit stays disputed (the submission is left unscored).
* **Single surviving proof** -> when the other replica can never report (its
  worker went stale/retired and the replica's retries are exhausted with no other
  eligible distinct owner, per the reassignment pass) the unit does NOT hang: it
  is accepted from the one proof, forwarded once, and a degradation warning is
  recorded. A replica that merely posts ``success=false`` (no proof) never
  triggers a dispute -- dispute requires TWO differing manifest hashes.

Late/foreign result posts cannot corrupt reconciliation: they are rejected by the
ownership gate in :class:`base.master.worker_assignment.WorkerAssignmentService`
before touching any replica state, so reconciliation only ever reads
legitimately-owned replica results.

All of this is gated behind ``compute.worker_plane_enabled``: with the flag OFF
the reconciler is never constructed and no gpu unit is ever replicated, disputed,
or audited.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import (
    WorkAssignment,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerFault,
    WorkResult,
)
from base.db.session import session_scope
from base.master.assignment import (
    CAPABILITY_GPU,
    EXECUTOR_KIND_PAYLOAD_KEY,
    EXECUTOR_KIND_VALIDATOR,
    unit_executor_kind,
)
from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY, PROOF_PAYLOAD_KEY

logger = logging.getLogger(__name__)

#: Suffix appended to a disputed unit's id to form its validator AUDIT unit id.
AUDIT_WORK_UNIT_SUFFIX = ":audit"

#: Audit-unit payload key naming the original (disputed) work-unit id.
AUDIT_OF_PAYLOAD_KEY = "audit_of_work_unit_id"
#: Audit-unit payload flag set once its outcome has been folded into faults, so a
#: resolved audit is not reprocessed on later passes.
AUDIT_RESOLVED_PAYLOAD_KEY = "audit_resolved"
#: Primary-unit payload flag recording that a unit was accepted from a single
#: surviving proof (degraded replication) -- the reconciliation warning channel
#: (mirrors the assignment-engine degradation marker for VAL-MASTER-007/017).
RECONCILE_DEGRADED_PAYLOAD_KEY = "worker_reconciliation_degraded"

_TERMINAL_REPLICA_STATUSES = (
    WorkAssignmentStatus.COMPLETED,
    WorkAssignmentStatus.FAILED,
)
_TERMINAL_UNIT_STATUSES = (
    WorkAssignmentStatus.COMPLETED,
    WorkAssignmentStatus.FAILED,
    WorkAssignmentStatus.DISPUTED,
)


def audit_work_unit_id(work_unit_id: str) -> str:
    """Deterministic id of the validator AUDIT unit for a disputed unit."""

    return f"{work_unit_id}{AUDIT_WORK_UNIT_SUFFIX}"


def _manifest_from_payload(payload: Mapping[str, Any] | None) -> str | None:
    """Read the manifest hash from a result payload's ExecutionProof envelope."""

    if not payload:
        return None
    proof = payload.get(PROOF_PAYLOAD_KEY)
    if isinstance(proof, Mapping):
        manifest = proof.get(MANIFEST_SHA256_PAYLOAD_KEY)
        if isinstance(manifest, str) and manifest:
            return manifest
    manifest = payload.get(MANIFEST_SHA256_PAYLOAD_KEY)
    return manifest if isinstance(manifest, str) and manifest else None


class ChallengeResultForwarder(Protocol):
    """Forward exactly one accepted result to the challenge (fold seam).

    Mirrors :class:`base.master.orchestration.ChallengeFoldTrigger`: the
    production HTTP implementation posts to the challenge's internal result route;
    tests substitute a fake that counts forwards.
    """

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Mapping[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class ReconciliationPassResult:
    """Observable outcome of one reconciliation pass.

    ``accepted`` lists units accepted (result forwarded once); ``single_replica``
    is the subset accepted from a single surviving proof (degraded, warned);
    ``disputed`` lists units marked disputed this pass; ``audit_units`` maps a
    disputed unit id to its created audit unit id; ``faults`` lists
    ``{work_unit_id}:{worker_id}`` pairs faulted from audit outcomes this pass.
    """

    accepted: list[str] = field(default_factory=list)
    single_replica: list[str] = field(default_factory=list)
    disputed: list[str] = field(default_factory=list)
    audit_units: dict[str, str] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)


class WorkerReconciliationService:
    """Reconcile worker replica results; dispute divergence; fold audit faults."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        result_forwarder: ChallengeResultForwarder | None = None,
        required_capability: str = CAPABILITY_GPU,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._result_forwarder = result_forwarder
        self._required_capability = required_capability
        self._now_fn = now_fn

    def transaction(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Open a single committed transaction over the control-plane DB."""

        return session_scope(self._session_factory)

    async def reconcile_once(
        self, *, session: AsyncSession | None = None
    ) -> ReconciliationPassResult:
        """Reconcile all worker-replicated units and fold completed audits.

        Idempotent: an already-accepted unit (primary ``completed``) is never
        re-forwarded, a disputed unit is never late-accepted, and a resolved
        audit is never re-faulted.
        """

        if session is not None:
            return await self._reconcile_in_session(session)
        async with session_scope(self._session_factory) as own_session:
            return await self._reconcile_in_session(own_session)

    async def _reconcile_in_session(
        self, session: AsyncSession
    ) -> ReconciliationPassResult:
        now = self._now_fn()
        accepted: list[str] = []
        single_replica: list[str] = []
        disputed: list[str] = []
        audit_units: dict[str, str] = {}
        faults: list[str] = []

        replicas = (await session.execute(select(WorkerAssignment))).scalars().all()
        groups: dict[tuple[str, str], list[WorkerAssignment]] = {}
        for replica in replicas:
            groups.setdefault(
                (replica.challenge_slug, replica.work_unit_id), []
            ).append(replica)

        for (slug, unit_id), rows in sorted(groups.items()):
            primary = (
                await session.execute(
                    select(WorkAssignment).where(
                        WorkAssignment.challenge_slug == slug,
                        WorkAssignment.work_unit_id == unit_id,
                    )
                )
            ).scalar_one_or_none()
            if primary is None:
                continue
            if WorkAssignmentStatus(primary.status) in _TERMINAL_UNIT_STATUSES:
                continue

            reported = [
                row
                for row in rows
                if WorkAssignmentStatus(row.status) == WorkAssignmentStatus.COMPLETED
                and row.result_success
                and row.manifest_sha256
            ]
            distinct = {row.manifest_sha256 for row in reported}
            non_terminal = [
                row
                for row in rows
                if WorkAssignmentStatus(row.status) not in _TERMINAL_REPLICA_STATUSES
            ]

            if len(distinct) >= 2:
                primary.status = WorkAssignmentStatus.DISPUTED
                primary.last_progress_at = now
                audit_id = await self._ensure_audit_unit(session, primary, now)
                disputed.append(unit_id)
                audit_units[unit_id] = audit_id
                continue

            # A replica that can still report must be awaited: dispute needs two
            # DIFFERING hashes, not a missing one.
            if non_terminal:
                continue
            # No proof at all: the assignment engine terminally fails the unit.
            if not reported:
                continue

            winner = reported[0]
            if not await self._forward(primary, winner):
                continue
            primary.status = WorkAssignmentStatus.COMPLETED
            primary.result_ref = str(winner.id)
            primary.last_progress_at = now
            accepted.append(unit_id)
            if len(reported) < len(rows):
                primary.payload = {
                    **(primary.payload or {}),
                    RECONCILE_DEGRADED_PAYLOAD_KEY: True,
                }
                single_replica.append(unit_id)
                logger.warning(
                    "worker unit %s reconciled from a single surviving proof "
                    "(%d of %d replicas reported); accepted with degraded "
                    "replication",
                    unit_id,
                    len(reported),
                    len(rows),
                )

        faults = await self._resolve_audits(session, replicas, now)

        await session.flush()
        return ReconciliationPassResult(
            accepted=accepted,
            single_replica=single_replica,
            disputed=disputed,
            audit_units=audit_units,
            faults=faults,
        )

    async def _resolve_audits(
        self,
        session: AsyncSession,
        replicas: Sequence[WorkerAssignment],
        now: datetime,
    ) -> list[str]:
        faults: list[str] = []
        completed = (
            (
                await session.execute(
                    select(WorkAssignment).where(
                        WorkAssignment.status == WorkAssignmentStatus.COMPLETED
                    )
                )
            )
            .scalars()
            .all()
        )
        for audit in completed:
            payload = audit.payload or {}
            if unit_executor_kind(payload) != EXECUTOR_KIND_VALIDATOR:
                continue
            if payload.get(AUDIT_RESOLVED_PAYLOAD_KEY):
                continue
            original_id = payload.get(AUDIT_OF_PAYLOAD_KEY)
            if not original_id:
                continue
            authoritative = await self._audit_manifest(session, audit)
            if authoritative is None:
                continue
            for replica in replicas:
                if (
                    replica.challenge_slug != audit.challenge_slug
                    or replica.work_unit_id != original_id
                    or not replica.manifest_sha256
                ):
                    continue
                if replica.manifest_sha256 != authoritative:
                    session.add(
                        WorkerFault(
                            worker_id=replica.worker_id,
                            work_unit_id=original_id,
                            challenge_slug=audit.challenge_slug,
                            detail=(
                                "manifest diverged from validator audit "
                                f"({replica.manifest_sha256} != {authoritative})"
                            ),
                            created_at=now,
                        )
                    )
                    faults.append(f"{original_id}:{replica.worker_id}")
            audit.payload = {**payload, AUDIT_RESOLVED_PAYLOAD_KEY: True}
        return faults

    async def _audit_manifest(
        self, session: AsyncSession, audit: WorkAssignment
    ) -> str | None:
        result = (
            (
                await session.execute(
                    select(WorkResult)
                    .where(WorkResult.assignment_id == audit.id)
                    .order_by(WorkResult.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        if result is None:
            return None
        return _manifest_from_payload(result.payload)

    async def _ensure_audit_unit(
        self, session: AsyncSession, primary: WorkAssignment, now: datetime
    ) -> str:
        audit_id = audit_work_unit_id(primary.work_unit_id)
        existing = (
            await session.execute(
                select(WorkAssignment).where(
                    WorkAssignment.challenge_slug == primary.challenge_slug,
                    WorkAssignment.work_unit_id == audit_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return audit_id
        payload = {
            key: value
            for key, value in (primary.payload or {}).items()
            if key != RECONCILE_DEGRADED_PAYLOAD_KEY
        }
        payload[EXECUTOR_KIND_PAYLOAD_KEY] = EXECUTOR_KIND_VALIDATOR
        payload[AUDIT_OF_PAYLOAD_KEY] = primary.work_unit_id
        session.add(
            WorkAssignment(
                challenge_slug=primary.challenge_slug,
                work_unit_id=audit_id,
                submission_ref=primary.submission_ref,
                payload=payload,
                required_capability=self._required_capability,
                status=WorkAssignmentStatus.PENDING,
                attempt_count=0,
                max_attempts=primary.max_attempts,
                checkpoint_ref=primary.checkpoint_ref,
                created_at=now,
                updated_at=now,
            )
        )
        return audit_id

    async def _forward(self, primary: WorkAssignment, winner: WorkerAssignment) -> bool:
        if self._result_forwarder is None:
            return True
        try:
            await self._result_forwarder.forward_result(
                challenge_slug=primary.challenge_slug,
                work_unit_id=primary.work_unit_id,
                submission_ref=primary.submission_ref,
                result_payload=dict(winner.result_payload or {}),
            )
        except Exception:
            logger.exception(
                "failed to forward accepted result for unit %s",
                primary.work_unit_id,
            )
            return False
        return True


__all__ = [
    "AUDIT_OF_PAYLOAD_KEY",
    "AUDIT_RESOLVED_PAYLOAD_KEY",
    "AUDIT_WORK_UNIT_SUFFIX",
    "RECONCILE_DEGRADED_PAYLOAD_KEY",
    "ChallengeResultForwarder",
    "ReconciliationPassResult",
    "WorkerReconciliationService",
    "audit_work_unit_id",
]
