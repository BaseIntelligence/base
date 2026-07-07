"""Worker-plane replica-creation ENGINE (architecture.md sec 3.3).

Layered on top of :class:`base.master.worker_assignment.WorkerAssignmentService`
(the low-level replica-row primitive ``create_worker_assignment``) and
:class:`base.master.worker_coordination.WorkerCoordinationService` (worker
eligibility via ``effective_status``). All of this is gated behind
``compute.worker_plane_enabled``: with the flag OFF the engine is never
constructed and gpu units route to validators exactly as legacy.

The engine reads PENDING ``work_assignments`` rows whose ``required_capability``
is gpu -- the validator :class:`base.master.assignment.AssignmentService` is
configured to SKIP these when the worker plane is on (``worker_plane_capabilities``)
-- and materializes ``worker_assignments`` replica rows for them:

* **No self-evaluation**: a unit whose ``submission_ref`` (the submitting miner
  hotkey) equals a worker's ``miner_hotkey`` is never assigned to that worker,
  even when it is the only capacity (the unit simply waits, mirroring the legacy
  no-eligible-executor behavior).
* **Replication R=** ``replication_factor`` (default 2) across DISTINCT owner
  hotkeys -- two workers of the same owner never both serve one unit.
* **Graceful degradation** to a single replica (with a durably recorded warning
  on the unit payload + a log record) when fewer than ``replication_factor``
  eligible distinct owners exist.
* **Per-worker gpu concurrency 1** (mirrors ``DEFAULT_CAPABILITY_CONCURRENCY``):
  a worker already holding an in-flight gpu replica is not given a second.
* **Heartbeat TTL** drives eligibility: a stale/retired worker is never chosen.

:meth:`reassign_stale_replicas` mirrors the legacy validator reassignment pass:
a replica whose lease deadline lapsed, or whose worker went stale/retired, is
moved to a different eligible worker (bounded by ``max_attempts`` per replica --
exhausted replicas are terminally ``failed`` and never recreated); a unit whose
replicas are ALL exhausted is itself marked ``failed``.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import (
    WorkAssignment,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerRegistration,
    WorkerStatus,
)
from base.db.session import session_scope
from base.master.assignment import (
    CAPABILITY_GPU,
    DEFAULT_CAPABILITY_CONCURRENCY,
    EXECUTOR_KIND_VALIDATOR,
    unit_executor_kind,
)
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_coordination import WorkerCoordinationService

logger = logging.getLogger(__name__)

DEFAULT_REPLICATION_FACTOR = 2

#: Payload key stamped onto a primary ``work_assignments`` unit whose worker
#: replication was degraded below ``replication_factor`` (observable warning,
#: alongside the emitted log record).
DEGRADED_REPLICATION_PAYLOAD_KEY = "worker_replication_degraded"

_ACTIVE_REPLICA_STATUSES = (
    WorkAssignmentStatus.ASSIGNED,
    WorkAssignmentStatus.RUNNING,
)


@dataclass(frozen=True)
class WorkerAssignmentPassResult:
    """Observable outcome of one worker-plane assignment pass.

    ``created`` maps each work-unit id to the worker ids it was replicated to
    this pass; ``degraded`` lists the units assigned fewer than
    ``replication_factor`` replicas (a warning was recorded for each).
    """

    created: dict[str, list[str]] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkerReassignmentPassResult:
    """Observable outcome of one worker-plane reassignment pass.

    ``reassigned`` lists the work-unit ids that had a stale replica moved to a
    new worker; ``failed_replicas`` lists ``{work_unit_id}:{worker_id}`` pairs
    terminally failed (retries exhausted); ``failed_units`` lists the units whose
    replicas are ALL exhausted (the unit itself is marked ``failed``).
    """

    reassigned: list[str] = field(default_factory=list)
    failed_replicas: list[str] = field(default_factory=list)
    failed_units: list[str] = field(default_factory=list)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class WorkerAssignmentEngine:
    """Create + maintain gpu work-unit replicas across distinct-owner workers."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        assignment_service: WorkerAssignmentService,
        worker_service: WorkerCoordinationService,
        replication_factor: int = DEFAULT_REPLICATION_FACTOR,
        required_capability: str = CAPABILITY_GPU,
        gpu_concurrency: int | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._assignment_service = assignment_service
        self._worker_service = worker_service
        self._replication_factor = max(1, replication_factor)
        self._required_capability = required_capability
        self._gpu_concurrency = (
            DEFAULT_CAPABILITY_CONCURRENCY.get(required_capability, 1)
            if gpu_concurrency is None
            else gpu_concurrency
        )
        self._now_fn = now_fn

    def transaction(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Open a single committed transaction over the control-plane DB.

        Used to compose :meth:`reassign_stale_replicas` and
        :meth:`assign_pending` into one atomic pass (mirrors
        ``AssignmentService.transaction``).
        """

        return session_scope(self._session_factory)

    # -- assignment ---------------------------------------------------------

    async def assign_pending(
        self, *, seed: int | None = None, session: AsyncSession | None = None
    ) -> WorkerAssignmentPassResult:
        """Replicate fresh gpu units across eligible distinct-owner workers.

        A unit that already has any replica row is left untouched (its existing
        replicas are maintained by :meth:`reassign_stale_replicas`). A unit with
        no eligible distinct-owner worker stays ``pending`` (never lost).
        """

        if session is not None:
            return await self._assign_pending_in_session(session, seed=seed)
        async with session_scope(self._session_factory) as own_session:
            return await self._assign_pending_in_session(own_session, seed=seed)

    async def _assign_pending_in_session(
        self, session: AsyncSession, *, seed: int | None
    ) -> WorkerAssignmentPassResult:
        rng = random.Random(seed)
        created: dict[str, list[str]] = {}
        degraded: list[str] = []

        active_workers = await self._active_workers(session)
        if not active_workers:
            return WorkerAssignmentPassResult(created={}, degraded=[])

        load = await self._inflight_load(session, active_workers)

        pending_units = (
            (
                await session.execute(
                    select(WorkAssignment)
                    .where(
                        WorkAssignment.required_capability == self._required_capability,
                        WorkAssignment.status == WorkAssignmentStatus.PENDING,
                    )
                    .order_by(WorkAssignment.created_at, WorkAssignment.work_unit_id)
                )
            )
            .scalars()
            .all()
        )

        for unit in pending_units:
            if unit_executor_kind(unit.payload) == EXECUTOR_KIND_VALIDATOR:
                continue
            existing = (
                await session.execute(
                    select(func.count())
                    .select_from(WorkerAssignment)
                    .where(
                        WorkerAssignment.work_unit_id == unit.work_unit_id,
                        WorkerAssignment.challenge_slug == unit.challenge_slug,
                    )
                )
            ).scalar_one()
            if existing:
                continue

            chosen = self._select_owners(
                active_workers,
                load=load,
                submitter_hotkey=unit.submission_ref,
                excluded_owners=set(),
                excluded_pubkeys=set(),
                count=self._replication_factor,
                rng=rng,
            )
            if not chosen:
                continue

            created_ids: list[str] = []
            for worker in chosen:
                replica = await self._assignment_service.create_worker_assignment(
                    work_unit_id=unit.work_unit_id,
                    challenge_slug=unit.challenge_slug,
                    submission_ref=unit.submission_ref,
                    worker_id=worker.worker_id,
                    worker_pubkey=worker.worker_pubkey,
                    miner_hotkey=worker.miner_hotkey,
                    payload=dict(unit.payload or {}),
                    required_capability=unit.required_capability,
                    max_attempts=unit.max_attempts,
                    checkpoint_ref=unit.checkpoint_ref,
                    session=session,
                )
                created_ids.append(replica.worker_id)
                load[worker.worker_pubkey] = load.get(worker.worker_pubkey, 0) + 1

            unit.status = WorkAssignmentStatus.ASSIGNED
            unit.attempt_count = (unit.attempt_count or 0) + 1
            created[unit.work_unit_id] = created_ids

            if len(chosen) < self._replication_factor:
                self._record_degradation(unit, replicas=len(chosen))
                degraded.append(unit.work_unit_id)

        await session.flush()
        return WorkerAssignmentPassResult(created=created, degraded=degraded)

    # -- reassignment -------------------------------------------------------

    async def reassign_stale_replicas(
        self, *, seed: int | None = None, session: AsyncSession | None = None
    ) -> WorkerReassignmentPassResult:
        """Reassign lapsed/stale replicas; fail exhausted ones (bounded retries).

        Mirrors the legacy validator reassignment: a replica whose deadline
        lapsed or whose worker is no longer ``active`` is moved to another
        eligible worker (self-evaluation and distinct-owner rules still apply),
        incrementing ``attempt_count``. A replica whose attempts are exhausted is
        terminally ``failed`` and never recreated; a unit whose replicas are ALL
        failed is itself marked ``failed``.
        """

        if session is not None:
            return await self._reassign_in_session(session, seed=seed)
        async with session_scope(self._session_factory) as own_session:
            return await self._reassign_in_session(own_session, seed=seed)

    async def _reassign_in_session(
        self, session: AsyncSession, *, seed: int | None
    ) -> WorkerReassignmentPassResult:
        rng = random.Random(seed)
        now = self._now_fn()
        reassigned: list[str] = []
        failed_replicas: list[str] = []
        failed_units: list[str] = []

        active_workers = await self._active_workers(session)
        active_pubkeys = {w.worker_pubkey for w in active_workers}

        replicas = (await session.execute(select(WorkerAssignment))).scalars().all()
        load: dict[str, int] = {w.worker_pubkey: 0 for w in active_workers}
        for row in replicas:
            if (
                WorkAssignmentStatus(row.status) in _ACTIVE_REPLICA_STATUSES
                and row.worker_pubkey in load
            ):
                load[row.worker_pubkey] += 1

        # Owners currently covering each unit with a non-terminal replica.
        covered: dict[str, set[str]] = {}
        for row in replicas:
            if WorkAssignmentStatus(row.status) in _ACTIVE_REPLICA_STATUSES:
                covered.setdefault(row.work_unit_id, set()).add(row.miner_hotkey)

        for row in replicas:
            if WorkAssignmentStatus(row.status) not in _ACTIVE_REPLICA_STATUSES:
                continue
            worker_active = row.worker_pubkey in active_pubkeys
            deadline = row.deadline_at
            deadline_passed = deadline is not None and _as_utc(deadline) < now
            if worker_active and not deadline_passed:
                continue

            unit_covered = covered.setdefault(row.work_unit_id, set())
            if (row.attempt_count or 0) >= row.max_attempts:
                row.status = WorkAssignmentStatus.FAILED
                row.last_progress_at = now
                unit_covered.discard(row.miner_hotkey)
                if row.worker_pubkey in load and load[row.worker_pubkey] > 0:
                    load[row.worker_pubkey] -= 1
                failed_replicas.append(f"{row.work_unit_id}:{row.worker_id}")
                continue

            replacement = self._select_owners(
                active_workers,
                load=load,
                submitter_hotkey=row.submission_ref,
                excluded_owners=unit_covered - {row.miner_hotkey},
                excluded_pubkeys={row.worker_pubkey},
                count=1,
                rng=rng,
            )
            if not replacement:
                continue
            new_worker = replacement[0]
            if row.worker_pubkey in load and load[row.worker_pubkey] > 0:
                load[row.worker_pubkey] -= 1
            unit_covered.discard(row.miner_hotkey)
            row.worker_id = new_worker.worker_id
            row.worker_pubkey = new_worker.worker_pubkey
            row.miner_hotkey = new_worker.miner_hotkey
            row.status = WorkAssignmentStatus.ASSIGNED
            row.attempt_count = (row.attempt_count or 0) + 1
            row.deadline_at = None
            row.last_progress_at = now
            load[new_worker.worker_pubkey] = load.get(new_worker.worker_pubkey, 0) + 1
            unit_covered.add(new_worker.miner_hotkey)
            reassigned.append(row.work_unit_id)

        failed_units = await self._fail_exhausted_units(session, replicas)
        await session.flush()
        return WorkerReassignmentPassResult(
            reassigned=reassigned,
            failed_replicas=failed_replicas,
            failed_units=failed_units,
        )

    async def _fail_exhausted_units(
        self, session: AsyncSession, replicas: Sequence[WorkerAssignment]
    ) -> list[str]:
        replicas_by_unit: dict[tuple[str, str], list[WorkerAssignment]] = {}
        for row in replicas:
            replicas_by_unit.setdefault(
                (row.challenge_slug, row.work_unit_id), []
            ).append(row)

        failed_units: list[str] = []
        primary_units = (
            (
                await session.execute(
                    select(WorkAssignment).where(
                        WorkAssignment.required_capability == self._required_capability,
                        WorkAssignment.status.in_(_ACTIVE_REPLICA_STATUSES),
                    )
                )
            )
            .scalars()
            .all()
        )
        for unit in primary_units:
            unit_replicas = replicas_by_unit.get(
                (unit.challenge_slug, unit.work_unit_id)
            )
            if not unit_replicas:
                continue
            if all(
                WorkAssignmentStatus(r.status) == WorkAssignmentStatus.FAILED
                for r in unit_replicas
            ):
                unit.status = WorkAssignmentStatus.FAILED
                failed_units.append(unit.work_unit_id)
        return failed_units

    # -- helpers ------------------------------------------------------------

    async def _active_workers(self, session: AsyncSession) -> list[WorkerRegistration]:
        now = self._now_fn()
        rows = (
            (
                await session.execute(
                    select(WorkerRegistration).order_by(
                        WorkerRegistration.created_at,
                        WorkerRegistration.worker_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [
            row
            for row in rows
            if self._worker_service.effective_status(row, now) == WorkerStatus.ACTIVE
        ]

    async def _inflight_load(
        self, session: AsyncSession, active_workers: list[WorkerRegistration]
    ) -> dict[str, int]:
        load = {worker.worker_pubkey: 0 for worker in active_workers}
        counts = (
            await session.execute(
                select(WorkerAssignment.worker_pubkey, func.count())
                .where(
                    WorkerAssignment.required_capability == self._required_capability,
                    WorkerAssignment.status.in_(_ACTIVE_REPLICA_STATUSES),
                )
                .group_by(WorkerAssignment.worker_pubkey)
            )
        ).all()
        for pubkey, count in counts:
            if pubkey in load:
                load[pubkey] = count
        return load

    def _select_owners(
        self,
        active_workers: list[WorkerRegistration],
        *,
        load: dict[str, int],
        submitter_hotkey: str,
        excluded_owners: set[str],
        excluded_pubkeys: set[str],
        count: int,
        rng: random.Random,
    ) -> list[WorkerRegistration]:
        """Pick up to ``count`` distinct-owner workers under the eligibility rules.

        Excludes the submitter's own workers (self-evaluation), owners already
        covering the unit, the current replica's worker, and any worker at its
        gpu concurrency cap. One worker per owner is considered (the lowest
        ``worker_pubkey``), so two workers of the same owner never both count.
        Selection is least-loaded-first with seeded-random tie-breaking, so a
        fixed seed + identical inputs yield an identical choice.
        """

        by_owner: dict[str, WorkerRegistration] = {}
        for worker in active_workers:
            if worker.miner_hotkey == submitter_hotkey:
                continue
            if worker.miner_hotkey in excluded_owners:
                continue
            if worker.worker_pubkey in excluded_pubkeys:
                continue
            if load.get(worker.worker_pubkey, 0) >= self._gpu_concurrency:
                continue
            current = by_owner.get(worker.miner_hotkey)
            if current is None or worker.worker_pubkey < current.worker_pubkey:
                by_owner[worker.miner_hotkey] = worker

        chosen: list[WorkerRegistration] = []
        remaining = dict(by_owner)
        for _ in range(count):
            if not remaining:
                break
            min_load = min(
                load.get(worker.worker_pubkey, 0) for worker in remaining.values()
            )
            tied = sorted(
                (
                    owner
                    for owner, worker in remaining.items()
                    if load.get(worker.worker_pubkey, 0) == min_load
                )
            )
            owner = rng.choice(tied)
            chosen.append(remaining.pop(owner))
        return chosen

    def _record_degradation(self, unit: WorkAssignment, *, replicas: int) -> None:
        payload = dict(unit.payload or {})
        payload[DEGRADED_REPLICATION_PAYLOAD_KEY] = replicas
        unit.payload = payload
        logger.warning(
            "worker replication degraded to R=%d for unit %s (wanted R=%d): "
            "only %d eligible distinct owner(s) available",
            replicas,
            unit.work_unit_id,
            self._replication_factor,
            replicas,
        )


@dataclass(frozen=True)
class WorkerEnginePassResult:
    """Combined outcome of one full worker-plane engine pass."""

    assignment: WorkerAssignmentPassResult
    reassignment: WorkerReassignmentPassResult


async def run_worker_assignment_pass(
    *,
    engine: WorkerAssignmentEngine,
    seed: int | None = None,
) -> WorkerEnginePassResult:
    """Run reassignment (reclaim stale replicas) then fresh assignment.

    Both steps run in ONE atomic transaction so a partial failure rolls back
    cleanly. Reassignment runs first so a replica freed this pass (worker gone
    stale/retired, deadline lapsed) is immediately eligible for a fresh unit's
    distinct-owner selection in the same pass.
    """

    async with engine.transaction() as session:
        reassignment = await engine.reassign_stale_replicas(seed=seed, session=session)
        assignment = await engine.assign_pending(seed=seed, session=session)
    return WorkerEnginePassResult(assignment=assignment, reassignment=reassignment)


__all__ = [
    "DEFAULT_REPLICATION_FACTOR",
    "DEGRADED_REPLICATION_PAYLOAD_KEY",
    "WorkerAssignmentEngine",
    "WorkerAssignmentPassResult",
    "WorkerEnginePassResult",
    "WorkerReassignmentPassResult",
    "run_worker_assignment_pass",
]
