"""Master keeps agent-challenge at R=1 for attested units (flag ON).

Guardrail for the Phala-attested agent-challenge integration: an attested
agent-challenge submission is bridged by :class:`MasterOrchestrationDriver` as
one ``cpu`` work unit per selected task and assigned to a SINGLE executor
(R=1). Even with the worker plane ON (``compute.worker_plane_enabled`` -> the
validator ``AssignmentService`` configured with ``worker_plane_capabilities=
{"gpu"}`` and a live ``WorkerAssignmentEngine`` + ``WorkerReconciliationService``
wired into ``run_once``), attestation carries in the result payload and never
turns an agent-challenge unit into a gpu-style replicated/reconciled unit:

* VAL-VERIFY-020: each selected task is exactly ONE cpu unit assigned to one
  validator across repeated passes -- no ``worker_assignments`` replica, no
  R=2. A sibling prism gpu unit in the SAME pass IS replicated to R=2, proving
  the worker plane is genuinely active (the R=1 result is not vacuous).
* VAL-VERIFY-021: attested agent-challenge units never enter the worker-plane
  reconciliation path -- no ``disputed`` unit, no validator AUDIT unit, no
  ``worker_faults`` -- and the pre-existing retry-exhaustion fold still
  finalizes a stuck unit exactly once (single ``failed``, single fold).

Runs on in-memory SQLite (fast); the worker-plane integration parity across
Postgres is already covered by ``test_reclaim_guard_symmetry_postgres``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerFault,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment
from base.master.assignment import (
    CAPABILITY_GPU,
    EXECUTOR_KIND_PAYLOAD_KEY,
    EXECUTOR_KIND_VALIDATOR,
    AssignmentService,
)
from base.master.orchestration import (
    WORK_UNIT_MAX_ATTEMPTS_REASON,
    ChallengePendingWork,
    MasterOrchestrationDriver,
)
from base.master.validator_coordination import ValidatorCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import (
    AUDIT_WORK_UNIT_SUFFIX,
    WorkerReconciliationService,
)
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
)

NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)
TTL = 120

_ACTIVE = (WorkAssignmentStatus.ASSIGNED, WorkAssignmentStatus.RUNNING)


@dataclass
class _FakeWorkSource:
    works: list[ChallengePendingWork] = field(default_factory=list)

    async def fetch_pending_work(self) -> list[ChallengePendingWork]:
        return list(self.works)


@dataclass
class _FakeFoldTrigger:
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)

    async def fold(
        self, *, challenge_slug: str, job_id: str, task_id: str, reason: str
    ) -> None:
        self.calls.append((challenge_slug, job_id, task_id, reason))


class _FakeForwarder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Any,
    ) -> None:
        self.calls.append(work_unit_id)


async def _setup() -> tuple[Any, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, create_session_factory(engine)


def _build_flag_on_driver(
    factory: Any,
    works: list[ChallengePendingWork],
    *,
    default_max_attempts: int = 3,
) -> tuple[MasterOrchestrationDriver, _FakeForwarder, _FakeFoldTrigger]:
    """Wire a full flag-ON driver (worker engine + reconciler both present)."""

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    worker_service = WorkerCoordinationService(
        factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(factory),
        heartbeat_ttl_seconds=TTL,
        now_fn=lambda: NOW,
    )
    worker_assignment_service = WorkerAssignmentService(
        factory, worker_service=worker_service, now_fn=lambda: NOW
    )
    worker_engine = WorkerAssignmentEngine(
        factory,
        assignment_service=worker_assignment_service,
        worker_service=worker_service,
        replication_factor=2,
        now_fn=lambda: NOW,
    )
    forwarder = _FakeForwarder()
    reconciler = WorkerReconciliationService(
        factory, result_forwarder=forwarder, now_fn=lambda: NOW
    )
    # Flag-ON semantics: gpu is owned by the worker plane (skipped by the
    # validator assign/reclaim); cpu (agent-challenge) is untouched by it.
    assignment_service = AssignmentService(
        factory,
        now_fn=lambda: NOW,
        default_max_attempts=default_max_attempts,
        worker_plane_capabilities=frozenset({CAPABILITY_GPU}),
    )
    validator_service = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
    fold = _FakeFoldTrigger()
    driver = MasterOrchestrationDriver(
        assignment_service=assignment_service,
        validator_service=validator_service,
        work_source=_FakeWorkSource(works=works),
        fold_trigger=fold,
        worker_assignment_engine=worker_engine,
        worker_reconciler=reconciler,
        seed=1,
    )
    return driver, forwarder, fold


async def _add_worker(factory: Any, *, worker_pubkey: str, miner_hotkey: str) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkerRegistration(
                worker_id=f"wid-{worker_pubkey}",
                worker_pubkey=worker_pubkey,
                miner_hotkey=miner_hotkey,
                binding_signature="sig",
                binding_nonce=f"nonce-{worker_pubkey}",
                provider="local",
                provider_instance_ref="local-1",
                capabilities=["gpu"],
                status=WorkerStatus.ACTIVE,
                last_heartbeat_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_validator(factory: Any, hotkey: str, capabilities: list[str]) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=list(capabilities),
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _set_validator_status(
    factory: Any, hotkey: str, status: ValidatorStatus
) -> None:
    async with session_scope(factory) as session:
        validator = (
            await session.execute(select(Validator).where(Validator.hotkey == hotkey))
        ).scalar_one()
        validator.status = status


async def _units(factory: Any) -> dict[str, WorkAssignment]:
    async with factory() as session:
        rows = (await session.execute(select(WorkAssignment))).scalars().all()
        return {r.work_unit_id: r for r in rows}


async def _replica_count(factory: Any, work_unit_id: str) -> int:
    async with factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(WorkerAssignment)
                .where(
                    WorkerAssignment.work_unit_id == work_unit_id,
                    WorkerAssignment.status.in_(_ACTIVE),
                )
            )
        ).scalar_one()


async def _worker_fault_count(factory: Any) -> int:
    async with factory() as session:
        return (
            await session.execute(select(func.count()).select_from(WorkerFault))
        ).scalar_one()


def _agent_work(
    task_ids: tuple[str, ...] = ("a", "b", "c"),
    *,
    job_id: str | None = "job-1",
) -> ChallengePendingWork:
    # Attestation rides in the result payload; the unit itself is a plain cpu
    # unit as far as the master is concerned.
    return ChallengePendingWork(
        challenge_slug="agent-challenge",
        submission_id="sub",
        submission_ref="miner-C",
        task_ids=task_ids,
        job_id=job_id,
        payload={"proof": {"tier": "phala-tdx"}},
    )


def _prism_work() -> ChallengePendingWork:
    return ChallengePendingWork(
        challenge_slug="prism",
        submission_id="psub",
        submission_ref="miner-P",
    )


# --------------------------------------------------------------------------- #
# VAL-VERIFY-020: R=1 preserved for attested agent-challenge units (flag ON)
# --------------------------------------------------------------------------- #
async def test_flag_on_keeps_agent_challenge_at_r1_while_gpu_replicates() -> None:
    engine, factory = await _setup()
    try:
        driver, forwarder, _fold = _build_flag_on_driver(
            factory, works=[_agent_work(), _prism_work()]
        )
        # Two distinct-owner gpu workers (neither owned by the prism submitter)
        # so the gpu primary genuinely replicates to R=2.
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey="miner-A")
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey="miner-B")
        # One cpu validator to serve the agent-challenge cpu units.
        await _add_validator(factory, "v1", ["cpu"])

        await driver.run_once()

        units = await _units(factory)
        cpu_ids = ["sub:a", "sub:b", "sub:c"]
        # Each selected task is exactly ONE cpu unit assigned to a single
        # validator (R=1): one WorkAssignment row, one hotkey, one attempt.
        for uid in cpu_ids:
            unit = units[uid]
            assert unit.required_capability == "cpu"
            assert unit.status == WorkAssignmentStatus.ASSIGNED
            assert unit.assigned_validator_hotkey == "v1"
            assert unit.attempt_count == 1
            # No worker-plane replica materialized for a cpu unit.
            assert await _replica_count(factory, uid) == 0

        # Sibling gpu unit IS replicated to R=2 -> the worker plane is genuinely
        # active this pass, so the cpu R=1 result above is not vacuous.
        assert units["psub"].assigned_validator_hotkey is None  # worker-owned
        assert await _replica_count(factory, "psub") == 2

        # Repeated passes never add a second replica or a second assignment to a
        # cpu unit (still R=1, attempt_count stable, no new units created).
        for _ in range(3):
            result = await driver.run_once()
            assert result.folded == []
            units = await _units(factory)
            for uid in cpu_ids:
                assert units[uid].status == WorkAssignmentStatus.ASSIGNED
                assert units[uid].assigned_validator_hotkey == "v1"
                assert units[uid].attempt_count == 1
                assert await _replica_count(factory, uid) == 0

        # No audit/replica unit id was ever created for a cpu submission.
        assert all(not uid.endswith(AUDIT_WORK_UNIT_SUFFIX) for uid in units)
        assert set(units) == {"sub:a", "sub:b", "sub:c", "psub"}
        assert forwarder.calls == []
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# VAL-VERIFY-021: no reconciliation / audit / dispute for agent-challenge units
# --------------------------------------------------------------------------- #
async def test_flag_on_no_reconciliation_or_audit_for_agent_challenge() -> None:
    engine, factory = await _setup()
    try:
        driver, forwarder, _fold = _build_flag_on_driver(factory, works=[_agent_work()])
        await _add_validator(factory, "v1", ["cpu"])

        result = await driver.run_once()

        # A reconciler ran this pass (flag ON) but produced NO artifacts for the
        # agent-challenge units: nothing accepted/disputed/audited/faulted.
        assert result.reconciliation is not None
        assert result.reconciliation.accepted == []
        assert result.reconciliation.disputed == []
        assert result.reconciliation.audit_units == {}
        assert result.reconciliation.faults == []

        units = await _units(factory)
        # No unit is disputed, no validator AUDIT unit exists, and no
        # agent-challenge unit carries a validator executor-kind marker.
        for unit in units.values():
            assert unit.status != WorkAssignmentStatus.DISPUTED
            assert not unit.work_unit_id.endswith(AUDIT_WORK_UNIT_SUFFIX)
            assert (unit.payload or {}).get(EXECUTOR_KIND_PAYLOAD_KEY) != (
                EXECUTOR_KIND_VALIDATOR
            )
        # No worker faults and no forwards for a cpu submission.
        assert await _worker_fault_count(factory) == 0
        assert forwarder.calls == []
    finally:
        await engine.dispose()


async def test_flag_on_fold_on_exhaustion_finalizes_once_no_dispute() -> None:
    engine, factory = await _setup()
    try:
        driver, _forwarder, fold = _build_flag_on_driver(
            factory,
            works=[_agent_work(task_ids=("t1",), job_id="job-x")],
            default_max_attempts=1,
        )
        await _add_validator(factory, "v1", ["cpu"])

        # Pass 1: bridge + assign (attempt 1 of 1).
        first = await driver.run_once()
        assert first.folded == []
        assert fold.calls == []

        # The only cpu validator crashes; the unit exhausts its single attempt.
        await _set_validator_status(factory, "v1", ValidatorStatus.OFFLINE)

        # Pass 2: retry-exhausted -> failed -> folded EXACTLY once, via the
        # pre-existing fold path (never a dispute/audit).
        second = await driver.run_once()
        assert second.reassignment.failed == ["sub:t1"]
        assert second.folded == ["sub:t1"]
        assert fold.calls == [
            ("agent-challenge", "job-x", "t1", WORK_UNIT_MAX_ATTEMPTS_REASON)
        ]
        assert second.reconciliation is not None
        assert second.reconciliation.disputed == []
        assert second.reconciliation.audit_units == {}

        units = await _units(factory)
        assert units["sub:t1"].status == WorkAssignmentStatus.FAILED
        assert set(units) == {"sub:t1"}  # no audit/replica sibling unit created
        assert await _replica_count(factory, "sub:t1") == 0
        assert await _worker_fault_count(factory) == 0

        # Pass 3: an already-folded terminal unit is not re-folded or audited.
        third = await driver.run_once()
        assert third.folded == []
        assert len(fold.calls) == 1
        assert third.reconciliation is not None
        assert third.reconciliation.disputed == []
    finally:
        await engine.dispose()
