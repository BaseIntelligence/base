"""Worker-plane reconciliation, disputes, and audit-fault attribution.

Covers the master-side reconciliation the reconciliation-and-disputes feature
adds on top of the worker replica plane (``WorkerAssignmentEngine`` +
``WorkerAssignmentService.post_result``):

* VAL-MASTER-008: matching manifest hashes => exactly one result forwarded to the
  challenge and the unit reaches ``completed``.
* VAL-MASTER-009: divergent manifest hashes => the unit is ``disputed``, NOTHING
  is forwarded (before or after audit), and a validator-executor audit unit is
  created (assignable to validators, never to workers).
* VAL-MASTER-010: the validator audit outcome writes ``worker_faults`` for the
  divergent worker(s) only, visible in the fleet view, without mutating status.
* VAL-MASTER-017: a single surviving proof (the other replica exhausted) resolves
  deterministically (accepted, forwarded once, degradation warning), and a mere
  ``success=false`` replica never triggers a dispute.
* VAL-MASTER-019: late/foreign result posts are rejected and reconciliation uses
  only legitimately-owned replica results.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.challenge_sdk.roles import Role, activate_role
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
from base.master.assignment_coordination import AssignmentCoordinationService
from base.master.worker_assignment import (
    WorkerAssignmentOwnershipError,
    WorkerAssignmentService,
)
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import (
    AUDIT_OF_PAYLOAD_KEY,
    RECONCILE_DEGRADED_PAYLOAD_KEY,
    WorkerReconciliationService,
    audit_work_unit_id,
)
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
)
from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY, PROOF_PAYLOAD_KEY


@pytest.fixture(autouse=True)
def _activate_master_role():
    with activate_role(Role.MASTER):
        yield


NOW = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)
TTL = 120

MINER_H = "miner-H"
MINER_A = "miner-A"
MINER_B = "miner-B"
MINER_C = "miner-C"

HASH_A = "a" * 64
HASH_B = "b" * 64

GPU_VALIDATOR = "gpu-val"


class FakeClock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def now(self) -> datetime:
        return self.moment


class _FakeForwarder:
    """Records every challenge result forward so tests can count them."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Mapping[str, Any],
    ) -> None:
        self.calls.append(
            {
                "challenge_slug": challenge_slug,
                "work_unit_id": work_unit_id,
                "submission_ref": submission_ref,
                "result_payload": result_payload,
            }
        )


@dataclass
class Env:
    db_engine: Any
    factory: Any
    clock: FakeClock
    worker_service: WorkerCoordinationService
    worker_assignment_service: WorkerAssignmentService
    engine: WorkerAssignmentEngine
    forwarder: _FakeForwarder
    reconciler: WorkerReconciliationService
    validator_plane: AssignmentService
    validator_coordination: AssignmentCoordinationService


async def _build_env() -> Env:
    db_engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with db_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = create_session_factory(db_engine)
    clock = FakeClock(NOW)
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    worker_service = WorkerCoordinationService(
        factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(factory),
        heartbeat_ttl_seconds=TTL,
        now_fn=clock.now,
    )
    worker_assignment_service = WorkerAssignmentService(
        factory, worker_service=worker_service, now_fn=clock.now
    )
    engine = WorkerAssignmentEngine(
        factory,
        assignment_service=worker_assignment_service,
        worker_service=worker_service,
        replication_factor=2,
        now_fn=clock.now,
    )
    forwarder = _FakeForwarder()
    reconciler = WorkerReconciliationService(
        factory, result_forwarder=forwarder, now_fn=clock.now
    )
    validator_plane = AssignmentService(
        factory,
        now_fn=clock.now,
        worker_plane_capabilities=frozenset({CAPABILITY_GPU}),
    )
    validator_coordination = AssignmentCoordinationService(factory, now_fn=clock.now)
    return Env(
        db_engine=db_engine,
        factory=factory,
        clock=clock,
        worker_service=worker_service,
        worker_assignment_service=worker_assignment_service,
        engine=engine,
        forwarder=forwarder,
        reconciler=reconciler,
        validator_plane=validator_plane,
        validator_coordination=validator_coordination,
    )


async def _add_worker(
    factory: Any,
    *,
    worker_pubkey: str,
    miner_hotkey: str,
    status: WorkerStatus = WorkerStatus.ACTIVE,
    last_heartbeat_at: datetime | None = NOW,
) -> str:
    worker_id = f"wid-{worker_pubkey}"
    async with session_scope(factory) as session:
        session.add(
            WorkerRegistration(
                worker_id=worker_id,
                worker_pubkey=worker_pubkey,
                miner_hotkey=miner_hotkey,
                binding_signature="sig",
                binding_nonce=f"nonce-{worker_pubkey}",
                provider="local",
                provider_instance_ref="local-1",
                capabilities=["gpu"],
                status=status,
                last_heartbeat_at=last_heartbeat_at,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    return worker_id


async def _add_gpu_unit(
    factory: Any,
    *,
    work_unit_id: str,
    submitter: str,
    max_attempts: int = 3,
) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkAssignment(
                challenge_slug="prism",
                work_unit_id=work_unit_id,
                submission_ref=submitter,
                payload={"run_spec": {"image": "img"}},
                required_capability="gpu",
                status=WorkAssignmentStatus.PENDING,
                attempt_count=0,
                max_attempts=max_attempts,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_gpu_validator(factory: Any, *, hotkey: str = GPU_VALIDATOR) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=["gpu"],
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


def _proof_payload(manifest: str) -> dict[str, Any]:
    return {
        PROOF_PAYLOAD_KEY: {
            "version": 1,
            "tier": 0,
            MANIFEST_SHA256_PAYLOAD_KEY: manifest,
        },
        MANIFEST_SHA256_PAYLOAD_KEY: manifest,
    }


async def _replicas(factory: Any, work_unit_id: str) -> list[WorkerAssignment]:
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(WorkerAssignment)
                    .where(WorkerAssignment.work_unit_id == work_unit_id)
                    .order_by(WorkerAssignment.miner_hotkey)
                )
            )
            .scalars()
            .all()
        )
        return list(rows)


async def _unit(factory: Any, work_unit_id: str) -> WorkAssignment:
    async with factory() as session:
        return (
            await session.execute(
                select(WorkAssignment).where(
                    WorkAssignment.work_unit_id == work_unit_id
                )
            )
        ).scalar_one()


async def _maybe_unit(factory: Any, work_unit_id: str) -> WorkAssignment | None:
    async with factory() as session:
        return (
            await session.execute(
                select(WorkAssignment).where(
                    WorkAssignment.work_unit_id == work_unit_id
                )
            )
        ).scalar_one_or_none()


async def _post_replica(
    env: Env,
    *,
    work_unit_id: str,
    miner_hotkey: str,
    manifest: str | None = None,
    success: bool = True,
) -> None:
    replica = next(
        r
        for r in await _replicas(env.factory, work_unit_id)
        if r.miner_hotkey == miner_hotkey
    )
    payload = _proof_payload(manifest) if manifest is not None else {"error": "boom"}
    await env.worker_assignment_service.post_result(
        assignment_id=str(replica.id),
        worker_pubkey=replica.worker_pubkey,
        success=success,
        payload=payload,
    )


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    e = await _build_env()
    try:
        yield e
    finally:
        await e.db_engine.dispose()


# VAL-MASTER-008
async def test_matching_hashes_forward_exactly_one(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)

    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_A)

    result = await env.reconciler.reconcile_once()
    # Reconciling again must not forward a second time.
    await env.reconciler.reconcile_once()

    assert result.accepted == ["U"]
    assert result.disputed == []
    assert len(env.forwarder.calls) == 1
    assert env.forwarder.calls[0]["work_unit_id"] == "U"
    unit = await _unit(env.factory, "U")
    assert unit.status == WorkAssignmentStatus.COMPLETED
    assert await _maybe_unit(env.factory, audit_work_unit_id("U")) is None


# VAL-MASTER-009
async def test_divergent_hashes_dispute_and_audit_unit(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)

    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_B)

    result = await env.reconciler.reconcile_once()

    assert result.disputed == ["U"]
    assert result.accepted == []
    assert env.forwarder.calls == []
    unit = await _unit(env.factory, "U")
    assert unit.status == WorkAssignmentStatus.DISPUTED

    audit = await _unit(env.factory, audit_work_unit_id("U"))
    assert audit.required_capability == CAPABILITY_GPU
    assert audit.payload[EXECUTOR_KIND_PAYLOAD_KEY] == EXECUTOR_KIND_VALIDATOR
    assert audit.payload[AUDIT_OF_PAYLOAD_KEY] == "U"
    assert audit.status == WorkAssignmentStatus.PENDING


# VAL-MASTER-009: the audit unit routes to a validator and never to a worker.
async def test_audit_unit_routes_to_validator_not_worker(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_B)
    await env.reconciler.reconcile_once()

    audit_id = audit_work_unit_id("U")

    # The worker engine never picks up the validator-executor audit unit.
    await env.engine.assign_pending(seed=1)
    assert await _replicas(env.factory, audit_id) == []

    # The validator plane assigns it to an online gpu validator.
    await _add_gpu_validator(env.factory)
    await env.validator_plane.assign_pending(seed=1)
    audit = await _unit(env.factory, audit_id)
    assert audit.assigned_validator_hotkey == GPU_VALIDATOR
    assert audit.status == WorkAssignmentStatus.ASSIGNED


# VAL-MASTER-010
async def test_audit_outcome_faults_divergent_worker_only(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)
    # A agrees with the validator (HASH_A); B diverges (HASH_B).
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_B)
    await env.reconciler.reconcile_once()

    # A validator replays the audit unit; its authoritative manifest is HASH_A.
    await _add_gpu_validator(env.factory)
    await env.validator_plane.assign_pending(seed=1)
    audit = await _unit(env.factory, audit_work_unit_id("U"))
    # Pull transitions ASSIGNED -> RUNNING required by coordination post_result.
    await env.validator_coordination.pull(hotkey=GPU_VALIDATOR)
    await env.validator_coordination.post_result(
        assignment_id=str(audit.id),
        hotkey=GPU_VALIDATOR,
        success=True,
        payload=_proof_payload(HASH_A),
    )

    result = await env.reconciler.reconcile_once()

    # Exactly the divergent worker (B) is faulted; A is not.
    faulted_workers = {pair.split(":")[-1] for pair in result.faults}
    assert faulted_workers == {"wid-wp-b"}

    async with env.factory() as session:
        faults = (await session.execute(select(WorkerFault))).scalars().all()
    assert len(faults) == 1
    assert faults[0].worker_id == "wid-wp-b"
    assert faults[0].work_unit_id == "U"

    faults_by_worker = await env.worker_service.faults_by_worker()
    assert "wid-wp-b" in faults_by_worker
    assert "wid-wp-a" not in faults_by_worker

    # The divergent worker's status is unchanged by fault recording.
    async with env.factory() as session:
        worker_b = (
            await session.execute(
                select(WorkerRegistration).where(
                    WorkerRegistration.worker_id == "wid-wp-b"
                )
            )
        ).scalar_one()
    assert WorkerStatus(worker_b.status) == WorkerStatus.ACTIVE

    # The disputed unit is never forwarded, before or after audit resolution.
    assert env.forwarder.calls == []
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.DISPUTED


# VAL-MASTER-010: faults are recorded once, not re-written each pass.
async def test_audit_faults_are_idempotent(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_B)
    await env.reconciler.reconcile_once()
    await _add_gpu_validator(env.factory)
    await env.validator_plane.assign_pending(seed=1)
    audit = await _unit(env.factory, audit_work_unit_id("U"))
    # Pull transitions ASSIGNED -> RUNNING required by coordination post_result.
    await env.validator_coordination.pull(hotkey=GPU_VALIDATOR)
    await env.validator_coordination.post_result(
        assignment_id=str(audit.id),
        hotkey=GPU_VALIDATOR,
        success=True,
        payload=_proof_payload(HASH_A),
    )

    await env.reconciler.reconcile_once()
    await env.reconciler.reconcile_once()

    async with env.factory() as session:
        faults = (await session.execute(select(WorkerFault))).scalars().all()
    assert len(faults) == 1


# VAL-MASTER-017
async def test_single_surviving_proof_accepts_with_warning(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(
        env.factory, work_unit_id="U", submitter=MINER_H, max_attempts=1
    )
    await env.engine.assign_pending(seed=1)

    # A posts a valid proof; B's worker goes stale and its retries are exhausted.
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    async with session_scope(env.factory) as session:
        worker_b = (
            await session.execute(
                select(WorkerRegistration).where(
                    WorkerRegistration.miner_hotkey == MINER_B
                )
            )
        ).scalar_one()
        worker_b.status = WorkerStatus.STALE
        worker_b.last_heartbeat_at = NOW - timedelta(seconds=TTL + 60)
    # B's replica lapses with no other eligible distinct owner -> FAILED.
    await env.engine.reassign_stale_replicas(seed=1)

    with caplog.at_level(logging.WARNING, logger="base.master.worker_reconciliation"):
        result = await env.reconciler.reconcile_once()

    assert result.accepted == ["U"]
    assert result.disputed == []
    assert result.single_replica == ["U"]
    assert len(env.forwarder.calls) == 1
    unit = await _unit(env.factory, "U")
    assert unit.status == WorkAssignmentStatus.COMPLETED
    assert unit.payload[RECONCILE_DEGRADED_PAYLOAD_KEY] is True
    assert await _maybe_unit(env.factory, audit_work_unit_id("U")) is None
    assert any("U" in record.getMessage() for record in caplog.records)


# VAL-MASTER-017: a bare success=false replica never triggers a dispute.
async def test_failed_replica_alone_never_disputes(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)

    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    # B reports an execution failure with no proof.
    await _post_replica(
        env, work_unit_id="U", miner_hotkey=MINER_B, manifest=None, success=False
    )

    result = await env.reconciler.reconcile_once()

    assert result.disputed == []
    assert result.accepted == ["U"]
    assert len(env.forwarder.calls) == 1
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.COMPLETED
    assert await _maybe_unit(env.factory, audit_work_unit_id("U")) is None


# VAL-MASTER-017: reconciliation waits while a replica may still report.
async def test_reconciliation_waits_for_in_flight_replica(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)

    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    # B has not reported and is still in flight.

    result = await env.reconciler.reconcile_once()

    assert result.accepted == []
    assert result.disputed == []
    assert env.forwarder.calls == []
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.ASSIGNED


# VAL-MASTER-019
async def test_late_and_foreign_posts_rejected(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_worker(env.factory, worker_pubkey="wp-c", miner_hotkey=MINER_C)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)

    # Capture W1 = A's original replica id, then lapse it and reassign to C.
    a_replica = next(
        r for r in await _replicas(env.factory, "U") if r.miner_hotkey == MINER_A
    )
    a_replica_id = str(a_replica.id)
    async with session_scope(env.factory) as session:
        row = (
            await session.execute(
                select(WorkerAssignment).where(WorkerAssignment.id == a_replica.id)
            )
        ).scalar_one()
        row.deadline_at = NOW - timedelta(seconds=10)
        # Make A's worker stale so the replica is reclaimed to a fresh owner.
        worker_a = (
            await session.execute(
                select(WorkerRegistration).where(
                    WorkerRegistration.miner_hotkey == MINER_A
                )
            )
        ).scalar_one()
        worker_a.status = WorkerStatus.STALE
        worker_a.last_heartbeat_at = NOW - timedelta(seconds=TTL + 60)
    await env.engine.reassign_stale_replicas(seed=1)

    # The replica is now owned by C (W2); A's old post is foreign now.
    reassigned = next(
        r for r in await _replicas(env.factory, "U") if str(r.id) == a_replica_id
    )
    assert reassigned.miner_hotkey == MINER_C

    # W1 (A) posting for its OLD replica is rejected on ownership.
    with pytest.raises(WorkerAssignmentOwnershipError):
        await env.worker_assignment_service.post_result(
            assignment_id=a_replica_id,
            worker_pubkey="wp-a",
            success=True,
            payload=_proof_payload(HASH_B),
        )
    # A foreign worker (B) posting for C's replica is rejected too.
    with pytest.raises(WorkerAssignmentOwnershipError):
        await env.worker_assignment_service.post_result(
            assignment_id=a_replica_id,
            worker_pubkey="wp-b",
            success=True,
            payload=_proof_payload(HASH_B),
        )

    # The replica state is intact: still owned by C, still running, no manifest.
    reassigned = next(
        r for r in await _replicas(env.factory, "U") if str(r.id) == a_replica_id
    )
    assert reassigned.miner_hotkey == MINER_C
    assert reassigned.manifest_sha256 is None
    assert WorkAssignmentStatus(reassigned.status) == WorkAssignmentStatus.ASSIGNED

    # Both legitimately-owned replicas now post equal hashes -> clean accept.
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_C, manifest=HASH_A)
    result = await env.reconciler.reconcile_once()

    assert result.accepted == ["U"]
    assert len(env.forwarder.calls) == 1
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.COMPLETED


# VAL-MASTER-019: a late post to an already-reconciled replica is a no-op.
async def test_late_post_to_reconciled_unit_is_noop(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_A, manifest=HASH_A)
    await _post_replica(env, work_unit_id="U", miner_hotkey=MINER_B, manifest=HASH_A)
    await env.reconciler.reconcile_once()

    a_replica = next(
        r for r in await _replicas(env.factory, "U") if r.miner_hotkey == MINER_A
    )
    outcome = await env.worker_assignment_service.post_result(
        assignment_id=str(a_replica.id),
        worker_pubkey="wp-a",
        success=True,
        payload=_proof_payload(HASH_B),
    )
    assert outcome.idempotent is True
    # The stored manifest is unchanged and no extra forward happened.
    a_replica = next(
        r for r in await _replicas(env.factory, "U") if r.miner_hotkey == MINER_A
    )
    assert a_replica.manifest_sha256 == HASH_A
    assert len(env.forwarder.calls) == 1
