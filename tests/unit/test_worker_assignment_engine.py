"""Worker-plane replica-creation ENGINE behavior (architecture.md sec 3.3).

Covers the master-side gpu routing the engine adds on top of the
worker-agent-runtime seams (``WorkerAssignmentService.create_worker_assignment``,
``WorkerCoordinationService.effective_status``):

* VAL-MASTER-003: gpu units replicate only to ACTIVE workers.
* VAL-MASTER-004 / VAL-MASTER-005: self-evaluation exclusion, holding even when
  the submitter's own worker is the sole capacity (the unit waits).
* VAL-MASTER-006: R=2 across DISTINCT owner hotkeys; a same-owner pair never
  satisfies R=2.
* VAL-MASTER-007: graceful degradation to R=1 with a recorded warning.
* VAL-MASTER-011: per-replica deadline/reassignment; a unit with ALL replicas
  attempts-exhausted ends ``failed``.
* VAL-MASTER-012: heartbeat TTL drives active->stale eligibility (and recovery).
* VAL-MASTER-013: flag OFF routes gpu units to validators; flag ON routes them
  away from validators to worker replicas.
* VAL-MASTER-014: flag OFF leaves the worker assignment surface inert (404).
* VAL-MASTER-020: per-worker gpu concurrency is 1.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment
from base.master.app_proxy import create_proxy_app
from base.master.assignment import AssignmentService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import (
    DEGRADED_REPLICATION_PAYLOAD_KEY,
    WorkerAssignmentEngine,
    run_worker_assignment_pass,
)
from base.master.worker_coordination import WorkerCoordinationService
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
)

NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)
TTL = 120

# Submitter H and the distinct owner hotkeys A/B/C (none is the submitter).
MINER_H = "miner-H"
MINER_A = "miner-A"
MINER_B = "miner-B"
MINER_C = "miner-C"


class FakeClock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def now(self) -> datetime:
        return self.moment


class _FakeNonceStore:
    async def reserve(self, **_: Any) -> None:
        return None


class _FakeCache:
    def get(self) -> dict[str, int]:
        return {}


@dataclass
class Env:
    db_engine: Any
    factory: Any
    clock: FakeClock
    worker_service: WorkerCoordinationService
    worker_assignment_service: WorkerAssignmentService
    engine: WorkerAssignmentEngine


async def _build_env(*, replication_factor: int = 2) -> Env:
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
        replication_factor=replication_factor,
        now_fn=clock.now,
    )
    return Env(
        db_engine=db_engine,
        factory=factory,
        clock=clock,
        worker_service=worker_service,
        worker_assignment_service=worker_assignment_service,
        engine=engine,
    )


async def _add_worker(
    factory: Any,
    *,
    worker_pubkey: str,
    miner_hotkey: str,
    status: WorkerStatus = WorkerStatus.ACTIVE,
    last_heartbeat_at: datetime | None = NOW,
    created_at: datetime = NOW,
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
                created_at=created_at,
                updated_at=created_at,
            )
        )
    return worker_id


async def _add_gpu_unit(
    factory: Any,
    *,
    work_unit_id: str,
    submitter: str,
    max_attempts: int = 3,
    created_at: datetime = NOW,
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
                created_at=created_at,
                updated_at=created_at,
            )
        )


async def _replicas(factory: Any, work_unit_id: str) -> list[WorkerAssignment]:
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(WorkerAssignment)
                    .where(WorkerAssignment.work_unit_id == work_unit_id)
                    .order_by(WorkerAssignment.worker_id)
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


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    e = await _build_env()
    try:
        yield e
    finally:
        await e.db_engine.dispose()


# VAL-MASTER-003
async def test_gpu_units_replicate_only_to_active_workers(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_worker(
        env.factory,
        worker_pubkey="wp-pending",
        miner_hotkey=MINER_C,
        status=WorkerStatus.PENDING,
        last_heartbeat_at=None,
    )
    await _add_worker(
        env.factory,
        worker_pubkey="wp-stale",
        miner_hotkey="miner-D",
        status=WorkerStatus.STALE,
        last_heartbeat_at=NOW - timedelta(seconds=TTL + 5),
    )
    await _add_worker(
        env.factory,
        worker_pubkey="wp-retired",
        miner_hotkey="miner-E",
        status=WorkerStatus.RETIRED,
    )
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    await env.engine.assign_pending(seed=1)

    replicas = await _replicas(env.factory, "U")
    assert len(replicas) == 2
    assert {r.worker_pubkey for r in replicas} == {"wp-a", "wp-b"}
    inactive = {"wp-pending", "wp-stale", "wp-retired"}
    assert all(r.worker_pubkey not in inactive for r in replicas)


# VAL-MASTER-004
async def test_self_evaluation_exclusion(env: Env) -> None:
    # H owns an active, capable worker, but it must never serve H's own unit.
    await _add_worker(env.factory, worker_pubkey="wp-h", miner_hotkey=MINER_H)
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    # Repeated passes never place the unit on H's worker.
    await env.engine.assign_pending(seed=1)
    await env.engine.assign_pending(seed=1)

    replicas = await _replicas(env.factory, "U")
    assert len(replicas) == 2
    owners = {r.miner_hotkey for r in replicas}
    assert MINER_H not in owners
    assert owners == {MINER_A, MINER_B}


# VAL-MASTER-005
async def test_self_evaluation_holds_when_only_own_worker_available(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-h", miner_hotkey=MINER_H)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    await env.engine.assign_pending(seed=1)

    # The unit is never assigned to H's worker; it simply waits.
    assert await _replicas(env.factory, "U") == []
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.PENDING

    # A distinct-owner worker activates -> the unit is assigned promptly.
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await env.engine.assign_pending(seed=1)

    replicas = await _replicas(env.factory, "U")
    assert [r.miner_hotkey for r in replicas] == [MINER_A]
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.ASSIGNED


# VAL-MASTER-006
async def test_replication_two_across_distinct_owners(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    await env.engine.assign_pending(seed=1)

    replicas = await _replicas(env.factory, "U")
    assert len(replicas) == 2
    assert len({r.miner_hotkey for r in replicas}) == 2
    unit = await _unit(env.factory, "U")
    assert DEGRADED_REPLICATION_PAYLOAD_KEY not in unit.payload


# VAL-MASTER-006 (negative): two workers of the SAME owner never satisfy R=2.
async def test_same_owner_pair_never_satisfies_r2(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a1", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-a2", miner_hotkey=MINER_A)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    await env.engine.assign_pending(seed=1)

    replicas = await _replicas(env.factory, "U")
    assert len(replicas) == 1
    assert replicas[0].miner_hotkey == MINER_A


# VAL-MASTER-007
async def test_degradation_to_r1_records_warning(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    # Only one distinct eligible owner: H's worker is excluded by self-eval.
    await _add_worker(env.factory, worker_pubkey="wp-h", miner_hotkey=MINER_H)
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    with caplog.at_level(
        logging.WARNING, logger="base.master.worker_assignment_engine"
    ):
        result = await env.engine.assign_pending(seed=1)

    replicas = await _replicas(env.factory, "U")
    assert [r.miner_hotkey for r in replicas] == [MINER_A]
    assert result.degraded == ["U"]
    # The warning is durably recorded on the unit AND emitted as a log record.
    unit = await _unit(env.factory, "U")
    assert unit.payload[DEGRADED_REPLICATION_PAYLOAD_KEY] == 1
    assert any(
        "degraded" in record.message and "U" in record.getMessage()
        for record in caplog.records
    )


# VAL-MASTER-011: a lapsed replica reassigns to another eligible worker.
async def test_lapsed_replica_reassigned_to_other_worker(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)
    await env.engine.assign_pending(seed=1)

    # A's replica lease lapses; a fresh distinct-owner worker C is available.
    async with session_scope(env.factory) as session:
        row = (
            await session.execute(
                select(WorkerAssignment).where(
                    WorkerAssignment.work_unit_id == "U",
                    WorkerAssignment.miner_hotkey == MINER_A,
                )
            )
        ).scalar_one()
        row.deadline_at = NOW - timedelta(seconds=10)
    await _add_worker(env.factory, worker_pubkey="wp-c", miner_hotkey=MINER_C)

    result = await env.engine.reassign_stale_replicas(seed=1)

    assert result.reassigned == ["U"]
    replicas = await _replicas(env.factory, "U")
    owners = {r.miner_hotkey for r in replicas}
    # B's replica untouched; A's replica moved to the new distinct owner C.
    assert owners == {MINER_B, MINER_C}
    moved = next(r for r in replicas if r.miner_hotkey == MINER_C)
    assert moved.attempt_count == 2
    assert moved.deadline_at is None
    assert WorkAssignmentStatus(moved.status) == WorkAssignmentStatus.ASSIGNED
    untouched = next(r for r in replicas if r.miner_hotkey == MINER_B)
    assert untouched.attempt_count == 1


# VAL-MASTER-011: all replicas exhausted -> the unit itself fails.
async def test_all_replicas_exhausted_fails_unit(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(
        env.factory, work_unit_id="U", submitter=MINER_H, max_attempts=1
    )
    await env.engine.assign_pending(seed=1)

    # Both replicas are at their attempt cap and lapse with no other capacity.
    async with session_scope(env.factory) as session:
        rows = (
            (
                await session.execute(
                    select(WorkerAssignment).where(WorkerAssignment.work_unit_id == "U")
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            row.deadline_at = NOW - timedelta(seconds=10)

    result = await env.engine.reassign_stale_replicas(seed=1)

    assert result.failed_units == ["U"]
    assert len(result.failed_replicas) == 2
    replicas = await _replicas(env.factory, "U")
    assert all(
        WorkAssignmentStatus(r.status) == WorkAssignmentStatus.FAILED for r in replicas
    )
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.FAILED


# VAL-MASTER-012
async def test_heartbeat_ttl_drives_eligibility(env: Env) -> None:
    worker_pubkey = "wp-a"
    await _add_worker(env.factory, worker_pubkey=worker_pubkey, miner_hotkey=MINER_A)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    # Clock advances past the TTL -> the worker is stale and not assignable.
    env.clock.moment = NOW + timedelta(seconds=TTL + 10)
    async with env.factory() as session:
        worker = (
            await session.execute(
                select(WorkerRegistration).where(
                    WorkerRegistration.worker_pubkey == worker_pubkey
                )
            )
        ).scalar_one()
        assert (
            env.worker_service.effective_status(worker, env.clock.now())
            == WorkerStatus.STALE
        )

    await env.engine.assign_pending(seed=1)
    assert await _replicas(env.factory, "U") == []
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.PENDING

    # The worker heartbeats again -> active -> regains eligibility.
    async with session_scope(env.factory) as session:
        worker = (
            await session.execute(
                select(WorkerRegistration).where(
                    WorkerRegistration.worker_pubkey == worker_pubkey
                )
            )
        ).scalar_one()
        worker.last_heartbeat_at = env.clock.now()
        assert (
            env.worker_service.effective_status(worker, env.clock.now())
            == WorkerStatus.ACTIVE
        )

    await env.engine.assign_pending(seed=1)
    replicas = await _replicas(env.factory, "U")
    assert [r.worker_pubkey for r in replicas] == [worker_pubkey]


# VAL-MASTER-020
async def test_per_worker_gpu_concurrency_is_one(env: Env) -> None:
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_gpu_unit(env.factory, work_unit_id="unit-a", submitter=MINER_H)
    await _add_gpu_unit(env.factory, work_unit_id="unit-b", submitter=MINER_H)

    await env.engine.assign_pending(seed=1)

    # The single worker holds one in-flight replica; the second unit waits.
    assert len(await _replicas(env.factory, "unit-a")) == 1
    assert await _replicas(env.factory, "unit-b") == []
    assert (await _unit(env.factory, "unit-b")).status == WorkAssignmentStatus.PENDING

    # The first replica finishes -> the worker is eligible for the queued unit.
    async with session_scope(env.factory) as session:
        row = (
            await session.execute(
                select(WorkerAssignment).where(
                    WorkerAssignment.work_unit_id == "unit-a"
                )
            )
        ).scalar_one()
        row.status = WorkAssignmentStatus.COMPLETED

    await env.engine.assign_pending(seed=1)
    replicas = await _replicas(env.factory, "unit-b")
    assert [r.worker_pubkey for r in replicas] == ["wp-a"]


# VAL-MASTER-013: flag OFF routes gpu units to validators (never to workers).
async def test_flag_off_routes_gpu_to_validator(env: Env) -> None:
    validator_plane = AssignmentService(env.factory, now_fn=env.clock.now)
    async with session_scope(env.factory) as session:
        session.add(
            Validator(
                hotkey="gpu-val",
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=["gpu"],
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )
    # An active worker exists but the flag is OFF, so it is ignored.
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    await validator_plane.assign_pending(seed=1)

    unit = await _unit(env.factory, "U")
    assert unit.status == WorkAssignmentStatus.ASSIGNED
    assert unit.assigned_validator_hotkey == "gpu-val"
    assert await _replicas(env.factory, "U") == []


# VAL-MASTER-013: flag ON routes gpu units AWAY from validators to workers.
async def test_flag_on_routes_gpu_away_from_validators(env: Env) -> None:
    validator_plane = AssignmentService(
        env.factory, now_fn=env.clock.now, worker_plane_capabilities=frozenset({"gpu"})
    )
    async with session_scope(env.factory) as session:
        session.add(
            Validator(
                hotkey="gpu-val",
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=["gpu"],
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )
    await _add_worker(env.factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
    await _add_worker(env.factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
    await _add_gpu_unit(env.factory, work_unit_id="U", submitter=MINER_H)

    # The validator plane skips the gpu unit; the engine materializes replicas.
    await validator_plane.assign_pending(seed=1)
    assert (await _unit(env.factory, "U")).assigned_validator_hotkey is None
    assert (await _unit(env.factory, "U")).status == WorkAssignmentStatus.PENDING

    await run_worker_assignment_pass(engine=env.engine, seed=1)
    replicas = await _replicas(env.factory, "U")
    assert len(replicas) == 2
    assert {r.miner_hotkey for r in replicas} == {MINER_A, MINER_B}


# VAL-MASTER-014: with the plane OFF the worker assignment surface is inert (404).
async def test_flag_off_worker_assignment_surface_inert() -> None:
    db_engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with db_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    app = create_proxy_app(
        registry=object(),
        nonce_store=_FakeNonceStore(),
        metagraph_cache=_FakeCache(),  # type: ignore[arg-type]
        worker_assignment_service=None,
        worker_assignment_verifier=None,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        pull = await client.post("/v1/workers/assignments/pull", json={})
        assert pull.status_code == 404
        result = await client.post("/v1/workers/assignments/abc/result", json={})
        assert result.status_code == 404
        health = await client.get("/health")
        assert health.status_code == 200
    finally:
        await client.aclose()
        await db_engine.dispose()
