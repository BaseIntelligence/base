"""Postgres parity for worker-plane reconciliation, disputes, and audit faults.

Mirrors the SQLite unit behavior of :class:`WorkerReconciliationService` against
the throwaway test Postgres, confirming the ``disputed`` terminal status, the
validator-executor audit unit, and the ``worker_faults`` fault row persist and
reconcile identically across backends (VAL-MASTER-008/009/010).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
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
from base.master.assignment import CAPABILITY_GPU, AssignmentService
from base.master.assignment_coordination import AssignmentCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import (
    AUDIT_OF_PAYLOAD_KEY,
    WorkerReconciliationService,
    audit_work_unit_id,
)
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
)
from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY, PROOF_PAYLOAD_KEY

pytestmark = pytest.mark.postgres

NOW = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)
TTL = 120
HASH_A = "a" * 64
HASH_B = "b" * 64
GPU_VALIDATOR = "gpu-val"


class _Forwarder:
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


def _proof_payload(manifest: str) -> dict[str, Any]:
    return {
        PROOF_PAYLOAD_KEY: {
            "version": 1,
            "tier": 0,
            MANIFEST_SHA256_PAYLOAD_KEY: manifest,
        },
        MANIFEST_SHA256_PAYLOAD_KEY: manifest,
    }


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


async def _add_gpu_unit(factory: Any, *, work_unit_id: str, submitter: str) -> None:
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
                max_attempts=3,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _replicas(factory: Any, work_unit_id: str) -> list[WorkerAssignment]:
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(WorkerAssignment).where(
                        WorkerAssignment.work_unit_id == work_unit_id
                    )
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


def _build(factory: Any) -> tuple[Any, ...]:
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    worker_service = WorkerCoordinationService(
        factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(factory),
        heartbeat_ttl_seconds=TTL,
        now_fn=lambda: NOW,
    )
    assignment_service = WorkerAssignmentService(
        factory, worker_service=worker_service, now_fn=lambda: NOW
    )
    engine = WorkerAssignmentEngine(
        factory,
        assignment_service=assignment_service,
        worker_service=worker_service,
        replication_factor=2,
        now_fn=lambda: NOW,
    )
    forwarder = _Forwarder()
    reconciler = WorkerReconciliationService(
        factory, result_forwarder=forwarder, now_fn=lambda: NOW
    )
    validator_plane = AssignmentService(
        factory,
        now_fn=lambda: NOW,
        worker_plane_capabilities=frozenset({CAPABILITY_GPU}),
    )
    coordination = AssignmentCoordinationService(factory, now_fn=lambda: NOW)
    return (
        worker_service,
        assignment_service,
        engine,
        forwarder,
        reconciler,
        validator_plane,
        coordination,
    )


async def _post(
    service: WorkerAssignmentService, factory: Any, unit: str, owner: str, h: str
) -> None:
    replica = next(r for r in await _replicas(factory, unit) if r.miner_hotkey == owner)
    await service.post_result(
        assignment_id=str(replica.id),
        worker_pubkey=replica.worker_pubkey,
        success=True,
        payload=_proof_payload(h),
    )


async def test_dispute_and_audit_faults_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine_db = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine_db)
    (
        worker_service,
        assignment_service,
        engine,
        forwarder,
        reconciler,
        validator_plane,
        coordination,
    ) = _build(factory)
    try:
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey="miner-A")
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey="miner-B")
        await _add_gpu_unit(factory, work_unit_id="U", submitter="miner-H")
        await engine.assign_pending(seed=1)

        await _post(assignment_service, factory, "U", "miner-A", HASH_A)
        await _post(assignment_service, factory, "U", "miner-B", HASH_B)

        result = await reconciler.reconcile_once()
        assert result.disputed == ["U"]
        assert forwarder.calls == []
        assert (await _unit(factory, "U")).status == WorkAssignmentStatus.DISPUTED

        audit = await _unit(factory, audit_work_unit_id("U"))
        assert audit.required_capability == CAPABILITY_GPU
        assert audit.payload[AUDIT_OF_PAYLOAD_KEY] == "U"

        # A validator replays; its authoritative manifest agrees with A.
        async with session_scope(factory) as session:
            session.add(
                Validator(
                    hotkey=GPU_VALIDATOR,
                    uid=None,
                    status=ValidatorStatus.ONLINE,
                    capabilities=["gpu"],
                    version="1.0.0",
                    registered_at=NOW,
                    last_heartbeat_at=NOW,
                )
            )
        await validator_plane.assign_pending(seed=1)
        audit = await _unit(factory, audit_work_unit_id("U"))
        assert audit.assigned_validator_hotkey == GPU_VALIDATOR
        await coordination.post_result(
            assignment_id=str(audit.id),
            hotkey=GPU_VALIDATOR,
            success=True,
            payload=_proof_payload(HASH_A),
        )

        result = await reconciler.reconcile_once()
        assert {p.split(":")[-1] for p in result.faults} == {"wid-wp-b"}

        async with factory() as session:
            faults = (await session.execute(select(WorkerFault))).scalars().all()
        assert len(faults) == 1
        assert faults[0].worker_id == "wid-wp-b"
        assert faults[0].work_unit_id == "U"

        # Fault recording leaves the worker's status unchanged.
        async with factory() as session:
            worker_b = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_id == "wid-wp-b"
                    )
                )
            ).scalar_one()
        assert WorkerStatus(worker_b.status) == WorkerStatus.ACTIVE
        # The disputed unit is never forwarded, even after audit resolution.
        assert forwarder.calls == []
        assert (await _unit(factory, "U")).status == WorkAssignmentStatus.DISPUTED
    finally:
        await engine_db.dispose()


async def test_matching_hashes_accept_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine_db = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine_db)
    (
        _worker_service,
        assignment_service,
        engine,
        forwarder,
        reconciler,
        _validator_plane,
        _coordination,
    ) = _build(factory)
    try:
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey="miner-A")
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey="miner-B")
        await _add_gpu_unit(factory, work_unit_id="U", submitter="miner-H")
        await engine.assign_pending(seed=1)

        await _post(assignment_service, factory, "U", "miner-A", HASH_A)
        await _post(assignment_service, factory, "U", "miner-B", HASH_A)

        result = await reconciler.reconcile_once()
        await reconciler.reconcile_once()

        assert result.accepted == ["U"]
        assert forwarder.calls == ["U"]
        assert (await _unit(factory, "U")).status == WorkAssignmentStatus.COMPLETED
    finally:
        await engine_db.dispose()
