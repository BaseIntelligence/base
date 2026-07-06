from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.worker_coordination import (
    WorkerCoordinationService,
    WorkerRebindError,
)
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
    WorkerReplayError,
    worker_binding_message,
)

pytestmark = pytest.mark.postgres

BASE_TS = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
TTL = 120
MINER_H1 = "pg-miner-H1"
MINER_H2 = "pg-miner-H2"


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


def _sign(hotkey: str, message: bytes) -> str:
    return hashlib.sha256(hotkey.encode() + b":" + message).hexdigest()


def _fake_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message)


def _service(session_factory: object, clock: _Clock) -> WorkerCoordinationService:
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER_H1, MINER_H2], validator_permits=[False, False], stakes=[0.0, 0.0]
    )
    return WorkerCoordinationService(
        session_factory,  # type: ignore[arg-type]
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(session_factory),  # type: ignore[arg-type]
        signature_verifier=_fake_verifier,
        heartbeat_ttl_seconds=TTL,
        now_fn=clock.now,
    )


async def _register(
    service: WorkerCoordinationService,
    *,
    worker_pubkey: str,
    miner_hotkey: str,
    nonce: str,
    provider: str = "local",
) -> WorkerRegistration:
    message = worker_binding_message(
        worker_pubkey=worker_pubkey, miner_hotkey=miner_hotkey, nonce=nonce
    )
    return await service.register(
        worker_pubkey=worker_pubkey,
        miner_hotkey=miner_hotkey,
        binding_signature=_sign(miner_hotkey, message),
        nonce=nonce,
        provider=provider,
        provider_instance_ref="ref-1",
        capabilities=["gpu"],
    )


# VAL-MASTER-001
async def test_register_pending_then_heartbeat_active_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = _service(session_factory, clock)
    try:
        worker = await _register(
            service, worker_pubkey="pg-wp-1", miner_hotkey=MINER_H1, nonce="n-1"
        )
        assert WorkerStatus(worker.status) == WorkerStatus.PENDING

        rows = await service.list_workers()
        assert [service.effective_status(r, clock.now()).value for r in rows] == [
            "pending"
        ]

        clock.value = BASE_TS + timedelta(seconds=10)
        updated, now = await service.heartbeat(
            worker_id=worker.worker_id, worker_pubkey="pg-wp-1"
        )
        assert now == clock.value
        assert service.effective_status(updated, now) == WorkerStatus.ACTIVE

        active = await service.active_workers(MINER_H1)
        assert [w.worker_pubkey for w in active] == ["pg-wp-1"]
    finally:
        await engine.dispose()


# VAL-MASTER-002 (c): replayed binding nonce rejected on postgres
async def test_replayed_binding_nonce_rejected_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    service = _service(session_factory, _Clock(BASE_TS))
    try:
        await _register(
            service, worker_pubkey="pg-wp-r", miner_hotkey=MINER_H1, nonce="dup"
        )
        with pytest.raises(WorkerReplayError):
            await _register(
                service, worker_pubkey="pg-wp-r2", miner_hotkey=MINER_H1, nonce="dup"
            )
        async with session_factory() as session:
            count = await session.scalar(
                select(func.count(WorkerRegistration.id)).where(
                    WorkerRegistration.worker_pubkey == "pg-wp-r2"
                )
            )
        assert count == 0
    finally:
        await engine.dispose()


# VAL-MASTER-016
async def test_active_filtering_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = _service(session_factory, clock)
    try:
        active = await _register(
            service, worker_pubkey="pg-h1-active", miner_hotkey=MINER_H1, nonce="a"
        )
        stale = await _register(
            service, worker_pubkey="pg-h1-stale", miner_hotkey=MINER_H1, nonce="b"
        )
        other = await _register(
            service, worker_pubkey="pg-h2-active", miner_hotkey=MINER_H2, nonce="c"
        )

        await service.heartbeat(worker_id=stale.worker_id, worker_pubkey="pg-h1-stale")
        clock.value = BASE_TS + timedelta(seconds=TTL + 10)
        await service.heartbeat(
            worker_id=active.worker_id, worker_pubkey="pg-h1-active"
        )
        await service.heartbeat(worker_id=other.worker_id, worker_pubkey="pg-h2-active")

        h1 = await service.active_workers(MINER_H1)
        assert [w.worker_pubkey for w in h1] == ["pg-h1-active"]
        h2 = await service.active_workers(MINER_H2)
        assert [w.worker_pubkey for w in h2] == ["pg-h2-active"]
        assert await service.active_workers("pg-unknown") == []
    finally:
        await engine.dispose()


# VAL-MASTER-021
async def test_cross_owner_rebind_rejected_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = _service(session_factory, clock)
    try:
        worker = await _register(
            service, worker_pubkey="pg-wp-rb", miner_hotkey=MINER_H1, nonce="rb-1"
        )
        await service.heartbeat(worker_id=worker.worker_id, worker_pubkey="pg-wp-rb")

        with pytest.raises(WorkerRebindError):
            await _register(
                service, worker_pubkey="pg-wp-rb", miner_hotkey=MINER_H2, nonce="rb-2"
            )

        async with session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(WorkerRegistration).where(
                            WorkerRegistration.worker_pubkey == "pg-wp-rb"
                        )
                    )
                )
                .scalars()
                .all()
            )
        assert len(rows) == 1
        assert rows[0].miner_hotkey == MINER_H1
    finally:
        await engine.dispose()


# VAL-MASTER-018
async def test_retired_worker_terminal_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = _service(session_factory, clock)
    try:
        worker = await _register(
            service, worker_pubkey="pg-wp-ret", miner_hotkey=MINER_H1, nonce="ret-1"
        )
        await service.heartbeat(worker_id=worker.worker_id, worker_pubkey="pg-wp-ret")

        async with session_scope(session_factory) as session:
            row = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_pubkey == "pg-wp-ret"
                    )
                )
            ).scalar_one()
            row.status = WorkerStatus.RETIRED

        # sole worker of its hotkey, yet never listed active.
        assert await service.active_workers(MINER_H1) == []

        # heartbeat does not resurrect it.
        clock.value = BASE_TS + timedelta(seconds=5)
        updated, now = await service.heartbeat(
            worker_id=worker.worker_id, worker_pubkey="pg-wp-ret"
        )
        assert WorkerStatus(updated.status) == WorkerStatus.RETIRED
        assert service.effective_status(updated, now) == WorkerStatus.RETIRED
        assert await service.active_workers(MINER_H1) == []
    finally:
        await engine.dispose()
