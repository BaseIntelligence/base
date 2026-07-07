from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.app_proxy import create_proxy_app
from base.master.worker_coordination import (
    WorkerCoordinationService,
    WorkerRebindError,
)
from base.security.validator_auth import canonical_validator_request
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
    WorkerReplayError,
    WorkerSignedRequestVerifier,
    worker_binding_message,
)

pytestmark = pytest.mark.postgres

BASE_TS = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
BASE_EPOCH = BASE_TS.timestamp()
TTL = 120
MINER_H1 = "pg-miner-H1"
MINER_H2 = "pg-miner-H2"
VALIDATOR = "pg-val-permit"
INTERNAL_BRIDGE_TOKEN = "pg-prism-bridge-token"


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


# ---------------------------------------------------------------------------
# Admission fleet-read dual-auth over HTTP against real postgres (15433).
# master-admission-fleetread-auth: internal bridge bearer OR signed request on
# GET /v1/workers/active; the full fleet GET /v1/workers stays signed-only.
# ---------------------------------------------------------------------------


class _TokenRegistry:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_token(self, slug: str) -> str:
        if slug == "prism":
            return self._token
        raise RuntimeError(f"no token for {slug!r}")


class _HttpClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, UTC)


class FakeNonceStore:
    async def reserve(self, **_kwargs: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


def _signed_headers(
    *,
    method: str,
    path: str,
    query_string: str,
    hotkey: str,
    nonce: str,
    epoch: float,
) -> dict[str, str]:
    ts = str(int(epoch))
    canonical = canonical_validator_request(
        method=method,
        path=path,
        query_string=query_string,
        timestamp=ts,
        nonce=nonce,
        body=b"",
    )
    return {
        "X-Hotkey": hotkey,
        "X-Signature": _sign(hotkey, canonical.encode()),
        "X-Nonce": nonce,
        "X-Timestamp": ts,
    }


async def test_active_admission_dual_auth_over_http_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _HttpClock(BASE_EPOCH)
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER_H1, MINER_H2, VALIDATOR],
        validator_permits=[False, False, True],
        stakes=[0.0, 0.0, 100.0],
    )
    service = WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        signature_verifier=_fake_verifier,
        heartbeat_ttl_seconds=TTL,
        now_fn=clock.now,
    )
    verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        eligibility=CoordinationReadEligibility(session_factory, cache),
        signature_verifier=_fake_verifier,
        ttl_seconds=300,
        now_fn=clock.time,
    )
    app = create_proxy_app(
        registry=_TokenRegistry(INTERNAL_BRIDGE_TOKEN),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        worker_service=service,
        worker_verifier=verifier,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        # Two active workers on distinct owners (proves per-hotkey filtering).
        w1 = await _register(
            service, worker_pubkey="pg-http-h1", miner_hotkey=MINER_H1, nonce="h1"
        )
        w2 = await _register(
            service, worker_pubkey="pg-http-h2", miner_hotkey=MINER_H2, nonce="h2"
        )
        await service.heartbeat(worker_id=w1.worker_id, worker_pubkey="pg-http-h1")
        await service.heartbeat(worker_id=w2.worker_id, worker_pubkey="pg-http-h2")

        bearer = {"Authorization": f"Bearer {INTERNAL_BRIDGE_TOKEN}"}

        # (a) internal bearer returns exactly that hotkey's ACTIVE workers.
        resp = await client.get(
            "/v1/workers/active", params={"hotkey": MINER_H1}, headers=bearer
        )
        assert resp.status_code == 200
        assert [w["worker_pubkey"] for w in resp.json()["workers"]] == ["pg-http-h1"]

        # (b) wrong bearer / missing bearer with no signature => 401/403.
        wrong = await client.get(
            "/v1/workers/active",
            params={"hotkey": MINER_H1},
            headers={"Authorization": "Bearer nope"},
        )
        assert wrong.status_code in (401, 403)
        missing = await client.get("/v1/workers/active", params={"hotkey": MINER_H1})
        assert missing.status_code in (401, 403)

        # (c) signed-request path still works unchanged.
        signed = await client.get(
            "/v1/workers/active",
            params={"hotkey": MINER_H1},
            headers=_signed_headers(
                method="GET",
                path="/v1/workers/active",
                query_string=f"hotkey={MINER_H1}",
                hotkey=VALIDATOR,
                nonce="sig-active-1",
                epoch=clock.time(),
            ),
        )
        assert signed.status_code == 200
        assert [w["worker_pubkey"] for w in signed.json()["workers"]] == ["pg-http-h1"]

        # (d) full fleet view rejects the internal bearer (signed-only).
        fleet_bearer = await client.get("/v1/workers", headers=bearer)
        assert fleet_bearer.status_code in (401, 403)
        fleet_signed = await client.get(
            "/v1/workers",
            headers=_signed_headers(
                method="GET",
                path="/v1/workers",
                query_string="",
                hotkey=VALIDATOR,
                nonce="sig-fleet-1",
                epoch=clock.time(),
            ),
        )
        assert fleet_signed.status_code == 200
    finally:
        await client.aclose()
        await engine.dispose()
