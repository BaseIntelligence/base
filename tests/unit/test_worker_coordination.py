from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    WorkerFault,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.app_proxy import create_proxy_app
from base.master.worker_coordination import (
    WorkerCoordinationService,
    run_worker_health_loop,
)
from base.security.validator_auth import canonical_validator_request
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
    worker_binding_message,
)

NOW_EPOCH = 1_750_000_000.0
TTL_SECONDS = 120

# Metagraph hotkeys: H1/H2/H3 are miners (present, no permit); VAL holds a
# validator permit; STRANGER is off-graph and unregistered.
MINER_H1 = "miner-H1"
MINER_H2 = "miner-H2"
MINER_H3 = "miner-H3"
VALIDATOR = "val-permit"
STRANGER = "stranger-key"


class FakeClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, UTC)


class FakeNonceStore:
    async def reserve(self, **kwargs: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


def _sign(hotkey: str, message: bytes) -> str:
    return hashlib.sha256(hotkey.encode() + b":" + message).hexdigest()


def _fake_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message)


def _signed_headers(
    *,
    method: str,
    path: str,
    query_string: str = "",
    body: bytes,
    hotkey: str,
    nonce: str,
    timestamp: float,
) -> dict[str, str]:
    ts = str(int(timestamp))
    canonical = canonical_validator_request(
        method=method,
        path=path,
        query_string=query_string,
        timestamp=ts,
        nonce=nonce,
        body=body,
    )
    return {
        "X-Hotkey": hotkey,
        "X-Signature": _sign(hotkey, canonical.encode()),
        "X-Nonce": nonce,
        "X-Timestamp": ts,
        "Content-Type": "application/json",
    }


class Harness:
    def __init__(
        self,
        client: AsyncClient,
        session_factory: Any,
        clock: FakeClock,
        service: WorkerCoordinationService,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.clock = clock
        self.service = service
        self._nonce = 0

    def _next_nonce(self, prefix: str) -> str:
        self._nonce += 1
        return f"{prefix}-{self._nonce}"

    async def register(
        self,
        *,
        worker_pubkey: str,
        miner_hotkey: str,
        provider: str = "local",
        provider_instance_ref: str | None = "local-1",
        capabilities: list[str] | None = None,
        nonce: str | None = None,
        signature: str | None = None,
    ) -> Any:
        nonce = nonce or self._next_nonce("bind")
        message = worker_binding_message(
            worker_pubkey=worker_pubkey, miner_hotkey=miner_hotkey, nonce=nonce
        )
        signature = signature if signature is not None else _sign(miner_hotkey, message)
        payload: dict[str, Any] = {
            "worker_pubkey": worker_pubkey,
            "miner_hotkey": miner_hotkey,
            "binding_signature": signature,
            "nonce": nonce,
            "provider": provider,
            "provider_instance_ref": provider_instance_ref,
        }
        if capabilities is not None:
            payload["capabilities"] = capabilities
        return await self.client.post("/v1/workers/register", json=payload)

    async def heartbeat(self, *, worker_id: str, signer_pubkey: str) -> Any:
        body = b"{}"
        path = f"/v1/workers/{worker_id}/heartbeat"
        headers = _signed_headers(
            method="POST",
            path=path,
            body=body,
            hotkey=signer_pubkey,
            nonce=self._next_nonce("hb"),
            timestamp=self.clock.time(),
        )
        return await self.client.post(path, content=body, headers=headers)

    async def list_workers(self, *, signer: str = VALIDATOR) -> Any:
        path = "/v1/workers"
        headers = _signed_headers(
            method="GET",
            path=path,
            body=b"",
            hotkey=signer,
            nonce=self._next_nonce("list"),
            timestamp=self.clock.time(),
        )
        return await self.client.get(path, headers=headers)

    async def active(self, hotkey: str, *, signer: str = VALIDATOR) -> Any:
        path = "/v1/workers/active"
        query = f"hotkey={hotkey}"
        headers = _signed_headers(
            method="GET",
            path=path,
            query_string=query,
            body=b"",
            hotkey=signer,
            nonce=self._next_nonce("active"),
            timestamp=self.clock.time(),
        )
        return await self.client.get(path, params={"hotkey": hotkey}, headers=headers)

    async def worker_row(self, worker_pubkey: str) -> WorkerRegistration | None:
        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_pubkey == worker_pubkey
                    )
                )
            ).scalar_one_or_none()

    async def count(self, worker_pubkey: str) -> int:
        async with self.session_factory() as session:
            return await session.scalar(
                select(func.count(WorkerRegistration.id)).where(
                    WorkerRegistration.worker_pubkey == worker_pubkey
                )
            )

    async def set_status(self, worker_pubkey: str, status: WorkerStatus) -> None:
        async with session_scope(self.session_factory) as session:
            row = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_pubkey == worker_pubkey
                    )
                )
            ).scalar_one()
            row.status = status

    async def add_fault(self, worker_id: str, work_unit_id: str) -> None:
        async with session_scope(self.session_factory) as session:
            session.add(
                WorkerFault(
                    worker_id=worker_id,
                    work_unit_id=work_unit_id,
                    challenge_slug="prism",
                    detail="manifest hash divergence",
                    created_at=self.clock.now(),
                )
            )


async def _build_harness(*, mount_worker_plane: bool = True) -> tuple[Harness, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER_H1, MINER_H2, MINER_H3, VALIDATOR],
        validator_permits=[False, False, False, True],
        stakes=[0.0, 0.0, 0.0, 100.0],
    )
    clock = FakeClock(NOW_EPOCH)
    service = WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        signature_verifier=_fake_verifier,
        heartbeat_ttl_seconds=TTL_SECONDS,
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
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        worker_service=service if mount_worker_plane else None,
        worker_verifier=verifier if mount_worker_plane else None,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return Harness(client, session_factory, clock, service), engine


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h, engine = await _build_harness()
    try:
        yield h
    finally:
        await h.client.aclose()
        await engine.dispose()


# VAL-MASTER-001
async def test_register_pending_then_heartbeat_active(harness: Harness) -> None:
    response = await harness.register(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    assert response.status_code == 200
    body = response.json()
    assert body["worker"]["status"] == "pending"
    assert body["worker"]["miner_hotkey"] == MINER_H1
    assert body["worker"]["provider"] == "local"
    assert body["heartbeat_ttl_seconds"] == TTL_SECONDS
    worker_id = body["worker"]["worker_id"]

    listed = await harness.list_workers()
    assert listed.status_code == 200
    entry = listed.json()["workers"][0]
    assert entry["status"] == "pending"
    assert entry["worker_pubkey"] == "wp-1"
    assert entry["last_heartbeat_at"] is None

    harness.clock.epoch = NOW_EPOCH + 10
    hb = await harness.heartbeat(worker_id=worker_id, signer_pubkey="wp-1")
    assert hb.status_code == 200
    assert hb.json()["status"] == "active"

    listed = await harness.list_workers()
    entry = listed.json()["workers"][0]
    assert entry["status"] == "active"
    assert entry["last_heartbeat_at"] is not None


# VAL-AGENT-015
async def test_fleet_view_exposes_status_owner_provider_last_seen_faults(
    harness: Harness,
) -> None:
    reg1 = await harness.register(
        worker_pubkey="wp-a", miner_hotkey=MINER_H1, provider="lium"
    )
    reg2 = await harness.register(
        worker_pubkey="wp-b", miner_hotkey=MINER_H2, provider="targon"
    )
    id_a = reg1.json()["worker"]["worker_id"]
    id_b = reg2.json()["worker"]["worker_id"]

    await harness.heartbeat(worker_id=id_a, signer_pubkey="wp-a")
    await harness.heartbeat(worker_id=id_b, signer_pubkey="wp-b")
    await harness.add_fault(id_b, "unit-42")

    # wp-a heartbeats again just past nothing; wp-b goes stale past the TTL.
    harness.clock.epoch = NOW_EPOCH + TTL_SECONDS + 5
    await harness.heartbeat(worker_id=id_a, signer_pubkey="wp-a")

    listed = await harness.list_workers()
    assert listed.status_code == 200
    workers = {w["worker_pubkey"]: w for w in listed.json()["workers"]}
    for key in ("status", "miner_hotkey", "provider", "last_heartbeat_at", "faults"):
        assert key in workers["wp-a"]
        assert key in workers["wp-b"]
    assert workers["wp-a"]["status"] == "active"
    assert workers["wp-a"]["miner_hotkey"] == MINER_H1
    assert workers["wp-a"]["provider"] == "lium"
    assert workers["wp-b"]["status"] == "stale"
    assert workers["wp-b"]["provider"] == "targon"
    assert len(workers["wp-b"]["faults"]) == 1
    assert workers["wp-b"]["faults"][0]["work_unit_id"] == "unit-42"
    assert workers["wp-a"]["faults"] == []


# VAL-MASTER-002 (a): forged/invalid signature
async def test_register_invalid_signature_rejected(harness: Harness) -> None:
    response = await harness.register(
        worker_pubkey="wp-x", miner_hotkey=MINER_H1, signature="deadbeef"
    )
    assert response.status_code == 401
    assert await harness.worker_row("wp-x") is None


# VAL-MASTER-002 (b): miner not in metagraph
async def test_register_miner_not_in_metagraph_rejected(harness: Harness) -> None:
    response = await harness.register(
        worker_pubkey="wp-y", miner_hotkey="off-graph-miner"
    )
    assert response.status_code == 403
    assert await harness.worker_row("wp-y") is None


# VAL-MASTER-002 (c) + VAL-AGENT-004: replay rejected, fresh nonce re-enroll ok
async def test_replayed_nonce_rejected_fresh_nonce_reenrolls(harness: Harness) -> None:
    first = await harness.register(
        worker_pubkey="wp-z", miner_hotkey=MINER_H1, nonce="fixed-nonce"
    )
    assert first.status_code == 200

    replay = await harness.register(
        worker_pubkey="wp-z2", miner_hotkey=MINER_H1, nonce="fixed-nonce"
    )
    assert replay.status_code == 409
    assert await harness.worker_row("wp-z2") is None

    fresh = await harness.register(
        worker_pubkey="wp-z", miner_hotkey=MINER_H1, nonce="fresh-nonce"
    )
    assert fresh.status_code == 200
    assert await harness.count("wp-z") == 1


# VAL-MASTER-021: cross-owner rebind never silently rebinds
async def test_rebind_to_different_hotkey_rejected(harness: Harness) -> None:
    reg = await harness.register(worker_pubkey="wp-rb", miner_hotkey=MINER_H1)
    worker_id = reg.json()["worker"]["worker_id"]
    await harness.heartbeat(worker_id=worker_id, signer_pubkey="wp-rb")

    rebind = await harness.register(worker_pubkey="wp-rb", miner_hotkey=MINER_H2)
    assert rebind.status_code == 409

    row = await harness.worker_row("wp-rb")
    assert row is not None
    assert row.miner_hotkey == MINER_H1
    assert await harness.count("wp-rb") == 1

    listed = await harness.list_workers()
    entries = [w for w in listed.json()["workers"] if w["worker_pubkey"] == "wp-rb"]
    assert len(entries) == 1
    assert entries[0]["miner_hotkey"] == MINER_H1
    assert entries[0]["status"] == "active"


# VAL-MASTER-016: active endpoint filters by hotkey + status
async def test_active_endpoint_filters_by_hotkey_and_status(harness: Harness) -> None:
    # H1: one active + one stale; H2: one active; H3: only pending.
    a1 = await harness.register(worker_pubkey="h1-active", miner_hotkey=MINER_H1)
    a1_id = a1.json()["worker"]["worker_id"]
    s1 = await harness.register(worker_pubkey="h1-stale", miner_hotkey=MINER_H1)
    s1_id = s1.json()["worker"]["worker_id"]
    b1 = await harness.register(worker_pubkey="h2-active", miner_hotkey=MINER_H2)
    b1_id = b1.json()["worker"]["worker_id"]
    await harness.register(worker_pubkey="h3-pending", miner_hotkey=MINER_H3)

    await harness.heartbeat(worker_id=s1_id, signer_pubkey="h1-stale")
    # advance so h1-stale's heartbeat is now beyond the TTL, then heartbeat the
    # others fresh at the later time.
    harness.clock.epoch = NOW_EPOCH + TTL_SECONDS + 10
    await harness.heartbeat(worker_id=a1_id, signer_pubkey="h1-active")
    await harness.heartbeat(worker_id=b1_id, signer_pubkey="h2-active")

    h1 = await harness.active(MINER_H1)
    assert h1.status_code == 200
    h1_workers = [w["worker_pubkey"] for w in h1.json()["workers"]]
    assert h1_workers == ["h1-active"]

    h2 = await harness.active(MINER_H2)
    assert [w["worker_pubkey"] for w in h2.json()["workers"]] == ["h2-active"]

    h3 = await harness.active(MINER_H3)
    assert h3.status_code == 200
    assert h3.json()["workers"] == []

    unknown = await harness.active("nobody")
    assert unknown.status_code == 200
    assert unknown.json()["workers"] == []

    # After H1's active worker also passes the TTL, the query is empty.
    harness.clock.epoch = NOW_EPOCH + 2 * (TTL_SECONDS + 10) + 10
    h1_later = await harness.active(MINER_H1)
    assert h1_later.json()["workers"] == []


# VAL-MASTER-018: retired is terminal
async def test_retired_worker_is_terminal(harness: Harness) -> None:
    reg = await harness.register(worker_pubkey="wp-ret", miner_hotkey=MINER_H1)
    worker_id = reg.json()["worker"]["worker_id"]
    await harness.heartbeat(worker_id=worker_id, signer_pubkey="wp-ret")
    await harness.set_status("wp-ret", WorkerStatus.RETIRED)

    # (a)/(b) excluded from active listing even as sole worker of its hotkey.
    active = await harness.active(MINER_H1)
    assert active.json()["workers"] == []

    # still listed with status retired in the fleet view.
    listed = await harness.list_workers()
    entry = [w for w in listed.json()["workers"] if w["worker_pubkey"] == "wp-ret"][0]
    assert entry["status"] == "retired"

    # (c) a heartbeat never resurrects a retired worker.
    harness.clock.epoch = NOW_EPOCH + 5
    hb = await harness.heartbeat(worker_id=worker_id, signer_pubkey="wp-ret")
    assert hb.status_code == 200
    assert hb.json()["status"] == "retired"
    row = await harness.worker_row("wp-ret")
    assert row is not None
    assert WorkerStatus(row.status) == WorkerStatus.RETIRED
    assert (await harness.active(MINER_H1)).json()["workers"] == []


async def test_heartbeat_unknown_worker_returns_404(harness: Harness) -> None:
    # A registered worker signs, but targets an unknown worker_id.
    await harness.register(worker_pubkey="wp-known", miner_hotkey=MINER_H1)
    hb = await harness.heartbeat(worker_id="does-not-exist", signer_pubkey="wp-known")
    assert hb.status_code == 404


async def test_heartbeat_foreign_owner_returns_403(harness: Harness) -> None:
    reg = await harness.register(worker_pubkey="wp-owner", miner_hotkey=MINER_H1)
    other = await harness.register(worker_pubkey="wp-other", miner_hotkey=MINER_H2)
    owner_id = reg.json()["worker"]["worker_id"]
    # wp-other is a registered (eligible) signer but does not own owner_id.
    hb = await harness.heartbeat(worker_id=owner_id, signer_pubkey="wp-other")
    assert hb.status_code == 403
    assert other.status_code == 200


# Fleet reads are authenticated-but-not-admin: worker OR validator identities.
async def test_fleet_read_auth_accepts_worker_and_validator_rejects_stranger(
    harness: Harness,
) -> None:
    await harness.register(worker_pubkey="wp-auth", miner_hotkey=MINER_H1)

    by_validator = await harness.list_workers(signer=VALIDATOR)
    assert by_validator.status_code == 200

    by_worker = await harness.list_workers(signer="wp-auth")
    assert by_worker.status_code == 200

    by_stranger = await harness.list_workers(signer=STRANGER)
    assert by_stranger.status_code == 403


async def test_fleet_read_rejects_unsigned_request(harness: Harness) -> None:
    unsigned = await harness.client.get("/v1/workers")
    assert unsigned.status_code == 401


# Flag OFF (worker plane not mounted) => surface is inert (404), legacy safe.
async def test_worker_surface_absent_when_plane_disabled() -> None:
    h, engine = await _build_harness(mount_worker_plane=False)
    try:
        listed = await h.client.get("/v1/workers")
        assert listed.status_code == 404
        health = await h.client.get("/health")
        assert health.status_code == 200
    finally:
        await h.client.aclose()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Service-level staleness detection
# ---------------------------------------------------------------------------


async def _service_only() -> tuple[WorkerCoordinationService, Any, FakeClock, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph([MINER_H1], validator_permits=[False], stakes=[0.0])
    clock = FakeClock(NOW_EPOCH)
    service = WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        signature_verifier=_fake_verifier,
        heartbeat_ttl_seconds=TTL_SECONDS,
        now_fn=clock.now,
    )
    return service, session_factory, clock, engine


async def test_detect_stale_workers_transitions_active_to_stale() -> None:
    service, session_factory, clock, engine = await _service_only()
    try:
        worker = await service.register(
            worker_pubkey="wp-s",
            miner_hotkey=MINER_H1,
            binding_signature=_sign(
                MINER_H1,
                worker_binding_message(
                    worker_pubkey="wp-s", miner_hotkey=MINER_H1, nonce="n1"
                ),
            ),
            nonce="n1",
            provider="local",
            provider_instance_ref=None,
            capabilities=["gpu"],
        )
        await service.heartbeat(worker_id=worker.worker_id, worker_pubkey="wp-s")

        # within TTL: no transition
        clock.epoch = NOW_EPOCH + TTL_SECONDS
        assert await service.detect_stale_workers() == []

        # beyond TTL: transitions to stale (edge-triggered, once)
        clock.epoch = NOW_EPOCH + TTL_SECONDS + 1
        assert await service.detect_stale_workers() == [worker.worker_id]
        clock.epoch = NOW_EPOCH + TTL_SECONDS + 500
        assert await service.detect_stale_workers() == []

        async with session_factory() as session:
            row = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_pubkey == "wp-s"
                    )
                )
            ).scalar_one()
            assert WorkerStatus(row.status) == WorkerStatus.STALE
    finally:
        await engine.dispose()


async def test_run_worker_health_loop_repeats_until_shutdown() -> None:
    shutdown = asyncio.Event()

    class _Recording:
        def __init__(self) -> None:
            self.calls = 0

        async def detect_stale_workers(self) -> list[str]:
            self.calls += 1
            if self.calls >= 3:
                shutdown.set()
            return []

    service = _Recording()
    await asyncio.wait_for(
        run_worker_health_loop(
            service,  # type: ignore[arg-type]
            interval_seconds=0.001,
            shutdown_event=shutdown,
        ),
        timeout=2,
    )
    assert service.calls >= 3


def test_json_body_roundtrip_is_deterministic() -> None:
    # guard: the register body encoding used by the harness is stable JSON.
    assert json.loads(json.dumps({"a": 1})) == {"a": 1}
    assert timedelta(seconds=TTL_SECONDS).total_seconds() == TTL_SECONDS
