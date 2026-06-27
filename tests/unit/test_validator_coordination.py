from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorHealthEvent,
    ValidatorHealthEventType,
    ValidatorStatus,
)
from base.db.session import create_engine, create_session_factory, session_scope
from base.master.app_admin import create_admin_app
from base.master.app_proxy import create_proxy_app
from base.master.validator_coordination import (
    ValidatorCoordinationService,
    build_validator_health_lifespan,
    run_validator_health_loop,
)
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
    canonical_validator_request,
)

NOW_EPOCH = 1_750_000_000.0
HEARTBEAT_INTERVAL = 45
HEARTBEAT_TIMEOUT = 100
ADMIN_TOKEN = "admin-secret-token"


def _as_utc(value: datetime) -> datetime:
    """Normalize a possibly-naive datetime (SQLite) to aware UTC."""

    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


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


def _sign(hotkey: str, canonical: str) -> str:
    return hashlib.sha256(f"{hotkey}:{canonical}".encode()).hexdigest()


def _verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message.decode())


def signed_headers(
    *,
    method: str = "POST",
    path: str,
    body: bytes,
    hotkey: str = "permitted",
    nonce: str,
    timestamp: float,
) -> dict[str, str]:
    ts = str(int(timestamp))
    canonical = canonical_validator_request(
        method=method,
        path=path,
        query_string="",
        timestamp=ts,
        nonce=nonce,
        body=body,
    )
    return {
        "X-Hotkey": hotkey,
        "X-Signature": _sign(hotkey, canonical),
        "X-Nonce": nonce,
        "X-Timestamp": ts,
        "Content-Type": "application/json",
    }


class _Harness:
    def __init__(
        self,
        client: AsyncClient,
        session_factory: Any,
        clock: FakeClock,
        service: ValidatorCoordinationService,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.clock = clock
        self.service = service
        self._nonce = 0

    async def list_validators(self, *, token: str | None = None):
        headers = {"X-Admin-Token": token} if token is not None else {}
        return await self.client.get("/v1/validators", headers=headers)

    def _next_nonce(self, prefix: str) -> str:
        self._nonce += 1
        return f"{prefix}-{self._nonce}"

    async def register(
        self,
        *,
        capabilities: list[str],
        version: str | None,
        hotkey: str = "permitted",
        last_seen_meta: dict[str, Any] | None = None,
    ):
        payload: dict[str, Any] = {"capabilities": capabilities, "version": version}
        if last_seen_meta is not None:
            payload["last_seen_meta"] = last_seen_meta
        import json

        body = json.dumps(payload).encode()
        path = "/v1/validators/register"
        headers = signed_headers(
            path=path,
            body=body,
            hotkey=hotkey,
            nonce=self._next_nonce("reg"),
            timestamp=self.clock.time(),
        )
        return await self.client.post(path, content=body, headers=headers)

    async def heartbeat(
        self,
        *,
        hotkey: str = "permitted",
        last_seen_meta: dict[str, Any] | None = None,
    ):
        import json

        payload: dict[str, Any] = {}
        if last_seen_meta is not None:
            payload["last_seen_meta"] = last_seen_meta
        body = json.dumps(payload).encode()
        path = "/v1/validators/heartbeat"
        headers = signed_headers(
            path=path,
            body=body,
            hotkey=hotkey,
            nonce=self._next_nonce("hb"),
            timestamp=self.clock.time(),
        )
        return await self.client.post(path, content=body, headers=headers)

    async def get_validator(self, hotkey: str = "permitted") -> Validator | None:
        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one_or_none()

    async def count_validators(self, hotkey: str = "permitted") -> int:
        async with self.session_factory() as session:
            return await session.scalar(
                select(func.count(Validator.id)).where(Validator.hotkey == hotkey)
            )

    async def events(self, hotkey: str = "permitted") -> list[ValidatorHealthEventType]:
        async with self.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(ValidatorHealthEvent)
                        .where(ValidatorHealthEvent.validator_hotkey == hotkey)
                        .order_by(ValidatorHealthEvent.created_at)
                    )
                )
                .scalars()
                .all()
            )
        return [row.event for row in rows]

    async def set_offline(self, hotkey: str = "permitted") -> None:
        async with session_scope(self.session_factory) as session:
            validator = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            validator.status = ValidatorStatus.OFFLINE


async def _build_harness(
    factory: str, *, admin_token: str = ADMIN_TOKEN
) -> tuple[_Harness, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        ["permitted", "permitted2"],
        validator_permits=[True, True],
        stakes=[100.0, 50.0],
    )
    clock = FakeClock(NOW_EPOCH)
    verifier = ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(session_factory),
        eligibility=MetagraphValidatorEligibility(cache),
        signature_verifier=_verifier,
        ttl_seconds=300,
        now_fn=clock.time,
    )
    service = ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL,
        heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT,
        now_fn=clock.now,
    )

    if factory == "proxy":
        app = create_proxy_app(
            registry=object(),
            nonce_store=FakeNonceStore(),
            metagraph_cache=FakeCache(),  # type: ignore[arg-type]
            validator_service=service,
            validator_verifier=verifier,
            admin_token_provider=lambda: admin_token,
        )
    else:
        app = create_admin_app(
            registry=object(),
            runtime_controller=object(),  # type: ignore[arg-type]
            validator_service=service,
            validator_verifier=verifier,
            admin_token_provider=lambda: admin_token,
        )

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return _Harness(client, session_factory, clock, service), engine


@pytest.fixture
async def harness() -> AsyncIterator[_Harness]:
    h, engine = await _build_harness("proxy")
    try:
        yield h
    finally:
        await h.client.aclose()
        await engine.dispose()


# VAL-VREG-001
async def test_register_marks_online_with_capabilities_and_version(
    harness: _Harness,
) -> None:
    response = await harness.register(capabilities=["cpu"], version="1.2.3")
    assert response.status_code == 200
    body = response.json()
    assert body["heartbeat_interval_seconds"] == HEARTBEAT_INTERVAL
    assert body["heartbeat_interval_seconds"] > 0
    assert body["validator"]["status"] == "online"
    assert body["validator"]["capabilities"] == ["cpu"]
    assert body["validator"]["version"] == "1.2.3"

    validator = await harness.get_validator()
    assert validator is not None
    assert validator.status == ValidatorStatus.ONLINE
    assert validator.capabilities == ["cpu"]
    assert validator.version == "1.2.3"
    assert validator.registered_at is not None
    assert validator.last_heartbeat_at is not None


# VAL-VREG-002
async def test_register_records_gpu_capability(harness: _Harness) -> None:
    gpu = await harness.register(
        capabilities=["cpu", "gpu"], version="2.0.0", hotkey="permitted"
    )
    cpu = await harness.register(
        capabilities=["cpu"], version="2.0.0", hotkey="permitted2"
    )
    assert gpu.status_code == 200
    assert cpu.status_code == 200
    assert gpu.json()["validator"]["capabilities"] == ["cpu", "gpu"]
    assert cpu.json()["validator"]["capabilities"] == ["cpu"]

    gpu_row = await harness.get_validator("permitted")
    cpu_row = await harness.get_validator("permitted2")
    assert gpu_row is not None and cpu_row is not None
    assert gpu_row.capabilities == ["cpu", "gpu"]
    assert cpu_row.capabilities == ["cpu"]


# VAL-VREG-003
async def test_register_emits_registered_and_online_events(harness: _Harness) -> None:
    await harness.register(capabilities=["cpu"], version="1.0.0")
    events = await harness.events()
    assert ValidatorHealthEventType.REGISTERED in events
    assert ValidatorHealthEventType.ONLINE in events
    assert events[0] == ValidatorHealthEventType.REGISTERED


# VAL-VREG-004
async def test_reregister_is_idempotent_upsert(harness: _Harness) -> None:
    first = await harness.register(capabilities=["cpu"], version="1.0.0")
    assert first.status_code == 200
    original = await harness.get_validator()
    assert original is not None
    original_registered_at = original.registered_at

    harness.clock.epoch = NOW_EPOCH + 30
    second = await harness.register(capabilities=["cpu", "gpu"], version="2.0.0")
    assert second.status_code == 200

    assert await harness.count_validators() == 1
    updated = await harness.get_validator()
    assert updated is not None
    assert updated.capabilities == ["cpu", "gpu"]
    assert updated.version == "2.0.0"
    assert updated.status == ValidatorStatus.ONLINE
    assert updated.last_heartbeat_at is not None
    assert _as_utc(updated.registered_at) == _as_utc(original_registered_at)
    assert _as_utc(updated.last_heartbeat_at) == harness.clock.now()
    # registered event is emitted only on the first registration (edge-triggered).
    events = await harness.events()
    assert events.count(ValidatorHealthEventType.REGISTERED) == 1


# VAL-VREG-014
async def test_heartbeat_refreshes_liveness_and_returns_status_now(
    harness: _Harness,
) -> None:
    await harness.register(capabilities=["cpu"], version="1.0.0")
    harness.clock.epoch = NOW_EPOCH + 20
    response = await harness.heartbeat()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "online"
    assert "now" in body and body["now"]

    validator = await harness.get_validator()
    assert validator is not None
    assert validator.status == ValidatorStatus.ONLINE
    assert validator.last_heartbeat_at is not None
    assert _as_utc(validator.last_heartbeat_at) == harness.clock.now()
    assert await harness.count_validators() == 1


# VAL-VREG-015
async def test_offline_validator_heartbeat_flips_online(harness: _Harness) -> None:
    await harness.register(capabilities=["cpu"], version="1.0.0")
    await harness.set_offline()
    before = await harness.events()

    harness.clock.epoch = NOW_EPOCH + 20
    response = await harness.heartbeat()
    assert response.status_code == 200
    assert response.json()["status"] == "online"

    validator = await harness.get_validator()
    assert validator is not None
    assert validator.status == ValidatorStatus.ONLINE

    after = await harness.events()
    assert (
        after.count(ValidatorHealthEventType.ONLINE)
        == before.count(ValidatorHealthEventType.ONLINE) + 1
    )
    assert after[-1] == ValidatorHealthEventType.ONLINE


# VAL-VREG-016
async def test_heartbeat_persists_optional_liveness_metadata(
    harness: _Harness,
) -> None:
    await harness.register(capabilities=["cpu"], version="1.0.0")

    harness.clock.epoch = NOW_EPOCH + 10
    meta = {"broker": "ok", "concurrency": 4}
    with_meta = await harness.heartbeat(last_seen_meta=meta)
    assert with_meta.status_code == 200
    validator = await harness.get_validator()
    assert validator is not None
    assert validator.last_seen_meta == meta

    harness.clock.epoch = NOW_EPOCH + 20
    without_meta = await harness.heartbeat()
    assert without_meta.status_code == 200
    validator = await harness.get_validator()
    assert validator is not None
    assert validator.last_seen_meta == meta


# VAL-VREG-017
async def test_heartbeat_unknown_hotkey_returns_404_and_creates_no_row(
    harness: _Harness,
) -> None:
    response = await harness.heartbeat(hotkey="permitted")
    assert response.status_code == 404
    assert await harness.count_validators("permitted") == 0


async def test_register_wired_into_admin_app() -> None:
    h, engine = await _build_harness("admin")
    try:
        response = await h.register(capabilities=["cpu"], version="1.0.0")
        assert response.status_code == 200
        assert response.json()["validator"]["status"] == "online"
        heartbeat = await h.heartbeat()
        assert heartbeat.status_code == 200
    finally:
        await h.client.aclose()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Crash / offline detection (service-level)
# ---------------------------------------------------------------------------


async def _build_service(
    *, timeout: int = HEARTBEAT_TIMEOUT, epoch: float = NOW_EPOCH
) -> tuple[ValidatorCoordinationService, Any, FakeClock, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)
    clock = FakeClock(epoch)
    service = ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL,
        heartbeat_timeout_seconds=timeout,
        now_fn=clock.now,
    )
    return service, session_factory, clock, engine


async def _status(session_factory: Any, hotkey: str) -> ValidatorStatus:
    async with session_factory() as session:
        row = (
            await session.execute(select(Validator).where(Validator.hotkey == hotkey))
        ).scalar_one()
        return ValidatorStatus(row.status)


async def _events_for(
    session_factory: Any, hotkey: str
) -> list[ValidatorHealthEventType]:
    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(ValidatorHealthEvent)
                    .where(ValidatorHealthEvent.validator_hotkey == hotkey)
                    .order_by(ValidatorHealthEvent.created_at)
                )
            )
            .scalars()
            .all()
        )
    return [row.event for row in rows]


# VAL-VREG-018
async def test_detection_marks_stale_validator_offline_with_crash_event() -> None:
    service, session_factory, clock, engine = await _build_service()
    try:
        await service.register(
            hotkey="permitted", uid=1, capabilities=["cpu"], version="1.0.0"
        )
        assert await _status(session_factory, "permitted") == ValidatorStatus.ONLINE

        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 1
        transitioned = await service.detect_offline_validators()

        assert transitioned == ["permitted"]
        assert await _status(session_factory, "permitted") == ValidatorStatus.OFFLINE
        events = await _events_for(session_factory, "permitted")
        assert events.count(ValidatorHealthEventType.CRASH_DETECTED) == 1
        assert events[-1] == ValidatorHealthEventType.CRASH_DETECTED
    finally:
        await engine.dispose()


# VAL-VREG-019
async def test_detection_leaves_recent_validator_online() -> None:
    service, session_factory, clock, engine = await _build_service()
    try:
        await service.register(
            hotkey="permitted", uid=1, capabilities=["cpu"], version="1.0.0"
        )
        # exactly at the boundary: now - last_heartbeat_at == timeout (not > timeout)
        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT
        transitioned = await service.detect_offline_validators()

        assert transitioned == []
        assert await _status(session_factory, "permitted") == ValidatorStatus.ONLINE
        events = await _events_for(session_factory, "permitted")
        assert ValidatorHealthEventType.CRASH_DETECTED not in events
        assert ValidatorHealthEventType.OFFLINE not in events
    finally:
        await engine.dispose()


# VAL-VREG-020
async def test_detection_is_edge_triggered_no_duplicate_events() -> None:
    service, session_factory, clock, engine = await _build_service()
    try:
        await service.register(
            hotkey="permitted", uid=1, capabilities=["cpu"], version="1.0.0"
        )
        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 1

        first = await service.detect_offline_validators()
        assert first == ["permitted"]
        # repeated passes over the already-offline validator are no-ops
        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 500
        second = await service.detect_offline_validators()
        third = await service.detect_offline_validators()
        assert second == []
        assert third == []

        events = await _events_for(session_factory, "permitted")
        assert events.count(ValidatorHealthEventType.CRASH_DETECTED) == 1
    finally:
        await engine.dispose()


# VAL-VREG-022
async def test_health_events_are_append_only_and_time_ordered() -> None:
    service, session_factory, clock, engine = await _build_service()
    try:
        # registered -> online (T0)
        await service.register(
            hotkey="permitted", uid=1, capabilities=["cpu"], version="1.0.0"
        )
        async with session_factory() as session:
            registered_row = (
                await session.execute(
                    select(ValidatorHealthEvent)
                    .where(
                        ValidatorHealthEvent.validator_hotkey == "permitted",
                        ValidatorHealthEvent.event
                        == ValidatorHealthEventType.REGISTERED,
                    )
                    .order_by(ValidatorHealthEvent.created_at)
                )
            ).scalar_one()
            registered_created_at = registered_row.created_at

        # crash_detected (T1)
        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 1
        await service.detect_offline_validators()

        # online recovery (T2)
        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 50
        await service.heartbeat(hotkey="permitted")

        # full ordered audit trail
        async with session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(ValidatorHealthEvent)
                        .where(ValidatorHealthEvent.validator_hotkey == "permitted")
                        .order_by(ValidatorHealthEvent.created_at)
                    )
                )
                .scalars()
                .all()
            )

        events = [row.event for row in rows]
        assert events == [
            ValidatorHealthEventType.REGISTERED,
            ValidatorHealthEventType.ONLINE,
            ValidatorHealthEventType.CRASH_DETECTED,
            ValidatorHealthEventType.ONLINE,
        ]
        created = [_as_utc(row.created_at) for row in rows]
        assert created == sorted(created)
        # append-only: the earliest event was not mutated by later transitions
        assert _as_utc(rows[0].created_at) == _as_utc(registered_created_at)
    finally:
        await engine.dispose()


# VAL-VREG-021
async def test_admin_list_validators_is_token_gated_and_reports_status(
    harness: _Harness,
) -> None:
    await harness.register(capabilities=["cpu", "gpu"], version="3.1.0")

    missing = await harness.list_validators()
    assert missing.status_code in (401, 403)
    assert "permitted" not in missing.text

    bad = await harness.list_validators(token="wrong-token")
    assert bad.status_code in (401, 403)

    ok = await harness.list_validators(token=ADMIN_TOKEN)
    assert ok.status_code == 200
    body = ok.json()
    assert len(body["validators"]) == 1
    view = body["validators"][0]
    assert view["hotkey"] == "permitted"
    assert view["status"] == "online"
    assert view["capabilities"] == ["cpu", "gpu"]
    assert view["version"] == "3.1.0"
    assert view["last_heartbeat_at"] is not None


async def test_admin_list_validators_reflects_offline_status(harness: _Harness) -> None:
    await harness.register(capabilities=["cpu"], version="1.0.0")
    harness.clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 1
    await harness.service.detect_offline_validators()

    ok = await harness.list_validators(token=ADMIN_TOKEN)
    assert ok.status_code == 200
    view = ok.json()["validators"][0]
    assert view["status"] == "offline"


async def test_admin_list_validators_wired_into_admin_app() -> None:
    h, engine = await _build_harness("admin")
    try:
        await h.register(capabilities=["cpu"], version="1.0.0")
        unauthorized = await h.list_validators()
        assert unauthorized.status_code in (401, 403)
        ok = await h.list_validators(token=ADMIN_TOKEN)
        assert ok.status_code == 200
        assert ok.json()["validators"][0]["hotkey"] == "permitted"
    finally:
        await h.client.aclose()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Background detection loop
# ---------------------------------------------------------------------------


class _RecordingService:
    def __init__(self, stop_after: int, shutdown: asyncio.Event) -> None:
        self.calls = 0
        self._stop_after = stop_after
        self._shutdown = shutdown

    async def detect_offline_validators(self) -> list[str]:
        self.calls += 1
        if self.calls >= self._stop_after:
            self._shutdown.set()
        return []


async def test_run_validator_health_loop_repeats_until_shutdown() -> None:
    shutdown = asyncio.Event()
    service = _RecordingService(stop_after=3, shutdown=shutdown)
    await asyncio.wait_for(
        run_validator_health_loop(
            service,  # type: ignore[arg-type]
            interval_seconds=0.001,
            shutdown_event=shutdown,
        ),
        timeout=2,
    )
    assert service.calls >= 3


async def test_run_validator_health_loop_survives_pass_exception() -> None:
    shutdown = asyncio.Event()
    calls: list[int] = []

    class _FlakyService:
        async def detect_offline_validators(self) -> list[str]:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            shutdown.set()
            return []

    await asyncio.wait_for(
        run_validator_health_loop(
            _FlakyService(),  # type: ignore[arg-type]
            interval_seconds=0.001,
            shutdown_event=shutdown,
        ),
        timeout=2,
    )
    assert len(calls) == 2


async def test_build_validator_health_lifespan_disabled_returns_none() -> None:
    shutdown = asyncio.Event()
    service = _RecordingService(stop_after=1, shutdown=shutdown)
    assert build_validator_health_lifespan(None, 1.0) is None
    assert build_validator_health_lifespan(service, None) is None  # type: ignore[arg-type]
    assert build_validator_health_lifespan(service, 0) is None  # type: ignore[arg-type]


async def test_build_validator_health_lifespan_runs_and_stops() -> None:
    shutdown = asyncio.Event()
    service = _RecordingService(stop_after=10_000, shutdown=shutdown)
    lifespan = build_validator_health_lifespan(service, 0.001)  # type: ignore[arg-type]
    assert lifespan is not None
    async with lifespan(object()):  # type: ignore[arg-type]
        await asyncio.sleep(0.02)
    assert service.calls >= 1
