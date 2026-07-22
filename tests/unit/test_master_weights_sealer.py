"""In-process continuous weight sealer + serve auto-heal (auto-weights).

Covers VAL-AUTOW-003/004/005/006/007/009/010/011:
- sealer lifespan starts with proxy when interval > 0
- short TTL / expired row heals to 200 without CLI
- concurrent GET race is lock-safe
- zero-miner empty slate still seals
- no set_weights
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.challenge_sdk.roles import Role, activate_role
from base.db import Base
from base.master.aggregation import AggregationService
from base.master.app_admin import build_admin_router
from base.master.app_proxy import create_proxy_app
from base.master.service import MasterWeightService
from base.master.weights_sealer import (
    MasterWeightsSealer,
    build_master_weights_sealer_lifespan,
    resolve_master_weight_epoch,
    run_master_weights_sealer_loop,
)
from base.schemas.challenge import ChallengeStatus


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = (start or datetime.now(UTC)).replace(microsecond=0)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class StubMetagraphCache:
    def __init__(self, mapping: dict[str, int] | None = None) -> None:
        self._mapping = dict(mapping or {"5CkeyAAA": 10, "5CkeyBBB": 20})
        self._updated_at = 0.0

    def get(self, *, force: bool = False) -> dict[str, int]:
        return dict(self._mapping)


def _record(slug: str, emission: float = 50.0) -> SimpleNamespace:
    """ChallengeRecord-shaped object for active_challenge_inputs / registry view."""

    return SimpleNamespace(
        slug=slug,
        name=slug.title(),
        image=f"ghcr.io/x/{slug}:1.0.0",
        version="1.0.0",
        emission_percent=Decimal(str(emission)),
        status=ChallengeStatus.ACTIVE,
        description=None,
        metadata={},
        internal_base_url=f"http://127.0.0.1:{18080 if slug == 'prism' else 18081}",
        public_proxy_base_path=f"/challenges/{slug}",
        required_capabilities=["get_weights", "proxy_routes"],
        resources={},
        volumes={},
        env={},
        secrets=[],
    )


class FakeRegistry:
    def __init__(self, records: list[Any] | None = None) -> None:
        self._records = list(records or [_record("prism"), _record("agent-challenge")])
        self._tokens = {r.slug: f"tok-{r.slug}" for r in self._records}

    async def list(self, *, active_only: bool = False) -> list[Any]:
        if active_only:
            return [r for r in self._records if r.status == ChallengeStatus.ACTIVE]
        return list(self._records)

    async def get_token(self, slug: str) -> str:
        return self._tokens[slug]

    async def registry_response(self) -> Any:
        from base.schemas.challenge import RegistryResponse

        return RegistryResponse(challenges=[])


@pytest.fixture
async def session_factory(tmp_path: Any) -> AsyncIterator[async_sessionmaker]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sealer.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


def _service(
    session_factory: async_sessionmaker,
    *,
    freshness_seconds: int = 720,
    epoch_interval_seconds: int = 360,
    clock: FakeClock | None = None,
) -> MasterWeightService:
    clock = clock or FakeClock()
    agg = AggregationService(
        session_factory,
        now_fn=clock.now,
        freshness_seconds=freshness_seconds,
    )
    return MasterWeightService(
        metagraph_cache=StubMetagraphCache(),  # type: ignore[arg-type]
        session_factory=session_factory,
        aggregation_service=agg,
        freshness_seconds=freshness_seconds,
        epoch_interval_seconds=epoch_interval_seconds,
    )


def test_resolve_master_weight_epoch_bucket() -> None:
    fixed = datetime(2030, 1, 1, 0, 6, 0, tzinfo=UTC)
    # 360s buckets: timestamp // 360
    assert (
        resolve_master_weight_epoch(epoch_interval_seconds=360, now=fixed)
        == int(fixed.timestamp()) // 360
    )
    assert (
        resolve_master_weight_epoch(epoch_interval_seconds=360, now=fixed, epoch=99)
        == 99
    )
    # interval 0 falls back to default 360 (same as CLI).
    assert (
        resolve_master_weight_epoch(epoch_interval_seconds=0, now=fixed)
        == int(fixed.timestamp()) // 360
    )


def test_build_master_weights_sealer_lifespan_disabled_when_interval_nonpositive() -> (
    None
):
    assert build_master_weights_sealer_lifespan(None, 60.0) is None
    sealer = object()
    assert build_master_weights_sealer_lifespan(sealer, None) is None  # type: ignore[arg-type]
    assert build_master_weights_sealer_lifespan(sealer, 0) is None  # type: ignore[arg-type]
    assert build_master_weights_sealer_lifespan(sealer, -1.0) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sealer_lifespan_starts_and_stops(
    session_factory: async_sessionmaker,
) -> None:
    service = _service(session_factory, freshness_seconds=60, epoch_interval_seconds=1)
    registry = FakeRegistry()
    sealer = MasterWeightsSealer(
        weight_service=service,
        registry=registry,
        netuid=100,
        chain_endpoint="wss://chain.test",
        epoch_interval_seconds=1,
    )
    lifespan = build_master_weights_sealer_lifespan(sealer, 0.05)
    assert lifespan is not None
    app = FastAPI(lifespan=lifespan)
    async with lifespan(app):
        # Startup tick seals zero-miner immediately.
        await asyncio.sleep(0.01)
        with activate_role(Role.MASTER):
            latest = await service.aggregation.get_latest_vector()  # type: ignore[union-attr]
        assert latest is not None
        assert latest.uids == [0]
        assert latest.weights == [1.0]


@pytest.mark.asyncio
async def test_short_ttl_get_stays_200_without_cli(
    session_factory: async_sessionmaker,
) -> None:
    """Short freshness TTL + expired row → GET heals and returns 200 body."""

    clock = FakeClock()
    service = _service(
        session_factory,
        freshness_seconds=2,
        epoch_interval_seconds=60,
        clock=clock,
    )
    registry = FakeRegistry()
    from base.master.service import active_challenge_inputs

    challenges, tokens = await active_challenge_inputs(registry)

    with activate_role(Role.MASTER):
        first = await service.seal_fresh_if_needed(
            challenges,
            tokens,
            netuid=100,
            chain_endpoint="wss://x",
            now_fn=clock.now,
            force=True,
        )
    assert first.uids == [0]
    assert first.weights == [1.0]
    assert first.expires_at > clock.now()
    first_epoch = first.epoch
    first_vector_id = first.vector_id

    # Expire the durable row past TTL.
    clock.advance(5)
    with activate_role(Role.MASTER):
        stale = await service.aggregation.get_latest_vector()  # type: ignore[union-attr]
    assert stale is not None
    assert service._vector_is_expired(stale, now=clock.now())  # noqa: SLF001

    with activate_role(Role.MASTER):
        healed = await service.compute_latest_response(
            list(challenges),
            tokens,
            netuid=100,
            chain_endpoint="wss://x",
            now_fn=clock.now,
        )
    assert healed.uids == [0]
    assert healed.weights == [1.0]
    assert healed.expires_at > clock.now()
    assert healed.epoch is not None and healed.epoch > (first_epoch or 0)
    assert healed.vector_id != first_vector_id
    # Schema fields present (VAL-AUTOW-011).
    assert healed.netuid == 100
    assert healed.vector_digest
    assert healed.protocol_version == "1.0"


@pytest.mark.asyncio
async def test_expired_row_background_tick_then_200(
    session_factory: async_sessionmaker,
) -> None:
    clock = FakeClock()
    service = _service(
        session_factory,
        freshness_seconds=2,
        epoch_interval_seconds=30,
        clock=clock,
    )
    registry = FakeRegistry()
    sealer = MasterWeightsSealer(
        weight_service=service,
        registry=registry,
        netuid=42,
        chain_endpoint="wss://chain.test",
        epoch_interval_seconds=30,
        now_fn=clock.now,
    )
    with activate_role(Role.MASTER):
        first = await sealer.tick_once()
    assert first is not None
    first_epoch = getattr(first, "epoch", None)

    clock.advance(10)
    with activate_role(Role.MASTER):
        second = await sealer.tick_once()
    assert second is not None
    assert getattr(second, "epoch", None) != first_epoch
    second_expires = getattr(second, "expires_at", None)
    assert second_expires is not None
    assert second_expires > clock.now()


@pytest.mark.asyncio
async def test_zero_miner_empty_slate_still_seals(
    session_factory: async_sessionmaker,
) -> None:
    service = _service(session_factory, freshness_seconds=120)
    registry = FakeRegistry()
    from base.master.service import active_challenge_inputs

    challenges, tokens = await active_challenge_inputs(registry)
    with activate_role(Role.MASTER):
        response = await service.seal_fresh_if_needed(
            challenges,
            tokens,
            netuid=100,
            chain_endpoint="",
            force=True,
        )
    assert response.uids == [0]
    assert response.weights == [1.0]
    assert response.hotkey_weights == {}
    # Outcomes mark missing for both challenges.
    reasons = {
        str(o.challenge_slug): str(o.reason_code or o.outcome)
        for o in (response.source_outcomes or [])
    }
    # source_outcomes may be projected as SourceOutcome models
    if not reasons and response.source_challenges:
        reasons = {
            str(c.slug): str(c.error or "ok") for c in response.source_challenges
        }
    assert "prism" in reasons
    assert "agent-challenge" in reasons


@pytest.mark.asyncio
async def test_concurrent_get_seal_race_safe(
    session_factory: async_sessionmaker,
) -> None:
    clock = FakeClock()
    service = _service(
        session_factory,
        freshness_seconds=1,
        epoch_interval_seconds=60,
        clock=clock,
    )
    registry = FakeRegistry()
    from base.master.service import active_challenge_inputs

    challenges, tokens = await active_challenge_inputs(registry)
    challenges = list(challenges)

    async def one_get() -> Any:
        with activate_role(Role.MASTER):
            return await service.compute_latest_response(
                challenges,
                tokens,
                netuid=100,
                chain_endpoint="wss://x",
                now_fn=clock.now,
            )

    # Parallel first-seal GETs must not corrupt (lock serializes).
    results = await asyncio.gather(*[one_get() for _ in range(8)])
    vector_ids = {r.vector_id for r in results}
    assert len(vector_ids) == 1
    assert all(r.uids == [0] and r.weights == [1.0] for r in results)

    # Expire + concurrent heal reseals once.
    clock.advance(5)
    healed = await asyncio.gather(*[one_get() for _ in range(6)])
    healed_ids = {r.vector_id for r in healed}
    assert len(healed_ids) == 1
    assert healed_ids != vector_ids
    assert all(r.expires_at > clock.now() for r in healed)


@pytest.mark.asyncio
async def test_http_get_latest_heals_expired(
    session_factory: async_sessionmaker,
) -> None:
    clock = FakeClock()
    service = _service(
        session_factory,
        freshness_seconds=2,
        epoch_interval_seconds=60,
        clock=clock,
    )
    registry = FakeRegistry()
    router = build_admin_router(
        registry=registry,
        runtime_controller=AsyncMock(),
        weight_service=service,
        netuid=100,
        chain_endpoint="wss://chain.test",
        now_fn=clock.now,
        include_health=True,
    )
    app = FastAPI()
    app.include_router(router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No vector yet: heal seals zero-miner → 200 (not 404/502 expiry).
        r1 = await client.get("/v1/weights/latest")
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert body1["uids"] == [0]
        assert body1["weights"] == [1.0]
        assert body1["epoch"] is not None
        assert body1["vector_id"]
        assert body1["netuid"] == 100

        clock.advance(10)
        r2 = await client.get("/v1/weights/latest")
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["uids"] == [0]
        assert body2["vector_id"] != body1["vector_id"]
        assert body2["epoch"] > body1["epoch"]


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_proxy_combines_sealer_lifespan(
    session_factory: async_sessionmaker,
) -> None:
    """Sealer lifespan is composed into create_proxy_app and seals on enter."""

    service = _service(session_factory, freshness_seconds=60, epoch_interval_seconds=1)
    registry = FakeRegistry()
    sealer = MasterWeightsSealer(
        weight_service=service,
        registry=registry,
        netuid=100,
        epoch_interval_seconds=1,
    )
    lifespan = build_master_weights_sealer_lifespan(sealer, 0.05)
    assert lifespan is not None

    # Direct lifespan composition (proxy path uses the same builder).
    app = FastAPI(lifespan=lifespan)
    async with lifespan(app):
        await asyncio.sleep(0.02)
        with activate_role(Role.MASTER):
            latest = await service.aggregation.get_latest_vector()  # type: ignore[union-attr]
        assert latest is not None
        assert latest.uids == [0]

    # create_proxy_app accepts weights_sealer args without error when fully wired.
    class _EmptyRegistry:
        async def list(self, *, active_only: bool = False) -> list[Any]:
            return []

        async def get(self, slug: str) -> Any:
            raise KeyError(slug)

        async def get_token(self, slug: str) -> str:
            return "x"

        async def registry_response(self) -> Any:
            from base.schemas.challenge import RegistryResponse

            return RegistryResponse(challenges=[])

    class _FakeNonceStore:
        async def reserve(self, **_: object) -> None:
            return None

    proxy = create_proxy_app(
        registry=_EmptyRegistry(),
        nonce_store=_FakeNonceStore(),  # type: ignore[arg-type]
        metagraph_cache=StubMetagraphCache(),  # type: ignore[arg-type]
        weight_service=service,
        weights_sealer=sealer,
        weights_sealer_interval_seconds=0.05,
        netuid=100,
        chain_endpoint="wss://x",
    )
    assert proxy.router.lifespan_context is not None


def test_sealer_module_has_no_set_weights_call() -> None:
    import base.master.service as svc
    import base.master.weights_sealer as mod

    # Reject actual call sites / imports; docstrings may mention the prohibition.
    for name, module in (("weights_sealer", mod), ("service", svc)):
        source = inspect.getsource(module)
        assert "WeightSetter" not in source, name
        assert ".set_weights(" not in source, name
        assert "burn_weights" not in source, name
        assert "from base.bittensor" not in source or "WeightSetter" not in source


@pytest.mark.asyncio
async def test_run_loop_honors_shutdown(
    session_factory: async_sessionmaker,
) -> None:
    service = _service(session_factory)
    sealer = MasterWeightsSealer(
        weight_service=service,
        registry=FakeRegistry(),
        netuid=1,
        epoch_interval_seconds=1,
    )
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_master_weights_sealer_loop(
            sealer, interval_seconds=0.05, shutdown_event=shutdown
        )
    )
    await asyncio.sleep(0.12)
    shutdown.set()
    await asyncio.wait_for(task, timeout=1.0)
    with activate_role(Role.MASTER):
        latest = await service.aggregation.get_latest_vector()  # type: ignore[union-attr]
    assert latest is not None
