"""Tests for the signed per-validator challenge subscription endpoint.

Covers VAL-VDIR-SUB-001/002/003: a metagraph-eligible validator sets its
challenge subscriptions via a signed ``POST /v1/validators/subscriptions``
(persisted + returned); unknown/inactive slugs are rejected (422) without
mutating the stored set; and unsigned/forged/stale/replayed/ineligible requests
are rejected (401/403/409).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import Base, Validator
from base.db.session import create_engine, create_session_factory
from base.master.app_proxy import create_proxy_app
from base.master.validator_coordination import ValidatorCoordinationService
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
    canonical_validator_request,
)

NOW_EPOCH = 1_750_000_000.0
ADMIN_TOKEN = "admin-secret-token"
SUBS_PATH = "/v1/validators/subscriptions"


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


class FakeRegistry:
    """Minimal challenge registry exposing an async ``list(active_only=...)``."""

    def __init__(
        self,
        active: tuple[str, ...] = ("agent-challenge", "prism"),
        inactive: tuple[str, ...] = ("retired-challenge",),
    ) -> None:
        self.active = list(active)
        self.inactive = list(inactive)

    async def list(self, *, active_only: bool = False) -> list[Any]:
        slugs = self.active if active_only else [*self.active, *self.inactive]
        return [SimpleNamespace(slug=slug) for slug in slugs]


def _sign(hotkey: str, canonical: str) -> str:
    return hashlib.sha256(f"{hotkey}:{canonical}".encode()).hexdigest()


def _verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message.decode())


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

    def _next_nonce(self, prefix: str = "sub") -> str:
        self._nonce += 1
        return f"{prefix}-{self._nonce}"

    async def post_subscriptions(
        self,
        *,
        slugs: list[str],
        hotkey: str = "permitted",
        nonce: str | None = None,
        timestamp: float | None = None,
        signed: bool = True,
        signature: str | None = None,
    ):
        body = json.dumps({"slugs": slugs}, separators=(",", ":")).encode()
        ts = str(int(self.clock.time() if timestamp is None else timestamp))
        nonce = nonce or self._next_nonce()
        if not signed:
            return await self.client.post(
                SUBS_PATH, content=body, headers={"Content-Type": "application/json"}
            )
        canonical = canonical_validator_request(
            method="POST",
            path=SUBS_PATH,
            query_string="",
            timestamp=ts,
            nonce=nonce,
            body=body,
        )
        sig = signature if signature is not None else _sign(hotkey, canonical)
        headers = {
            "X-Hotkey": hotkey,
            "X-Signature": sig,
            "X-Nonce": nonce,
            "X-Timestamp": ts,
            "Content-Type": "application/json",
        }
        return await self.client.post(SUBS_PATH, content=body, headers=headers)

    async def stored_subscriptions(self, hotkey: str = "permitted") -> list[str] | None:
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one_or_none()
            return None if row is None else list(row.subscriptions)


async def _build_harness() -> tuple[_Harness, Any]:
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
    service = ValidatorCoordinationService(session_factory, now_fn=clock.now)
    app = create_proxy_app(
        registry=FakeRegistry(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        validator_service=service,
        validator_verifier=verifier,
        admin_token_provider=lambda: ADMIN_TOKEN,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return _Harness(client, session_factory, clock, service), engine


@pytest.fixture
async def harness() -> AsyncIterator[_Harness]:
    h, engine = await _build_harness()
    try:
        yield h
    finally:
        await h.client.aclose()
        await engine.dispose()


async def _register(harness: _Harness, hotkey: str = "permitted") -> None:
    await harness.service.register(
        hotkey=hotkey, uid=1, capabilities=["cpu"], version="1.0.0"
    )


# VAL-VDIR-SUB-001
async def test_signed_subscription_persists_and_is_returned(harness: _Harness) -> None:
    await _register(harness)
    response = await harness.post_subscriptions(slugs=["agent-challenge", "prism"])

    assert response.status_code == 200
    body = response.json()
    assert body["subscriptions"] == ["agent-challenge", "prism"]
    assert body["validator"]["subscriptions"] == ["agent-challenge", "prism"]
    assert body["validator"]["hotkey"] == "permitted"
    assert await harness.stored_subscriptions() == ["agent-challenge", "prism"]


# VAL-VDIR-SUB-001 (empty clears the subscription)
async def test_empty_subscription_clears_set(harness: _Harness) -> None:
    await _register(harness)
    await harness.post_subscriptions(slugs=["prism"])
    assert await harness.stored_subscriptions() == ["prism"]

    response = await harness.post_subscriptions(slugs=[])
    assert response.status_code == 200
    assert response.json()["subscriptions"] == []
    assert await harness.stored_subscriptions() == []


# VAL-VDIR-SUB-001 (duplicate slugs are de-duplicated, order preserved)
async def test_subscription_is_deduped_order_preserved(harness: _Harness) -> None:
    await _register(harness)
    response = await harness.post_subscriptions(
        slugs=["prism", "prism", "agent-challenge"]
    )
    assert response.status_code == 200
    assert await harness.stored_subscriptions() == ["prism", "agent-challenge"]


# VAL-VDIR-SUB-002
async def test_unknown_slug_rejected_422_without_mutation(harness: _Harness) -> None:
    await _register(harness)
    await harness.post_subscriptions(slugs=["prism"])
    assert await harness.stored_subscriptions() == ["prism"]

    response = await harness.post_subscriptions(slugs=["agent-challenge", "nope"])
    assert response.status_code == 422
    # the stored set is unchanged (no partial mutation)
    assert await harness.stored_subscriptions() == ["prism"]


# VAL-VDIR-SUB-002 (a known-but-inactive slug is rejected too)
async def test_inactive_slug_rejected_422(harness: _Harness) -> None:
    await _register(harness)
    response = await harness.post_subscriptions(slugs=["retired-challenge"])
    assert response.status_code == 422
    assert await harness.stored_subscriptions() == []


async def test_subscription_unknown_hotkey_returns_404(harness: _Harness) -> None:
    # Eligible + signed, but never registered -> no validators row.
    response = await harness.post_subscriptions(slugs=["prism"])
    assert response.status_code == 404
    assert await harness.stored_subscriptions() is None


# VAL-VDIR-SUB-003
async def test_unsigned_request_rejected_401(harness: _Harness) -> None:
    await _register(harness)
    response = await harness.post_subscriptions(slugs=["prism"], signed=False)
    assert response.status_code == 401
    assert await harness.stored_subscriptions() == []


# VAL-VDIR-SUB-003
async def test_forged_signature_rejected_401(harness: _Harness) -> None:
    await _register(harness)
    response = await harness.post_subscriptions(slugs=["prism"], signature="0xdeadbeef")
    assert response.status_code == 401
    assert await harness.stored_subscriptions() == []


# VAL-VDIR-SUB-003
async def test_stale_timestamp_rejected_401(harness: _Harness) -> None:
    await _register(harness)
    response = await harness.post_subscriptions(
        slugs=["prism"], timestamp=NOW_EPOCH - 1000
    )
    assert response.status_code == 401
    assert await harness.stored_subscriptions() == []


# VAL-VDIR-SUB-003
async def test_replayed_nonce_rejected_409(harness: _Harness) -> None:
    await _register(harness)
    first = await harness.post_subscriptions(slugs=["prism"], nonce="reused-nonce")
    assert first.status_code == 200
    second = await harness.post_subscriptions(
        slugs=["agent-challenge"], nonce="reused-nonce"
    )
    assert second.status_code == 409
    # the replay did not overwrite the first (committed) set
    assert await harness.stored_subscriptions() == ["prism"]


# VAL-VDIR-SUB-003
async def test_ineligible_hotkey_rejected_403(harness: _Harness) -> None:
    response = await harness.post_subscriptions(slugs=["prism"], hotkey="outsider")
    assert response.status_code == 403
