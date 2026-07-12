"""Authenticated challenge raw-weight push ingress tests.

Covers VAL-WEIGHT-011..027/086/087/095/101 and VAL-SDK-017 schema boundaries:
valid push, credential isolation, auth failures, signature binding, exact
idempotence, conflicting duplicates, revision ordering, epoch sealing,
freshness, digests, hotkey-only values, size limits, and media type.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.challenge_sdk.roles import Capability, Role, RoleContractError, activate_role
from base.challenge_sdk.schemas import RawWeightPushRequest
from base.db import (
    Base,
    RawWeightNonce,
    RawWeightSnapshot,
    session_scope,
)
from base.master.app_proxy import create_proxy_app
from base.master.raw_weight_ingress import (
    ChallengeCredentialStore,
    RawWeightIngressService,
    canonical_challenge_push_request,
    sign_challenge_push_request,
)

# Use a wall-clock aligned base so RawWeightPushRequest schema
# (expires_at must be after real UTC now) accepts fixtures.
NOW_EPOCH = datetime.now(UTC).timestamp()
SLUG = "prism"
TOKEN = "challenge-secret-token-prism"
OTHER_SLUG = "challenge-b"
OTHER_TOKEN = "challenge-secret-token-b"
HOTKEY = "5CkeyABC"


class FakeClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, UTC)

    def advance(self, seconds: float) -> None:
        self.epoch += float(seconds)


class FakeRegistry:
    def __init__(self, tokens: dict[str, str]) -> None:
        self._tokens = dict(tokens)

    def get_token(self, slug: str) -> str:
        if slug not in self._tokens:
            raise KeyError(slug)
        return self._tokens[slug]

    def get(self, slug: str) -> Any:
        if slug not in self._tokens:
            raise KeyError(slug)

        class _Record:
            token_hash = hashlib.sha256(self._tokens[slug].encode()).hexdigest()

        return _Record()


class FakeNonceStore:
    async def reserve(self, **_: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


def _build_body(
    *,
    slug: str = SLUG,
    epoch: int = 42,
    revision: int = 1,
    weights: dict[str, float] | None = None,
    nonce: str = "n-1",
    computed_at: datetime | None = None,
    expires_at: datetime | None = None,
    protocol_version: str = "1.0",
    bad_digest: str | None = None,
    include_digest: bool = True,
    extra: dict[str, Any] | None = None,
    clock: FakeClock | None = None,
) -> bytes:
    now = clock.now() if clock is not None else datetime.fromtimestamp(NOW_EPOCH, UTC)
    computed = (computed_at or now).replace(microsecond=0)
    expires = (expires_at or (computed + timedelta(minutes=5))).replace(microsecond=0)
    body: dict[str, Any] = {
        "protocol_version": protocol_version,
        "challenge_slug": slug,
        "epoch": epoch,
        "revision": revision,
        "computed_at": computed.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "nonce": nonce,
        "weights": weights if weights is not None else {HOTKEY: 1.0},
    }
    if extra:
        body.update(extra)
    if include_digest:
        body["payload_digest"] = bad_digest or RawWeightPushRequest.compute_digest(
            {k: v for k, v in body.items() if k != "payload_digest"}
        )
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _signed_headers(
    *,
    path: str,
    body: bytes,
    token: str = TOKEN,
    slug: str = SLUG,
    timestamp: int | None = None,
    clock: FakeClock | None = None,
    method: str = "POST",
    signature: str | None = None,
) -> dict[str, str]:
    if timestamp is None:
        ts = int(clock.time() if clock is not None else NOW_EPOCH)
    else:
        ts = timestamp
    canonical = canonical_challenge_push_request(
        method=method,
        path=path,
        challenge_slug=slug,
        timestamp=str(ts),
        body=body,
    )
    sig = signature or sign_challenge_push_request(token=token, canonical=canonical)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Base-Challenge-Slug": slug,
        "X-Signature": sig,
        "X-Timestamp": str(ts),
    }


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    # File-backed SQLite supports concurrent writers under unique constraints.
    db_path = tmp_path / "raw_weight.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, autoflush=False
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    clock = FakeClock(NOW_EPOCH)
    registry = FakeRegistry({SLUG: TOKEN, OTHER_SLUG: OTHER_TOKEN})
    service = RawWeightIngressService(
        session_factory,
        credential_store=ChallengeCredentialStore(registry),
        now_fn=clock.now,
        max_clock_skew_seconds=30,
        max_future_epoch_ahead=2,
    )
    app = create_proxy_app(
        registry=registry,
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        now_fn=clock.now,
        raw_weight_ingress_service=service,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "session_factory": session_factory,
            "clock": clock,
            "service": service,
            "engine": engine,
        }
    await engine.dispose()


async def _count_snapshots(session_factory: Any, slug: str = SLUG) -> int:
    async with session_scope(session_factory) as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(RawWeightSnapshot)
                    .where(RawWeightSnapshot.challenge_slug == slug)
                )
            ).scalar_one()
        )


async def _selected(
    session_factory: Any, *, slug: str = SLUG, epoch: int = 42
) -> RawWeightSnapshot | None:
    async with session_scope(session_factory) as session:
        return (
            await session.execute(
                select(RawWeightSnapshot).where(
                    RawWeightSnapshot.challenge_slug == slug,
                    RawWeightSnapshot.epoch == epoch,
                    RawWeightSnapshot.is_selected_source.is_(True),
                )
            )
        ).scalar_one_or_none()


@pytest.mark.asyncio
async def test_valid_push_persists_snapshot_and_ack(harness: dict[str, Any]) -> None:
    client: AsyncClient = harness["client"]
    clock: FakeClock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock)
    headers = _signed_headers(path=path, body=body, clock=clock)
    response = await client.post(path, content=body, headers=headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["idempotent"] is False
    assert payload["challenge_slug"] == SLUG
    assert payload["epoch"] == 42
    assert payload["revision"] == 1
    assert payload["payload_digest"] == json.loads(body)["payload_digest"]
    assert payload["snapshot_id"]
    assert await _count_snapshots(session_factory) == 1
    selected = await _selected(session_factory)
    assert selected is not None
    assert selected.revision == 1


@pytest.mark.asyncio
async def test_exact_retry_is_idempotent(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, nonce="exact-1")
    headers = _signed_headers(path=path, body=body, clock=clock)
    first = await client.post(path, content=body, headers=headers)
    second = await client.post(path, content=body, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["snapshot_id"] == second.json()["snapshot_id"]
    assert second.json()["idempotent"] is True
    assert await _count_snapshots(session_factory) == 1
    async with session_scope(session_factory) as session:
        nonces = (
            await session.execute(select(func.count()).select_from(RawWeightNonce))
        ).scalar_one()
    assert int(nonces) == 1


@pytest.mark.asyncio
async def test_concurrent_exact_delivery_single_snapshot(
    harness: dict[str, Any],
) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, nonce="conc-1")
    headers = _signed_headers(path=path, body=body, clock=clock)

    async def one() -> int:
        response = await client.post(path, content=body, headers=headers)
        return response.status_code

    statuses = await asyncio.gather(*[one() for _ in range(10)])
    assert set(statuses) == {200}, statuses
    assert await _count_snapshots(session_factory) == 1


@pytest.mark.asyncio
async def test_conflicting_duplicate_is_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body1 = _build_body(clock=clock, nonce="c1", weights={HOTKEY: 1.0})
    headers1 = _signed_headers(path=path, body=body1, clock=clock)
    first = await client.post(path, content=body1, headers=headers1)
    assert first.status_code == 200
    body2 = _build_body(clock=clock, nonce="c2", weights={HOTKEY: 2.0})
    headers2 = _signed_headers(path=path, body=body2, clock=clock)
    second = await client.post(path, content=body2, headers=headers2)
    assert second.status_code == 409
    assert await _count_snapshots(session_factory) == 1
    selected = await _selected(session_factory)
    assert selected is not None
    assert float(selected.weights[HOTKEY]) == 1.0


@pytest.mark.asyncio
async def test_revision_ordering_and_history(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    for rev, weight in ((1, 1.0), (3, 3.0)):
        body = _build_body(
            clock=clock, revision=rev, nonce=f"r{rev}", weights={HOTKEY: weight}
        )
        headers = _signed_headers(path=path, body=body, clock=clock)
        response = await client.post(path, content=body, headers=headers)
        assert response.status_code == 200
    # Lower revision after higher is history-preserving but not selected.
    body2 = _build_body(clock=clock, revision=2, nonce="r2", weights={HOTKEY: 2.0})
    headers2 = _signed_headers(path=path, body=body2, clock=clock)
    response2 = await client.post(path, content=body2, headers=headers2)
    assert response2.status_code == 200
    assert await _count_snapshots(session_factory) == 3
    selected = await _selected(session_factory)
    assert selected is not None
    assert selected.revision == 3


@pytest.mark.asyncio
async def test_post_seal_revision_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    service: RawWeightIngressService = harness["service"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, revision=1, nonce="seal-1")
    headers = _signed_headers(path=path, body=body, clock=clock)
    response = await client.post(path, content=body, headers=headers)
    assert response.status_code == 200
    caps = (Capability.MASTER_RAW_WEIGHT_INGRESS,)
    with activate_role(Role.MASTER, capabilities=caps):
        await service.seal_epoch(42)
    body2 = _build_body(clock=clock, revision=2, nonce="seal-2")
    headers2 = _signed_headers(path=path, body=body2, clock=clock)
    response = await client.post(path, content=body2, headers=headers2)
    assert response.status_code == 409
    assert await _count_snapshots(session_factory) == 1


@pytest.mark.asyncio
async def test_stale_epoch_and_future_epoch_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    service: RawWeightIngressService = harness["service"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, epoch=42, nonce="e42")
    headers = _signed_headers(path=path, body=body, clock=clock)
    response = await client.post(path, content=body, headers=headers)
    assert response.status_code == 200
    caps = (Capability.MASTER_RAW_WEIGHT_INGRESS,)
    with activate_role(Role.MASTER, capabilities=caps):
        await service.seal_epoch(42)
    stale = _build_body(clock=clock, epoch=41, nonce="e41")
    stale_headers = _signed_headers(path=path, body=stale, clock=clock)
    stale_response = await client.post(path, content=stale, headers=stale_headers)
    assert stale_response.status_code == 409
    future = _build_body(clock=clock, epoch=99, nonce="e99")
    future_headers = _signed_headers(path=path, body=future, clock=clock)
    future_response = await client.post(path, content=future, headers=future_headers)
    assert future_response.status_code == 422
    assert await _count_snapshots(session_factory) == 1


@pytest.mark.asyncio
async def test_credential_isolation_and_cross_slug(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path_a = f"/internal/v1/challenges/{SLUG}/raw-weights"
    path_b = f"/internal/v1/challenges/{OTHER_SLUG}/raw-weights"
    body_a = _build_body(clock=clock, slug=SLUG, nonce="iso-a")
    body_b = _build_body(clock=clock, slug=OTHER_SLUG, nonce="iso-b")
    ok_a = await client.post(
        path_a,
        content=body_a,
        headers=_signed_headers(
            path=path_a,
            body=body_a,
            token=TOKEN,
            slug=SLUG,
            clock=clock,
        ),
    )
    ok_b = await client.post(
        path_b,
        content=body_b,
        headers=_signed_headers(
            path=path_b, body=body_b, token=OTHER_TOKEN, slug=OTHER_SLUG, clock=clock
        ),
    )
    assert ok_a.status_code == 200
    assert ok_b.status_code == 200
    # Prism token on challenge-b route + body
    cross = _build_body(clock=clock, slug=OTHER_SLUG, nonce="iso-cross")
    bad = await client.post(
        path_b,
        content=cross,
        headers=_signed_headers(
            path=path_b, body=cross, token=TOKEN, slug=OTHER_SLUG, clock=clock
        ),
    )
    assert bad.status_code == 401
    assert await _count_snapshots(session_factory, SLUG) == 1
    assert await _count_snapshots(session_factory, OTHER_SLUG) == 1


@pytest.mark.asyncio
async def test_missing_malformed_credentials_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, nonce="auth-1")
    # No auth
    r1 = await client.post(
        path, content=body, headers={"Content-Type": "application/json"}
    )
    assert r1.status_code == 401
    # Bad bearer
    headers = _signed_headers(path=path, body=body, clock=clock)
    headers["Authorization"] = "Bearer wrong-token"
    r2 = await client.post(path, content=body, headers=headers)
    assert r2.status_code == 401
    # X-Role does not authorize
    headers = _signed_headers(path=path, body=body, clock=clock)
    headers.pop("Authorization")
    headers["X-Role"] = "challenge"
    r3 = await client.post(path, content=body, headers=headers)
    assert r3.status_code == 401
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_signature_binds_method_path_body(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, nonce="sig-1")
    # Wrong path in signature domain
    headers = _signed_headers(
        path="/internal/v1/challenges/other/raw-weights", body=body, clock=clock
    )
    r1 = await client.post(path, content=body, headers=headers)
    assert r1.status_code == 401
    # Tampered body after signing
    headers = _signed_headers(path=path, body=body, clock=clock)
    tampered = _build_body(clock=clock, nonce="sig-1", weights={HOTKEY: 9.0})
    r2 = await client.post(path, content=tampered, headers=headers)
    assert r2.status_code == 401
    # Bad signature bits
    headers = _signed_headers(path=path, body=body, clock=clock, signature="deadbeef")
    r3 = await client.post(path, content=body, headers=headers)
    assert r3.status_code == 401
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_freshness_window_enforced(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    # Expired relative to receipt (expires well in the past beyond skew)
    past = clock.now() - timedelta(hours=1)
    body = _build_body(
        clock=clock,
        nonce="fresh-1",
        computed_at=past,
        expires_at=past + timedelta(minutes=1),
    )
    # Schema no longer hard-rejects expires_at vs wall clock; server receipt
    # policy still rejects out-of-window deliveries (422).
    r = await client.post(
        path, content=body, headers=_signed_headers(path=path, body=body, clock=clock)
    )
    assert r.status_code == 422
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_delayed_in_window_delivery_accepts_past_schema_expires(
    harness: dict[str, Any],
) -> None:
    """VAL-WEIGHT-019: receipt policy, not schema dual-policy, is authoritative.

    Payload computed/expires are within 5 minutes of receipt, but absolute
    wall-clock ``expires_at`` may already be in the past relative to a delayed
    client parse. Schema must accept; server receipt-window must accept when
    ``receipt < expires_at + skew``.
    """

    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"

    # expires_at is 10s in the "past" relative to receipt; skew is 30s so still
    # inside ``expires_at + skew``.
    computed = clock.now() - timedelta(seconds=40)
    expires = clock.now() - timedelta(seconds=10)
    body = _build_body(
        clock=clock,
        nonce="delayed-window-1",
        computed_at=computed,
        expires_at=expires,
    )
    # Schema itself must parse (no wall-clock future gate).
    payload = RawWeightPushRequest.model_validate_json(body.decode())
    assert payload.expires_at < datetime.now(UTC)

    r = await client.post(
        path, content=body, headers=_signed_headers(path=path, body=body, clock=clock)
    )
    assert r.status_code == 200, r.text
    assert await _count_snapshots(session_factory) == 1
    selected = await _selected(session_factory)
    assert selected is not None
    assert selected.nonce == "delayed-window-1"


@pytest.mark.asyncio
async def test_concurrent_multi_revision_highest_wins(
    harness: dict[str, Any],
) -> None:
    """VAL-WEIGHT-017/095: concurrent revisions converge on max revision selected."""

    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"

    async def push(rev: int) -> int:
        body = _build_body(
            clock=clock,
            revision=rev,
            nonce=f"race-r{rev}",
            weights={HOTKEY: float(rev)},
        )
        headers = _signed_headers(path=path, body=body, clock=clock)
        response = await client.post(path, content=body, headers=headers)
        return response.status_code

    # Race lower and higher revisions; after all complete the selected source
    # must be the highest revision even if lower commits last on some schedules.
    statuses = await asyncio.gather(*[push(rev) for rev in (1, 5, 3, 4, 2)])
    assert set(statuses) == {200}, statuses
    assert await _count_snapshots(session_factory) == 5
    selected = await _selected(session_factory)
    assert selected is not None
    assert selected.revision == 5
    assert float(selected.weights[HOTKEY]) == 5.0

    # A later higher revision still wins selection.
    body = _build_body(clock=clock, revision=6, nonce="race-r6", weights={HOTKEY: 6.0})
    r6 = await client.post(
        path, content=body, headers=_signed_headers(path=path, body=body, clock=clock)
    )
    assert r6.status_code == 200
    selected = await _selected(session_factory)
    assert selected is not None
    assert selected.revision == 6


@pytest.mark.asyncio
async def test_digest_required_and_verified(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, nonce="dig-1", bad_digest="0" * 64)
    r = await client.post(
        path, content=body, headers=_signed_headers(path=path, body=body, clock=clock)
    )
    assert r.status_code == 422
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_uid_keys_and_negative_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    # UID-only key
    body1 = _build_body(clock=clock, nonce="uid-1", weights={"123": 1.0})
    # May fail body construction on digest/schema before post; force raw payload.
    computed = clock.now().replace(microsecond=0)
    expires = computed + timedelta(minutes=5)
    raw = {
        "protocol_version": "1.0",
        "challenge_slug": SLUG,
        "epoch": 42,
        "revision": 1,
        "computed_at": computed.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "nonce": "uid-1",
        "weights": {"123": 1.0},
    }
    raw["payload_digest"] = RawWeightPushRequest.compute_digest(
        {k: v for k, v in raw.items() if k != "payload_digest"}
    )
    body1 = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    r1 = await client.post(
        path, content=body1, headers=_signed_headers(path=path, body=body1, clock=clock)
    )
    assert r1.status_code == 422
    # Negative weight is rejected by the strict schema.
    with pytest.raises(ValidationError):
        RawWeightPushRequest.model_validate(
            {
                **raw,
                "nonce": "neg-1",
                "weights": {HOTKEY: -1.0},
                "payload_digest": "0" * 64,
            }
        )
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_unknown_fields_and_media_type_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, nonce="uf-1", extra={"unknown": True})
    r1 = await client.post(
        path, content=body, headers=_signed_headers(path=path, body=body, clock=clock)
    )
    assert r1.status_code == 422
    good = _build_body(clock=clock, nonce="uf-2")
    headers = _signed_headers(path=path, body=good, clock=clock)
    headers["Content-Type"] = "text/plain"
    r2 = await client.post(path, content=good, headers=headers)
    assert r2.status_code == 415
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_payload_too_large_rejected(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    session_factory = harness["session_factory"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    # Over body byte limit
    huge_key = "A" + ("b" * 64)
    weights = {f"{huge_key}{i}": 1.0 for i in range(1)}
    body = _build_body(clock=clock, nonce="big-1", weights=weights)
    # Force oversized body
    oversized = body + (b" " * 300_000)
    headers = _signed_headers(path=path, body=oversized, clock=clock)
    r = await client.post(path, content=oversized, headers=headers)
    assert r.status_code in {413, 401, 422}
    assert await _count_snapshots(session_factory) == 0


@pytest.mark.asyncio
async def test_role_contract_requires_master(harness: dict[str, Any]) -> None:
    service: RawWeightIngressService = harness["service"]
    clock: FakeClock = harness["clock"]
    body = _build_body(clock=clock, nonce="role-1")
    with activate_role(Role.VALIDATOR):
        with pytest.raises(RoleContractError):
            await service.accept_push(
                route_slug=SLUG,
                method="POST",
                path=f"/internal/v1/challenges/{SLUG}/raw-weights",
                authorization=f"Bearer {TOKEN}",
                content_type="application/json",
                raw_body=body,
                signature="x",
                timestamp_header=str(int(clock.time())),
                challenge_slug_header=SLUG,
            )


@pytest.mark.asyncio
async def test_slug_body_route_mismatch_forbidden(harness: dict[str, Any]) -> None:
    client = harness["client"]
    clock = harness["clock"]
    path = f"/internal/v1/challenges/{SLUG}/raw-weights"
    body = _build_body(clock=clock, slug=OTHER_SLUG, nonce="mis-1")
    headers = _signed_headers(path=path, body=body, token=TOKEN, slug=SLUG, clock=clock)
    r = await client.post(path, content=body, headers=headers)
    assert r.status_code == 403
