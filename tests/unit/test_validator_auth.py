from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import Base, ValidatorRequestNonce
from base.db.session import create_engine, create_session_factory
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorIdentity,
    ValidatorSignedRequestVerifier,
    build_validator_auth_dependency,
    canonical_validator_request,
)

NOW_EPOCH = 1_750_000_000.0


def _sign(hotkey: str, canonical: str) -> str:
    return hashlib.sha256(f"{hotkey}:{canonical}".encode()).hexdigest()


def _verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message.decode())


def signed_headers(
    *,
    method: str = "POST",
    path: str = "/protected",
    query: str = "",
    body: bytes = b"{}",
    hotkey: str = "permitted",
    nonce: str = "nonce-1",
    timestamp: str | None = None,
    signature: str | None = None,
    sign_method: str | None = None,
    sign_query: str | None = None,
) -> dict[str, str]:
    ts = timestamp if timestamp is not None else str(int(NOW_EPOCH))
    canonical = canonical_validator_request(
        method=sign_method or method,
        path=path,
        query_string=sign_query if sign_query is not None else query,
        timestamp=ts,
        nonce=nonce,
        body=body,
    )
    sig = signature if signature is not None else _sign(hotkey, canonical)
    return {
        "X-Hotkey": hotkey,
        "X-Signature": sig,
        "X-Nonce": nonce,
        "X-Timestamp": ts,
    }


@pytest.fixture
async def auth_client() -> AsyncIterator[tuple[AsyncClient, object]]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        ["permitted", "no_permit"],
        validator_permits=[True, False],
        stakes=[100.0, 5.0],
    )
    eligibility = MetagraphValidatorEligibility(cache)
    nonce_store = SqlAlchemyValidatorNonceStore(session_factory)
    verifier = ValidatorSignedRequestVerifier(
        nonce_store=nonce_store,
        eligibility=eligibility,
        signature_verifier=_verifier,
        ttl_seconds=300,
        now_fn=lambda: NOW_EPOCH,
    )
    dependency = build_validator_auth_dependency(verifier)

    app = FastAPI()

    @app.post("/protected")
    async def protected(
        identity: ValidatorIdentity = Depends(dependency),
    ) -> dict[str, object]:
        return {"hotkey": identity.hotkey, "uid": identity.uid, "nonce": identity.nonce}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory
    await engine.dispose()


async def _nonce_count(session_factory) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count(ValidatorRequestNonce.id)))


# VAL-VREG-007: permitted hotkey + valid signature is accepted.
async def test_permitted_hotkey_valid_signature_accepted(auth_client) -> None:
    client, session_factory = auth_client
    response = await client.post("/protected", content=b"{}", headers=signed_headers())
    assert response.status_code == 200
    assert response.json() == {"hotkey": "permitted", "uid": 0, "nonce": "nonce-1"}
    assert await _nonce_count(session_factory) == 1


# VAL-VREG-005: hotkey absent from the metagraph -> 403, no nonce reserved.
async def test_absent_hotkey_rejected_403(auth_client) -> None:
    client, session_factory = auth_client
    response = await client.post(
        "/protected",
        content=b"{}",
        headers=signed_headers(hotkey="absent"),
    )
    assert response.status_code == 403
    assert await _nonce_count(session_factory) == 0


# VAL-VREG-006: on metagraph but no validator permit -> 403.
async def test_unpermitted_hotkey_rejected_403(auth_client) -> None:
    client, session_factory = auth_client
    response = await client.post(
        "/protected",
        content=b"{}",
        headers=signed_headers(hotkey="no_permit"),
    )
    assert response.status_code == 403
    assert await _nonce_count(session_factory) == 0


# VAL-VREG-008: missing any signing header -> 401, no nonce reserved.
@pytest.mark.parametrize(
    "missing", ["X-Hotkey", "X-Signature", "X-Nonce", "X-Timestamp"]
)
async def test_missing_header_rejected_401(auth_client, missing: str) -> None:
    client, session_factory = auth_client
    headers = signed_headers()
    headers.pop(missing)
    response = await client.post("/protected", content=b"{}", headers=headers)
    assert response.status_code == 401
    assert await _nonce_count(session_factory) == 0


# VAL-VREG-009: forged signature -> 401, nonce NOT consumed (reusable later).
async def test_forged_signature_rejected_401(auth_client) -> None:
    client, session_factory = auth_client
    forged = await client.post(
        "/protected",
        content=b"{}",
        headers=signed_headers(signature="deadbeef"),
    )
    assert forged.status_code == 401
    assert await _nonce_count(session_factory) == 0

    valid = await client.post(
        "/protected", content=b"{}", headers=signed_headers(nonce="nonce-1")
    )
    assert valid.status_code == 200


# VAL-VREG-010: tampered body (body-hash mismatch) -> 401.
async def test_tampered_body_rejected_401(auth_client) -> None:
    client, session_factory = auth_client
    headers = signed_headers(body=b'{"a":1}')
    response = await client.post("/protected", content=b'{"a":2}', headers=headers)
    assert response.status_code == 401
    assert await _nonce_count(session_factory) == 0


# VAL-VREG-011: timestamp skew boundary (>300s rejected, <=300s accepted).
@pytest.mark.parametrize(
    ("offset", "expected"),
    [(-301, 401), (301, 401), (-299, 200), (299, 200)],
)
async def test_timestamp_skew_boundary(auth_client, offset: int, expected: int) -> None:
    client, _ = auth_client
    ts = str(int(NOW_EPOCH) + offset)
    response = await client.post(
        "/protected",
        content=b"{}",
        headers=signed_headers(timestamp=ts, nonce=f"skew-{offset}"),
    )
    assert response.status_code == expected


async def test_non_numeric_timestamp_rejected_401(auth_client) -> None:
    client, _ = auth_client
    for bad in ("not-a-number", "inf", "1e999"):
        response = await client.post(
            "/protected",
            content=b"{}",
            headers=signed_headers(timestamp=bad, nonce=f"ts-{bad}"),
        )
        assert response.status_code == 401


# VAL-VREG-012 / VAL-WEIGHT-003: exact body+nonce retry is idempotent; changed
# bytes under the same nonce remain a conflict. A fresh nonce always succeeds.
async def test_replayed_nonce_rejected(auth_client) -> None:
    client, session_factory = auth_client
    first = await client.post(
        "/protected", content=b"{}", headers=signed_headers(nonce="replay")
    )
    exact = await client.post(
        "/protected", content=b"{}", headers=signed_headers(nonce="replay")
    )
    assert first.status_code == 200
    assert exact.status_code == 200
    assert await _nonce_count(session_factory) == 1

    conflict_body = b'{"x":1}'
    conflict = await client.post(
        "/protected",
        content=conflict_body,
        headers=signed_headers(nonce="replay", body=conflict_body),
    )
    assert conflict.status_code == 409
    assert await _nonce_count(session_factory) == 1

    fresh = await client.post(
        "/protected", content=b"{}", headers=signed_headers(nonce="fresh")
    )
    assert fresh.status_code == 200
    assert await _nonce_count(session_factory) == 2


# VAL-VREG-013: canonical string binds method/path; query order is normalized.
async def test_canonical_binds_method_rejects_mismatch(auth_client) -> None:
    client, _ = auth_client
    response = await client.post(
        "/protected",
        content=b"{}",
        headers=signed_headers(sign_method="GET", nonce="method-bind"),
    )
    assert response.status_code == 401


async def test_canonical_query_order_normalized_accepted(auth_client) -> None:
    client, _ = auth_client
    response = await client.post(
        "/protected?b=2&a=1",
        content=b"{}",
        headers=signed_headers(
            path="/protected",
            query="a=1&b=2",
            sign_query="b=2&a=1",
            nonce="query-order",
        ),
    )
    assert response.status_code == 200
