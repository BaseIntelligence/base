from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent_challenge.db import Base, database
from agent_challenge.models import RequestNonce
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.security import (
    SignedRequestAuth,
    body_sha256,
    build_owner_signed_auth_dependency,
    build_signed_auth_dependency,
    canonical_request_string,
)

NOW = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)


@pytest.fixture
async def signed_auth_client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    calls: list[tuple[str, str, str]] = []

    async def session_override() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    def verifier(hotkey: str, message: str, signature: str) -> bool:
        calls.append((hotkey, message, signature))
        return signature == "valid-signature"

    settings = ChallengeSettings(
        owner_hotkey="owner-hotkey",
        signing_ttl_seconds=300,
        shared_token="test-token",
    )
    app = FastAPI()

    signed_auth = build_signed_auth_dependency(
        settings,
        verifier=verifier,
        now_provider=lambda: NOW,
    )
    owner_auth = build_owner_signed_auth_dependency(
        settings,
        verifier=verifier,
        now_provider=lambda: NOW,
    )

    @app.post("/signed")
    async def signed_endpoint(auth: SignedRequestAuth = Depends(signed_auth)) -> dict[str, str]:
        return {
            "hotkey": auth.hotkey,
            "canonical_request": auth.canonical_request,
            "body_sha256": auth.body_sha256,
        }

    @app.post("/owner")
    async def owner_endpoint(auth: SignedRequestAuth = Depends(owner_auth)) -> dict[str, str]:
        return {"hotkey": auth.hotkey}

    app.dependency_overrides[database.session_dependency] = session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory, calls
    await engine.dispose()


def signed_headers(
    *,
    hotkey: str = "miner-hotkey",
    nonce: str = "nonce-1",
    timestamp: str | None = None,
    signature: str = "valid-signature",
) -> dict[str, str]:
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp or NOW.isoformat(),
    }


async def test_signed_request_uses_canonical_path_query_and_raw_body_hash(signed_auth_client):
    client, session_factory, calls = signed_auth_client
    raw_body = b'{"z":2,"a":1}'

    response = await client.post(
        "/signed?b=2&a=hello+world&a=first",
        content=raw_body,
        headers=signed_headers(),
    )

    assert response.status_code == 200
    expected_canonical = canonical_request_string(
        method="POST",
        path="/signed",
        query_string="b=2&a=hello+world&a=first",
        timestamp=NOW.isoformat(),
        nonce="nonce-1",
        raw_body=raw_body,
    )
    assert response.json() == {
        "hotkey": "miner-hotkey",
        "canonical_request": expected_canonical,
        "body_sha256": hashlib.sha256(raw_body).hexdigest(),
    }
    assert calls == [("miner-hotkey", expected_canonical, "valid-signature")]
    assert expected_canonical == (
        "POST\n"
        "/signed?a=first&a=hello+world&b=2\n"
        f"{NOW.isoformat()}\n"
        "nonce-1\n"
        f"{body_sha256(raw_body)}"
    )
    async with session_factory() as session:
        nonce_count = await session.scalar(select(func.count(RequestNonce.id)))
    assert nonce_count == 1


async def test_replayed_nonce_returns_conflict(signed_auth_client):
    client, session_factory, _calls = signed_auth_client
    headers = signed_headers(nonce="same-nonce")

    first = await client.post("/signed", content=b"{}", headers=headers)
    second = await client.post("/signed", content=b"{}", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {"detail": "replayed request"}
    async with session_factory() as session:
        nonce_count = await session.scalar(select(func.count(RequestNonce.id)))
    assert nonce_count == 1


@pytest.mark.parametrize(
    ("headers", "expected_detail"),
    [
        (
            signed_headers(timestamp=(NOW - timedelta(seconds=301)).isoformat()),
            "invalid signed request",
        ),
        (signed_headers(timestamp="inf"), "invalid signed request"),
        (signed_headers(timestamp="1e999"), "invalid signed request"),
        (signed_headers(signature="invalid-signature"), "invalid signed request"),
        ({"X-Hotkey": "miner-hotkey"}, "invalid signed request"),
    ],
)
async def test_auth_failures_return_generic_unauthorized(
    signed_auth_client,
    headers: dict[str, str],
    expected_detail: str,
):
    client, _session_factory, _calls = signed_auth_client

    response = await client.post("/signed", content=b"{}", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": expected_detail}


async def test_owner_dependency_requires_exact_owner_hotkey(signed_auth_client):
    client, _session_factory, calls = signed_auth_client

    wrong_owner = await client.post(
        "/owner",
        content=b"{}",
        headers=signed_headers(hotkey="miner-hotkey", nonce="owner-nonce-1"),
    )
    right_owner = await client.post(
        "/owner",
        content=b"{}",
        headers=signed_headers(hotkey="owner-hotkey", nonce="owner-nonce-2"),
    )

    assert wrong_owner.status_code == 403
    assert wrong_owner.json() == {"detail": "forbidden"}
    assert right_owner.status_code == 200
    assert right_owner.json() == {"hotkey": "owner-hotkey"}
    assert [call[0] for call in calls] == ["owner-hotkey"]
