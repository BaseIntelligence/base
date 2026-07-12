from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from base.master.app_proxy import create_proxy_app
from base.master.registry import ChallengeRegistry
from base.schemas.challenge import ChallengeCreate, ChallengeStatus
from base.security.miner_auth import NonceReplayError

ALLOWED_ORIGIN = "https://joinbase.ai"
VERCEL_PREVIEW_ORIGIN = (
    "https://platform-lz6erslee-mathismassimino-6459s-projects.vercel.app"
)


def _payload(slug: str = "demo") -> dict[str, Any]:
    return {
        "slug": slug,
        "name": "Demo",
        "image": "ghcr.io/baseintelligence/demo:1.0.0@sha256:" + ("c" * 64),
        "version": "1.0.0",
        "emission_percent": "40.0",
    }


class FakeNonceStore:
    def __init__(self) -> None:
        self.keys: set[tuple[int, str, str, str]] = set()

    async def reserve(self, **kwargs: Any) -> None:
        key = (
            int(kwargs["netuid"]),
            str(kwargs["challenge_slug"]),
            str(kwargs["hotkey"]),
            str(kwargs["nonce"]),
        )
        if key in self.keys:
            raise NonceReplayError("nonce already used")
        self.keys.add(key)


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


def _proxy_app(registry: ChallengeRegistry, **kwargs: Any) -> FastAPI:
    return create_proxy_app(
        registry=registry,
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_cors_allows_joinbase_origin_on_public_read() -> None:
    client = TestClient(_proxy_app(ChallengeRegistry()))

    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_cors_allows_vercel_preview_origin() -> None:
    client = TestClient(_proxy_app(ChallengeRegistry()))

    response = client.get("/health", headers={"Origin": VERCEL_PREVIEW_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == VERCEL_PREVIEW_ORIGIN


def test_cors_preflight_returns_allow_headers() -> None:
    client = TestClient(_proxy_app(ChallengeRegistry()))

    response = client.options(
        "/health",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "GET" in response.headers["access-control-allow-methods"]


def test_cors_omits_header_for_disallowed_origin() -> None:
    client = TestClient(_proxy_app(ChallengeRegistry()))

    response = client.get("/health", headers={"Origin": "https://evil.example"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_header_present_on_forwarded_challenge_response() -> None:
    registry = ChallengeRegistry()
    registry.create(
        ChallengeCreate(
            **{
                **_payload("demo"),
                "internal_base_url": "http://challenge-demo:8000",
            }
        )
    )
    registry.set_status("demo", ChallengeStatus.ACTIVE)

    challenge_app = FastAPI()

    @challenge_app.get("/leaderboard")
    async def leaderboard(request: Request) -> dict[str, Any]:
        return {"ok": True}

    @asynccontextmanager
    async def client_factory():
        transport = httpx.ASGITransport(app=challenge_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://challenge-demo:8000"
        ) as client:
            yield client

    proxy_client = TestClient(_proxy_app(registry, client_factory=client_factory))
    response = proxy_client.get(
        "/challenges/demo/leaderboard", headers={"Origin": ALLOWED_ORIGIN}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_cors_origins_are_configurable_via_param() -> None:
    client = TestClient(
        _proxy_app(
            ChallengeRegistry(),
            allowed_cors_origins=["https://custom.example"],
        )
    )

    allowed = client.get("/health", headers={"Origin": "https://custom.example"})
    default_now_blocked = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})

    assert allowed.headers["access-control-allow-origin"] == "https://custom.example"
    assert "access-control-allow-origin" not in default_now_blocked.headers
