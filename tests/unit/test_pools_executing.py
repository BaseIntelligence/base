"""RED contract: master aggregator GET /v1/pools/executing.

Fans out to embedded challenges' GET /v1/execution-pool/live:
  - Prism            → http://127.0.0.1:18080
  - Agent Challenge  → http://127.0.0.1:18081

(see deploy/compose/docker-compose.yml embed topology).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from base.master.app_proxy import create_proxy_app
from base.master.registry import ChallengeRegistry
from base.schemas.challenge import ChallengeCreate, ChallengeStatus
from base.security.miner_auth import NonceReplayError

PRISM_LIVE = "http://127.0.0.1:18080/v1/execution-pool/live"
AC_LIVE = "http://127.0.0.1:18081/v1/execution-pool/live"

_PINNED = "a" * 64


class _NonceStore:
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


class _Cache:
    def get(self) -> dict[str, int]:
        return {}


def _embed_registry() -> ChallengeRegistry:
    """Two active challenges on the compose embed loopback ports."""

    registry = ChallengeRegistry()
    registry.create(
        ChallengeCreate(
            slug="prism",
            name="PRISM",
            image=(f"ghcr.io/baseintelligence/prism:1.0.0@sha256:{_PINNED}"),
            version="1.0.0",
            emission_percent=Decimal("30"),
            status=ChallengeStatus.ACTIVE,
            internal_base_url="http://127.0.0.1:18080",
        )
    )
    registry.create(
        ChallengeCreate(
            slug="agent-challenge",
            name="Agent Challenge",
            image=(f"ghcr.io/baseintelligence/agent-challenge:latest@sha256:{_PINNED}"),
            version="1.0.0",
            emission_percent=Decimal("70"),
            status=ChallengeStatus.ACTIVE,
            internal_base_url="http://127.0.0.1:18081",
        )
    )
    return registry


def _proxy_client(
    *,
    registry: ChallengeRegistry | None = None,
) -> TestClient:
    @asynccontextmanager
    async def client_factory():
        # Default factory is unused for the aggregator fan-out (direct loopback
        # httpx calls to each challenge internal_base_url). Keep a no-op client
        # so create_proxy_app still constructs.
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(404, json={"detail": "unused"})
            ),
            base_url="http://unused.invalid",
        ) as client:
            yield client

    return TestClient(
        create_proxy_app(
            registry=registry or _embed_registry(),
            nonce_store=_NonceStore(),
            metagraph_cache=_Cache(),  # type: ignore[arg-type]
            client_factory=client_factory,
        )
    )


def _assert_no_score_fields(payload: Any, *, path: str = "$") -> None:
    """Pool payload must never expose score / weight / emission fields."""

    forbidden_keys = {
        "score",
        "scores",
        "weight",
        "weights",
        "emission",
        "emission_percent",
        "raw_score",
        "final_score",
        "normalized_score",
        "incentive",
    }
    if isinstance(payload, dict):
        lowered = {str(k).lower() for k in payload}
        leaked = forbidden_keys & lowered
        assert not leaked, f"score-like keys at {path}: {sorted(leaked)}"
        for key, value in payload.items():
            _assert_no_score_fields(value, path=f"{path}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _assert_no_score_fields(item, path=f"{path}[{index}]")


@respx.mock
def test_pools_executing_route_exists_on_proxy_app() -> None:
    """GET /v1/pools/executing is a first-class master route (not challenge proxy)."""

    respx.get(PRISM_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={"units": [{"unit_id": "prism-u1", "status": "executing"}]},
        )
    )
    respx.get(AC_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={"units": [{"unit_id": "ac-u1", "status": "executing"}]},
        )
    )

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/json")


@respx.mock
def test_pools_executing_fans_out_to_both_embed_challenges() -> None:
    """Aggregator hits Prism :18080 and Agent Challenge :18081 live pool endpoints."""

    prism_route = respx.get(PRISM_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "prism-unit-1",
                        "status": "executing",
                        "hotkey": "5PrismHotkey",
                    }
                ]
            },
        )
    )
    ac_route = respx.get(AC_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "ac-unit-1",
                        "status": "executing",
                        "hotkey": "5AcHotkey",
                    }
                ]
            },
        )
    )

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    assert prism_route.called
    assert ac_route.called
    assert prism_route.call_count == 1
    assert ac_route.call_count == 1


@respx.mock
def test_pools_executing_response_shape_keyed_by_challenge_slug() -> None:
    """Response is per-challenge entries keyed by slug, each listing executing units."""

    respx.get(PRISM_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "prism-unit-1",
                        "status": "executing",
                        "started_at": "2026-01-01T00:00:00Z",
                    }
                ]
            },
        )
    )
    respx.get(AC_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "ac-unit-1",
                        "status": "executing",
                        "started_at": "2026-01-01T00:00:01Z",
                    },
                    {
                        "unit_id": "ac-unit-2",
                        "status": "executing",
                        "started_at": "2026-01-01T00:00:02Z",
                    },
                ]
            },
        )
    )

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    body = response.json()

    # Top-level may be the challenges map itself or wrap under a known key.
    challenges = body.get("challenges", body)
    assert isinstance(challenges, dict)
    assert "prism" in challenges
    assert "agent-challenge" in challenges

    prism_entry = challenges["prism"]
    ac_entry = challenges["agent-challenge"]
    assert isinstance(prism_entry, dict)
    assert isinstance(ac_entry, dict)

    # Healthy entries expose currently-executing units (no fabricated placeholders).
    prism_units = prism_entry.get("units", prism_entry.get("executing"))
    ac_units = ac_entry.get("units", ac_entry.get("executing"))
    assert isinstance(prism_units, list)
    assert isinstance(ac_units, list)
    assert len(prism_units) == 1
    assert len(ac_units) == 2
    assert prism_units[0]["unit_id"] == "prism-unit-1"
    assert {u["unit_id"] for u in ac_units} == {"ac-unit-1", "ac-unit-2"}

    # Healthy entries must not carry an error object.
    assert "error" not in prism_entry
    assert "error" not in ac_entry

    _assert_no_score_fields(body)


@respx.mock
def test_pools_executing_partial_failure_returns_200_with_error_object() -> None:
    """One challenge down: 200 + real data for healthy + explicit error for failed.

    Must NEVER fabricate/placeholder units for the failed challenge.
    """

    respx.get(PRISM_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "prism-healthy-1",
                        "status": "executing",
                    }
                ]
            },
        )
    )
    # Agent Challenge unreachable / timed out.
    respx.get(AC_LIVE).mock(side_effect=httpx.ConnectError("connection refused"))

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    body = response.json()
    challenges = body.get("challenges", body)

    prism_entry = challenges["prism"]
    ac_entry = challenges["agent-challenge"]

    prism_units = prism_entry.get("units", prism_entry.get("executing"))
    assert isinstance(prism_units, list)
    assert len(prism_units) == 1
    assert prism_units[0]["unit_id"] == "prism-healthy-1"
    assert "error" not in prism_entry

    # Failed challenge: explicit error object, no fabricated units.
    assert "error" in ac_entry
    error_obj = ac_entry["error"]
    assert error_obj is not None
    assert isinstance(error_obj, dict)
    # Error must be descriptive (code and/or message).
    assert error_obj.get("code") or error_obj.get("message") or error_obj.get("detail")

    failed_units = ac_entry.get("units", ac_entry.get("executing", []))
    if failed_units is None:
        failed_units = []
    assert failed_units == [], (
        "partial failure must not fabricate placeholder units for the down challenge"
    )

    _assert_no_score_fields(body)


@respx.mock
def test_pools_executing_partial_failure_when_prism_down() -> None:
    """Symmetric partial failure: Prism down, AC healthy."""

    respx.get(PRISM_LIVE).mock(side_effect=httpx.ReadTimeout("timed out"))
    respx.get(AC_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={"units": [{"unit_id": "ac-only-1", "status": "executing"}]},
        )
    )

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    body = response.json()
    challenges = body.get("challenges", body)

    prism_entry = challenges["prism"]
    ac_entry = challenges["agent-challenge"]

    assert "error" in prism_entry
    assert isinstance(prism_entry["error"], dict)
    prism_units = prism_entry.get("units", prism_entry.get("executing", [])) or []
    assert prism_units == []

    ac_units = ac_entry.get("units", ac_entry.get("executing"))
    assert isinstance(ac_units, list)
    assert ac_units[0]["unit_id"] == "ac-only-1"
    assert "error" not in ac_entry

    _assert_no_score_fields(body)


@respx.mock
def test_pools_executing_payload_never_includes_score_fields() -> None:
    """No score/weight/emission fields anywhere in the pool payload."""

    respx.get(PRISM_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "p1",
                        "status": "executing",
                        "hotkey": "hk1",
                    }
                ]
            },
        )
    )
    respx.get(AC_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "a1",
                        "status": "executing",
                        "hotkey": "hk2",
                    }
                ]
            },
        )
    )

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    body = response.json()
    _assert_no_score_fields(body)

    # Explicit top-level absence (defense in depth beyond recursive walk).
    blob = response.text.lower()
    for token in (
        '"score"',
        '"scores"',
        '"weight"',
        '"weights"',
        '"emission"',
        '"emission_percent"',
        '"raw_score"',
        '"final_score"',
        '"normalized_score"',
        '"incentive"',
    ):
        assert token not in blob, f"forbidden score token present: {token}"


@respx.mock
def test_pools_executing_does_not_require_live_network() -> None:
    """Contract is fully mockable — no real sockets to 18080/18081."""

    respx.get(PRISM_LIVE).mock(return_value=httpx.Response(200, json={"units": []}))
    respx.get(AC_LIVE).mock(return_value=httpx.Response(200, json={"units": []}))

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    # Route must exist and succeed against mocks only.
    assert response.status_code == 200
    body = response.json()
    challenges = body.get("challenges", body)
    assert set(challenges) >= {"prism", "agent-challenge"}


@respx.mock
def test_pools_executing_accepts_prism_jobs_shape() -> None:
    """Prism live pool returns {"jobs": [...]} — must not be dropped as empty units.

    Agent Challenge uses {"units": [...]}. Aggregator normalizes both under units.
    """

    respx.get(PRISM_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "eval_job_id": "j1",
                        "status": "running",
                        "latest_event": {"type": "progress"},
                    }
                ]
            },
        )
    )
    respx.get(AC_LIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "units": [
                    {
                        "unit_id": "ac-u1",
                        "status": "executing",
                    }
                ]
            },
        )
    )

    client = _proxy_client()
    response = client.get("/v1/pools/executing")

    assert response.status_code == 200
    body = response.json()
    challenges = body.get("challenges", body)

    prism_entry = challenges["prism"]
    ac_entry = challenges["agent-challenge"]
    assert "error" not in prism_entry
    assert "error" not in ac_entry

    prism_units = prism_entry.get("units", prism_entry.get("executing"))
    ac_units = ac_entry.get("units", ac_entry.get("executing"))
    assert isinstance(prism_units, list)
    assert isinstance(ac_units, list)
    assert len(prism_units) == 1, (
        "Prism jobs[] must be normalized into units (not silently dropped)"
    )
    assert prism_units[0]["eval_job_id"] == "j1"
    assert prism_units[0]["status"] == "running"
    assert len(ac_units) == 1
    assert ac_units[0]["unit_id"] == "ac-u1"

    _assert_no_score_fields(body)
