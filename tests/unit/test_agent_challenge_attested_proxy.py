from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from base.config.settings import MasterSettings
from base.master.app_proxy import (
    _is_agent_challenge_enabled_mode_allowed_route,
    _is_blocked_agent_challenge_proxy_path,
    create_proxy_app,
)
from base.master.registry import ChallengeRegistry
from base.schemas.challenge import ChallengeCreate, ChallengeStatus
from base.security.miner_auth import NonceReplayError


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


@dataclass(frozen=True)
class _SignedRoute:
    method: str
    path: str
    upstream_path: str
    upstream_status: int


SIGNED_ROUTES = (
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions",
        "/submissions",
        201,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/review/prepare",
        "/submissions/sub-1/review/prepare",
        200,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/review/retry",
        "/submissions/sub-1/review/retry",
        201,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/review/deployed",
        "/submissions/sub-1/review/deployed",
        200,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/review/cancel",
        "/submissions/sub-1/review/cancel",
        200,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/eval/prepare",
        "/submissions/sub-1/eval/prepare",
        200,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/eval/retry",
        "/submissions/sub-1/eval/retry",
        201,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/eval/cancel",
        "/submissions/sub-1/eval/cancel",
        200,
    ),
    _SignedRoute(
        "POST",
        "/challenges/agent-challenge/submissions/sub-1/eval/failure",
        "/submissions/sub-1/eval/failure",
        200,
    ),
    _SignedRoute(
        "GET",
        "/challenges/agent-challenge/submissions/sub-1/review/report",
        "/submissions/sub-1/review/report",
        200,
    ),
    _SignedRoute(
        "GET",
        "/challenges/agent-challenge/submissions/sub-1/review/history",
        "/submissions/sub-1/review/history",
        200,
    ),
    _SignedRoute(
        "GET",
        "/challenges/agent-challenge/submissions/sub-1/eval/status",
        "/submissions/sub-1/eval/status",
        200,
    ),
)


def _registry() -> ChallengeRegistry:
    registry = ChallengeRegistry()
    registry.create(
        ChallengeCreate(
            slug="agent-challenge",
            name="Agent Challenge",
            image="ghcr.io/baseintelligence/agent-challenge:latest@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            version="1.0.0",
            emission_percent=Decimal("100"),
            status=ChallengeStatus.ACTIVE,
            internal_base_url="http://challenge-agent-challenge:8000",
        )
    )
    return registry


def _proxy_client(
    handler: httpx.AsyncBaseTransport | Any,
    *,
    attested_routes_enabled: bool = True,
) -> TestClient:
    @asynccontextmanager
    async def client_factory():
        transport = (
            handler
            if isinstance(handler, httpx.AsyncBaseTransport)
            else httpx.MockTransport(handler)
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://challenge-agent-challenge:8000",
        ) as client:
            yield client

    return TestClient(
        create_proxy_app(
            registry=_registry(),
            nonce_store=_NonceStore(),
            metagraph_cache=_Cache(),  # type: ignore[arg-type]
            client_factory=client_factory,
            agent_challenge_attested_routes_enabled=attested_routes_enabled,
        )
    )


@pytest.mark.parametrize("route", SIGNED_ROUTES)
def test_exact_attested_signed_route_preserves_canonical_upstream_bytes(
    route: _SignedRoute,
) -> None:
    captured: dict[str, Any] = {}
    upstream_body = (
        b'{"schema_version":1,"opaque":"upstream\\u0000bytes","route":"'
        + route.upstream_path.encode()
        + b'"}'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = request.url.query
        captured["body"] = await request.aread()
        captured["headers"] = request.headers
        return httpx.Response(
            route.upstream_status,
            content=upstream_body,
            headers={"content-type": "application/vnd.base.attested+json"},
        )

    client = _proxy_client(handler)
    request_body = (
        b'{"schema_version":1,"expected_id":"opaque","approval_id":"operator-1",'
        b'"binary":"\\u0000\\u00ff"}'
    )
    response = client.request(
        route.method,
        f"{route.path}?z=last&a=first",
        content=request_body,
        headers={
            "Content-Type": "application/vnd.base.signed+json",
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "miner-signature",
            "X-Nonce": "miner-nonce",
            "X-Timestamp": "1700000000",
        },
    )

    assert response.status_code == route.upstream_status
    assert response.content == upstream_body
    assert response.headers["content-type"] == "application/vnd.base.attested+json"
    assert captured["method"] == route.method
    assert captured["path"] == route.upstream_path
    assert captured["query"] == b"z=last&a=first"
    assert captured["body"] == request_body
    headers = captured["headers"]
    assert headers["content-type"] == "application/vnd.base.signed+json"
    assert headers["x-hotkey"] == "miner-hotkey"
    assert headers["x-signature"] == "miner-signature"
    assert headers["x-nonce"] == "miner-nonce"
    assert headers["x-timestamp"] == "1700000000"


def test_attested_signed_route_strips_caller_authority_and_proxy_headers() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"ok": True})

    client = _proxy_client(handler)
    response = client.post(
        "/challenges/agent-challenge/submissions/sub-1/review/retry",
        content=b'{"expected_assignment_id":"assignment-1","approval_id":"approval-1"}',
        headers={
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "miner-signature",
            "X-Nonce": "miner-nonce",
            "X-Timestamp": "1700000000",
            "Authorization": "Bearer caller-capability",
            "Proxy-Authorization": "Basic caller-proxy",
            "X-Admin-Token": "caller-admin",
            "X-Base-Admin-Token": "caller-base-admin",
            "X-Base-Internal-Token": "caller-internal",
            "X-Internal-Authorization": "caller-internal-auth",
            "X-Base-Verified-Hotkey": "caller-verified",
            "X-Base-Verified-Future": "caller-future-trust",
            "X-Base-Request-Hash": "caller-hash",
            "X-Trust-Level": "caller-trust",
            "X-Trusted-Proxy": "caller-trusted-proxy",
            "X-Base-Trust-Result": "caller-base-trust",
            "X-RA-TLS-Peer-Key": "caller-peer",
            "X-RATLS-Peer-Certificate": "caller-peer-cert",
            "X-Review-Verified": "true",
            "X-Review-Verification": "passed",
            "X-Attestation-Verified": "true",
            "X-Allowlist-Digest": "caller-allowlist",
            "X-Measurement-MRTD": "caller-measurement",
            "Forwarded": "for=caller",
            "Via": "caller-proxy",
            "X-Forwarded-For": "198.51.100.7",
            "X-Forwarded-Host": "evil.example",
            "X-Forwarded-Proto": "https",
            "X-Real-IP": "198.51.100.8",
            "X-Proxy-Trust": "caller-proxy-trust",
            "X-Base-Proxy": "false",
            "X-Base-Challenge-Slug": "prism",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    headers: httpx.Headers = captured["headers"]
    assert headers["x-hotkey"] == "miner-hotkey"
    assert headers["x-signature"] == "miner-signature"
    assert headers["x-nonce"] == "miner-nonce"
    assert headers["x-timestamp"] == "1700000000"
    assert headers["x-public-header"] == "preserved"
    assert headers.get_list("x-base-proxy") == ["true"]
    assert headers.get_list("x-base-challenge-slug") == ["agent-challenge"]
    forbidden = {
        "authorization",
        "proxy-authorization",
        "x-admin-token",
        "x-base-admin-token",
        "x-base-internal-token",
        "x-internal-authorization",
        "x-base-verified-hotkey",
        "x-base-verified-future",
        "x-base-request-hash",
        "x-trust-level",
        "x-trusted-proxy",
        "x-base-trust-result",
        "x-ra-tls-peer-key",
        "x-ratls-peer-certificate",
        "x-review-verified",
        "x-review-verification",
        "x-attestation-verified",
        "x-allowlist-digest",
        "x-measurement-mrtd",
        "forwarded",
        "via",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
        "x-proxy-trust",
    }
    assert forbidden.isdisjoint(headers)


@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    (
        (
            "GET",
            "/challenges/agent-challenge/review/v1/assignments/assignment-1",
            "/review/v1/assignments/assignment-1",
        ),
        (
            "GET",
            "/challenges/agent-challenge/review/v1/assignments/assignment-1/artifact",
            "/review/v1/assignments/assignment-1/artifact",
        ),
        (
            "GET",
            "/challenges/agent-challenge/review/v1/assignments/assignment-1/rules",
            "/review/v1/assignments/assignment-1/rules",
        ),
        (
            "POST",
            "/challenges/agent-challenge/review/v1/assignments/assignment-1/model-call-started",
            "/review/v1/assignments/assignment-1/model-call-started",
        ),
        (
            "POST",
            "/challenges/agent-challenge/review/v1/assignments/assignment-1/failure",
            "/review/v1/assignments/assignment-1/failure",
        ),
        (
            "POST",
            "/challenges/agent-challenge/review/v1/assignments/assignment-1/report",
            "/review/v1/assignments/assignment-1/report",
        ),
    ),
)
@pytest.mark.parametrize("attested_routes_enabled", (False, True))
def test_review_capability_routes_preserve_authorization_bearer(
    method: str,
    path: str,
    upstream_path: str,
    attested_routes_enabled: bool,
) -> None:
    """Measured-review guest Bearer must survive proxy on closed /review/v1 table.

    Residual RuntimeError class-only guest failures after public_logs=true map to
    assignment fetch 401 when Authorization is stripped generically.
    """

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(200, json={"ok": True})

    client = _proxy_client(handler, attested_routes_enabled=attested_routes_enabled)
    response = client.request(
        method,
        path,
        content=b'{"marker":true}' if method == "POST" else None,
        headers={
            "Authorization": "Bearer ra_assignment-1.deadbeef",
            "Proxy-Authorization": "Basic should-still-strip",
            "X-Admin-Token": "should-strip",
            "X-Base-Verified-Hotkey": "should-strip",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    assert captured["method"] == method
    assert captured["path"] == upstream_path
    headers: httpx.Headers = captured["headers"]
    assert headers["authorization"] == "Bearer ra_assignment-1.deadbeef"
    assert headers["x-public-header"] == "preserved"
    assert "proxy-authorization" not in headers
    assert "x-admin-token" not in headers
    assert "x-base-verified-hotkey" not in headers


def test_review_capability_authorization_not_preserved_on_signed_prepare() -> None:
    """Signed miner routes still strip Authorization (only signature headers)."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"ok": True})

    client = _proxy_client(handler, attested_routes_enabled=False)
    response = client.post(
        "/challenges/agent-challenge/submissions/sub-1/review/prepare",
        content=b"{}",
        headers={
            "Authorization": "Bearer should-not-forward",
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "miner-signature",
            "X-Nonce": "miner-nonce",
            "X-Timestamp": "1700000000",
        },
    )
    assert response.status_code == 200
    headers: httpx.Headers = captured["headers"]
    assert "authorization" not in headers
    assert headers["x-hotkey"] == "miner-hotkey"


@pytest.mark.parametrize(
    "path",
    (
        "/challenges/agent-challenge/submissions/sub-1/review/prepare/",
        "/challenges/agent-challenge/submissions/sub-1/review//prepare",
        "/challenges/agent-challenge/submissions//sub-1/review/prepare",
        "/challenges/agent-challenge/submissions/sub-1/review/%70repare",
        "/challenges/agent-challenge/submissions/%73ub-1/review/prepare",
        "/challenges/%61gent-challenge/submissions/sub-1/review/prepare",
        "/challenges/agent-challenge/submissions/sub-1/eval/status/",
    ),
)
def test_attested_signed_route_rejects_noncanonical_path_neighbors(path: str) -> None:
    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"unexpected": True})

    response = _proxy_client(handler).request(
        "GET" if "status" in path else "POST",
        path,
        headers={
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "miner-signature",
            "X-Nonce": "miner-nonce",
            "X-Timestamp": "1700000000",
        },
    )

    assert response.status_code == 404
    assert upstream_calls == []


@pytest.mark.parametrize("slug_alias", ("Agent%20Challenge", "AGENT-CHALLENGE"))
def test_attested_private_routes_reject_agent_challenge_name_aliases(
    slug_alias: str,
) -> None:
    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"unexpected": True})

    response = _proxy_client(handler).get(
        f"/challenges/{slug_alias}/review/v1/assignments/assignment-1/artifact",
        headers={"Authorization": "Bearer caller-capability"},
    )

    assert response.status_code == 404
    assert upstream_calls == []


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/submissions/sub-1/review/prepare"),
        ("PUT", "/submissions/sub-1/review/retry"),
        ("POST", "/submissions/sub-1/review/report"),
        # GET review/history is signature-preserved (signed_get_routes) — not private.
        ("POST", "/submissions/sub-1/review/history"),
        # Public GET review/tee is allowlisted; wrong method + neighbors stay denied.
        ("POST", "/submissions/sub-1/review/tee"),
        ("PUT", "/submissions/sub-1/review/tee"),
        ("GET", "/submissions/sub-1/review/tee/extra"),
        ("GET", "/submissions/sub-1/review/math"),
        ("GET", "/submissions/sub-1/eval/prepare"),
        ("POST", "/submissions/sub-1/eval/status"),
        ("POST", "/submissions/sub-1/eval/result"),
        ("POST", "/submissions/sub-1/eval/key-release"),
        # Miner env/launch shapes are allowlisted + signature-preserved (FIX G).
        # Wrong methods on those shapes stay denied.
        ("DELETE", "/submissions/sub-1/env"),
        ("POST", "/submissions/sub-1/env"),
        ("PUT", "/submissions/sub-1/env/confirm-empty"),
        ("GET", "/submissions/sub-1/env/confirm-empty"),
        ("GET", "/submissions/sub-1/launch"),
        ("PUT", "/submissions/sub-1/launch"),
        # review/v1 guest capability table is allowlisted + Authorization preserved
        # (see test_review_capability_routes_preserve_authorization_bearer). Neighbor
        # aliases and unconstrained assignment paths remain blocked below.
        ("GET", "/review/v1/assignments"),
        ("POST", "/review/v1/assignments"),
        ("GET", "/review/v1/assignments/assignment-1/extra"),
        ("POST", "/review/v1/assignments/assignment-1/unknown"),
        ("GET", "/internal/v1/reviews/session-1/report"),
        ("GET", "/internal/v1/reviews/session-1/evidence/object-1"),
        ("POST", "/internal/v1/reviews/session-1/approvals"),
        ("GET", "/key-release/nonce"),
        ("POST", "/key-release/release"),
        ("GET", "/keyrelease/nonce"),
        ("POST", "/keyrelease/release"),
        ("GET", "/nonce"),
        ("POST", "/release"),
        # Fall-through aliases that a deny-list leave-behind would still forward.
        ("GET", "/results"),
        ("POST", "/results"),
        ("GET", "/result"),
        ("POST", "/result"),
        ("GET", "/submissions/sub-1/results"),
        ("POST", "/submissions/sub-1/results"),
        ("GET", "/submissions/sub-1/result"),
        ("POST", "/submissions/sub-1/result"),
        ("GET", "/capability"),
        ("POST", "/capability/token"),
        ("GET", "/capabilities/token"),
        ("POST", "/assignments/assignment-1"),
        ("GET", "/assignment/assignment-1"),
        ("GET", "/evidence/object-1"),
        ("GET", "/submissions/sub-1/evidence/object-1"),
        ("POST", "/key_release/release"),
        ("GET", "/direct-result"),
        ("POST", "/direct/result"),
        ("GET", "/anything-private"),
        ("POST", "/evals/run-1/result"),
    ),
)
@pytest.mark.parametrize(
    "prefix",
    (
        "/challenges/agent-challenge",
        "/v1/challenges/agent-challenge",
    ),
)
def test_attested_private_neighbors_and_aliases_are_local_404(
    method: str,
    path: str,
    prefix: str,
) -> None:
    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"unexpected": True})

    client = _proxy_client(handler)
    response = client.request(
        method,
        f"{prefix}{path}",
        content=b'{"caller_trust":true}',
        headers={
            "Authorization": "Bearer caller-capability",
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "miner-signature",
            "X-Nonce": "miner-nonce",
            "X-Timestamp": "1700000000",
            "X-Allowlist-Digest": "caller-allowlist",
            "X-Measurement-MRTD": "caller-measurement",
            "X-RA-TLS-Peer-Key": "caller-peer",
            "X-Review-Verified": "true",
            "X-Base-Verified-Hotkey": "caller-verified",
        },
    )

    assert response.status_code == 404
    assert upstream_calls == []


@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    (
        (
            "GET",
            "/challenges/agent-challenge/submissions/sub-1/status",
            "/submissions/sub-1/status",
        ),
        (
            "GET",
            "/challenges/agent-challenge/submissions/sub-1/events",
            "/submissions/sub-1/events",
        ),
        # Public TEE math (safe subset) — same class as status/events.
        (
            "GET",
            "/challenges/agent-challenge/submissions/sub-1/review/tee",
            "/submissions/sub-1/review/tee",
        ),
        (
            "GET",
            "/challenges/agent-challenge/benchmarks/tasks",
            "/benchmarks/tasks",
        ),
        (
            "GET",
            "/challenges/agent-challenge/benchmarks",
            "/benchmarks",
        ),
    ),
)
def test_attested_public_status_and_benchmark_routes_remain_forwardable(
    method: str,
    path: str,
    upstream_path: str,
) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            content=b'{"schema_version":1,"safe":true}',
            headers={"content-type": "application/json"},
        )

    response = _proxy_client(handler).request(
        method,
        path,
        headers={
            "Authorization": "Bearer caller-capability",
            "X-Allowlist-Digest": "caller-allowlist",
            "X-Measurement-MRTD": "caller-measurement",
            "X-RA-TLS-Peer-Key": "caller-peer",
            "X-Review-Verified": "true",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    assert response.content == b'{"schema_version":1,"safe":true}'
    assert captured["method"] == method
    assert captured["path"] == upstream_path
    headers: httpx.Headers = captured["headers"]
    assert headers["x-public-header"] == "preserved"
    assert headers.get_list("x-base-proxy") == ["true"]
    assert headers.get_list("x-base-challenge-slug") == ["agent-challenge"]
    assert "authorization" not in headers
    assert "x-allowlist-digest" not in headers
    assert "x-measurement-mrtd" not in headers
    assert "x-ra-tls-peer-key" not in headers
    assert "x-review-verified" not in headers


def test_attested_signed_upstream_auth_error_is_preserved_without_rewriting() -> None:
    upstream_body = b'{"detail":{"code":"invalid_signed_request"}}'

    async def handler(request: httpx.Request) -> httpx.Response:
        assert "x-signature" not in request.headers
        return httpx.Response(
            401,
            content=upstream_body,
            headers={"content-type": "application/problem+json"},
        )

    client = _proxy_client(handler)
    response = client.post(
        "/challenges/agent-challenge/submissions/sub-1/eval/prepare",
        content=b'{"schema_version":1}',
    )

    assert response.status_code == 401
    assert response.content == upstream_body
    assert response.headers["content-type"] == "application/problem+json"


def test_attested_proxy_flag_defaults_off_and_keeps_generic_legacy_behavior() -> None:
    """Flag default remains off; legacy unpinned paths stay open."""
    assert MasterSettings().agent_challenge_attested_routes_enabled is False
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(404, content=b'{"detail":"Not Found"}')

    client = _proxy_client(handler, attested_routes_enabled=False)
    # Neighbor status path stays generically forwardable when flag is off
    # (not part of the signed-review/eval row).
    response = client.get(
        "/challenges/agent-challenge/submissions/sub-1/status",
        headers={
            "X-Forwarded-For": "198.51.100.7",
            "X-Review-Legacy-Metadata": "legacy-value",
        },
    )

    assert response.status_code == 404
    assert response.content == b'{"detail":"Not Found"}'
    assert captured["path"] == "/submissions/sub-1/status"
    assert captured["headers"]["x-forwarded-for"] == "198.51.100.7"
    assert captured["headers"]["x-review-legacy-metadata"] == "legacy-value"


@pytest.mark.parametrize(
    "path",
    (
        "/challenges/agent-challenge/openapi.json",
        "/challenges/agent-challenge/docs",
        "/challenges/agent-challenge/redoc",
        "/challenges/agent-challenge/leaderboard",
    ),
)
def test_attested_mode_allows_public_discovery_read_routes(path: str) -> None:
    """Flag-on must not 404 joinbase readiness openapi/docs/leaderboard."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, content=b'{"ok":true}')

    response = _proxy_client(handler, attested_routes_enabled=True).get(path)
    assert response.status_code == 200
    assert captured["path"] == path.removeprefix("/challenges/agent-challenge")


def test_flag_off_still_preserves_miner_signature_headers_on_review_prepare() -> None:
    """Auth-binding residual 401: even with attested flag off, minersign headers
    must reach dual-flag agent-challenge for POST review/prepare (and exact
    eval/review signed neighbors). Previously SENSITIVE_REQUEST_HEADERS stripped
    them when flag was false, so joinbase returned HTTP 401 while submit (in the
    legacy preserve set) still returned 201.
    """

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        captured["body"] = await request.aread()
        # Simulate challenge delivering one-time capability after valid signature.
        return httpx.Response(
            200,
            content=(
                b'{"schema_version":1,"session_id":"rs_test","assignment_id":"ra_test",'
                b'"attempt":1,"assignment":{},"review_session_token":"delivered-once"}'
            ),
            headers={"content-type": "application/json"},
        )

    client = _proxy_client(handler, attested_routes_enabled=False)
    request_body = b"{}"
    response = client.post(
        "/challenges/agent-challenge/submissions/1/review/prepare",
        content=request_body,
        headers={
            "Content-Type": "application/json",
            "X-Hotkey": "5D7D4EGayNMinerHotkeyExampleForTestOnly",
            "X-Signature": "0x" + ("ab" * 32),
            "X-Nonce": "fresh-nonce-review-prepare",
            "X-Timestamp": "1700000000",
            "X-Forwarded-For": "198.51.100.7",
            "X-Attestation-Verified": "should-not-elevate",
            "X-Base-Verified-Hotkey": "forged",
        },
    )

    assert response.status_code == 200
    assert b"review_session_token" in response.content
    assert captured["path"] == "/submissions/1/review/prepare"
    assert captured["body"] == request_body
    headers: httpx.Headers = captured["headers"]
    assert headers["x-hotkey"] == "5D7D4EGayNMinerHotkeyExampleForTestOnly"
    assert headers["x-signature"].startswith("0x")
    assert headers["x-nonce"] == "fresh-nonce-review-prepare"
    assert headers["x-timestamp"] == "1700000000"
    # Signature headers alone unblocked the residual 401; trust-header
    # stripping is the separate fail-closed surface when the flagged mode is on.


def test_flag_off_still_preserves_miner_signature_headers_on_review_history() -> None:
    """Live residual v11: miner GET review/history long-poll needs sign headers.

    Even when agent_challenge_attested_routes_enabled is off (joinbase dual-flag
    residual), the master must forward X-Hotkey/X-Signature/X-Nonce/X-Timestamp
    for exact GET submissions/{id}/review/history. Omitting this row from
    signed_get_routes caused HTTP 401 on history polls until ad-hoc hotpatch.
    """

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            content=b'{"schema_version":1,"events":[],"cursor":null}',
            headers={"content-type": "application/json"},
        )

    client = _proxy_client(handler, attested_routes_enabled=False)
    response = client.get(
        "/challenges/agent-challenge/submissions/1/review/history",
        headers={
            "X-Hotkey": "5D7D4EGayNMinerHotkeyExampleForTestOnly",
            "X-Signature": "0x" + ("cd" * 32),
            "X-Nonce": "fresh-nonce-review-history",
            "X-Timestamp": "1700000001",
            "X-Forwarded-For": "198.51.100.9",
            "X-Attestation-Verified": "should-not-elevate",
            "X-Base-Verified-Hotkey": "forged",
            "Authorization": "Bearer should-not-forward",
        },
    )

    assert response.status_code == 200
    assert b'"events"' in response.content
    assert captured["method"] == "GET"
    assert captured["path"] == "/submissions/1/review/history"
    headers: httpx.Headers = captured["headers"]
    assert headers["x-hotkey"] == "5D7D4EGayNMinerHotkeyExampleForTestOnly"
    assert headers["x-signature"].startswith("0x")
    assert headers["x-nonce"] == "fresh-nonce-review-history"
    assert headers["x-timestamp"] == "1700000001"
    # Authorization remains stripped on signed miner routes (Bearer is guest).
    assert "authorization" not in headers


@pytest.mark.parametrize("attested_routes_enabled", [True, False])
@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    (
        (
            "GET",
            "/challenges/agent-challenge/submissions/13/env",
            "/submissions/13/env",
        ),
        (
            "PUT",
            "/challenges/agent-challenge/submissions/13/env",
            "/submissions/13/env",
        ),
        (
            "POST",
            "/challenges/agent-challenge/submissions/13/env/confirm-empty",
            "/submissions/13/env/confirm-empty",
        ),
        (
            "POST",
            "/challenges/agent-challenge/submissions/13/launch",
            "/submissions/13/launch",
        ),
    ),
)
def test_miner_env_launch_routes_allowed_and_preserve_signature_headers(
    attested_routes_enabled: bool,
    method: str,
    path: str,
    upstream_path: str,
) -> None:
    """FIX G: env/launch must work under attested allowlist AND keep miner sig headers.

    Prod sets agent_challenge_attested_routes_enabled=true. Without these shapes in
    the enabled-mode allowlist the public edge returns local 404 Proxy path not found.
    Even after allowlisting, signed PUT/POST would residual 401 if X-Hotkey/X-Signature
    were stripped. Both flag states must forward + preserve the four miner headers.
    """

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        captured["body"] = await request.aread()
        return httpx.Response(
            200,
            content=b'{"ok":true,"env":"forwarded"}',
            headers={"content-type": "application/json"},
        )

    client = _proxy_client(handler, attested_routes_enabled=attested_routes_enabled)
    request_body = b'{"OPENROUTER_API_KEY":"sk-test"}' if method == "PUT" else b""
    response = client.request(
        method,
        path,
        content=request_body,
        headers={
            "Content-Type": "application/json",
            "X-Hotkey": "5D7D4EGayNMinerHotkeyExampleForTestOnly",
            "X-Signature": "0x" + ("ef" * 32),
            "X-Nonce": "fresh-nonce-miner-env",
            "X-Timestamp": "1700000013",
            "Authorization": "Bearer should-not-forward",
            "X-Base-Verified-Hotkey": "forged",
            "X-Allowlist-Digest": "caller-allowlist",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "env": "forwarded"}
    assert captured["method"] == method
    assert captured["path"] == upstream_path
    if method == "PUT":
        assert captured["body"] == request_body
    headers: httpx.Headers = captured["headers"]
    assert headers["x-hotkey"] == "5D7D4EGayNMinerHotkeyExampleForTestOnly"
    assert headers["x-signature"].startswith("0x")
    assert headers["x-nonce"] == "fresh-nonce-miner-env"
    assert headers["x-timestamp"] == "1700000013"
    assert "authorization" not in headers
    if attested_routes_enabled:
        assert "x-base-verified-hotkey" not in headers
        assert "x-allowlist-digest" not in headers


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "submissions/13/env"),
        ("PUT", "submissions/13/env"),
        ("POST", "submissions/13/env/confirm-empty"),
        ("POST", "submissions/13/launch"),
    ),
)
def test_enabled_mode_allowlist_accepts_miner_env_launch_shapes(
    method: str,
    path: str,
) -> None:
    assert _is_agent_challenge_enabled_mode_allowed_route(
        "agent-challenge",
        method,
        path,
    )
    assert not _is_blocked_agent_challenge_proxy_path(
        "agent-challenge",
        method,
        path,
        attested_routes_enabled=True,
    )


def test_forged_trust_headers_do_not_elevate_private_routes() -> None:
    """VAL-ACAT-047/048: forged trust headers never open private aliases."""

    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"unexpected": True})

    client = _proxy_client(handler)
    trust_headers = {
        "Authorization": "Bearer caller-capability",
        "X-Attestation-Verified": "true",
        "X-RA-TLS-Peer-Key": "forged",
        "X-Base-Verified-Hotkey": "forged-hotkey",
        "X-Trust-Level": "admin",
        "X-Review-Verified": "true",
        "X-Allowlist-Digest": "forged-allowlist",
        "X-Measurement-MRTD": "forged-mrtd",
        "X-Base-Internal-Token": "forged-internal",
    }
    private_paths = (
        "/challenges/agent-challenge/internal/v1/reviews/session-1/report",
        "/challenges/agent-challenge/key-release/release",
        "/challenges/agent-challenge/keyrelease/release",
        "/challenges/agent-challenge/capability/token",
        "/challenges/agent-challenge/submissions/sub-1/eval/result",
        "/challenges/agent-challenge/llm/v1/chat/completions",
    )
    for path in private_paths:
        response = client.post(
            path,
            content=b'{"forged":true}',
            headers=trust_headers,
        )
        assert response.status_code == 404, path
    assert upstream_calls == []


def test_public_submit_strips_trust_headers_non_elevating() -> None:
    """VAL-ACAT-048: strip trust headers on allowlisted public paths."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["path"] = request.url.path
        return httpx.Response(201, json={"accepted": True})

    client = _proxy_client(handler)
    response = client.post(
        "/challenges/agent-challenge/submissions",
        content=b'{"schema_version":1,"agent_zip_sha256":"' + b"a" * 64 + b'"}',
        headers={
            "Content-Type": "application/vnd.base.signed+json",
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "miner-signature",
            "X-Nonce": "miner-nonce",
            "X-Timestamp": "1700000000",
            "X-Attestation-Verified": "true",
            "X-RA-TLS-Peer-Key": "forged",
            "X-Base-Verified-Hotkey": "forged",
            "X-Trust-Level": "admin",
            "X-Public-Header": "ok",
        },
    )
    assert response.status_code == 201
    assert captured["path"] == "/submissions"
    headers: httpx.Headers = captured["headers"]
    assert headers["x-hotkey"] == "miner-hotkey"
    assert headers["x-public-header"] == "ok"
    assert "x-attestation-verified" not in headers
    assert "x-ra-tls-peer-key" not in headers
    assert "x-base-verified-hotkey" not in headers
    assert "x-trust-level" not in headers


@pytest.mark.parametrize(
    "submission_id",
    ("sub-1", "42", "abc-def"),
)
def test_enabled_mode_allowlist_allows_public_review_tee_get(
    submission_id: str,
) -> None:
    """VAL-PLATATM-001: GET submissions/{id}/review/tee is public allowlisted."""

    path = f"submissions/{submission_id}/review/tee"
    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "GET",
            path,
        )
        is True
    )


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "internal/v1/reviews/session-1/report"),
        ("GET", "internal/v1/reviews/session-1/evidence/object-1"),
        ("POST", "internal/v1/reviews/session-1/approvals"),
        ("POST", "submissions/sub-1/review/tee"),
        ("PUT", "submissions/sub-1/review/tee"),
        ("GET", "submissions/sub-1/review/tee/extra"),
        ("GET", "submissions/sub-1/review/math"),
        ("GET", "key-release/nonce"),
        ("POST", "key-release/release"),
        ("GET", "evidence/object-1"),
        ("GET", "submissions/sub-1/evidence/object-1"),
    ),
)
def test_enabled_mode_allowlist_denies_internal_and_tee_neighbors(
    method: str,
    path: str,
) -> None:
    """VAL-PLATATM-002: internal + non-public tee neighbors remain denied."""

    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            method,
            path,
        )
        is False
    )


def test_public_review_tee_get_strips_trust_headers() -> None:
    """VAL-PLATATM-002: trust header strip unchanged on public GET review/tee."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            content=b'{"available":false}',
            headers={"content-type": "application/json"},
        )

    response = _proxy_client(handler).get(
        "/challenges/agent-challenge/submissions/sub-1/review/tee",
        headers={
            "Authorization": "Bearer caller-capability",
            "X-Attestation-Verified": "true",
            "X-RA-TLS-Peer-Key": "forged",
            "X-Base-Verified-Hotkey": "forged",
            "X-Trust-Level": "admin",
            "X-Allowlist-Digest": "caller-allowlist",
            "X-Measurement-MRTD": "caller-measurement",
            "X-Review-Verified": "true",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    assert response.content == b'{"available":false}'
    assert captured["path"] == "/submissions/sub-1/review/tee"
    headers: httpx.Headers = captured["headers"]
    assert headers["x-public-header"] == "preserved"
    assert headers.get_list("x-base-proxy") == ["true"]
    assert headers.get_list("x-base-challenge-slug") == ["agent-challenge"]
    assert "authorization" not in headers
    assert "x-attestation-verified" not in headers
    assert "x-ra-tls-peer-key" not in headers
    assert "x-base-verified-hotkey" not in headers
    assert "x-trust-level" not in headers
    assert "x-allowlist-digest" not in headers
    assert "x-measurement-mrtd" not in headers
    assert "x-review-verified" not in headers


# ---------------------------------------------------------------------------
# Realtime eval telemetry capability + public pool/SSE reads (attested allowlist)
# ---------------------------------------------------------------------------


_EVAL_CAPABILITY_POST_ROUTES = (
    (
        "POST",
        "/challenges/agent-challenge/evaluation/v1/runs/run-1/progress",
        "/evaluation/v1/runs/run-1/progress",
    ),
    (
        "POST",
        "/challenges/agent-challenge/evaluation/v1/runs/run-1/telemetry-session",
        "/evaluation/v1/runs/run-1/telemetry-session",
    ),
    (
        "POST",
        "/challenges/agent-challenge/evaluation/v1/runs/run-1/result",
        "/evaluation/v1/runs/run-1/result",
    ),
)

_PUBLIC_POOL_SSE_GET_ROUTES = (
    (
        "GET",
        "/challenges/agent-challenge/submissions/sub-1/task-events",
        "/submissions/sub-1/task-events",
    ),
    (
        "GET",
        "/challenges/agent-challenge/submissions/sub-1/task-events/stream",
        "/submissions/sub-1/task-events/stream",
    ),
    (
        "GET",
        "/challenges/agent-challenge/v1/execution-pool/live",
        "/v1/execution-pool/live",
    ),
)


@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    _EVAL_CAPABILITY_POST_ROUTES,
)
def test_eval_capability_routes_allowed_when_attested_flag_on(
    method: str,
    path: str,
    upstream_path: str,
) -> None:
    """Eval-run Bearer capability POSTs must be allowlisted under attested mode."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(200, json={"ok": True})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.request(
        method,
        path,
        content=b'{"schema_version":1,"marker":true}',
        headers={
            "Authorization": "Bearer eval_run_token.deadbeef",
            "Content-Type": "application/json",
            "X-Telemetry-Session": "sess-1",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    assert captured["method"] == method
    assert captured["path"] == upstream_path


@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    _EVAL_CAPABILITY_POST_ROUTES,
)
def test_eval_capability_routes_preserve_authorization_and_strip_trust(
    method: str,
    path: str,
    upstream_path: str,
) -> None:
    """Mirror review-capability: forward Authorization; strip client trust headers."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(200, json={"ok": True})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.request(
        method,
        path,
        content=b'{"schema_version":1}',
        headers={
            "Authorization": "Bearer eval_run_token.deadbeef",
            "Proxy-Authorization": "Basic should-still-strip",
            "X-Admin-Token": "should-strip",
            "X-Base-Admin-Token": "should-strip",
            "X-Base-Internal-Token": "should-strip",
            "X-Internal-Authorization": "should-strip",
            "X-Base-Verified-Hotkey": "should-strip",
            "X-Base-Verified-Future": "should-strip",
            "X-Base-Request-Hash": "should-strip",
            "X-Trust-Level": "should-strip",
            "X-Trusted-Proxy": "should-strip",
            "X-Base-Trust-Result": "should-strip",
            "X-RA-TLS-Peer-Key": "should-strip",
            "X-RATLS-Peer-Certificate": "should-strip",
            "X-Review-Verified": "true",
            "X-Attestation-Verified": "true",
            "X-Allowlist-Digest": "should-strip",
            "X-Measurement-MRTD": "should-strip",
            "Forwarded": "for=caller",
            "X-Forwarded-For": "198.51.100.7",
            "X-Real-IP": "198.51.100.8",
            "X-Public-Header": "preserved",
            "X-Telemetry-Session": "sess-capability-1",
        },
    )

    assert response.status_code == 200
    assert captured["method"] == method
    assert captured["path"] == upstream_path
    headers: httpx.Headers = captured["headers"]
    assert headers["authorization"] == "Bearer eval_run_token.deadbeef"
    assert headers["x-public-header"] == "preserved"
    assert headers.get("x-telemetry-session") == "sess-capability-1"
    assert "proxy-authorization" not in headers
    assert "x-admin-token" not in headers
    assert "x-base-admin-token" not in headers
    assert "x-base-internal-token" not in headers
    assert "x-internal-authorization" not in headers
    assert "x-base-verified-hotkey" not in headers
    assert "x-base-verified-future" not in headers
    assert "x-base-request-hash" not in headers
    assert "x-trust-level" not in headers
    assert "x-trusted-proxy" not in headers
    assert "x-base-trust-result" not in headers
    assert "x-ra-tls-peer-key" not in headers
    assert "x-ratls-peer-certificate" not in headers
    assert "x-review-verified" not in headers
    assert "x-attestation-verified" not in headers
    assert "x-allowlist-digest" not in headers
    assert "x-measurement-mrtd" not in headers
    assert "forwarded" not in headers
    assert "x-forwarded-for" not in headers
    assert "x-real-ip" not in headers


@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    _PUBLIC_POOL_SSE_GET_ROUTES,
)
def test_public_pool_and_task_event_reads_allowed_when_attested_flag_on(
    method: str,
    path: str,
    upstream_path: str,
) -> None:
    """Public pool live + task-events (incl. SSE stream) must be allowlisted."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            content=b'{"schema_version":1,"units":[]}',
            headers={"content-type": "application/json"},
        )

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.request(
        method,
        path,
        headers={
            "Authorization": "Bearer should-not-forward-on-public-read",
            "X-Allowlist-Digest": "caller-allowlist",
            "X-Base-Verified-Hotkey": "forged",
            "X-Attestation-Verified": "true",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    assert captured["method"] == method
    assert captured["path"] == upstream_path
    headers: httpx.Headers = captured["headers"]
    assert headers["x-public-header"] == "preserved"
    assert headers.get_list("x-base-proxy") == ["true"]
    assert headers.get_list("x-base-challenge-slug") == ["agent-challenge"]
    # Public reads are not capability routes — Authorization stays stripped.
    assert "authorization" not in headers
    assert "x-allowlist-digest" not in headers
    assert "x-base-verified-hotkey" not in headers
    assert "x-attestation-verified" not in headers


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/challenges/agent-challenge/internal/v1/reviews/session-1/report"),
        ("POST", "/challenges/agent-challenge/internal/v1/eval/runs/run-1/progress"),
        ("GET", "/challenges/agent-challenge/internal/v1/execution-pool/live"),
        ("POST", "/challenges/agent-challenge/internal/v1/anything"),
    ),
)
def test_attested_mode_denies_internal_v1_paths_not_proxied(
    method: str,
    path: str,
) -> None:
    """Any /internal/v1/... under agent-challenge stays fail-closed (not proxied)."""

    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"unexpected": True})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.request(
        method,
        path,
        content=b'{"forged":true}' if method == "POST" else None,
        headers={
            "Authorization": "Bearer caller-capability",
            "X-Base-Internal-Token": "forged-internal",
            "X-Attestation-Verified": "true",
            "X-Base-Verified-Hotkey": "forged",
        },
    )

    assert response.status_code == 404
    assert upstream_calls == []


def test_attested_mode_denies_arbitrary_non_allowlisted_ac_path() -> None:
    """Arbitrary non-allowlisted AC path is local 404 (fail-closed)."""

    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"unexpected": True})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.get(
        "/challenges/agent-challenge/telemetry/v1/debug/dump",
        headers={"Authorization": "Bearer caller-capability"},
    )

    assert response.status_code == 404
    assert upstream_calls == []


def test_eval_capability_allowlist_helper_accepts_exact_shapes() -> None:
    """Direct allowlist helper pins exact evaluation/v1/runs/{id}/{action} shapes."""

    for action in ("progress", "telemetry-session", "result"):
        path = f"evaluation/v1/runs/run-42/{action}"
        assert (
            _is_agent_challenge_enabled_mode_allowed_route(
                "agent-challenge",
                "POST",
                path,
            )
            is True
        )


def test_public_pool_sse_allowlist_helper_accepts_exact_shapes() -> None:
    """Direct allowlist helper pins task-events + execution-pool/live GETs."""

    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "GET",
            "submissions/sub-9/task-events",
        )
        is True
    )
    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "GET",
            "submissions/sub-9/task-events/stream",
        )
        is True
    )
    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "GET",
            "v1/execution-pool/live",
        )
        is True
    )


def test_review_capability_still_preserves_authorization_adjacent_regression() -> None:
    """Adjacent: measured-review guest Bearer semantics unchanged by eval telemetry."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(200, json={"ok": True})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.get(
        "/challenges/agent-challenge/review/v1/assignments/assignment-1/artifact",
        headers={
            "Authorization": "Bearer ra_assignment-1.deadbeef",
            "Proxy-Authorization": "Basic should-still-strip",
            "X-Admin-Token": "should-strip",
            "X-Base-Verified-Hotkey": "should-strip",
            "X-Public-Header": "preserved",
        },
    )

    assert response.status_code == 200
    assert captured["method"] == "GET"
    assert captured["path"] == "/review/v1/assignments/assignment-1/artifact"
    headers: httpx.Headers = captured["headers"]
    assert headers["authorization"] == "Bearer ra_assignment-1.deadbeef"
    assert headers["x-public-header"] == "preserved"
    assert "proxy-authorization" not in headers
    assert "x-admin-token" not in headers
    assert "x-base-verified-hotkey" not in headers


def test_validator_assignment_progress_not_confused_with_eval_telemetry() -> None:
    """Adjacent: master POST /v1/assignments/{id}/progress is not eval telemetry.

    Validator lease heartbeat (assignment_coordination) must remain a distinct
    master route — never treated as an agent-challenge evaluation capability
    path and never allowlisted as challenge-local telemetry.
    """

    # Challenge-local lookalike must stay denied under attested mode.
    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "POST",
            "v1/assignments/asg-1/progress",
        )
        is False
    )
    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "POST",
            "assignments/asg-1/progress",
        )
        is False
    )

    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"unexpected": True})

    client = _proxy_client(handler, attested_routes_enabled=True)

    # Challenge-prefixed lookalike is fail-closed (not proxied as eval telemetry).
    challenge_lookalike = client.post(
        "/challenges/agent-challenge/v1/assignments/asg-1/progress",
        content=b'{"checkpoint_ref":"x"}',
        headers={"Authorization": "Bearer eval_run_token.deadbeef"},
    )
    assert challenge_lookalike.status_code == 404
    assert upstream_calls == []

    # Master validator lease path is NOT the challenge proxy surface. Without
    # assignment_coordination_service the route is absent (404) — never forwarded
    # upstream as agent-challenge evaluation progress.
    master_progress = client.post(
        "/v1/assignments/asg-1/progress",
        content=b'{"checkpoint_ref":"x","meta":{}}',
        headers={
            "Authorization": "Bearer should-not-become-eval-capability",
            "Content-Type": "application/json",
        },
    )
    assert master_progress.status_code == 404
    assert upstream_calls == []
    # Must not be rewritten into evaluation/v1/runs/... either.
    assert b"evaluation" not in master_progress.content.lower()


# ---------------------------------------------------------------------------
# T12: FE public reads must stay reachable in both attested-flag states
# ---------------------------------------------------------------------------


# Public FE catalog + by-hash surfaces the joinbase UI hits (base.ts paths).
_FE_PUBLIC_GET_PATHS = (
    "submissions",
    "submissions/count",
    "submissions/42",
    "submissions/sub-1",
    "submissions/42/versions",
    "submissions/42/status",
    "submissions/42/events",
    "submissions/42/task-events",
    "submissions/42/task-events/stream",
    # T9 may add hash lookup under /submissions/by-hash/{hash}; allow prefix now.
    "submissions/by-hash/ed7e204a0123456789abcdef0123456789abcdef0123456789abcdef01234567",
    "agents/ed7e204a0123456789abcdef0123456789abcdef0123456789abcdef01234567/evaluation",
    "agents/ed7e204a0123456789abcdef0123456789abcdef0123456789abcdef01234567/source",
    (
        "agents/ed7e204a0123456789abcdef0123456789abcdef0123456789abcdef01234567"
        "/source/download"
    ),
    "benchmarks",
    "benchmarks/tasks",
)


@pytest.mark.parametrize("path", _FE_PUBLIC_GET_PATHS)
def test_enabled_mode_allowlist_allows_frontend_public_reads(path: str) -> None:
    """T12/S1: attested ON allowlists FE list/detail/events/task-events/by-hash."""

    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            "GET",
            path,
        )
        is True
    )
    assert (
        _is_blocked_agent_challenge_proxy_path(
            "agent-challenge",
            "GET",
            path,
            attested_routes_enabled=True,
        )
        is False
    )


@pytest.mark.parametrize("path", _FE_PUBLIC_GET_PATHS)
def test_legacy_flag_off_does_not_block_frontend_public_reads(path: str) -> None:
    """T12/S3: attested OFF keeps legacy open (no allowlist block) for FE paths."""

    assert (
        _is_blocked_agent_challenge_proxy_path(
            "agent-challenge",
            "GET",
            path,
            attested_routes_enabled=False,
        )
        is False
    )


@pytest.mark.parametrize(
    "path",
    (
        "submissions",
        "submissions/count",
        "submissions/42",
        "submissions/42/events",
        "submissions/42/task-events",
        "submissions/42/task-events/stream",
        "submissions/by-hash/abc123",
        "agents/abc123/evaluation",
        "agents/abc123/source",
        "agents/abc123/source/download",
        "benchmarks",
        "benchmarks/tasks",
    ),
)
def test_attested_proxy_forwards_frontend_public_gets(path: str) -> None:
    """T12/S1 surface: attested ON proxies FE GETs upstream (not local 404)."""

    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["method"] = request.method
        return httpx.Response(200, json={"ok": True, "path": path})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.get(
        f"/challenges/agent-challenge/{path}",
        headers={"X-Public-Header": "preserved"},
    )

    assert response.status_code == 200, response.text
    assert captured["method"] == "GET"
    assert captured["path"] == f"/{path}"


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "key-release/nonce"),
        ("POST", "key-release/release"),
        ("GET", "internal/v1/reviews/session-1/report"),
        ("GET", "submissions/42/evidence/object-1"),
        ("GET", "evidence/object-1"),
        ("GET", "owner/submissions/42/revalidate"),
        ("POST", "submissions/42/status"),
        ("DELETE", "submissions/42"),
        ("GET", "submissions/42/status/extra"),
        ("GET", "agents/abc/source/download/extra"),
        ("POST", "agents/abc/evaluation"),
        ("GET", "agents/abc/secrets"),
    ),
)
def test_enabled_mode_still_denies_non_frontend_neighbors(
    method: str,
    path: str,
) -> None:
    """T12/S2: allowlist widen must not open key-release/internal/evidence/neighbors."""

    assert (
        _is_agent_challenge_enabled_mode_allowed_route(
            "agent-challenge",
            method,
            path,
        )
        is False
    )
    assert (
        _is_blocked_agent_challenge_proxy_path(
            "agent-challenge",
            method,
            path,
            attested_routes_enabled=True,
        )
        is True
    )

    upstream_calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"unexpected": True})

    client = _proxy_client(handler, attested_routes_enabled=True)
    response = client.request(
        method,
        f"/challenges/agent-challenge/{path}",
        content=b"{}" if method != "GET" else None,
        headers={"Authorization": "Bearer should-not-matter"},
    )
    assert response.status_code == 404
    assert upstream_calls == []
