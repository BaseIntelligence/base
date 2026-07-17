from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from base.config.settings import MasterSettings
from base.master.app_proxy import create_proxy_app
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
        ("GET", "/submissions/sub-1/eval/prepare"),
        ("POST", "/submissions/sub-1/eval/status"),
        ("POST", "/submissions/sub-1/eval/result"),
        ("POST", "/submissions/sub-1/eval/key-release"),
        ("GET", "/submissions/sub-1/env"),
        ("PUT", "/submissions/sub-1/env"),
        ("POST", "/submissions/sub-1/env/confirm-empty"),
        ("POST", "/submissions/sub-1/launch"),
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
        ("POST", "/evaluation/v1/runs/run-1/result"),
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
        (
            "GET",
            "/challenges/agent-challenge/benchmarks/tasks",
            "/benchmarks/tasks",
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
