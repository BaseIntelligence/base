"""Streaming pass-through + real-upstream-status logging for the LLM gateway.

The eval agent calls the gateway with ``stream=true`` and needs incremental
bytes; the gateway forwards those through a :class:`StreamingResponse` WITHOUT
buffering the whole completion, retrying only pre-first-byte. Stream=false
callers (including the ``llm_review`` safety gates) stay on the unchanged
buffered path. On any upstream error the gateway now logs the REAL upstream
status (status + source only, never a body/header/key) instead of collapsing it
into an opaque 502 with no trace.

Providers are the deterministic mock (no network egress); the mock's ``stream``
serves canned SSE and can script a status sequence to drive the pre-first-byte
retry deterministically.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from base.master.app_proxy import create_proxy_app
from base.master.llm_gateway import (
    DEFAULT_PROVIDER_BASE_URL,
    GatewayTokenAuthority,
    LLMGatewayService,
    MockLLMProvider,
    ProviderResponse,
    SourceRoute,
    StreamingProviderResponse,
)

YUNWU_KEY = "sk-yunwu-server-secret-key"
OTHER_KEY = "sk-other-upstream-secret"
TOKEN_SECRET = "gateway-hmac-secret"
MODEL = "claude-opus-4-8"

SSE_CHUNKS = (
    b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
    b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
    b"data: [DONE]\n\n",
)


class FakeNonceStore:
    async def reserve(self, **_: object) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


async def _aiter(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _build(
    provider: MockLLMProvider,
) -> tuple[LLMGatewayService, GatewayTokenAuthority]:
    authority = GatewayTokenAuthority(TOKEN_SECRET)
    service = LLMGatewayService(
        providers={"yunwu": provider},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={
            "agent": SourceRoute(provider="yunwu", model=MODEL),
            "llm_review": SourceRoute(provider="yunwu", model=MODEL),
        },
    )
    return service, authority


def _make_client(service: LLMGatewayService) -> AsyncClient:
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def _stream_body() -> bytes:
    return json.dumps(
        {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    ).encode()


def _buffered_body(**extra: object) -> bytes:
    body: dict[str, object] = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(extra)
    return json.dumps(body).encode()


# 2) Buffered non-retryable persists -> 502 + the REAL status is logged.
async def test_buffered_upstream_5xx_surfaced_502_and_logs_real_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = MockLLMProvider(
        name="yunwu",
        base_url=DEFAULT_PROVIDER_BASE_URL,
        response_factory=lambda _req: ProviderResponse(
            status_code=529,
            body=json.dumps({"error": f"overloaded {YUNWU_KEY}"}).encode(),
        ),
    )
    service, authority = _build(provider)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    with caplog.at_level(logging.WARNING, logger="base.master.llm_gateway.gateway"):
        async with _make_client(service) as client:
            resp = await client.post(
                "/llm/v1/chat/completions",
                content=_buffered_body(),
                headers={"X-Gateway-Token": token},
            )
    assert resp.status_code == 502
    assert YUNWU_KEY not in resp.text
    assert "status=529" in caplog.text
    assert "stream=False" in caplog.text
    assert "source=agent" in caplog.text
    assert YUNWU_KEY not in caplog.text


# 3) 429 is still surfaced as 429 and never retried.
async def test_buffered_upstream_429_surfaced_and_not_retried() -> None:
    provider = MockLLMProvider(
        name="yunwu",
        base_url=DEFAULT_PROVIDER_BASE_URL,
        response_factory=lambda _req: ProviderResponse(
            status_code=429, body=b'{"error": "slow down"}'
        ),
    )
    service, authority = _build(provider)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    async with _make_client(service) as client:
        resp = await client.post(
            "/llm/v1/chat/completions",
            content=_buffered_body(),
            headers={"X-Gateway-Token": token},
        )
    assert resp.status_code == 429
    assert provider.call_count == 1


# 4) Streaming pass-through: caller stream=true -> incremental SSE relayed, with
#    secret/framing headers stripped exactly like the buffered path.
async def test_streaming_passthrough_relays_sse_and_strips_headers() -> None:
    def _factory(_req: object) -> StreamingProviderResponse:
        return StreamingProviderResponse(
            status_code=200,
            headers={
                "content-type": "text/event-stream",
                "Authorization": f"Bearer {YUNWU_KEY}",
                "content-length": "999",
                "content-encoding": "gzip",
                "transfer-encoding": "chunked",
                "x-trace-id": "keep-me",
            },
            aiter_bytes=_aiter(*SSE_CHUNKS),
        )

    provider = MockLLMProvider(
        name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL, stream_factory=_factory
    )
    service, authority = _build(provider)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    async with _make_client(service) as client:
        resp = await client.post(
            "/llm/v1/chat/completions",
            content=_stream_body(),
            headers={"X-Gateway-Token": token},
        )
    assert resp.status_code == 200
    assert resp.content == b"".join(SSE_CHUNKS)
    assert resp.headers["content-type"] == "text/event-stream"
    lowered = {k.lower() for k in resp.headers}
    assert "authorization" not in lowered
    assert "content-encoding" not in lowered
    assert "transfer-encoding" not in lowered
    # content-length is stripped so the body is relayed chunked (httpx may
    # recompute one for the fully-buffered client view, but never the upstream
    # value 999).
    assert resp.headers.get("content-length") != "999"
    assert resp.headers.get("x-trace-id") == "keep-me"
    # The injected provider key never appears in any relayed header value.
    assert all(YUNWU_KEY not in v for v in resp.headers.values())
    # The upstream body carried stream:true + the resolved model server-side.
    assert provider.requests[-1].json_body()["stream"] is True
    assert provider.requests[-1].json_body()["model"] == MODEL


# 5) Streaming pre-first-byte retry: 529 then 200+SSE -> the 200 body is streamed.
async def test_streaming_retries_pre_first_byte_then_streams() -> None:
    provider = MockLLMProvider(
        name="yunwu",
        base_url=DEFAULT_PROVIDER_BASE_URL,
        status_sequence=[529, 200],
        max_attempts=4,
    )
    service, authority = _build(provider)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    async with _make_client(service) as client:
        resp = await client.post(
            "/llm/v1/chat/completions",
            content=_stream_body(),
            headers={"X-Gateway-Token": token},
        )
    assert resp.status_code == 200
    assert b"data: [DONE]" in resp.content
    assert provider.stream_attempts == 2  # retried once pre-first-byte, then 200


# 6) Streaming non-retryable persists -> 502 + real status logged, NO body bytes.
async def test_streaming_non_retryable_surfaces_502_without_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = MockLLMProvider(
        name="yunwu",
        base_url=DEFAULT_PROVIDER_BASE_URL,
        status_sequence=[500],
        max_attempts=4,
    )
    service, authority = _build(provider)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    with caplog.at_level(logging.WARNING, logger="base.master.llm_gateway.gateway"):
        async with _make_client(service) as client:
            resp = await client.post(
                "/llm/v1/chat/completions",
                content=_stream_body(),
                headers={"X-Gateway-Token": token},
            )
    assert resp.status_code == 502
    # No upstream body byte reached the caller: the response is the controlled
    # error JSON, never an SSE frame.
    assert b"data:" not in resp.content
    assert json.loads(resp.content)["detail"]
    assert "status=500" in caplog.text
    assert "stream=True" in caplog.text
    assert provider.stream_attempts == 4  # bounded to max_attempts


# 7) Regression: a stream=false reviewer-style call still returns buffered JSON.
async def test_stream_false_reviewer_call_returns_buffered_json() -> None:
    provider = MockLLMProvider(name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL)
    service, authority = _build(provider)
    token = authority.issue(
        validator_hotkey="v1", assignment_id="a1", source="llm_review"
    )
    async with _make_client(service) as client:
        resp = await client.post(
            "/llm/v1/chat/completions",
            content=_buffered_body(
                stream=False,
                tools=[{"type": "function", "function": {"name": "submit_verdict"}}],
                tool_choice="auto",
            ),
            headers={"X-Gateway-Token": token},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    payload = resp.json()
    assert payload["provider"] == "yunwu"
    assert payload["object"] == "chat.completion"
    assert provider.requests[-1].json_body()["model"] == MODEL


# 8) Secret non-leak: the injected provider key appears in NO log and NO relayed
#    response header, even when the upstream echoes it back.
async def test_streaming_never_leaks_injected_key_in_logs_or_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _factory(_req: object) -> StreamingProviderResponse:
        return StreamingProviderResponse(
            status_code=200,
            headers={
                "content-type": "text/event-stream",
                "Authorization": f"Bearer {YUNWU_KEY}",
                "x-upstream-echo": OTHER_KEY,
            },
            aiter_bytes=_aiter(*SSE_CHUNKS),
        )

    provider = MockLLMProvider(
        name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL, stream_factory=_factory
    )
    service, authority = _build(provider)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    with caplog.at_level(logging.DEBUG, logger="base.master.llm_gateway.gateway"):
        async with _make_client(service) as client:
            resp = await client.post(
                "/llm/v1/chat/completions",
                content=_stream_body(),
                headers={"X-Gateway-Token": token},
            )
    assert resp.status_code == 200
    assert YUNWU_KEY not in caplog.text
    assert token not in caplog.text
    assert all(YUNWU_KEY not in v for v in resp.headers.values())
    # The stripped upstream Authorization header is not relayed at all.
    assert "authorization" not in {k.lower() for k in resp.headers}
