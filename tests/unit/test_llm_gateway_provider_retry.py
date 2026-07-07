"""Upstream-retry resilience tests for :class:`HttpLLMProvider.forward`.

In production the master gateway forwards to a yunwu upstream that intermittently
drops the connection mid-response (``httpx.RemoteProtocolError``) or returns a
transient 5xx; with no retry the gateway collapsed these to caller-facing 502s.
Because the gateway is fully buffered (no bytes reach the caller until an attempt
fully succeeds), retrying a POST completion on a FRESH client is safe. These
tests pin the bounded retry policy: transient transport errors / 5xx are retried
up to ``max_attempts`` with exponential backoff; a 429 / other 4xx is never
retried; a persistent failure surfaces after the bounded attempts.

The mock provider is unchanged (no network, no retry); these tests drive the
real HTTP provider through an injected ``client_factory`` backed by
``httpx.MockTransport`` (no network egress), mirroring the existing gateway/proxy
test fixtures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from base.master.llm_gateway.providers import (
    HttpLLMProvider,
    ProviderConfig,
    ProviderRequest,
    build_providers,
)

BASE_URL = "https://yunwu.test/v1"


def _request(path: str = "chat/completions") -> ProviderRequest:
    return ProviderRequest(
        method="POST",
        path=path,
        headers={
            "Authorization": "Bearer sk-yunwu-server-secret-key",
            "Content-Type": "application/json",
        },
        body=b'{"model": "claude-opus-4-8", "messages": []}',
    )


def _ok(body: bytes = b'{"ok": true}') -> httpx.Response:
    return httpx.Response(
        200, content=body, headers={"content-type": "application/json"}
    )


def _status(code: int) -> httpx.Response:
    return httpx.Response(
        code, content=b'{"error": "x"}', headers={"content-type": "application/json"}
    )


class ScriptedUpstream:
    """An injected ``client_factory`` whose attempts follow a scripted list.

    Each script entry is either an ``Exception`` (raised inside the request, as a
    transient transport failure) or an ``httpx.Response`` (returned). Attempts
    beyond the script reuse its last entry, so a single-entry script models a
    persistent failure. ``attempts`` counts client builds, i.e. the number of
    ``forward()`` attempts, which lets tests assert the retry bound.
    """

    def __init__(self, script: list[Exception | httpx.Response]) -> None:
        self._script = script
        self.attempts = 0

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[httpx.AsyncClient]:
        index = self.attempts
        self.attempts += 1
        entry = self._script[min(index, len(self._script) - 1)]

        async def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(entry, Exception):
                if isinstance(entry, httpx.RequestError):
                    entry.request = request
                raise entry
            return entry

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=BASE_URL
        ) as client:
            yield client


def _provider(upstream: ScriptedUpstream, *, max_attempts: int = 3) -> HttpLLMProvider:
    return HttpLLMProvider(
        name="yunwu",
        base_url=BASE_URL,
        client_factory=upstream,
        max_attempts=max_attempts,
        retry_backoff_seconds=0.0,
    )


async def test_retries_remote_protocol_error_then_succeeds() -> None:
    # The exact production failure: peer closes mid-response twice, then succeeds.
    upstream = ScriptedUpstream(
        [
            httpx.RemoteProtocolError("peer closed connection"),
            httpx.RemoteProtocolError("peer closed connection"),
            _ok(b'{"ok": true, "attempt": 3}'),
        ]
    )
    response = await _provider(upstream, max_attempts=3).forward(_request())
    assert response.status_code == 200
    assert response.body == b'{"ok": true, "attempt": 3}'
    assert upstream.attempts == 3  # (a) succeeds, (b) bounded: two retries + success


async def test_retries_transient_5xx_then_succeeds() -> None:
    upstream = ScriptedUpstream([_status(502), _ok()])
    response = await _provider(upstream, max_attempts=3).forward(_request())
    assert response.status_code == 200
    assert upstream.attempts == 2


async def test_persistent_transient_exception_exhausts_and_reraises() -> None:
    upstream = ScriptedUpstream([httpx.RemoteProtocolError("always drops")])
    with pytest.raises(httpx.RemoteProtocolError):
        await _provider(upstream, max_attempts=3).forward(_request())
    assert upstream.attempts == 3  # (c) bounded to max_attempts, then surfaces


async def test_persistent_transient_5xx_exhausts_and_returns_last_response() -> None:
    upstream = ScriptedUpstream([_status(503)])
    response = await _provider(upstream, max_attempts=3).forward(_request())
    assert response.status_code == 503  # surfaced (gateway maps 5xx -> 502)
    assert upstream.attempts == 3


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422, 429])
async def test_non_transient_status_is_not_retried(code: int) -> None:
    upstream = ScriptedUpstream([_status(code), _ok()])
    response = await _provider(upstream, max_attempts=3).forward(_request())
    assert response.status_code == code
    assert upstream.attempts == 1  # (d) returned on first attempt, never retried


async def test_max_attempts_one_disables_retry() -> None:
    upstream = ScriptedUpstream([httpx.RemoteProtocolError("drop")])
    with pytest.raises(httpx.RemoteProtocolError):
        await _provider(upstream, max_attempts=1).forward(_request())
    assert upstream.attempts == 1


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("read"),
        httpx.WriteError("write"),
        httpx.ConnectError("connect"),
        httpx.ConnectTimeout("connect-timeout"),
        httpx.ReadTimeout("read-timeout"),
        httpx.PoolTimeout("pool-timeout"),
    ],
)
async def test_each_listed_transient_exception_is_retried(exc: Exception) -> None:
    upstream = ScriptedUpstream([exc, _ok()])
    response = await _provider(upstream, max_attempts=3).forward(_request())
    assert response.status_code == 200
    assert upstream.attempts == 2


async def test_non_transient_transport_exception_is_not_retried() -> None:
    # A client-side protocol bug is NOT transient and must surface immediately.
    upstream = ScriptedUpstream([httpx.LocalProtocolError("bad request framing")])
    with pytest.raises(httpx.LocalProtocolError):
        await _provider(upstream, max_attempts=3).forward(_request())
    assert upstream.attempts == 1


async def test_backoff_between_attempts_follows_exponential_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from base.master.llm_gateway import providers as providers_module

    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(providers_module.asyncio, "sleep", fake_sleep)
    upstream = ScriptedUpstream(
        [
            httpx.RemoteProtocolError("d1"),
            httpx.RemoteProtocolError("d2"),
            _ok(),
        ]
    )
    provider = HttpLLMProvider(
        name="yunwu",
        base_url=BASE_URL,
        client_factory=upstream,
        max_attempts=3,
        retry_backoff_seconds=0.25,
        # Deterministic (identity) jitter so the wait equals the pure exponential
        # delay; production defaults to full jitter (uniform in [0, delay]).
        jitter=lambda delay: delay,
    )
    response = await provider.forward(_request())
    assert response.status_code == 200
    assert delays == [0.25, 0.5]  # exponential-ish, bounded added latency


async def test_retry_logs_are_redacted_free_of_body_and_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    secret = "sk-yunwu-server-secret-key"
    upstream = ScriptedUpstream([httpx.RemoteProtocolError(f"boom {secret}"), _ok()])
    with caplog.at_level(logging.WARNING, logger="base.master.llm_gateway.providers"):
        response = await _provider(upstream, max_attempts=3).forward(_request())
    assert response.status_code == 200
    # The retry log names only the exception class + counters, never the secret.
    assert secret not in caplog.text
    assert "RemoteProtocolError" in caplog.text


def test_default_retry_policy_on_real_provider() -> None:
    provider = build_providers(ProviderConfig(mode="real"))["yunwu"]
    assert isinstance(provider, HttpLLMProvider)
    assert provider.max_attempts == 4
    assert provider.retry_backoff_seconds == 0.25


def test_build_providers_passes_retry_policy() -> None:
    provider = build_providers(
        ProviderConfig(mode="real", retry_attempts=5, retry_backoff_seconds=0.1)
    )["yunwu"]
    assert isinstance(provider, HttpLLMProvider)
    assert provider.max_attempts == 5
    assert provider.retry_backoff_seconds == 0.1


def test_backoff_delay_schedule() -> None:
    provider = _provider(ScriptedUpstream([_ok()]))
    provider.retry_backoff_seconds = 0.25
    assert provider._backoff_delay(0) == 0.25
    assert provider._backoff_delay(1) == 0.5
    assert provider._backoff_delay(2) == 1.0


async def test_full_jitter_default_bounds_wait_within_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default (full) jitter draws uniformly in [0, delay]; assert every
    # sampled wait stays within the exponential ceiling so backoff is bounded.
    from base.master.llm_gateway import providers as providers_module

    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(providers_module.asyncio, "sleep", fake_sleep)
    upstream = ScriptedUpstream(
        [
            httpx.RemoteProtocolError("d1"),
            httpx.RemoteProtocolError("d2"),
            _ok(),
        ]
    )
    provider = HttpLLMProvider(
        name="yunwu",
        base_url=BASE_URL,
        client_factory=upstream,
        max_attempts=3,
        retry_backoff_seconds=0.25,
    )
    response = await provider.forward(_request())
    assert response.status_code == 200
    assert len(delays) == 2
    assert 0.0 <= delays[0] <= 0.25
    assert 0.0 <= delays[1] <= 0.5


async def test_buffered_retries_529_then_succeeds() -> None:
    # The newly-broadened set: Anthropic 529 "overloaded" is now retried.
    upstream = ScriptedUpstream([_status(529), _ok()])
    response = await _provider(upstream, max_attempts=4).forward(_request())
    assert response.status_code == 200
    assert upstream.attempts == 2


@pytest.mark.parametrize("code", [500, 520, 522, 524, 529])
async def test_new_transient_5xx_codes_are_retried(code: int) -> None:
    upstream = ScriptedUpstream([_status(code), _ok()])
    response = await _provider(upstream, max_attempts=4).forward(_request())
    assert response.status_code == 200
    assert upstream.attempts == 2


# --- Streaming forward (HttpLLMProvider.stream) --------------------------------

SSE_CHUNKS = (
    b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
    b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
    b"data: [DONE]\n\n",
)


def _sse(status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        content=b"".join(SSE_CHUNKS),
        headers={"content-type": "text/event-stream"},
    )


async def _collect(aiter: AsyncIterator[bytes]) -> bytes:
    out = b""
    async for chunk in aiter:
        out += chunk
    return out


async def test_stream_passthrough_yields_upstream_chunks() -> None:
    upstream = ScriptedUpstream([_sse()])
    async with _provider(upstream, max_attempts=4).stream(_request()) as sp:
        assert sp.status_code == 200
        assert sp.headers.get("content-type") == "text/event-stream"
        body = await _collect(sp.aiter_bytes)
    assert body == b"".join(SSE_CHUNKS)
    assert upstream.attempts == 1


async def test_stream_retries_pre_first_byte_5xx_then_streams() -> None:
    upstream = ScriptedUpstream([_status(529), _sse()])
    async with _provider(upstream, max_attempts=4).stream(_request()) as sp:
        assert sp.status_code == 200
        body = await _collect(sp.aiter_bytes)
    assert body == b"".join(SSE_CHUNKS)
    assert upstream.attempts == 2  # retried once pre-first-byte, then streamed


async def test_stream_retries_transient_transport_error_then_streams() -> None:
    upstream = ScriptedUpstream([httpx.ConnectError("boom"), _sse()])
    async with _provider(upstream, max_attempts=4).stream(_request()) as sp:
        assert sp.status_code == 200
        body = await _collect(sp.aiter_bytes)
    assert body == b"".join(SSE_CHUNKS)
    assert upstream.attempts == 2


async def test_stream_persistent_5xx_yields_last_status_bounded() -> None:
    upstream = ScriptedUpstream([_status(500)])
    async with _provider(upstream, max_attempts=4).stream(_request()) as sp:
        assert sp.status_code == 500
    assert upstream.attempts == 4  # bounded to max_attempts, then surfaced


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422, 429])
async def test_stream_non_transient_status_is_not_retried(code: int) -> None:
    upstream = ScriptedUpstream([_status(code), _sse()])
    async with _provider(upstream, max_attempts=4).stream(_request()) as sp:
        assert sp.status_code == code
    assert upstream.attempts == 1  # returned on first open, never retried
