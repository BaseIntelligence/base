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
    assert provider.max_attempts == 3
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
