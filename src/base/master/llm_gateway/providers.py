"""LLM provider abstraction for the master gateway.

A provider is the seam between the gateway and an upstream LLM API. Tests use
the deterministic :class:`MockLLMProvider` (records the request it received and
returns a canned response); deploys use :class:`HttpLLMProvider` (a real
``httpx`` client). The provider in use is chosen by config
(:func:`build_providers`), so the test suite never makes a real network call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

#: Default provider name + upstream base for local/dev + tests. Production values
#: come from master config (``GatewaySettings.providers``); there is NO hardcoded
#: provider-enforcement constant. yunwu is OpenAI-compatible.
DEFAULT_PROVIDER_NAME = "yunwu"
DEFAULT_PROVIDER_BASE_URL = "https://yunwu.ai/v1"

ProviderMode = Literal["mock", "real"]

logger = logging.getLogger(__name__)

#: Default bounded-retry policy for :class:`HttpLLMProvider`. yunwu intermittently
#: drops a connection mid-response (``httpx.RemoteProtocolError``) or returns a
#: transient upstream 5xx; because the gateway is fully buffered (no bytes reach
#: the caller until an attempt fully succeeds) retrying a fresh request is safe.
#: Four attempts (three retries) rides out a short burst of edge overload without
#: unbounded latency.
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25

#: Upstream status codes treated as transient and retried. These are the edge /
#: upstream overload signals seen in front of yunwu: Cloudflare ``520`` (unknown
#: origin error), ``522`` (origin connection timeout), ``524`` (origin response
#: timeout); Anthropic ``529`` ("overloaded"); the generic ``500`` a slow/loaded
#: origin also emits; plus the classic ``502``/``503``/``504``. Retrying them is
#: safe because the buffered path delivers no partial bytes before an attempt
#: fully succeeds, and the streaming path only retries pre-first-byte (never once
#: a chunk has reached the caller). A ``429`` (rate limited) and any other 4xx
#: are NOT retried: the gateway surfaces them as-is.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
    {500, 502, 503, 504, 520, 522, 524, 529}
)

#: Streaming-forward httpx timeout: bounded connect/write/pool, but a generous
#: per-read (chunk) budget so a long-but-flowing completion is fine while a
#: stalled stream (no chunk within ``read``) still fails. The buffered path keeps
#: its own (shorter) whole-response timeout unchanged.
STREAMING_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)

#: Transient httpx transport failures that warrant a retry on a fresh client.
#: Deliberately enumerated (rather than the broad ``httpx.TransportError``) so a
#: client-side ``LocalProtocolError`` / ``UnsupportedProtocol`` is NOT retried.
RETRYABLE_TRANSPORT_EXCEPTIONS: tuple[type[httpx.HTTPError], ...] = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
)


def compose_provider_url(base_url: str, path: str) -> str:
    """Append a gateway ``{path}`` suffix to a provider base URL."""

    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True)
class ProviderRequest:
    """A forward request handed to a provider after key injection."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class ProviderResponse:
    """The provider's upstream response, returned to the gateway."""

    status_code: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)
    media_type: str | None = "application/json"


@dataclass(frozen=True)
class StreamingProviderResponse:
    """An upstream *streaming* response handed back to the gateway.

    Unlike :class:`ProviderResponse` the body is not buffered: ``aiter_bytes``
    is an open async byte iterator that the caller drains incrementally. The
    provider yields this WHILE its upstream stream is still open (see
    :meth:`HttpLLMProvider.stream`), so upstream chunks reach the caller without
    waiting for the whole completion. ``status_code``/``headers`` are the real
    upstream values (headers are stripped by the gateway before relay).
    """

    status_code: int
    headers: Mapping[str, str]
    aiter_bytes: AsyncIterator[bytes]


@dataclass
class RecordedProviderRequest:
    """A request captured by :class:`MockLLMProvider` for test assertions."""

    provider: str
    method: str
    path: str
    url: str
    headers: dict[str, str]
    body: bytes

    def header(self, name: str) -> str | None:
        """Case-insensitive lookup of a captured request header."""

        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None

    def json_body(self) -> dict[str, object]:
        """Decode the captured JSON body (``{}`` when absent/invalid)."""

        if not self.body:
            return {}
        try:
            parsed = json.loads(self.body)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


class LLMProvider(Protocol):
    """Forwards a key-injected request to an upstream LLM provider."""

    name: str
    base_url: str

    def compose_url(self, path: str) -> str: ...

    async def forward(self, request: ProviderRequest) -> ProviderResponse: ...

    def stream(
        self, request: ProviderRequest
    ) -> AbstractAsyncContextManager[StreamingProviderResponse]:
        """Open a streaming forward as an async context manager.

        The manager yields a :class:`StreamingProviderResponse` whose
        ``aiter_bytes`` must be consumed WHILE the manager is open (the upstream
        stream closes on exit).
        """
        ...


ClientFactory = Callable[[], AbstractAsyncContextManager[httpx.AsyncClient]]


class MockLLMProvider:
    """Deterministic provider used in tests; records inbound requests.

    Never performs network I/O. By default it returns a deterministic
    OpenAI-style chat completion so identical requests yield byte-identical
    responses; ``response_factory`` overrides this to simulate upstream
    failures (e.g. 429/5xx) without leaking secrets.

    Streaming is served by :meth:`stream`: ``stream_factory`` yields a
    caller-supplied :class:`StreamingProviderResponse` verbatim, ``status_sequence``
    scripts a bounded pre-first-byte retry (each transient
    :data:`RETRYABLE_STATUS_CODES` entry is retried up to ``max_attempts``,
    mirroring :meth:`HttpLLMProvider.stream`), and the default is a single 200
    whose body is canned SSE chunks. The buffered :meth:`forward` path is
    unchanged.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        response_factory: Callable[[ProviderRequest], ProviderResponse] | None = None,
        stream_factory: Callable[[ProviderRequest], StreamingProviderResponse]
        | None = None,
        status_sequence: Sequence[int] | None = None,
        max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.requests: list[RecordedProviderRequest] = []
        self._response_factory = response_factory
        self._stream_factory = stream_factory
        self._status_sequence = (
            list(status_sequence) if status_sequence is not None else None
        )
        #: Total streaming open attempts (>= 1); ``1`` disables scripted retry.
        self.max_attempts = max(1, max_attempts)
        #: Streaming open attempts recorded so a retry test can assert the bound.
        self.stream_attempts = 0

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def compose_url(self, path: str) -> str:
        return compose_provider_url(self.base_url, path)

    async def forward(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(
            RecordedProviderRequest(
                provider=self.name,
                method=request.method,
                path=request.path,
                url=self.compose_url(request.path),
                headers=dict(request.headers),
                body=request.body,
            )
        )
        if self._response_factory is not None:
            return self._response_factory(request)
        return self._default_response(request)

    def _default_response(self, request: ProviderRequest) -> ProviderResponse:
        model = ""
        if request.body:
            try:
                payload = json.loads(request.body)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                model = str(payload.get("model") or "")
        body = json.dumps(
            {
                "id": f"mock-{self.name}-completion",
                "object": "chat.completion",
                "model": model,
                "provider": self.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"[mock:{self.name}] deterministic completion",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            sort_keys=True,
        ).encode("utf-8")
        return ProviderResponse(
            status_code=200, body=body, media_type="application/json"
        )

    @asynccontextmanager
    async def stream(
        self, request: ProviderRequest
    ) -> AsyncIterator[StreamingProviderResponse]:
        """Serve a deterministic streaming response (no network egress).

        Mirrors :meth:`HttpLLMProvider.stream`'s shape so the gateway streaming
        path is exercised without httpx. A scripted ``status_sequence`` retries
        each transient entry pre-first-byte up to ``max_attempts`` (an error
        status yields an EMPTY body so no bytes reach the caller before the
        gateway surfaces it); a ``2xx`` yields canned SSE chunks.
        """

        self.requests.append(
            RecordedProviderRequest(
                provider=self.name,
                method=request.method,
                path=request.path,
                url=self.compose_url(request.path),
                headers=dict(request.headers),
                body=request.body,
            )
        )
        if self._stream_factory is not None:
            self.stream_attempts += 1
            yield self._stream_factory(request)
            return

        sequence = self._status_sequence or [200]
        last_index = self.max_attempts - 1
        for attempt in range(self.max_attempts):
            self.stream_attempts += 1
            status = sequence[min(attempt, len(sequence) - 1)]
            if status in RETRYABLE_STATUS_CODES and attempt != last_index:
                continue
            yield StreamingProviderResponse(
                status_code=status,
                headers={"content-type": "text/event-stream"},
                aiter_bytes=(
                    _aiter_bytes(self._default_stream_chunks(request))
                    if status < 400
                    else _aiter_bytes([])
                ),
            )
            return

    def _default_stream_chunks(self, request: ProviderRequest) -> list[bytes]:
        model = ""
        if request.body:
            try:
                payload = json.loads(request.body)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                model = str(payload.get("model") or "")
        chunks: list[bytes] = []
        for delta, finish in (
            ({"role": "assistant", "content": f"[mock:{self.name}] "}, None),
            ({"content": "streamed completion"}, "stop"),
        ):
            payload_chunk = json.dumps(
                {
                    "id": f"mock-{self.name}-stream",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "provider": self.name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                },
                sort_keys=True,
            ).encode("utf-8")
            chunks.append(b"data: " + payload_chunk + b"\n\n")
        chunks.append(b"data: [DONE]\n\n")
        return chunks


async def _aiter_bytes(chunks: Sequence[bytes]) -> AsyncIterator[bytes]:
    """Async byte iterator over an in-memory chunk list (mock streaming body)."""

    for chunk in chunks:
        yield chunk


@asynccontextmanager
async def _default_http_client_factory(
    timeout_seconds: float | httpx.Timeout,
) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        timeout=timeout_seconds, follow_redirects=False
    ) as client:
        yield client


class HttpLLMProvider:
    """Real provider that forwards to an upstream LLM API over ``httpx``.

    Constructed (but never invoked against the network) in tests; the live
    deploy selects it via config.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        timeout_seconds: float = 30.0,
        client_factory: ClientFactory | None = None,
        max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        #: Total attempts (>= 1); ``1`` disables retry.
        self.max_attempts = max(1, max_attempts)
        #: Base backoff; the wait after attempt ``i`` is ``base * 2**i`` seconds.
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._client_factory = client_factory
        #: Full-jitter transform applied to each backoff delay to avoid a
        #: thundering herd of simultaneous retries under shared upstream
        #: overload. Injectable so tests can make the wait deterministic (e.g.
        #: identity ``lambda d: d`` or ``lambda d: 0.0``); the default draws
        #: uniformly in ``[0, delay]``.
        self._jitter: Callable[[float], float] = (
            jitter if jitter is not None else (lambda delay: random.uniform(0.0, delay))
        )

    def compose_url(self, path: str) -> str:
        return compose_provider_url(self.base_url, path)

    def _client(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        if self._client_factory is not None:
            return self._client_factory()
        return _default_http_client_factory(self.timeout_seconds)

    def _stream_client(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        """Client for the streaming path (per-read timeout, not whole-response).

        Uses the injected factory when present (tests); otherwise a fresh client
        with :data:`STREAMING_TIMEOUT` so a stalled stream fails while a
        long-but-flowing completion does not.
        """

        if self._client_factory is not None:
            return self._client_factory()
        return _default_http_client_factory(STREAMING_TIMEOUT)

    async def forward(self, request: ProviderRequest) -> ProviderResponse:
        """Forward the request, retrying transient upstream failures.

        Each attempt uses a FRESH client (no pool reuse). A transient transport
        failure (:data:`RETRYABLE_TRANSPORT_EXCEPTIONS`) or a transient upstream
        status (:data:`RETRYABLE_STATUS_CODES`) is retried up to ``max_attempts``
        with exponential backoff; a persistent transport failure re-raises the
        last exception and a persistent transient status returns the last
        response (both surface as a controlled 502 at the gateway). A 429 / other
        4xx (and any 2xx/3xx) returns immediately and is never retried.

        Retrying is safe: the gateway is fully buffered, so no partial bytes are
        ever delivered to the caller before an attempt fully succeeds.
        """

        url = self.compose_url(request.path)
        headers = dict(request.headers)
        last_index = self.max_attempts - 1
        for attempt in range(self.max_attempts):
            try:
                async with self._client() as client:
                    response = await client.request(
                        request.method,
                        url,
                        content=request.body,
                        headers=headers,
                    )
            except RETRYABLE_TRANSPORT_EXCEPTIONS as exc:
                if attempt == last_index:
                    raise
                # Log only the exception class + attempt counters: the message,
                # request body, URL, and headers are never logged, so no injected
                # provider key can leak from this module's logger.
                logger.warning(
                    "gateway upstream transient error (%s); retrying attempt %d/%d",
                    type(exc).__name__,
                    attempt + 2,
                    self.max_attempts,
                )
                await self._backoff(attempt)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt != last_index:
                logger.warning(
                    "gateway upstream transient status %d; retrying attempt %d/%d",
                    response.status_code,
                    attempt + 2,
                    self.max_attempts,
                )
                await self._backoff(attempt)
                continue

            return ProviderResponse(
                status_code=response.status_code,
                body=response.content,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )

        raise RuntimeError(  # pragma: no cover - loop always returns or raises
            "retry loop exited without returning a response"
        )

    @asynccontextmanager
    async def stream(
        self, request: ProviderRequest
    ) -> AsyncIterator[StreamingProviderResponse]:
        """Open a streaming forward, retrying transient failures pre-first-byte.

        Mirrors :meth:`forward` but yields the upstream response as an OPEN
        stream so incremental bytes reach the caller without buffering the whole
        completion. The retry contract is deliberately narrower than the
        buffered path: a transient upstream status (:data:`RETRYABLE_STATUS_CODES`)
        or transport failure (:data:`RETRYABLE_TRANSPORT_EXCEPTIONS`) seen while
        OPENING the stream is retried up to ``max_attempts`` with jittered
        backoff, and the ``async with`` stream/client contexts close on each
        ``continue`` so no partial body is delivered. Once
        :class:`StreamingProviderResponse` is yielded the stream is committed:
        the caller consumes ``aiter_bytes`` while this context is held open and a
        mid-stream failure is NEVER retried (it would double-deliver bytes).

        The per-read timeout (:data:`STREAMING_TIMEOUT`) fails a stalled stream
        without capping a long-but-flowing completion.
        """

        url = self.compose_url(request.path)
        headers = dict(request.headers)
        last_index = self.max_attempts - 1
        for attempt in range(self.max_attempts):
            # The OPEN (sending the request + reading response status/headers) is
            # the only part that may be retried. It is wrapped in its own stack so
            # a transient failure closes the client/stream before the next
            # attempt; the ``yield`` below is deliberately OUTSIDE this try/except
            # so an error raised WHILE the caller consumes the body (thrown back
            # into this generator at the yield) can NEVER re-enter the retry loop.
            stack = AsyncExitStack()
            try:
                client = await stack.enter_async_context(self._stream_client())
                response = await stack.enter_async_context(
                    client.stream(
                        request.method,
                        url,
                        content=request.body,
                        headers=headers,
                    )
                )
            except RETRYABLE_TRANSPORT_EXCEPTIONS as exc:
                await stack.aclose()
                if attempt == last_index:
                    raise
                # Log only the exception class + attempt counters (no body, URL,
                # or headers) so no injected provider key can leak.
                logger.warning(
                    "gateway upstream transient error (%s) (streamed); "
                    "retrying attempt %d/%d",
                    type(exc).__name__,
                    attempt + 2,
                    self.max_attempts,
                )
                await self._backoff(attempt)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt != last_index:
                await stack.aclose()
                logger.warning(
                    "gateway upstream transient status %d (streamed); "
                    "retrying attempt %d/%d",
                    response.status_code,
                    attempt + 2,
                    self.max_attempts,
                )
                await self._backoff(attempt)
                continue

            try:
                yield StreamingProviderResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    aiter_bytes=response.aiter_bytes(),
                )
            finally:
                await stack.aclose()
            return

        raise RuntimeError(  # pragma: no cover - loop always yields or raises
            "stream retry loop exited without yielding a response"
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential ceiling (seconds) for a failed 0-based ``attempt``.

        Pure and jitter-free: with the default base (0.25s) this yields 0.25s,
        0.5s, then 1.0s. :meth:`_backoff` applies the (full-)jitter transform to
        this ceiling before sleeping.
        """

        return self.retry_backoff_seconds * (2**attempt)

    async def _backoff(self, attempt: int) -> None:
        wait = self._jitter(self._backoff_delay(attempt))
        if wait > 0:
            await asyncio.sleep(wait)


def _default_provider_registry() -> dict[str, str]:
    return {DEFAULT_PROVIDER_NAME: DEFAULT_PROVIDER_BASE_URL}


@dataclass(frozen=True)
class ProviderConfig:
    """Config that selects the provider implementation + the provider registry.

    ``providers`` maps a provider name to its upstream base URL. It is fully
    config-driven (no hardcoded provider list): the master builds it from
    ``GatewaySettings.providers``; the default is yunwu-only for local/dev + tests.
    """

    mode: ProviderMode = "mock"
    providers: Mapping[str, str] = field(default_factory=_default_provider_registry)
    timeout_seconds: float = 30.0
    #: Bounded upstream-retry policy for the real HTTP provider (transient drops /
    #: 5xx). Ignored by the mock provider.
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS


def build_providers(config: ProviderConfig) -> dict[str, LLMProvider]:
    """Build the configured providers, keyed by name, selected by ``config.mode``.

    ``mock`` returns deterministic in-process providers (no egress); ``real``
    constructs HTTP clients pinned to each configured upstream base.
    """

    if config.mode == "mock":
        return {
            name: MockLLMProvider(name=name, base_url=base_url)
            for name, base_url in config.providers.items()
        }
    return {
        name: HttpLLMProvider(
            name=name,
            base_url=base_url,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.retry_attempts,
            retry_backoff_seconds=config.retry_backoff_seconds,
        )
        for name, base_url in config.providers.items()
    }
