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
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
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
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25

#: Upstream status codes treated as transient and retried. A 429 (rate limited)
#: and any other 4xx are NOT retried: the gateway surfaces them as-is.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})

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


ClientFactory = Callable[[], AbstractAsyncContextManager[httpx.AsyncClient]]


class MockLLMProvider:
    """Deterministic provider used in tests; records inbound requests.

    Never performs network I/O. By default it returns a deterministic
    OpenAI-style chat completion so identical requests yield byte-identical
    responses; ``response_factory`` overrides this to simulate upstream
    failures (e.g. 429/5xx) without leaking secrets.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        response_factory: Callable[[ProviderRequest], ProviderResponse] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.requests: list[RecordedProviderRequest] = []
        self._response_factory = response_factory

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
async def _default_http_client_factory(
    timeout_seconds: float,
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
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        #: Total attempts (>= 1); ``1`` disables retry.
        self.max_attempts = max(1, max_attempts)
        #: Base backoff; the wait after attempt ``i`` is ``base * 2**i`` seconds.
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._client_factory = client_factory

    def compose_url(self, path: str) -> str:
        return compose_provider_url(self.base_url, path)

    def _client(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        if self._client_factory is not None:
            return self._client_factory()
        return _default_http_client_factory(self.timeout_seconds)

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

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential wait (seconds) after a failed 0-based ``attempt``.

        With the default base (0.25s) this yields 0.25s then 0.5s.
        """

        return self.retry_backoff_seconds * (2**attempt)

    async def _backoff(self, attempt: int) -> None:
        delay = self._backoff_delay(attempt)
        if delay > 0:
            await asyncio.sleep(delay)


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
