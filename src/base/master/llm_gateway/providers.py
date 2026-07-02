"""LLM provider abstraction for the master gateway.

A provider is the seam between the gateway and an upstream LLM API. Tests use
the deterministic :class:`MockLLMProvider` (records the request it received and
returns a canned response); deploys use :class:`HttpLLMProvider` (a real
``httpx`` client). The provider in use is chosen by config
(:func:`build_providers`), so the test suite never makes a real network call.
"""

from __future__ import annotations

import json
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
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client_factory = client_factory

    def compose_url(self, path: str) -> str:
        return compose_provider_url(self.base_url, path)

    def _client(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        if self._client_factory is not None:
            return self._client_factory()
        return _default_http_client_factory(self.timeout_seconds)

    async def forward(self, request: ProviderRequest) -> ProviderResponse:
        async with self._client() as client:
            response = await client.request(
                request.method,
                self.compose_url(request.path),
                content=request.body,
                headers=dict(request.headers),
            )
        return ProviderResponse(
            status_code=response.status_code,
            body=response.content,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type"),
        )


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
        )
        for name, base_url in config.providers.items()
    }
