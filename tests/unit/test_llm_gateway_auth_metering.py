"""Auth lifecycle, redaction, metering, and safe upstream passthrough.

Covers the master LLM gateway scoped-token lifecycle binding, per-(validator,
assignment) usage metering with no secret material, key/token redaction across
logs/responses/errors, and safe surfacing of upstream 429/5xx. A single
source-driven ``/llm/v1`` route resolves the provider (yunwu) from the token.
Providers are always the deterministic mock (no egress).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from base.master.app_proxy import create_proxy_app
from base.master.llm_gateway import (
    BASE_LLM_GATEWAY_URL_ENV,
    DEFAULT_PROVIDER_BASE_URL,
    GatewayTokenAuthority,
    InMemoryAssignmentResolver,
    InMemoryUsageRecorder,
    LLMGatewayService,
    MockLLMProvider,
    ProviderResponse,
    SourceRoute,
)

YUNWU_KEY = "sk-yunwu-server-secret-key"
TOKEN_SECRET = "gateway-hmac-secret"
MODEL = "claude-opus-4-8"


class FakeNonceStore:
    async def reserve(self, **_: object) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


class Clock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch


class Harness:
    def __init__(
        self,
        client: AsyncClient,
        service: LLMGatewayService,
        yunwu: MockLLMProvider,
        authority: GatewayTokenAuthority,
        recorder: InMemoryUsageRecorder,
        resolver: InMemoryAssignmentResolver,
    ) -> None:
        self.client = client
        self.service = service
        self.yunwu = yunwu
        self.authority = authority
        self.recorder = recorder
        self.resolver = resolver

    def token(
        self,
        *,
        validator_hotkey: str = "validator-1",
        assignment_id: str = "assignment-1",
        ttl_seconds: int = 3_600,
    ) -> str:
        self.resolver.activate(validator_hotkey, assignment_id)
        return self.authority.issue(
            validator_hotkey=validator_hotkey,
            assignment_id=assignment_id,
            ttl_seconds=ttl_seconds,
            source="agent",
        )

    async def post(
        self,
        *,
        body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        path: str = "chat/completions",
    ):
        content = json.dumps(body or {}).encode()
        return await self.client.post(
            f"/llm/v1/{path}",
            content=content,
            headers=headers or {},
        )


def _build(
    clock: Clock,
    *,
    yunwu_response: ProviderResponse | None = None,
) -> tuple[
    LLMGatewayService,
    MockLLMProvider,
    GatewayTokenAuthority,
    InMemoryUsageRecorder,
    InMemoryAssignmentResolver,
]:
    yunwu = MockLLMProvider(
        name="yunwu",
        base_url=DEFAULT_PROVIDER_BASE_URL,
        response_factory=(lambda _req: yunwu_response)
        if yunwu_response is not None
        else None,
    )
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=clock.time)
    recorder = InMemoryUsageRecorder()
    resolver = InMemoryAssignmentResolver()
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={"agent": SourceRoute(provider="yunwu", model=MODEL)},
        usage_recorder=recorder,
        assignment_resolver=resolver,
    )
    return service, yunwu, authority, recorder, resolver


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    clock = Clock(1_750_000_000.0)
    service, yunwu, authority, recorder, resolver = _build(clock)
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield Harness(client, service, yunwu, authority, recorder, resolver)
    finally:
        await client.aclose()


def _body(model: str = "agent-sent-placeholder") -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


async def test_successful_call_records_metering_row_without_secret(
    harness: Harness,
) -> None:
    token = harness.token(validator_hotkey="val-A", assignment_id="assign-A")
    response = await harness.post(body=_body(), headers={"X-Gateway-Token": token})
    assert response.status_code == 200
    assert len(harness.recorder.records) == 1
    record = harness.recorder.records[0]
    assert record.validator_hotkey == "val-A"
    assert record.assignment_id == "assign-A"
    assert record.provider == "yunwu"
    # Metering records the resolved model the gateway forwarded.
    assert record.model == MODEL
    assert record.total_tokens == 2
    serialized = json.dumps(record.__dict__)
    assert YUNWU_KEY not in serialized
    assert token not in serialized


async def test_provider_key_never_returned_to_caller(harness: Harness) -> None:
    response = await harness.post(
        body=_body(),
        headers={"X-Gateway-Token": harness.token()},
    )
    assert response.status_code == 200
    assert YUNWU_KEY not in response.text
    assert all(YUNWU_KEY not in value for value in response.headers.values())


async def test_key_and_token_redacted_in_logs(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    token = harness.token()
    with caplog.at_level(logging.DEBUG):
        ok = await harness.post(body=_body(), headers={"X-Gateway-Token": token})
        assert ok.status_code == 200
    assert YUNWU_KEY not in caplog.text
    assert token not in caplog.text


async def test_key_redacted_in_logs_on_upstream_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock(1_750_000_000.0)

    def _boom(_req: object) -> ProviderResponse:
        raise RuntimeError(f"upstream boom carrying {YUNWU_KEY}")

    yunwu = MockLLMProvider(
        name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL, response_factory=_boom
    )
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=clock.time)
    resolver = InMemoryAssignmentResolver()
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={"agent": SourceRoute(provider="yunwu", model=MODEL)},
        assignment_resolver=resolver,
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    resolver.activate("v1", "a1")
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    transport = ASGITransport(app=app)
    with caplog.at_level(logging.DEBUG):
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/llm/v1/chat/completions",
                content=json.dumps(_body()).encode(),
                headers={"X-Gateway-Token": token},
            )
    assert response.status_code == 502
    assert YUNWU_KEY not in response.text
    assert YUNWU_KEY not in caplog.text


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({}, (401, 403)),  # missing token
        ({"X-Gateway-Token": "garbage"}, (401, 403)),
    ],
)
async def test_error_bodies_never_leak_secrets(
    harness: Harness,
    headers: dict[str, str],
    expected: tuple[int, ...],
) -> None:
    token = harness.token()
    resolved = {k: (token if v == "TOKEN" else v) for k, v in headers.items()}
    response = await harness.post(body=_body(), headers=resolved)
    assert response.status_code in expected
    assert YUNWU_KEY not in response.text
    assert token not in response.text


async def test_upstream_429_surfaced_safely_without_metering() -> None:
    clock = Clock(1_750_000_000.0)
    upstream = ProviderResponse(
        status_code=429,
        body=json.dumps({"error": f"rate limited; key={YUNWU_KEY}"}).encode(),
        headers={"Authorization": f"Bearer {YUNWU_KEY}"},
    )
    service, _yunwu, authority, recorder, resolver = _build(
        clock, yunwu_response=upstream
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    resolver.activate("v1", "a1")
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/llm/v1/chat/completions",
            content=json.dumps(_body()).encode(),
            headers={"X-Gateway-Token": token},
        )
    assert response.status_code == 429
    assert YUNWU_KEY not in response.text
    assert all(YUNWU_KEY not in v for v in response.headers.values())
    assert recorder.records == []


async def test_upstream_5xx_surfaced_safely_without_metering() -> None:
    clock = Clock(1_750_000_000.0)
    upstream = ProviderResponse(
        status_code=503,
        body=json.dumps({"error": f"backend down key={YUNWU_KEY}"}).encode(),
    )
    service, _yunwu, authority, recorder, resolver = _build(
        clock, yunwu_response=upstream
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    resolver.activate("v1", "a1")
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/llm/v1/chat/completions",
            content=json.dumps(_body()).encode(),
            headers={"X-Gateway-Token": token},
        )
    assert response.status_code == 502
    assert YUNWU_KEY not in response.text
    assert recorder.records == []


async def test_unconfigured_provider_route_rejected() -> None:
    """A token whose source maps to an unconfigured provider -> 502, no call."""
    clock = Clock(1_750_000_000.0)
    yunwu = MockLLMProvider(name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL)
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=clock.time)
    recorder = InMemoryUsageRecorder()
    resolver = InMemoryAssignmentResolver()
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={"ghost": SourceRoute(provider="not-configured")},
        usage_recorder=recorder,
        assignment_resolver=resolver,
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    resolver.activate("v1", "a1")
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="ghost")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/llm/v1/chat/completions",
            content=json.dumps(_body()).encode(),
            headers={"X-Gateway-Token": token},
        )
    assert response.status_code in (404, 502)
    assert yunwu.call_count == 0
    assert recorder.records == []


async def test_consumption_contract_base_url_and_token_only(harness: Harness) -> None:
    assert BASE_LLM_GATEWAY_URL_ENV == "BASE_LLM_GATEWAY_URL"
    # A caller carrying ONLY a scoped gateway token (no provider key) succeeds.
    env = {
        "BASE_LLM_GATEWAY_URL": "http://gateway/llm/v1",
        "BASE_GATEWAY_TOKEN": harness.token(),
    }
    assert not any(key.endswith("_API_KEY") for key in env)
    response = await harness.post(
        body=_body(),
        headers={"X-Gateway-Token": env["BASE_GATEWAY_TOKEN"]},
    )
    assert response.status_code == 200
    assert harness.yunwu.requests[-1].header("Authorization") == f"Bearer {YUNWU_KEY}"


async def test_metering_failure_does_not_break_call_or_leak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock(1_750_000_000.0)

    class BoomRecorder:
        async def record(self, record: object) -> None:
            raise RuntimeError(f"db down near {YUNWU_KEY}")

    yunwu = MockLLMProvider(name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL)
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=clock.time)
    resolver = InMemoryAssignmentResolver()
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={"agent": SourceRoute(provider="yunwu", model=MODEL)},
        usage_recorder=BoomRecorder(),
        assignment_resolver=resolver,
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    resolver.activate("v1", "a1")
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", source="agent")
    transport = ASGITransport(app=app)
    with caplog.at_level(logging.DEBUG):
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/llm/v1/chat/completions",
                content=json.dumps(_body()).encode(),
                headers={"X-Gateway-Token": token},
            )
    assert response.status_code == 200
    assert YUNWU_KEY not in response.text
    assert YUNWU_KEY not in caplog.text


async def test_token_rejected_after_assignment_terminates(harness: Harness) -> None:
    token = harness.token(validator_hotkey="val-X", assignment_id="assign-X")
    active = await harness.post(body=_body(), headers={"X-Gateway-Token": token})
    assert active.status_code == 200
    assert harness.yunwu.call_count == 1
    assert len(harness.recorder.records) == 1

    harness.resolver.deactivate("val-X", "assign-X")
    rejected = await harness.post(body=_body(), headers={"X-Gateway-Token": token})
    assert rejected.status_code in (401, 403)
    assert harness.yunwu.call_count == 1
    assert len(harness.recorder.records) == 1
    assert YUNWU_KEY not in rejected.text
    assert token not in rejected.text
