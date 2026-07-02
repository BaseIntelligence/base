"""Behavioral tests for the master LLM gateway core (VAL-LLM-CODE-002/003).

A single source-driven route ``POST /llm/v1/{path}`` resolves the provider +
model from the token (yunwu-only), overwrites the request-body model, and injects
the provider key server-side; the caller holds only a scoped token. Providers are
always the deterministic mock (no network egress).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from base.master.app_proxy import create_proxy_app
from base.master.llm_gateway import (
    DEFAULT_PROVIDER_BASE_URL,
    GatewayTokenAuthority,
    HttpLLMProvider,
    LLMGatewayService,
    MockLLMProvider,
    ProviderConfig,
    ProviderResponse,
    SourceRoute,
    build_providers,
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
        clock: Clock,
    ) -> None:
        self.client = client
        self.service = service
        self.yunwu = yunwu
        self.authority = authority
        self.clock = clock

    def token(
        self,
        *,
        validator_hotkey: str = "validator-1",
        assignment_id: str = "assignment-1",
        ttl_seconds: int = 3_600,
        source: str | None = "agent",
    ) -> str:
        return self.authority.issue(
            validator_hotkey=validator_hotkey,
            assignment_id=assignment_id,
            ttl_seconds=ttl_seconds,
            source=source,
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


def _build_service(
    clock: Clock,
    *,
    yunwu_response: ProviderResponse | None = None,
) -> tuple[LLMGatewayService, MockLLMProvider, GatewayTokenAuthority]:
    yunwu = MockLLMProvider(
        name="yunwu",
        base_url=DEFAULT_PROVIDER_BASE_URL,
        response_factory=(lambda _req: yunwu_response)
        if yunwu_response is not None
        else None,
    )
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=clock.time)
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={
            "agent": SourceRoute(provider="yunwu", model=MODEL),
            "llm_review": SourceRoute(provider="yunwu", model=MODEL),
        },
    )
    return service, yunwu, authority


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    clock = Clock(1_750_000_000.0)
    service, yunwu, authority = _build_service(clock)
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield Harness(client, service, yunwu, authority, clock)
    finally:
        await client.aclose()


def _body(model: str = "agent-sent-placeholder") -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


# VAL-LLM-CODE-002
async def test_forwards_with_injected_key_and_resolved_model(harness: Harness) -> None:
    response = await harness.post(
        body=_body(),
        headers={"X-Gateway-Token": harness.token()},
    )
    assert response.status_code == 200
    assert harness.yunwu.call_count == 1
    recorded = harness.yunwu.requests[0]
    # Server injected the configured yunwu key; the caller sent no key.
    assert recorded.header("Authorization") == f"Bearer {YUNWU_KEY}"
    # The gateway overwrote the request-body model with the resolved model.
    assert recorded.json_body()["model"] == MODEL
    body = response.json()
    assert body["provider"] == "yunwu"


# VAL-LLM-CODE-002: the model is overwritten regardless of what the caller sent.
@pytest.mark.parametrize(
    "sent",
    [
        {"model": "deepseek-v4-pro", "messages": []},
        {"model": "gpt-4o", "messages": []},
        {"model": "", "messages": []},
        {"messages": []},
    ],
)
async def test_gateway_overwrites_any_caller_model(
    harness: Harness, sent: dict[str, object]
) -> None:
    response = await harness.post(
        body=sent, headers={"X-Gateway-Token": harness.token()}
    )
    assert response.status_code == 200
    assert harness.yunwu.requests[-1].json_body()["model"] == MODEL


# VAL-LLM-CODE-003: single /llm/v1 route replaces the provider-path routes.
async def test_only_llm_v1_route_exists(harness: Harness) -> None:
    ok = await harness.post(body=_body(), headers={"X-Gateway-Token": harness.token()})
    assert ok.status_code == 200

    # The old provider-path routes no longer exist (route not found -> 404).
    for legacy in (
        "/llm/deepseek/chat/completions",
        "/llm/openrouter/chat/completions",
    ):
        gone = await harness.client.post(
            legacy,
            content=json.dumps(_body()).encode(),
            headers={"X-Gateway-Token": harness.token()},
        )
        assert gone.status_code == 404
    assert harness.yunwu.call_count == 1


# VAL-LLM-CODE-004: the real provider targets the configured yunwu base (no
# api.deepseek.com / openrouter.ai literal), and the mock is the test default.
async def test_real_provider_targets_configured_base() -> None:
    real = build_providers(ProviderConfig(mode="real"))
    yunwu = real["yunwu"]
    assert isinstance(yunwu, HttpLLMProvider)
    assert yunwu.base_url == "https://yunwu.ai/v1"
    assert (
        yunwu.compose_url("chat/completions") == "https://yunwu.ai/v1/chat/completions"
    )
    mock = build_providers(ProviderConfig(mode="mock"))
    assert isinstance(mock["yunwu"], MockLLMProvider)


async def test_caller_never_supplies_provider_key(harness: Harness) -> None:
    # (a) Works with NO provider key at all.
    no_key = await harness.post(
        body=_body(),
        headers={"X-Gateway-Token": harness.token()},
    )
    assert no_key.status_code == 200

    # (b) A bogus caller-supplied key is NOT forwarded; the server key is.
    bogus = await harness.post(
        body=_body(),
        headers={
            "X-Gateway-Token": harness.token(),
            "Authorization": "Bearer bogus-caller-provider-key",
        },
    )
    assert bogus.status_code == 200
    forwarded_auth = harness.yunwu.requests[-1].header("Authorization")
    assert forwarded_auth == f"Bearer {YUNWU_KEY}"
    assert "bogus-caller-provider-key" not in str(forwarded_auth)


# VAL-LLM-CODE-002: the llm_review source resolves the same yunwu route.
@pytest.mark.parametrize(
    "review_body",
    [
        {
            "model": "whatever",
            "messages": [
                {"role": "system", "content": "reviewer"},
                {"role": "user", "content": "manifest"},
            ],
            "tools": [{"type": "function", "function": {"name": "submit_verdict"}}],
            "tool_choice": "auto",
        },
        {
            "model": "whatever",
            "messages": [{"role": "user", "content": "prism review"}],
            "tools": [{"type": "function", "function": {"name": "SubmitVerdict"}}],
            "tool_choice": {"type": "function", "function": {"name": "SubmitVerdict"}},
            "temperature": 0,
        },
    ],
)
async def test_llm_review_source_serves_both_review_consumers(
    harness: Harness, review_body: dict[str, object]
) -> None:
    response = await harness.post(
        body=review_body,
        headers={"X-Gateway-Token": harness.token(source="llm_review")},
    )
    assert response.status_code == 200
    recorded = harness.yunwu.requests[-1]
    assert recorded.header("Authorization") == f"Bearer {YUNWU_KEY}"
    assert recorded.json_body()["model"] == MODEL


async def test_missing_gateway_token_rejected(harness: Harness) -> None:
    response = await harness.post(body=_body(), headers={})
    assert response.status_code in (401, 403)
    assert harness.yunwu.call_count == 0
    assert YUNWU_KEY not in response.text


@pytest.mark.parametrize("token", ["garbage", "not.a.real.token", "....", "a.b.c"])
async def test_invalid_gateway_token_rejected(harness: Harness, token: str) -> None:
    response = await harness.post(body=_body(), headers={"X-Gateway-Token": token})
    assert response.status_code in (401, 403)
    assert harness.yunwu.call_count == 0


async def test_expired_gateway_token_rejected(harness: Harness) -> None:
    token = harness.token(ttl_seconds=10)
    harness.clock.epoch += 20
    response = await harness.post(body=_body(), headers={"X-Gateway-Token": token})
    assert response.status_code in (401, 403)
    assert harness.yunwu.call_count == 0
    assert "expir" in response.text.lower() or "invalid" in response.text.lower()


async def test_gateway_token_scoped_per_assignment_validator(harness: Harness) -> None:
    token = harness.token(validator_hotkey="validator-A", assignment_id="assignment-A")

    in_scope = await harness.post(
        body=_body(),
        headers={
            "X-Gateway-Token": token,
            "X-Gateway-Validator": "validator-A",
            "X-Gateway-Assignment": "assignment-A",
        },
    )
    assert in_scope.status_code == 200
    assert harness.yunwu.call_count == 1

    cross_assignment = await harness.post(
        body=_body(),
        headers={
            "X-Gateway-Token": token,
            "X-Gateway-Validator": "validator-A",
            "X-Gateway-Assignment": "assignment-B",
        },
    )
    assert cross_assignment.status_code == 403
    assert harness.yunwu.call_count == 1

    cross_validator = await harness.post(
        body=_body(),
        headers={
            "X-Gateway-Token": token,
            "X-Gateway-Validator": "validator-B",
            "X-Gateway-Assignment": "assignment-A",
        },
    )
    assert cross_validator.status_code == 403
    assert harness.yunwu.call_count == 1


async def test_upstream_failure_is_surfaced_without_leaking_secrets() -> None:
    clock = Clock(1_750_000_000.0)

    def _boom(_request: object) -> ProviderResponse:
        raise RuntimeError(f"upstream boom with {YUNWU_KEY}")

    yunwu = MockLLMProvider(
        name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL, response_factory=_boom
    )
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=clock.time)
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={"agent": SourceRoute(provider="yunwu", model=MODEL)},
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/llm/v1/chat/completions",
            content=json.dumps(_body()).encode(),
            headers={
                "X-Gateway-Token": authority.issue(
                    validator_hotkey="v1", assignment_id="a1", source="agent"
                )
            },
        )
    assert response.status_code == 502
    assert YUNWU_KEY not in response.text
