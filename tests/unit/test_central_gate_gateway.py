"""Central-gate (non-assignment-scoped) gateway token behavior.

A ``central-gate`` token authorizes the central safety gates (agent-challenge
analyzer LLM review + prism ``llm_review`` gate) to call the master LLM gateway
WITHOUT a live work assignment. The gateway treats it as active by valid
signature + unexpired ``exp`` alone, bypassing the assignment-lifecycle resolver,
resolves the provider from the token ``source``, and records usage keyed by the
token's principal/label. The standard assignment-scoped path is left UNCHANGED
(an inactive/unowned assignment still yields 403).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

import base.cli_app.main as cli_main
from base.cli_app.main import app
from base.master.app_proxy import create_proxy_app
from base.master.llm_gateway import (
    CENTRAL_GATE_KIND,
    DEFAULT_PROVIDER_BASE_URL,
    GatewayAssignmentInactiveError,
    GatewayTokenAuthority,
    InMemoryUsageRecorder,
    LLMGatewayService,
    MockLLMProvider,
    SourceRoute,
)

TOKEN_SECRET = "central-gate-hmac-secret"
YUNWU_KEY = "sk-yunwu-server-secret-key"
MODEL = "claude-opus-4-8"


class FakeNonceStore:
    async def reserve(self, **_: object) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


class ExplodingResolver:
    """A resolver that fails the test if it is ever consulted.

    Proves the central-gate path never reaches assignment resolution.
    """

    async def is_active(self, *, validator_hotkey: str, assignment_id: str) -> bool:
        raise AssertionError(
            "assignment resolver must NOT be consulted for a central-gate token"
        )


class InactiveResolver:
    async def is_active(self, *, validator_hotkey: str, assignment_id: str) -> bool:
        return False


def _service(resolver: object) -> tuple[LLMGatewayService, InMemoryUsageRecorder]:
    recorder = InMemoryUsageRecorder()
    service = LLMGatewayService(
        providers={
            "yunwu": MockLLMProvider(name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL),
        },
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=GatewayTokenAuthority(TOKEN_SECRET, now_fn=lambda: 1_000.0),
        sources={"llm_review": SourceRoute(provider="yunwu", model=MODEL)},
        usage_recorder=recorder,
        assignment_resolver=resolver,  # type: ignore[arg-type]
    )
    return service, recorder


@pytest.fixture
async def client_and_recorder() -> AsyncIterator[
    tuple[AsyncClient, LLMGatewayService, InMemoryUsageRecorder]
]:
    service, recorder = _service(ExplodingResolver())
    app_proxy = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    transport = ASGITransport(app=app_proxy)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield client, service, recorder
    finally:
        await client.aclose()


async def test_central_gate_token_bypasses_resolver_and_meters_principal_label(
    client_and_recorder: tuple[AsyncClient, LLMGatewayService, InMemoryUsageRecorder],
) -> None:
    client, service, recorder = client_and_recorder
    token = service.issue_central_gate_token(
        principal="central-gate", label="agent-challenge", source="llm_review"
    )
    response = await client.post(
        "/llm/v1/chat/completions",
        content=json.dumps(
            {"model": "whatever", "messages": [{"role": "user", "content": "hi"}]}
        ).encode(),
        headers={"X-Gateway-Token": token},
    )
    # The ExplodingResolver would have raised had it been consulted; a 200 proves
    # the central-gate path bypassed assignment resolution.
    assert response.status_code == 200, response.text
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.validator_hotkey == "central-gate"
    assert record.assignment_id == "agent-challenge"
    assert record.provider == "yunwu"
    assert record.model == MODEL
    # No secret material is recorded.
    assert YUNWU_KEY not in json.dumps(record.__dict__)
    assert token not in json.dumps(record.__dict__)
    # The gateway injected the yunwu key + resolved model server-side.
    forwarded = service.provider("yunwu").requests[-1]  # type: ignore[attr-defined]
    assert forwarded.header("authorization") == f"Bearer {YUNWU_KEY}"
    assert forwarded.json_body()["model"] == MODEL


async def test_central_gate_ensure_active_is_a_noop_without_consulting_resolver() -> (
    None
):
    service, _recorder = _service(ExplodingResolver())
    claims = service.token_authority.verify(
        service.issue_central_gate_token(
            principal="central-gate", label="prism", source="llm_review"
        )
    )
    assert claims.kind == CENTRAL_GATE_KIND
    # Must NOT raise (and must NOT consult the ExplodingResolver).
    await service.ensure_assignment_active(claims)


async def test_assignment_kind_still_rejects_inactive_assignment() -> None:
    service, _recorder = _service(InactiveResolver())
    claims = service.token_authority.verify(
        service.issue_token(validator_hotkey="v1", assignment_id="a1", source="agent")
    )
    with pytest.raises(GatewayAssignmentInactiveError):
        await service.ensure_assignment_active(claims)


def test_cli_mint_central_gate_token_prints_verifiable_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "cli-gateway-secret"
    monkeypatch.setattr(
        cli_main,
        "load_settings",
        lambda config: SimpleNamespace(
            gateway=SimpleNamespace(token_secret=secret, token_secret_file=None)
        ),
    )
    result = CliRunner().invoke(
        app,
        [
            "master",
            "mint-central-gate-token",
            "--label",
            "agent-challenge",
            "--ttl-seconds",
            "31536000",
        ],
    )
    assert result.exit_code == 0, result.output
    token = result.output.strip()
    # ONLY the token is printed (a single non-empty line, two HMAC parts).
    assert token and len(token.splitlines()) == 1
    assert len(token.split(".")) == 2

    claims = GatewayTokenAuthority(secret).verify(token)
    assert claims.kind == CENTRAL_GATE_KIND
    assert claims.validator_hotkey == "central-gate"
    assert claims.assignment_id == "agent-challenge"
    # The CLI stamps the default llm_review source into the token.
    assert claims.source == "llm_review"


def test_cli_mint_central_gate_token_accepts_source_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "cli-gateway-secret"
    monkeypatch.setattr(
        cli_main,
        "load_settings",
        lambda config: SimpleNamespace(
            gateway=SimpleNamespace(token_secret=secret, token_secret_file=None)
        ),
    )
    result = CliRunner().invoke(
        app,
        [
            "master",
            "mint-central-gate-token",
            "--label",
            "prism",
            "--source",
            "llm_review",
            "--model",
            "claude-opus-4-8",
        ],
    )
    assert result.exit_code == 0, result.output
    claims = GatewayTokenAuthority(secret).verify(result.output.strip())
    assert claims.source == "llm_review"
    assert claims.model == "claude-opus-4-8"
