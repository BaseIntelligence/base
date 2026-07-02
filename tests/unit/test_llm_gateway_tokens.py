"""Gateway token issuance/verification and provider seams (VAL-LLM-CODE-001/004).

Tokens carry optional ``source``/``model`` claims; the gateway resolves the
provider + model from the token's ``source`` against config (yunwu-only) and
injects the provider key + overwrites the request-body model. Providers are the
deterministic mock (no egress).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from base.master.llm_gateway import (
    ASSIGNMENT_KIND,
    CENTRAL_GATE_KIND,
    DEFAULT_PROVIDER_BASE_URL,
    GatewayTokenAuthority,
    GatewayTokenExpired,
    GatewayTokenInvalid,
    GatewayTokenScopeError,
    HttpLLMProvider,
    LLMGatewayService,
    MockLLMProvider,
    ProviderConfig,
    ProviderRequest,
    SourceRoute,
    UnknownProviderError,
    build_llm_gateway_service,
    compose_provider_url,
)

SECRET = "unit-secret"
YUNWU_KEY = "sk-yunwu-server-secret-key"


def test_issue_then_verify_defaults_to_assignment_kind() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=60)
    assert authority.verify(token).kind == ASSIGNMENT_KIND


def test_issue_central_gate_round_trips_kind_and_slots() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue_central_gate(
        principal="central-gate", label="agent-challenge", ttl_seconds=60
    )
    claims = authority.verify(token)
    assert claims.kind == CENTRAL_GATE_KIND
    # The principal/label reuse the v/a slots (no new claim columns).
    assert claims.validator_hotkey == "central-gate"
    assert claims.assignment_id == "agent-challenge"
    assert claims.expires_at == 1060


def test_central_gate_default_ttl_used_when_unset() -> None:
    authority = GatewayTokenAuthority(
        SECRET, now_fn=lambda: 1000.0, default_ttl_seconds=42
    )
    token = authority.issue_central_gate(principal="central-gate", label="prism")
    assert authority.verify(token).expires_at == 1042


# VAL-LLM-CODE-001: source/model claims round-trip; absence yields None.
def test_source_and_model_claims_round_trip() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue(
        validator_hotkey="v1",
        assignment_id="a1",
        ttl_seconds=60,
        source="agent",
        model="claude-opus-4-8",
    )
    claims = authority.verify(token)
    assert claims.source == "agent"
    assert claims.model == "claude-opus-4-8"


def test_missing_source_and_model_verify_as_none() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    # An "old" token minted without the new claims still verifies with None.
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=60)
    claims = authority.verify(token)
    assert claims.source is None
    assert claims.model is None
    # A token carrying only source (no model) leaves model None.
    only_source = authority.issue(
        validator_hotkey="v1", assignment_id="a1", ttl_seconds=60, source="llm_review"
    )
    only_claims = authority.verify(only_source)
    assert only_claims.source == "llm_review"
    assert only_claims.model is None


def test_central_gate_carries_source_claim() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue_central_gate(
        principal="central-gate", label="prism", source="llm_review"
    )
    claims = authority.verify(token)
    assert claims.kind == CENTRAL_GATE_KIND
    assert claims.source == "llm_review"
    assert claims.model is None


def test_unknown_kind_is_rejected_as_invalid() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    # Forge a payload carrying an unknown ``k`` claim, signed with the real secret
    # so only the kind check can reject it.
    import base64
    import json as _json

    payload = {"k": "rogue", "v": "v1", "a": "a1", "exp": 2000}
    payload_b64 = (
        base64.urlsafe_b64encode(
            _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    signature = authority._sign(payload_b64)
    with pytest.raises(GatewayTokenInvalid):
        authority.verify(f"{payload_b64}.{signature}")


class Clock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch


def test_issue_then_verify_round_trips_claims() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=60)
    claims = authority.verify(token)
    assert claims.validator_hotkey == "v1"
    assert claims.assignment_id == "a1"
    assert claims.expires_at == 1060


def test_empty_secret_rejected() -> None:
    with pytest.raises(ValueError):
        GatewayTokenAuthority("")


@pytest.mark.parametrize("token", ["", None, "one-part", "a.b.c", "x.", ".y"])
def test_malformed_token_is_invalid(token: str | None) -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    with pytest.raises(GatewayTokenInvalid):
        authority.verify(token)


def test_forged_signature_is_invalid() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=60)
    payload_b64, _signature = token.split(".")
    forged = f"{payload_b64}.deadbeef"
    with pytest.raises(GatewayTokenInvalid):
        authority.verify(forged)


def test_token_signed_by_other_secret_is_invalid() -> None:
    issuer = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = issuer.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=60)
    other = GatewayTokenAuthority("different-secret", now_fn=lambda: 1000.0)
    with pytest.raises(GatewayTokenInvalid):
        other.verify(token)


def test_expired_token_raises_expired() -> None:
    clock = Clock(1000.0)
    authority = GatewayTokenAuthority(SECRET, now_fn=clock.time)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=10)
    clock.epoch = 1011.0
    with pytest.raises(GatewayTokenExpired):
        authority.verify(token)


def test_scope_mismatch_raises_scope_error() -> None:
    authority = GatewayTokenAuthority(SECRET, now_fn=lambda: 1000.0)
    token = authority.issue(validator_hotkey="v1", assignment_id="a1", ttl_seconds=60)
    with pytest.raises(GatewayTokenScopeError):
        authority.verify(token, expected_validator="v2")
    with pytest.raises(GatewayTokenScopeError):
        authority.verify(token, expected_assignment="a2")
    # Matching scope passes.
    claims = authority.verify(token, expected_validator="v1", expected_assignment="a1")
    assert claims.assignment_id == "a1"


def test_compose_provider_url_strips_slashes() -> None:
    assert (
        compose_provider_url("https://yunwu.ai/v1/", "/chat/completions")
        == "https://yunwu.ai/v1/chat/completions"
    )


def test_mock_provider_is_deterministic_and_records_request() -> None:
    provider = MockLLMProvider(name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL)
    request = ProviderRequest(
        method="POST",
        path="chat/completions",
        headers={"Authorization": "Bearer server-key"},
        body=json.dumps({"model": "claude-opus-4-8"}).encode(),
    )

    first = asyncio.run(provider.forward(request))
    second = asyncio.run(provider.forward(request))

    assert first.body == second.body  # deterministic
    assert provider.call_count == 2
    recorded = provider.requests[0]
    assert recorded.header("authorization") == "Bearer server-key"
    assert recorded.url == "https://yunwu.ai/v1/chat/completions"
    assert recorded.json_body()["model"] == "claude-opus-4-8"


def test_build_service_real_mode_requires_provider_keys() -> None:
    with pytest.raises(ValueError, match="yunwu"):
        build_llm_gateway_service(
            api_keys={"yunwu": ""},
            token_secret=SECRET,
            provider_config=ProviderConfig(mode="real"),
        )


def test_build_service_real_mode_succeeds_with_keys() -> None:
    service = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=SECRET,
        provider_config=ProviderConfig(mode="real"),
    )
    assert isinstance(service, LLMGatewayService)
    assert isinstance(service.provider("yunwu"), HttpLLMProvider)


def test_build_service_mock_mode_allows_empty_keys() -> None:
    service = build_llm_gateway_service(
        api_keys={"yunwu": ""},
        token_secret=SECRET,
        provider_config=ProviderConfig(mode="mock"),
    )
    token = service.issue_token(validator_hotkey="v1", assignment_id="a1")
    assert service.token_authority.verify(token).validator_hotkey == "v1"


# VAL-LLM-CODE-002: source=agent resolves yunwu + claude-opus-4-8 from config,
# overwrites the request-body model, and injects the provider key.
def test_build_service_injects_key_and_resolves_model_from_source() -> None:
    service = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=SECRET,
        provider_config=ProviderConfig(mode="mock"),
        sources={"agent": SourceRoute(provider="yunwu", model="claude-opus-4-8")},
    )
    token = service.issue_token(
        validator_hotkey="v1", assignment_id="a1", source="agent"
    )
    call = service.authenticate(
        token=token, expected_validator=None, expected_assignment=None
    )
    assert call.provider == "yunwu"
    assert call.model == "claude-opus-4-8"

    response = asyncio.run(
        service.forward(
            provider=call.provider,
            model=call.model,
            # The caller sent a bogus model; the gateway overwrites it.
            body=json.dumps({"model": "whatever-the-agent-sent"}).encode(),
            path="chat/completions",
            caller_headers={},
        )
    )
    assert response.status_code == 200
    provider = service.provider("yunwu")
    assert isinstance(provider, MockLLMProvider)
    forwarded = provider.requests[-1]
    assert forwarded.header("Authorization") == f"Bearer {YUNWU_KEY}"
    # The forwarded body model was overwritten with the resolved model.
    assert forwarded.json_body()["model"] == "claude-opus-4-8"


def test_token_model_claim_overrides_source_route_model() -> None:
    service = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=SECRET,
        provider_config=ProviderConfig(mode="mock"),
        sources={"agent": SourceRoute(provider="yunwu", model="claude-opus-4-8")},
    )
    token = service.issue_token(
        validator_hotkey="v1", assignment_id="a1", source="agent", model="pinned-model"
    )
    call = service.authenticate(
        token=token, expected_validator=None, expected_assignment=None
    )
    assert call.model == "pinned-model"


def test_missing_source_falls_back_to_default_route() -> None:
    service = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=SECRET,
        provider_config=ProviderConfig(mode="mock"),
        sources={"agent": SourceRoute(provider="yunwu", model="claude-opus-4-8")},
    )
    # A token with no source claim -> default source "agent" -> yunwu route.
    token = service.issue_token(validator_hotkey="v1", assignment_id="a1")
    call = service.authenticate(
        token=token, expected_validator=None, expected_assignment=None
    )
    assert call.provider == "yunwu"
    assert call.model == "claude-opus-4-8"


def test_service_unknown_provider_raises() -> None:
    service = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=SECRET,
    )
    with pytest.raises(UnknownProviderError):
        service.provider("bogus")
    # A token whose source maps to a provider that is not configured is rejected
    # at authenticate time (server-side misconfiguration).
    token = service.issue_token(
        validator_hotkey="v1", assignment_id="a1", source="ghost"
    )
    service_with_ghost = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=SECRET,
        sources={"ghost": SourceRoute(provider="not-configured")},
    )
    ghost_token = service_with_ghost.issue_token(
        validator_hotkey="v1", assignment_id="a1", source="ghost"
    )
    with pytest.raises(UnknownProviderError):
        service_with_ghost.authenticate(
            token=ghost_token, expected_validator=None, expected_assignment=None
        )
    # A source with no route + default_provider present still resolves fine.
    assert (
        service.authenticate(
            token=token, expected_validator=None, expected_assignment=None
        ).provider
        == "yunwu"
    )
