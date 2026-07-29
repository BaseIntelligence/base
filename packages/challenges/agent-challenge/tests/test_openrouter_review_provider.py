"""OpenRouter analyzer provider — NO_PHALA only; gateway path unchanged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    GatewayReviewProvider,
    LlmProviderRateLimited,
    LlmProviderTimeout,
    LlmProviderUnavailable,
)
from agent_challenge.analyzer.openrouter_review_provider import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_PROVIDER_NAME,
    OpenRouterReviewProvider,
    resolve_openrouter_api_key,
    use_openrouter_review_provider,
)


def test_use_openrouter_only_when_no_phala() -> None:
    assert use_openrouter_review_provider(no_phala=False, llm_provider="openrouter") is False
    assert use_openrouter_review_provider(no_phala=False, llm_provider=None) is False
    assert use_openrouter_review_provider(no_phala=True, llm_provider=None) is True
    assert use_openrouter_review_provider(no_phala=True, llm_provider="openrouter") is True
    assert use_openrouter_review_provider(no_phala=True, llm_provider="gateway") is False


def test_resolve_key_order_explicit_then_env_then_opencode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHALLENGE_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert resolve_openrouter_api_key(explicit="  explicit-key  ", home=tmp_path) == "explicit-key"

    monkeypatch.setenv("CHALLENGE_OPENROUTER_API_KEY", "challenge-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "plain-key")
    env = dict(**__import__("os").environ)
    assert resolve_openrouter_api_key(explicit=None, environ=env, home=tmp_path) == "challenge-key"

    monkeypatch.delenv("CHALLENGE_OPENROUTER_API_KEY", raising=False)
    env = dict(**__import__("os").environ)
    assert resolve_openrouter_api_key(explicit=None, environ=env, home=tmp_path) == "plain-key"

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    auth = tmp_path / ".local" / "share" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    payload = {"openrouter": {"type": "api", "key": "file-key"}}
    auth.write_text(json.dumps(payload), encoding="utf-8")
    assert resolve_openrouter_api_key(explicit=None, environ={}, home=tmp_path) == "file-key"


def test_openrouter_provider_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "id": "gen-1",
                "model": DEFAULT_OPENROUTER_MODEL,
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_verdict",
                                        "arguments": json.dumps(
                                            {
                                                "verdict": "allow",
                                                "confidence": 0.9,
                                                "rationale": "Clean agent.",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    def fake_post(url, *, headers, json, timeout):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return Response()

    monkeypatch.setattr("agent_challenge.analyzer.openrouter_review_provider.httpx.post", fake_post)
    provider = OpenRouterReviewProvider(api_key="sk-test", model_name=DEFAULT_OPENROUTER_MODEL)
    response = provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        tool_choice="auto",
        timeout_seconds=5,
    )

    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert "X-Gateway-Token" not in captured["headers"]
    assert captured["json"]["model"] == DEFAULT_OPENROUTER_MODEL
    assert response.tool_calls[0].name == "submit_verdict"
    assert response.tool_calls[0].arguments["verdict"] == "allow"
    assert provider.provider_name == OPENROUTER_PROVIDER_NAME
    # Key never appears in repr
    assert "sk-test" not in repr(provider)


def test_openrouter_provider_fail_closed_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 500

        def json(self) -> dict[str, Any]:
            return {"error": "boom"}

    monkeypatch.setattr(
        "agent_challenge.analyzer.openrouter_review_provider.httpx.post",
        lambda *a, **k: Response(),
    )
    provider = OpenRouterReviewProvider(api_key="sk-test")
    with pytest.raises(LlmProviderUnavailable, match="HTTP 500"):
        provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)


def test_openrouter_provider_rate_limit_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 429

        def json(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(
        "agent_challenge.analyzer.openrouter_review_provider.httpx.post",
        lambda *a, **k: Response(),
    )
    provider = OpenRouterReviewProvider(api_key="sk-test")
    with pytest.raises(LlmProviderRateLimited):
        provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)

    def boom(*a, **k):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr("agent_challenge.analyzer.openrouter_review_provider.httpx.post", boom)
    with pytest.raises(LlmProviderTimeout):
        provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)


def test_openrouter_provider_missing_key_fail_closed() -> None:
    provider = OpenRouterReviewProvider(api_key=None)
    with pytest.raises(LlmProviderUnavailable, match="not configured"):
        provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)


def test_openrouter_cost_limit_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": "", "tool_calls": []}}],
                "cost": {"total_cost": 1.5},
            }

    monkeypatch.setattr(
        "agent_challenge.analyzer.openrouter_review_provider.httpx.post",
        lambda *a, **k: Response(),
    )
    provider = OpenRouterReviewProvider(api_key="sk-test", cost_limit_usd=1.0)
    with pytest.raises(LlmProviderUnavailable, match="cost limit"):
        provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)


def test_lifecycle_builds_gateway_when_no_phala_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.analyzer import lifecycle
    from agent_challenge.core import config as core_config

    mock_settings = MagicMock()
    mock_settings.no_phala = False
    mock_settings.llm_provider = "openrouter"
    mock_settings.llm_gateway_base_url = "http://master:19080"
    mock_settings.llm_gateway_token = "gw-token"
    mock_settings.openrouter_api_key = "sk-should-not-use"
    mock_settings.llm_model = DEFAULT_OPENROUTER_MODEL
    mock_settings.llm_cost_limit_usd = None
    monkeypatch.setattr(lifecycle, "settings", mock_settings)
    monkeypatch.setattr(core_config, "settings", mock_settings)

    provider = lifecycle._build_configured_review_provider()
    assert isinstance(provider, GatewayReviewProvider)
    assert provider.provider_name == "gateway"
    assert provider.model_name == GATEWAY_PLACEHOLDER_MODEL


def test_lifecycle_builds_openrouter_when_no_phala_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.analyzer import lifecycle

    mock_settings = MagicMock()
    mock_settings.no_phala = True
    mock_settings.llm_provider = "openrouter"
    mock_settings.llm_gateway_base_url = "http://master:19080"
    mock_settings.llm_gateway_token = "gw-token"
    mock_settings.openrouter_api_key = "sk-or-test"
    mock_settings.llm_model = "x-ai/grok-4.5"
    mock_settings.llm_cost_limit_usd = 2.0
    monkeypatch.setattr(lifecycle, "settings", mock_settings)

    provider = lifecycle._build_configured_review_provider()
    assert isinstance(provider, OpenRouterReviewProvider)
    assert provider.provider_name == "openrouter"
    assert provider.model_name == "x-ai/grok-4.5"


def test_llm_provider_ready_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.analyzer import lifecycle

    mock_settings = MagicMock()
    mock_settings.no_phala = True
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter_api_key = None
    mock_settings.llm_gateway_base_url = None
    mock_settings.llm_gateway_token = None
    monkeypatch.setattr(lifecycle, "settings", mock_settings)
    monkeypatch.setattr(
        lifecycle,
        "resolve_openrouter_api_key",
        lambda explicit=None: None,
    )
    assert lifecycle._llm_provider_ready() is False

    monkeypatch.setattr(
        lifecycle,
        "resolve_openrouter_api_key",
        lambda explicit=None: "sk-ok",
    )
    assert lifecycle._llm_provider_ready() is True
