"""OpenRouter-backed LLM review provider for NO_PHALA host-local mode.

Selected only when NO_PHALA is active and the operator sets
``CHALLENGE_LLM_PROVIDER=openrouter`` (default under NO_PHALA). The attested /
gateway path is untouched: when NO_PHALA is off, lifecycle always builds
``GatewayReviewProvider``.

Never logs or returns the API key. Fail closed on missing key, HTTP errors,
rate limits, timeouts, and optional cost-limit breach.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from agent_challenge.analyzer.llm_reviewer import (
    LlmProviderRateLimited,
    LlmProviderResponse,
    LlmProviderTimeout,
    LlmProviderUnavailable,
    _parse_provider_tool_calls,
    _redacted_response,
)

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "x-ai/grok-4.5"
OPENROUTER_PROVIDER_NAME = "openrouter"

# Env vars checked before local credential files (never logged).
_ENV_KEY_NAMES = (
    "CHALLENGE_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
)


def resolve_openrouter_api_key(
    *,
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> str | None:
    """Resolve OpenRouter API key without logging it.

    Order:
    1. ``explicit`` (settings field / caller)
    2. ``CHALLENGE_OPENROUTER_API_KEY`` then ``OPENROUTER_API_KEY``
    3. ``~/.local/share/opencode/auth.json`` → ``openrouter.key``
    4. ``~/.factory/`` small config files containing an openrouter key field
    """

    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    env = environ if environ is not None else os.environ
    for name in _ENV_KEY_NAMES:
        value = env.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    root = home if home is not None else Path.home()
    opencode_key = _key_from_opencode_auth(root / ".local" / "share" / "opencode" / "auth.json")
    if opencode_key:
        return opencode_key

    return _key_from_factory_dir(root / ".factory")


def _key_from_opencode_auth(path: Path) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    block = data.get("openrouter")
    if not isinstance(block, Mapping):
        return None
    for field in ("key", "apiKey", "api_key", "token"):
        value = block.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _key_from_factory_dir(factory_root: Path) -> str | None:
    if not factory_root.is_dir():
        return None
    # Prefer small known config filenames; never walk huge artifact trees.
    candidates = (
        factory_root / "auth.json",
        factory_root / "config.json",
        factory_root / "settings.json",
        factory_root / "keys.json",
    )
    for path in candidates:
        key = _key_from_json_file_shallow(path)
        if key:
            return key
    return None


def _key_from_json_file_shallow(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > 64_000:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _find_openrouter_key_in_mapping(data, depth=0)


def _find_openrouter_key_in_mapping(value: object, *, depth: int) -> str | None:
    if depth > 4 or not isinstance(value, Mapping):
        return None
    # Direct openrouter block
    block = value.get("openrouter")
    if isinstance(block, Mapping):
        for field in ("key", "apiKey", "api_key", "token"):
            candidate = block.get(field)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    for field in (
        "OPENROUTER_API_KEY",
        "openrouter_api_key",
        "openrouterKey",
        "openrouter_key",
    ):
        candidate = value.get(field)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for child in value.values():
        if isinstance(child, Mapping):
            found = _find_openrouter_key_in_mapping(child, depth=depth + 1)
            if found:
                return found
    return None


class OpenRouterReviewProvider:
    """LLM review provider that calls OpenRouter chat completions directly.

    Implements the same ``LlmReviewProvider`` contract as ``GatewayReviewProvider``
    so ``KimiLlmReviewer`` verdict parsing and fail-closed semantics are unchanged.
    """

    provider_name = OPENROUTER_PROVIDER_NAME

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        base_url: str = OPENROUTER_CHAT_COMPLETIONS_URL,
        cost_limit_usd: float | None = None,
    ) -> None:
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.cost_limit_usd = cost_limit_usd
        self._spent_usd = 0.0

    def __repr__(self) -> str:
        return (
            f"OpenRouterReviewProvider(model_name={self.model_name!r}, "
            f"api_key={'set' if self._api_key else 'missing'})"
        )

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | Mapping[str, Any],
        timeout_seconds: float,
    ) -> LlmProviderResponse:
        if not self._api_key:
            raise LlmProviderUnavailable("OpenRouter API key is not configured")
        if self.cost_limit_usd is not None and self._spent_usd >= self.cost_limit_usd:
            raise LlmProviderUnavailable("LLM cost limit exceeded")

        timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=30.0, pool=10.0)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter attribution headers (no secrets).
            "HTTP-Referer": "https://joinbase.ai",
            "X-Title": "agent-challenge-no-phala-review",
        }
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": tool_choice,
            "temperature": 0,
        }
        # Grok/OpenRouter: omit parallel_tool_calls (some endpoints 404 with it).
        try:
            response = httpx.post(
                self.base_url
                if self.base_url.endswith("/chat/completions")
                else f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise LlmProviderTimeout("OpenRouter request timed out") from exc
        except httpx.HTTPError as exc:
            raise LlmProviderUnavailable("OpenRouter request failed") from exc

        if response.status_code == 429:
            raise LlmProviderRateLimited("OpenRouter rate limit exceeded")
        if response.status_code in {401, 403}:
            raise LlmProviderUnavailable("OpenRouter authentication failed")
        if response.status_code >= 400:
            raise LlmProviderUnavailable(f"OpenRouter returned HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LlmProviderUnavailable("OpenRouter returned non-JSON body") from exc
        if not isinstance(data, Mapping):
            raise LlmProviderUnavailable("OpenRouter response is not a JSON object")

        self._accumulate_cost(data)
        if self.cost_limit_usd is not None and self._spent_usd > self.cost_limit_usd:
            raise LlmProviderUnavailable("LLM cost limit exceeded")

        choices = data.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        return LlmProviderResponse(
            content=str(message.get("content") or ""),
            tool_calls=_parse_provider_tool_calls(message),
            raw_response=_redacted_response(data),
            usage=data.get("usage") if isinstance(data.get("usage"), Mapping) else None,
            cost=data.get("cost") if isinstance(data.get("cost"), Mapping) else None,
        )

    def _accumulate_cost(self, data: Mapping[str, Any]) -> None:
        cost = data.get("cost")
        if isinstance(cost, Mapping):
            for key in ("total_cost", "total", "cost"):
                value = cost.get(key)
                if isinstance(value, (int, float)):
                    self._spent_usd += float(value)
                    return
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            value = usage.get("cost")
            if isinstance(value, (int, float)):
                self._spent_usd += float(value)


def use_openrouter_review_provider(
    *,
    no_phala: bool,
    llm_provider: str | None,
) -> bool:
    """Return True when analyzer LLM review should use OpenRouter.

    OpenRouter is never selected when NO_PHALA is off (gateway path stays default).
    Under NO_PHALA, default provider is openrouter unless operator forces gateway.
    """

    if not no_phala:
        return False
    normalized = (llm_provider or "openrouter").strip().lower()
    if normalized in {"", "openrouter", "or"}:
        return True
    if normalized == "gateway":
        return False
    # Unknown value under NO_PHALA: fail toward openrouter only if explicitly openrouter-like.
    return normalized.startswith("openrouter")


__all__ = [
    "DEFAULT_OPENROUTER_MODEL",
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "OPENROUTER_PROVIDER_NAME",
    "OpenRouterReviewProvider",
    "resolve_openrouter_api_key",
    "use_openrouter_review_provider",
]
