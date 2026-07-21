"""Miner / job submission env policy (keys and tokens only).

Submission env is a cheat surface when URL/proxy/host/gateway injection is
allowed. Product policy:

* admit **API keys / tokens / measured product secrets** only
* fail-closed reject ``*_URL`` / ``*_URI`` / ``*_ENDPOINT`` / ``*_HOST`` /
  ``*_PROXY``, ``DOCKER_HOST``, gateway names, and ``HTTP(S)_PROXY`` /
  ``ALL_PROXY`` / ``NO_PROXY``
* reject URL-shaped **values** on any admitted key
* control stream keys (``BASE_LOG_STREAM_*``) are never miner-settable

Used by the public env PUT path and the job ``_terminal_bench_env`` builder so
rejected keys never reach the job container even if an earlier layer slipped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

__all__ = [
    "MAX_MINER_ENV_KEYS",
    "MAX_MINER_ENV_TOTAL_BYTES",
    "MAX_MINER_ENV_VALUE_BYTES",
    "MINER_ENV_CONTROL_PREFIXES",
    "MINER_ENV_KEY_RE",
    "MINER_ENV_PRODUCT_ALLOWLIST",
    "MinerEnvValidationError",
    "is_allowed_miner_env_key",
    "is_forbidden_miner_env_key",
    "is_token_or_key_env_name",
    "looks_like_url_value",
    "sanitize_miner_env_for_job",
    "validate_miner_env",
]

MINER_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
MAX_MINER_ENV_KEYS = 64
MAX_MINER_ENV_VALUE_BYTES = 16 * 1024
MAX_MINER_ENV_TOTAL_BYTES = 128 * 1024

#: Product-required measured / session tokens (exact names).
MINER_ENV_PRODUCT_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "OPENROUTER_API_KEY",
        "LLM_COST_LIMIT",
        "EVAL_RUN_TOKEN",
        "REVIEW_SESSION_TOKEN",
    }
)

#: Exact proxy / host control names (case-insensitive match via upper()).
_FORBIDDEN_EXACT: Final[frozenset[str]] = frozenset(
    {
        "ALL_PROXY",
        "DOCKER_HOST",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "PROXY",
    }
)

#: Suffix patterns that always mark URL/host/proxy injection.
_FORBIDDEN_SUFFIXES: Final[tuple[str, ...]] = (
    "_URL",
    "_URI",
    "_ENDPOINT",
    "_HOST",
    "_PROXY",
)

#: Substrings that mark gateway / control injection surfaces.
_FORBIDDEN_SUBSTRINGS: Final[tuple[str, ...]] = (
    "GATEWAY",
    "DOCKER_HOST",
)

#: Miner cannot override dispatcher log-stream control.
MINER_ENV_CONTROL_PREFIXES: Final[tuple[str, ...]] = ("BASE_LOG_STREAM_",)

#: Values that look like absolute URLs (scheme://...).
_URL_VALUE_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://\S")

#: Secret-shaped suffixes that miners may legitimately supply.
_TOKEN_SUFFIXES: Final[tuple[str, ...]] = (
    "_API_KEY",
    "_API_TOKEN",
    "_ACCESS_TOKEN",
    "_SECRET_KEY",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_TOKEN",
    "_KEY",
)


class MinerEnvValidationError(ValueError):
    """Fail-closed miner env rejection with a public, non-echoing reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def is_forbidden_miner_env_key(name: str) -> bool:
    """Return True when ``name`` is a URL/proxy/host/gateway/control injection key."""
    upper = name.upper()
    if upper in _FORBIDDEN_EXACT:
        return True
    if any(upper.startswith(prefix) for prefix in MINER_ENV_CONTROL_PREFIXES):
        return True
    if any(marker in upper for marker in _FORBIDDEN_SUBSTRINGS):
        return True
    if any(upper.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES):
        return True
    return False


def is_token_or_key_env_name(name: str) -> bool:
    """Return True when ``name`` is an API key / token / product measured secret."""
    upper = name.upper()
    if upper in MINER_ENV_PRODUCT_ALLOWLIST:
        return True
    return any(upper.endswith(suffix) for suffix in _TOKEN_SUFFIXES)


def is_allowed_miner_env_key(name: str) -> bool:
    """Token/key shape and not a forbidden injection name."""
    if not MINER_ENV_KEY_RE.fullmatch(name):
        return False
    if is_forbidden_miner_env_key(name):
        return False
    return is_token_or_key_env_name(name)


def looks_like_url_value(value: str) -> bool:
    """Return True when ``value`` is shaped like an absolute URL."""
    return bool(_URL_VALUE_RE.match(value.strip()))


def validate_miner_env(env: Mapping[str, object]) -> dict[str, str]:
    """Validate miner submission env; raise :class:`MinerEnvValidationError` on reject.

    Returns a plain ``dict[str, str]`` of accepted keys. Reasons never echo values.
    """
    if len(env) > MAX_MINER_ENV_KEYS:
        raise MinerEnvValidationError("too many env vars")

    total_bytes = 0
    validated: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not MINER_ENV_KEY_RE.fullmatch(key):
            raise MinerEnvValidationError("invalid env var key")
        if is_forbidden_miner_env_key(key):
            raise MinerEnvValidationError("env var key not allowed")
        if not is_token_or_key_env_name(key):
            raise MinerEnvValidationError("env var key not allowed")
        if not isinstance(value, str):
            raise MinerEnvValidationError("env var values must be strings")
        if looks_like_url_value(value):
            raise MinerEnvValidationError("env var value must not be a URL")
        value_bytes = value.encode("utf-8")
        if len(value_bytes) > MAX_MINER_ENV_VALUE_BYTES:
            raise MinerEnvValidationError("env var value too large")
        total_bytes += len(value_bytes)
        if total_bytes > MAX_MINER_ENV_TOTAL_BYTES:
            raise MinerEnvValidationError("env var payload too large")
        validated[key] = value
    return validated


def sanitize_miner_env_for_job(miner_env: Mapping[str, str] | None) -> dict[str, str]:
    """Drop rejected miner keys before they reach the job container env.

    Fail-open keys are never invented; unknown or forbidden keys are stripped.
    """
    if not miner_env:
        return {}
    cleaned: dict[str, str] = {}
    for name, value in miner_env.items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        if not is_allowed_miner_env_key(name):
            continue
        if looks_like_url_value(value):
            continue
        cleaned[name] = value
    return cleaned
