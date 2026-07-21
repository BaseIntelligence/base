"""Hard-pinned review callback base URL (anti-cheat).

Measured review runtime and selfdeploy encrypt/deploy must call joinbase only.
Miners cannot redirect the review CVM callback via env or CLI in production.
Local/dev overrides require an explicit ``CHALLENGE_ALLOW_DEV_URLS=1`` flag.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# Production Base master hosts AC under this public challenge base.
# chain.platform.network is historically 502 and is not a valid review report target.
# Authority is intentionally immutable in prod — not miner-env.
PINNED_REVIEW_API_BASE_URL = "https://chain.joinbase.ai/challenges/agent-challenge"
DEFAULT_REVIEW_API_BASE_URL = PINNED_REVIEW_API_BASE_URL

#: Explicit non-prod override gate. Prod digests / measured paths bake this off.
ALLOW_DEV_REVIEW_URLS_ENV = "CHALLENGE_ALLOW_DEV_URLS"


class ReviewApiBaseUrlError(ValueError):
    """Review callback base URL is not the production joinbase pin."""


def normalize_review_api_base_url(value: str) -> str:
    """Strip whitespace and trailing slashes from a candidate base URL."""

    if not isinstance(value, str):
        raise ReviewApiBaseUrlError("REVIEW_API_BASE_URL must be a string")
    text = value.strip()
    if not text:
        raise ReviewApiBaseUrlError("REVIEW_API_BASE_URL must be non-empty")
    return text.rstrip("/")


def is_pinned_review_api_base_url(value: object) -> bool:
    """True when value normalizes exactly to the production joinbase pin."""

    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return normalize_review_api_base_url(value) == PINNED_REVIEW_API_BASE_URL
    except ReviewApiBaseUrlError:
        return False


def _truthy_flag(raw: object) -> bool:
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "on"}


def allow_dev_review_urls(
    environ: Mapping[str, str] | None = None,
    *,
    allow_dev: bool | None = None,
) -> bool:
    """Return whether an explicit non-prod URL override is authorized."""

    if allow_dev is not None:
        return bool(allow_dev)
    env = os.environ if environ is None else environ
    return _truthy_flag(env.get(ALLOW_DEV_REVIEW_URLS_ENV))


def assert_pinned_review_api_base_url(
    value: object,
    *,
    allow_dev: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the pinned base, or a validated dev override when explicitly allowed.

    Production (default): only the exact joinbase agent-challenge constant is
    accepted. Non-joinbase hosts, http://, IPs, localhost, and legacy
    platform.network are refused fail-closed.
    """

    if not isinstance(value, str) or not value.strip():
        # Absent/blank → force constant (honest default).
        return PINNED_REVIEW_API_BASE_URL
    normalized = normalize_review_api_base_url(value)
    if normalized == PINNED_REVIEW_API_BASE_URL:
        return PINNED_REVIEW_API_BASE_URL
    if allow_dev_review_urls(environ, allow_dev=allow_dev):
        if not normalized.startswith("https://"):
            raise ReviewApiBaseUrlError("REVIEW_API_BASE_URL dev override must be an https URL")
        # Dev still refuses empty-host / bare path garbage.
        if "://" not in normalized:
            raise ReviewApiBaseUrlError("REVIEW_API_BASE_URL dev override is malformed")
        return normalized
    raise ReviewApiBaseUrlError(
        "REVIEW_API_BASE_URL must be exactly "
        f"{PINNED_REVIEW_API_BASE_URL} in production "
        f"(set {ALLOW_DEV_REVIEW_URLS_ENV}=1 only for non-prod images)"
    )


def resolve_review_api_base_url(
    *,
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
    allow_dev: bool | None = None,
) -> str:
    """Resolve the effective review callback base for runtime/deploy.

    Priority of *candidates* (when present): explicit CLI/flag, then env
    ``REVIEW_API_BASE_URL``. Empty/absent → production pin. Non-pin candidates
    are refused in production and only accepted behind the explicit dev flag.
    """

    env: Mapping[str, Any]
    env = os.environ if environ is None else environ
    candidate: str | None = None
    if isinstance(explicit, str) and explicit.strip():
        candidate = explicit
    else:
        raw = env.get("REVIEW_API_BASE_URL") if hasattr(env, "get") else None
        if isinstance(raw, str) and raw.strip():
            candidate = raw
    if candidate is None:
        return PINNED_REVIEW_API_BASE_URL
    return assert_pinned_review_api_base_url(candidate, allow_dev=allow_dev, environ=env)


__all__ = [
    "ALLOW_DEV_REVIEW_URLS_ENV",
    "DEFAULT_REVIEW_API_BASE_URL",
    "PINNED_REVIEW_API_BASE_URL",
    "ReviewApiBaseUrlError",
    "allow_dev_review_urls",
    "assert_pinned_review_api_base_url",
    "is_pinned_review_api_base_url",
    "normalize_review_api_base_url",
    "resolve_review_api_base_url",
]
