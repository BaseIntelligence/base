"""Agent-challenge gateway-free digest gate after Base LLM gateway removal.

Base must never restore ``/llm/v1`` or inject gateway secrets. Pre-upgrade
agent-challenge images that still demand the removed gateway contract are
refused with a stable machine-readable diagnostic.

Gateway-free (attestation-only) digests may start/reconcile without gateway
env once they appear on the digest allowlist. The allowlist is the only unlock
mechanism — never a compatibility adapter.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

AGENT_CHALLENGE_SLUG = "agent-challenge"
AGENT_CHALLENGE_INCOMPATIBLE_CODE = "AGENT_CHALLENGE_INCOMPATIBLE_NO_LLM_GATEWAY"
AGENT_CHALLENGE_INCOMPATIBLE_MESSAGE = (
    "Current agent-challenge requires the removed LLM gateway contract and must "
    "be upgraded before registration, activation, seeding, or reconcile. Do not "
    "set a legacy gateway token or enable a compatibility adapter."
)

#: Operator env (comma-separated ``sha256:<64 hex>``) that extended the digest
#: allowlist for gateway-free / attestation-only agent-challenge images.
GATEWAY_FREE_DIGESTS_ENV = "BASE_AGENT_CHALLENGE_GATEWAY_FREE_DIGESTS"

#: Built-in empty set. Production unlock is via ``GATEWAY_FREE_DIGESTS_ENV`` or
#: tests/registry pins that call :func:`register_gateway_free_digest` only in
#: process-local unit fixtures (not product operator API).
_BUILTIN_GATEWAY_FREE_DIGESTS: frozenset[str] = frozenset()

_DIGEST_RE = re.compile(r"(?i)sha256:([0-9a-fA-F]{64})")

#: Env names that must never be injected into long-lived AC Compose services
#: and that mark an image contract as still gateway-shaped when present.
FORBIDDEN_GATEWAY_ENV_NAMES: frozenset[str] = frozenset(
    {
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "CENTRAL_GATEWAY_TOKEN",
        "CHALLENGE_LLM_GATEWAY_TOKEN",
        "CHALLENGE_LLM_GATEWAY_BASE_URL",
        "CHALLENGE_LLM_GATEWAY_TOKEN_FILE",
        "CHALLENGE_AGENT_GATEWAY_TOKEN",
        "CHALLENGE_AGENT_GATEWAY_TOKEN_FILE",
        "PRISM_LLM_GATEWAY_URL",
    }
)

#: Master-owned secrets that must never appear on the master process path for
#: agent-challenge (miner keys stay in Phala encrypted_env).
FORBIDDEN_MASTER_OPENROUTER_ENV_NAMES: frozenset[str] = frozenset(
    {
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_TOKEN",
        "OPEN_ROUTER_API_KEY",
        "OPENROUTER_KEY",
    }
)

_PROCESS_GATEWAY_FREE_DIGESTS: set[str] = set()


@dataclass(frozen=True)
class AgentChallengeIncompatibility:
    """Structured diagnostic returned by API/CLI preflight checks."""

    code: str = AGENT_CHALLENGE_INCOMPATIBLE_CODE
    message: str = AGENT_CHALLENGE_INCOMPATIBLE_MESSAGE
    challenge_slug: str = AGENT_CHALLENGE_SLUG

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "challenge_slug": self.challenge_slug,
        }


@dataclass(frozen=True)
class AgentChallengeActivationDecision:
    """Whether reconcile/seed/activate may start the long-lived AC service."""

    allowed: bool
    digest: str | None
    incompatibility: AgentChallengeIncompatibility | None

    @property
    def code(self) -> str | None:
        if self.incompatibility is None:
            return None
        return self.incompatibility.code


def is_agent_challenge_slug(slug: str | None) -> bool:
    return str(slug or "").strip().lower() == AGENT_CHALLENGE_SLUG


def agent_challenge_incompatibility() -> AgentChallengeIncompatibility:
    return AgentChallengeIncompatibility()


def normalize_image_digest(value: str | None) -> str | None:
    """Return lowercase ``sha256:<64 hex>`` if *value* contains a digest, else None."""

    if not value:
        return None
    match = _DIGEST_RE.search(str(value).strip())
    if match is None:
        return None
    return f"sha256:{match.group(1).lower()}"


def agent_challenge_image_digest(image: str | None) -> str | None:
    """Extract OCI content digest from a pinned image reference."""

    return normalize_image_digest(image)


def register_gateway_free_digest(digest: str) -> str:
    """Process-local allowlist insert for unit tests (not a product API)."""

    normalized = normalize_image_digest(digest)
    if normalized is None:
        raise ValueError(f"invalid agent-challenge gateway-free digest: {digest!r}")
    _PROCESS_GATEWAY_FREE_DIGESTS.add(normalized)
    return normalized


def clear_gateway_free_digest_registry() -> None:
    """Drop process-local test digests (tests only)."""

    _PROCESS_GATEWAY_FREE_DIGESTS.clear()


def _env_gateway_free_digests() -> frozenset[str]:
    raw = os.environ.get(GATEWAY_FREE_DIGESTS_ENV, "")
    if not raw.strip():
        return frozenset()
    found: set[str] = set()
    for part in raw.replace(";", ",").split(","):
        normalized = normalize_image_digest(part)
        if normalized is not None:
            found.add(normalized)
    return frozenset(found)


def gateway_free_agent_challenge_digests() -> frozenset[str]:
    """Union of built-in, env, and process-local gateway-free digests."""

    return frozenset(
        set(_BUILTIN_GATEWAY_FREE_DIGESTS)
        | _env_gateway_free_digests()
        | set(_PROCESS_GATEWAY_FREE_DIGESTS)
    )


def env_declares_gateway_contract(env: Mapping[str, Any] | None) -> bool:
    """True when container env still names removed Base gateway secrets."""

    if not env:
        return False
    for key in env:
        name = str(key).strip().upper()
        if name in FORBIDDEN_GATEWAY_ENV_NAMES:
            return True
        if name.startswith("BASE_GATEWAY") or name.startswith("BASE_LLM_GATEWAY"):
            return True
        if "LLM_GATEWAY" in name and name.endswith(
            ("URL", "TOKEN", "TOKEN_FILE", "BASE_URL")
        ):
            return True
    return False


def master_env_holds_openrouter_keys(env: Mapping[str, Any] | None) -> bool:
    """True when a env bag would place OpenRouter keys on the master process."""

    if not env:
        return False
    for key in env:
        name = str(key).strip().upper()
        if name in FORBIDDEN_MASTER_OPENROUTER_ENV_NAMES:
            return True
        if name.startswith("OPENROUTER_"):
            return True
    return False


def is_gateway_free_agent_challenge_digest(digest: str | None) -> bool:
    normalized = normalize_image_digest(digest)
    if normalized is None:
        return False
    return normalized in gateway_free_agent_challenge_digests()


def is_gateway_free_agent_challenge_image(
    image: str | None,
    *,
    env: Mapping[str, Any] | None = None,
) -> bool:
    """Return True only for allowlisted digests without gateway env residue."""

    digest = agent_challenge_image_digest(image)
    if digest is None:
        return False
    if not is_gateway_free_agent_challenge_digest(digest):
        return False
    if env_declares_gateway_contract(env):
        return False
    return True


def decide_agent_challenge_activation(
    *,
    image: str | None,
    env: Mapping[str, Any] | None = None,
    slug: str | None = AGENT_CHALLENGE_SLUG,
) -> AgentChallengeActivationDecision:
    """Gate seed/start/reconcile for agent-challenge images.

    Allowed only when the image reference is digest-pinned to an allowlisted
    gateway-free digest and the env bag does not reintroduce gateway secrets.
    """

    if slug is not None and not is_agent_challenge_slug(slug):
        return AgentChallengeActivationDecision(
            allowed=True,
            digest=agent_challenge_image_digest(image),
            incompatibility=None,
        )

    digest = agent_challenge_image_digest(image)
    if is_gateway_free_agent_challenge_image(image, env=env):
        return AgentChallengeActivationDecision(
            allowed=True,
            digest=digest,
            incompatibility=None,
        )
    return AgentChallengeActivationDecision(
        allowed=False,
        digest=digest,
        incompatibility=agent_challenge_incompatibility(),
    )


def should_refuse_agent_challenge(
    *,
    image: str | None = None,
    env: Mapping[str, Any] | None = None,
    slug: str | None = AGENT_CHALLENGE_SLUG,
) -> AgentChallengeIncompatibility | None:
    """Return the stable diagnostic when activation must be refused."""

    decision = decide_agent_challenge_activation(image=image, env=env, slug=slug)
    return decision.incompatibility


def agent_challenge_compose_env_is_gateway_free(env: Mapping[str, Any] | None) -> bool:
    """VAL-ACAT-046: long-lived AC Compose env must omit gateway secrets."""

    return not env_declares_gateway_contract(env)


def filter_forbidden_gateway_env(
    env: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Drop forbidden gateway keys from a candidate env mapping (copy)."""

    if not env:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in env.items():
        name = str(key).strip().upper()
        if name in FORBIDDEN_GATEWAY_ENV_NAMES:
            continue
        if name.startswith("BASE_GATEWAY") or name.startswith("BASE_LLM_GATEWAY"):
            continue
        if "LLM_GATEWAY" in name and name.endswith(
            ("URL", "TOKEN", "TOKEN_FILE", "BASE_URL")
        ):
            continue
        cleaned[str(key)] = value
    return cleaned
