"""Master LLM gateway wiring for decentralized task execution.

The agent's LLM calls are routed at the master LLM gateway: the eval runtime is
pointed at the gateway's OpenAI-compatible base URL and authenticates with a
per-assignment scoped token (delivered alongside the work-unit assignment). The
gateway resolves the provider and model server-side from the token's ``source``
claim, so the agent sends NO provider key and NO model name. The validator
therefore holds NO raw provider key, and no master-only env-decryption is
required to obtain LLM credentials at execution time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_challenge.sdk.config import ChallengeSettings

#: Path appended to the gateway base URL for the source-driven LLM route
#: (OpenAI-compatible; the agent appends ``chat/completions``).
LLM_GATEWAY_PATH = "/llm/v1"

#: Env var carrying the per-assignment scoped gateway token (never a raw
#: provider key). The eval runtime sends this to the gateway, which authenticates
#: the scoped token and injects the provider key + model server-side.
GATEWAY_TOKEN_ENV = "BASE_GATEWAY_TOKEN"

#: Env var the agent reads for the OpenAI-compatible gateway base URL.
BASE_LLM_GATEWAY_URL_ENV = "BASE_LLM_GATEWAY_URL"

#: Assignment-payload keys the master uses to deliver the scoped gateway token.
#: Mirrors the platform coordination contract (``gateway_token`` /
#: ``BASE_GATEWAY_TOKEN``) so the validator never invents its own.
GATEWAY_TOKEN_PAYLOAD_KEYS = ("gateway_token", GATEWAY_TOKEN_ENV)

#: Assignment-payload keys carrying an explicit gateway ROOT URL (falls back to
#: the validator-configured master gateway URL when absent). The ``/llm/v1``
#: route is composed from this root.
GATEWAY_BASE_URL_PAYLOAD_KEYS = ("gateway_url", "gateway_base_url")


class GatewayConfigError(ValueError):
    """A work unit's assignment payload cannot yield a master gateway config.

    Raised when a scoped gateway token or base URL is missing, so the production
    validator cycle can NEVER fall back to dispatching an eval run with
    ``gateway=None`` (which would let a raw provider key reach the eval
    container).
    """


@dataclass(frozen=True)
class GatewayExecutionConfig:
    """Per-assignment master LLM gateway configuration for an eval run.

    ``base_url`` is the gateway root (e.g. the master proxy URL); the ``/llm/v1``
    OpenAI-compatible route is composed from it. ``token`` is the per-assignment
    scoped gateway token. No raw provider key and no model name are ever part of
    this config - the gateway injects both from the token's ``source`` claim.
    """

    base_url: str
    token: str

    @property
    def llm_gateway_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{LLM_GATEWAY_PATH}"

    def agent_env(self) -> dict[str, str]:
        """Env injected into the eval runtime to route LLM calls via the gateway."""

        return {
            BASE_LLM_GATEWAY_URL_ENV: self.llm_gateway_url,
            GATEWAY_TOKEN_ENV: self.token,
        }

    @classmethod
    def from_assignment_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        base_url: str | None = None,
    ) -> GatewayExecutionConfig:
        """Build the gateway config from a work unit's master assignment payload.

        The scoped gateway token is taken from the payload (the master issues it
        per assignment); the base URL is taken from the payload or the
        validator-configured ``base_url`` fallback. Raises
        :class:`GatewayConfigError` when either is missing so the production
        cycle never dispatches without a gateway.
        """

        data = dict(payload or {})
        token = _first_present(data, GATEWAY_TOKEN_PAYLOAD_KEYS)
        resolved_base = base_url or _first_present(data, GATEWAY_BASE_URL_PAYLOAD_KEYS)
        if not token:
            raise GatewayConfigError("assignment payload is missing a scoped gateway token")
        if not resolved_base:
            raise GatewayConfigError("no master LLM gateway base URL configured for the assignment")
        return cls(base_url=str(resolved_base), token=str(token))


def agent_gateway_config_from_settings(
    settings: ChallengeSettings,
) -> GatewayExecutionConfig | None:
    """VAL-ACAT-013/014/050: Base LLM gateway injection is **removed**.

    Residual Settings fields (``llm_gateway_base_url``, ``agent_gateway_token``)
    are intentionally **ignored**. Production eval agents must not receive
    ``BASE_GATEWAY_TOKEN`` / ``BASE_LLM_GATEWAY_URL``; measured OpenRouter inside
    the eval CVM (or tools-only) is the only legal LLM path under policy.

    Always returns ``None`` so own_runner / combined-worker never re-inject Base
    gateway routing. ``settings`` is accepted for call-site compatibility.

    :class:`GatewayConfigError` from residual assignment builders is **never**
    the production success path (VAL-ACAT-050).
    """

    _ = settings  # residual gateway fields must not drive agent sandbox env
    return None


def production_agent_llm_gateway_config_forbidden() -> None:
    """Documented no-op fence: production eval-agent LLM never uses Base gateway."""

    return None


def _first_present(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value:
            return value
    return None
