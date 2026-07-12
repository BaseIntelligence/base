"""Explicit agent-challenge incompatibility diagnostic after gateway removal.

The current public agent-challenge contract still depends on the removed LLM
gateway. Base must refuse to activate/seed/reconcile it with a stable machine-
readable code and must never restore a hidden gateway compatibility path.
"""

from __future__ import annotations

from dataclasses import dataclass

AGENT_CHALLENGE_SLUG = "agent-challenge"
AGENT_CHALLENGE_INCOMPATIBLE_CODE = "AGENT_CHALLENGE_INCOMPATIBLE_NO_LLM_GATEWAY"
AGENT_CHALLENGE_INCOMPATIBLE_MESSAGE = (
    "Current agent-challenge requires the removed LLM gateway contract and must "
    "be upgraded before registration, activation, seeding, or reconcile. Do not "
    "set a legacy gateway token or enable a compatibility adapter."
)


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


def is_agent_challenge_slug(slug: str | None) -> bool:
    return str(slug or "").strip().lower() == AGENT_CHALLENGE_SLUG


def agent_challenge_incompatibility() -> AgentChallengeIncompatibility:
    return AgentChallengeIncompatibility()
