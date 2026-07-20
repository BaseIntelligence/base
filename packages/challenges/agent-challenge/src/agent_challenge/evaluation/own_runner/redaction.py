"""Captured-log secret redaction for the in-CVM orchestrator (isolation invariant).

The M2 orchestrator captures each trial's agent output, verifier stdout, and any
error text, then persists them to per-trial log files and streams them. Two
classes of secret must never survive into that captured/persisted output
(architecture sec 4 C2):

* the scoped LLM **gateway token** (``BASE_GATEWAY_TOKEN``) handed to the agent;
* any **miner-supplied env values** that surface in task stdout/stderr.

:class:`LogRedactor` replaces every occurrence of those secret *values* with a
stable placeholder. It is deliberately dependency-free (stdlib only) and operates
on any object with the trial log-channel attributes via
:func:`dataclasses.replace`, so it stays import-light for the lean canonical
image and never couples the redactor to the orchestrator's concrete types.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TypeVar

__all__ = [
    "REDACTED_GATEWAY_TOKEN",
    "REDACTED_MINER_ENV",
    "LogRedactor",
]

#: Placeholder written in place of the scoped gateway token.
REDACTED_GATEWAY_TOKEN = "[REDACTED_GATEWAY_TOKEN]"
#: Placeholder written in place of a miner-supplied env value.
REDACTED_MINER_ENV = "[REDACTED_MINER_ENV]"

_T = TypeVar("_T")

#: Trial log-channel attributes that may carry captured task output.
_LOG_FIELDS = ("agent_output", "verifier_stdout", "error_text")


class LogRedactor:
    """Redacts gateway-token / miner-env secret values from captured text.

    Longer secrets are redacted first so a secret that is a substring of another
    cannot leave a partial value behind. Empty/falsy secret values are ignored so
    an unset token or blank env value never redacts the entire output.
    """

    def __init__(
        self,
        *,
        gateway_token: str | None = None,
        miner_env_values: Iterable[str] = (),
    ) -> None:
        mapping: dict[str, str] = {}
        for value in miner_env_values:
            if value:
                mapping.setdefault(value, REDACTED_MINER_ENV)
        if gateway_token:
            # The gateway token wins if a miner also supplied the same value.
            mapping[gateway_token] = REDACTED_GATEWAY_TOKEN
        self._pairs: tuple[tuple[str, str], ...] = tuple(
            sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True)
        )

    @property
    def active(self) -> bool:
        """``True`` when at least one secret value will be redacted."""
        return bool(self._pairs)

    def redact(self, text: str | None) -> str | None:
        """Return ``text`` with every known secret value replaced (``None`` safe)."""
        if not text or not self._pairs:
            return text
        redacted = text
        for value, placeholder in self._pairs:
            redacted = redacted.replace(value, placeholder)
        return redacted

    def redact_outcome(self, outcome: _T) -> _T:
        """Return a copy of ``outcome`` with its captured log channels redacted.

        Operates on any dataclass carrying the trial log-channel fields
        (``agent_output``/``verifier_stdout``/``error_text``); non-log fields
        (scores, reason codes, identity) are preserved unchanged.
        """
        if not self._pairs:
            return outcome
        updates = {
            field: self.redact(getattr(outcome, field))
            for field in _LOG_FIELDS
            if hasattr(outcome, field)
        }
        if not updates:
            return outcome
        return replace(outcome, **updates)
