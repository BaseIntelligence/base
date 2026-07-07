"""Transport-agnostic loop primitives shared by coordination-plane agents.

Both the validator agent (:mod:`base.validator.agent.runtime`) and the
miner-funded worker agent (:mod:`base.worker.runtime`) run the same
register/heartbeat/pull/execute/post shape. The pure pieces of that loop live
here so the two agents share one implementation instead of copy-pasting it:
bounded exponential backoff, the transient-vs-permanent error classification,
the per-pass summary, and the shutdown-aware sleep helpers. This is a verbatim
extraction from the validator agent, so validator behavior is unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class BackoffPolicy:
    """Bounded exponential backoff for transient master/coordination failures.

    The delay grows geometrically with the number of consecutive failures and is
    capped at ``max_seconds`` so an agent retries a briefly-unavailable master
    without either giving up or busy-looping.
    """

    initial_seconds: float = 1.0
    max_seconds: float = 60.0
    multiplier: float = 2.0

    def delay(self, consecutive_failures: int) -> float:
        """Backoff delay after ``consecutive_failures`` failures (>=1)."""

        if consecutive_failures <= 0:
            return 0.0
        raw = self.initial_seconds * (self.multiplier ** (consecutive_failures - 1))
        return min(self.max_seconds, max(0.0, raw))


def is_transient_error(exc: BaseException) -> bool:
    """Whether a coordination failure is worth retrying with backoff.

    Transport errors (no status code) and ``429``/``5xx`` master responses are
    transient; a ``4xx`` (e.g. ``403`` ineligible, ``404`` not registered,
    ``401`` auth) is a permanent client error that should fail fast.
    """

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        return True
    return status_code == 429 or status_code >= 500


@dataclass(frozen=True)
class AgentCycleSummary:
    """Counts from one assignment-processing pass."""

    pulled: int
    completed: int
    failed: int


async def sleep_until(shutdown_event: asyncio.Event, seconds: float) -> None:
    """Sleep up to ``seconds``, waking early if ``shutdown_event`` fires."""

    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=seconds)
    except TimeoutError:
        return


async def backoff_sleep(shutdown_event: asyncio.Event | None, seconds: float) -> bool:
    """Sleep ``seconds``; return ``False`` if shutdown fired during the wait."""

    if shutdown_event is None:
        await asyncio.sleep(seconds)
        return True
    if shutdown_event.is_set():
        return False
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=seconds)
    except TimeoutError:
        return True
    return False
