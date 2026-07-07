"""Reusable, dependency-free retry policy for supervisor updaters.

A pure :class:`RetryPolicy` (exponential backoff with an optional jitter) plus a
mutable per-target :class:`RetryState` (attempt count, next-eligible monotonic
time, last error). Both are deterministic — ``now`` and the jitter fraction are
injected — so the backoff math and the state transitions are unit-testable
without sleeping or touching the clock.

Used by the master image-updater to space out retries between the supervisor's
fixed-cadence ticks and to give up (and alert) after a bounded number of failed
rollouts; a NEW desired digest resets the state so a fresh rollout is attempted
immediately.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

#: Injectable jitter source: returns a fraction in ``[0.0, 1.0)`` (default
#: :func:`random.random`). Injected so tests can pin the jittered delay.
JitterSource = Callable[[], float]


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff policy (pure, no clock, no I/O)."""

    max_attempts: int = 5
    base_delay: float = 60.0
    max_delay: float = 1800.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"RetryPolicy.max_attempts must be >= 1, got {self.max_attempts!r}"
            )
        if self.base_delay <= 0:
            raise ValueError(
                f"RetryPolicy.base_delay must be positive, got {self.base_delay!r}"
            )
        if self.max_delay < self.base_delay:
            raise ValueError(
                "RetryPolicy.max_delay must be >= base_delay "
                f"(max_delay={self.max_delay!r}, base_delay={self.base_delay!r})"
            )

    def compute_delay(self, attempt: int, *, jitter: float = 1.0) -> float:
        """Backoff delay (seconds) before a 1-based ``attempt``.

        Exponential ``base_delay * 2**(attempt-1)`` capped at ``max_delay``. When
        :attr:`jitter` is enabled the delay is decorrelated using an "equal
        jitter" scheme (``capped/2 + capped/2 * fraction``): it keeps a
        meaningful minimum backoff floor (never returns ~0) while spreading
        concurrent retries, with ``fraction`` a caller-supplied value in
        ``[0.0, 1.0]`` (injected for determinism).
        """
        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt!r}")
        raw = self.base_delay * (2.0 ** (attempt - 1))
        capped = min(raw, self.max_delay)
        if not self.jitter:
            return capped
        fraction = min(max(jitter, 0.0), 1.0)
        half = capped / 2.0
        return half + half * fraction


@dataclass
class RetryState:
    """Mutable per-target retry bookkeeping (attempts + next-eligible time)."""

    attempts: int = 0
    next_eligible_monotonic: float = 0.0
    last_error: str | None = None

    def record_failure(
        self,
        now: float,
        policy: RetryPolicy,
        *,
        error: str | None = None,
        jitter_source: JitterSource = random.random,
    ) -> None:
        """Count one failure and schedule the next-eligible time via backoff."""
        self.attempts += 1
        self.last_error = error
        fraction = jitter_source() if policy.jitter else 1.0
        self.next_eligible_monotonic = now + policy.compute_delay(
            self.attempts, jitter=fraction
        )

    def record_success(self) -> None:
        """Reset the state after a successful (or no-op) refresh."""
        self.attempts = 0
        self.next_eligible_monotonic = 0.0
        self.last_error = None

    def is_eligible(self, now: float) -> bool:
        """True once ``now`` has reached the scheduled next-eligible time."""
        return now >= self.next_eligible_monotonic

    def is_exhausted(self, policy: RetryPolicy) -> bool:
        """True once the failure budget (``max_attempts``) is used up."""
        return self.attempts >= policy.max_attempts


__all__ = ["JitterSource", "RetryPolicy", "RetryState"]
