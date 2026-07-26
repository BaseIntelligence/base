"""Continuous constation poller with gap budget and fail-closed limits (todo 16).

Attests at run **start**, **end**, and **randomized intervals**. Any observed
gap beyond ``gap_budget_seconds`` fails the run (TOCTOU / M2). Lium HTTP 429
and exhausted network retries are fail-closed (never silent skip).

All time and I/O are injectable for hermetic tests (fake clocks / transports).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from base.compute.constation_types import (
    ConstationFailCode,
    ConstationVerdict,
    FaultClass,
    PollSample,
)
from base.compute.lium import LiumAuthError, LiumError, LiumRateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

NowFn = Callable[[], float]
SleepFn = Callable[[float], Awaitable[None]]
RngFn = Callable[[], float]  # uniform [0, 1)
PollOnceFn = Callable[[str], Awaitable[PollSample | ConstationVerdict]]


class PollPhase(StrEnum):
    START = "start"
    INTERVAL = "interval"
    END = "end"


@dataclass(frozen=True, slots=True)
class PollerConfig:
    """Budgets and pacing for one submission's continuous constation."""

    gap_budget_seconds: float = 30.0
    min_interval_seconds: float = 5.0
    max_interval_seconds: float = 20.0
    max_polls: int = 64
    max_cost_units: float = 64.0
    cost_per_poll: float = 1.0
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_network_retries: int = 3
    rate_limit_per_second: float = 5.0

    def __post_init__(self) -> None:
        if self.gap_budget_seconds <= 0:
            raise ValueError("gap_budget_seconds must be positive")
        if self.min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive")
        if self.max_interval_seconds < self.min_interval_seconds:
            raise ValueError("max_interval_seconds must be >= min_interval_seconds")
        if self.max_polls < 2:
            raise ValueError("max_polls must be >= 2 (start + end)")
        if self.max_network_retries < 0:
            raise ValueError("max_network_retries must be >= 0")
        if self.rate_limit_per_second <= 0:
            raise ValueError("rate_limit_per_second must be positive")


@dataclass(frozen=True, slots=True)
class PollerResult:
    """Outcome of a continuous constation poll window."""

    ok: bool
    reason: ConstationFailCode
    fault_class: FaultClass | None
    samples: tuple[PollSample, ...]
    observed_max_gap_seconds: float
    gap_budget_seconds: float
    poll_count: int
    cost_units: float
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class _TokenBucket:
    """Simple rate limiter (tokens per second) with injectable clock."""

    rate_per_second: float
    now_fn: NowFn
    capacity: float = field(init=False)
    tokens: float = field(init=False)
    updated_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.capacity = max(self.rate_per_second, 1.0)
        self.tokens = self.capacity
        self.updated_at = self.now_fn()

    def try_take(self) -> float:
        """Return 0 if a token was taken, else seconds to wait."""
        now = self.now_fn()
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        missing = 1.0 - self.tokens
        return missing / self.rate_per_second


@dataclass
class ContinuousConstationPoller:
    """Run start / randomized / end polls under gap and cost budgets."""

    config: PollerConfig
    now_fn: NowFn
    sleep_fn: SleepFn
    rng_fn: RngFn
    _limiter: _TokenBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = _TokenBucket(
            rate_per_second=self.config.rate_limit_per_second,
            now_fn=self.now_fn,
        )

    async def run(
        self,
        *,
        duration_seconds: float,
        poll_once: PollOnceFn,
    ) -> PollerResult:
        """Execute continuous constation for ``duration_seconds`` of run time.

        ``poll_once(phase)`` performs one attest cycle and returns either a
        :class:`PollSample` or a terminal :class:`ConstationVerdict`.
        """
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")

        cfg = self.config
        samples: list[PollSample] = []
        cost = 0.0
        poll_count = 0
        observed_max_gap = 0.0
        last_success_at: float | None = None
        run_start = self.now_fn()
        run_end = run_start + duration_seconds

        async def _one(phase: str) -> PollerResult | None:
            nonlocal cost, poll_count, observed_max_gap, last_success_at
            if poll_count >= cfg.max_polls:
                return _fail(
                    ConstationFailCode.POLL_CAP_EXCEEDED,
                    samples,
                    observed_max_gap,
                    cfg,
                    poll_count,
                    cost,
                    detail=f"polls={poll_count}",
                )
            if cost + cfg.cost_per_poll > cfg.max_cost_units:
                return _fail(
                    ConstationFailCode.COST_CAP_EXCEEDED,
                    samples,
                    observed_max_gap,
                    cfg,
                    poll_count,
                    cost,
                    detail=f"cost={cost}",
                )

            wait = self._limiter.try_take()
            if wait > 0:
                await self.sleep_fn(wait)

            outcome = await self._poll_with_retry(poll_once, phase)
            if isinstance(outcome, ConstationVerdict):
                return _fail(
                    outcome.reason,
                    samples,
                    observed_max_gap,
                    cfg,
                    poll_count,
                    cost,
                    detail=outcome.detail,
                    fault_class=outcome.fault_class,
                )

            now = outcome.at_monotonic
            if last_success_at is not None:
                gap = now - last_success_at
                if gap > observed_max_gap:
                    observed_max_gap = gap
                if gap > cfg.gap_budget_seconds:
                    samples.append(outcome)
                    poll_count += 1
                    cost += cfg.cost_per_poll
                    return _fail(
                        ConstationFailCode.CONSTATION_GAP,
                        samples,
                        observed_max_gap,
                        cfg,
                        poll_count,
                        cost,
                        detail=(
                            f"observed={observed_max_gap} "
                            f"budget={cfg.gap_budget_seconds}"
                        ),
                    )
            last_success_at = now
            samples.append(outcome)
            poll_count += 1
            cost += cfg.cost_per_poll
            return None

        # Start
        failed = await _one(PollPhase.START)
        if failed is not None:
            return failed

        # Randomized mid-run intervals until wall clock reaches run_end
        while self.now_fn() < run_end:
            interval = self._next_interval()
            # Sleep in slices so gap detection can fire if poll_once is slow
            # relative to budget; still one logical wait between polls.
            await self.sleep_fn(interval)
            if self.now_fn() >= run_end:
                break
            # Pre-check idle gap before attempting poll (sidecar silence)
            if last_success_at is not None:
                idle = self.now_fn() - last_success_at
                if idle > observed_max_gap:
                    observed_max_gap = idle
                if idle > cfg.gap_budget_seconds:
                    return _fail(
                        ConstationFailCode.CONSTATION_GAP,
                        samples,
                        observed_max_gap,
                        cfg,
                        poll_count,
                        cost,
                        detail=(
                            f"observed={observed_max_gap} "
                            f"budget={cfg.gap_budget_seconds}"
                        ),
                    )
            failed = await _one(PollPhase.INTERVAL)
            if failed is not None:
                return failed

        # End
        failed = await _one(PollPhase.END)
        if failed is not None:
            return failed

        return PollerResult(
            ok=True,
            reason=ConstationFailCode.OK,
            fault_class=None,
            samples=tuple(samples),
            observed_max_gap_seconds=observed_max_gap,
            gap_budget_seconds=cfg.gap_budget_seconds,
            poll_count=poll_count,
            cost_units=cost,
        )

    def _next_interval(self) -> float:
        cfg = self.config
        span = cfg.max_interval_seconds - cfg.min_interval_seconds
        return cfg.min_interval_seconds + self.rng_fn() * span

    async def _poll_with_retry(
        self,
        poll_once: PollOnceFn,
        phase: str,
    ) -> PollSample | ConstationVerdict:
        cfg = self.config
        attempt = 0
        while True:
            try:
                return await poll_once(phase)
            except LiumAuthError:
                logger.warning("constation poll auth revoked phase=%s", phase)
                return ConstationVerdict(
                    ok=False,
                    reason=ConstationFailCode.LIUM_AUTH_REVOKED,
                    detail=f"phase={phase}",
                )
            except LiumRateLimitError:
                logger.warning("constation poll rate limited phase=%s", phase)
                return ConstationVerdict(
                    ok=False,
                    reason=ConstationFailCode.LIUM_RATE_LIMITED,
                    detail=f"phase={phase}",
                )
            except LiumError as exc:
                # Transport / 5xx — bounded retry then network_partition
                if attempt >= cfg.max_network_retries:
                    logger.warning(
                        "constation poll network exhausted phase=%s attempts=%s",
                        phase,
                        attempt + 1,
                    )
                    return ConstationVerdict(
                        ok=False,
                        reason=ConstationFailCode.NETWORK_PARTITION,
                        detail=f"phase={phase} err={type(exc).__name__}",
                    )
                delay = min(
                    cfg.backoff_max_seconds,
                    cfg.backoff_base_seconds * (2**attempt),
                )
                # full jitter
                delay = delay * self.rng_fn()
                attempt += 1
                await self.sleep_fn(delay)
            except Exception as exc:
                # Sidecar / unexpected — treat as miner-side attest failure
                logger.warning(
                    "constation poll sidecar/transport failure phase=%s err=%s",
                    phase,
                    type(exc).__name__,
                )
                return ConstationVerdict(
                    ok=False,
                    reason=ConstationFailCode.SIDECAR_ATTEST_FAILED,
                    detail=f"phase={phase} err={type(exc).__name__}",
                )


def _fail(
    reason: ConstationFailCode,
    samples: list[PollSample],
    observed_max_gap: float,
    cfg: PollerConfig,
    poll_count: int,
    cost: float,
    *,
    detail: str | None = None,
    fault_class: FaultClass | None = None,
) -> PollerResult:
    from base.compute.constation_types import fault_class_for

    return PollerResult(
        ok=False,
        reason=reason,
        fault_class=fault_class if fault_class is not None else fault_class_for(reason),
        samples=tuple(samples),
        observed_max_gap_seconds=observed_max_gap,
        gap_budget_seconds=cfg.gap_budget_seconds,
        poll_count=poll_count,
        cost_units=cost,
        detail=detail,
    )


__all__ = [
    "ContinuousConstationPoller",
    "PollPhase",
    "PollerConfig",
    "PollerResult",
]
