"""Unit tests for the reusable retry policy (RetryPolicy + RetryState).

Pure/deterministic: ``now`` and the jitter fraction are injected, so no clock
and no real sleeping are involved.
"""

from __future__ import annotations

import pytest

from base.supervisor.retry import RetryPolicy, RetryState

# ---------------------------------------------------------------------------
# RetryPolicy: construction validation
# ---------------------------------------------------------------------------


def test_policy_rejects_non_positive_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_policy_rejects_non_positive_base_delay() -> None:
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay=0.0)


def test_policy_rejects_max_delay_below_base_delay() -> None:
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(base_delay=100.0, max_delay=50.0)


# ---------------------------------------------------------------------------
# RetryPolicy: backoff math (no jitter)
# ---------------------------------------------------------------------------


def test_backoff_doubles_each_attempt_without_jitter() -> None:
    policy = RetryPolicy(base_delay=60.0, max_delay=100_000.0, jitter=False)
    assert policy.compute_delay(1) == 60.0
    assert policy.compute_delay(2) == 120.0
    assert policy.compute_delay(3) == 240.0
    assert policy.compute_delay(4) == 480.0


def test_backoff_is_capped_at_max_delay() -> None:
    policy = RetryPolicy(base_delay=60.0, max_delay=200.0, jitter=False)
    assert policy.compute_delay(1) == 60.0
    assert policy.compute_delay(2) == 120.0
    # 60 * 2**2 = 240 -> capped to 200; and every later attempt stays capped.
    assert policy.compute_delay(3) == 200.0
    assert policy.compute_delay(10) == 200.0


def test_compute_delay_rejects_attempt_below_one() -> None:
    policy = RetryPolicy(jitter=False)
    with pytest.raises(ValueError, match="attempt"):
        policy.compute_delay(0)


# ---------------------------------------------------------------------------
# RetryPolicy: jitter bounds
# ---------------------------------------------------------------------------


def test_jitter_stays_within_half_to_full_capped_delay() -> None:
    policy = RetryPolicy(base_delay=100.0, max_delay=100_000.0, jitter=True)
    capped = 100.0  # attempt 1
    assert policy.compute_delay(1, jitter=0.0) == capped / 2.0
    assert policy.compute_delay(1, jitter=1.0) == capped
    mid = policy.compute_delay(1, jitter=0.5)
    assert capped / 2.0 <= mid <= capped
    assert mid == pytest.approx(75.0)


def test_jitter_fraction_is_clamped_to_unit_interval() -> None:
    policy = RetryPolicy(base_delay=100.0, max_delay=100_000.0, jitter=True)
    # Out-of-range fractions are clamped, never producing <half or >capped.
    assert policy.compute_delay(1, jitter=-5.0) == 50.0
    assert policy.compute_delay(1, jitter=5.0) == 100.0


# ---------------------------------------------------------------------------
# RetryState: transitions
# ---------------------------------------------------------------------------


def test_fresh_state_is_eligible_and_not_exhausted() -> None:
    policy = RetryPolicy(max_attempts=3)
    state = RetryState()
    assert state.attempts == 0
    assert state.is_eligible(now=0.0)
    assert not state.is_exhausted(policy)


def test_record_failure_increments_and_schedules_backoff() -> None:
    policy = RetryPolicy(base_delay=60.0, max_delay=100_000.0, jitter=False)
    state = RetryState()

    state.record_failure(now=1_000.0, policy=policy, error="boom")
    assert state.attempts == 1
    assert state.last_error == "boom"
    assert state.next_eligible_monotonic == 1_060.0
    # Not eligible until the backoff elapses.
    assert not state.is_eligible(1_059.9)
    assert state.is_eligible(1_060.0)

    state.record_failure(now=1_060.0, policy=policy)
    assert state.attempts == 2
    assert state.next_eligible_monotonic == 1_060.0 + 120.0


def test_record_failure_uses_injected_jitter_source() -> None:
    policy = RetryPolicy(base_delay=100.0, max_delay=100_000.0, jitter=True)
    state = RetryState()
    state.record_failure(now=0.0, policy=policy, jitter_source=lambda: 0.0)
    # Equal-jitter floor at attempt 1 == capped/2 == 50.
    assert state.next_eligible_monotonic == 50.0


def test_record_success_resets_state() -> None:
    policy = RetryPolicy(base_delay=60.0, jitter=False)
    state = RetryState()
    state.record_failure(now=5.0, policy=policy, error="boom")
    state.record_success()
    assert state.attempts == 0
    assert state.next_eligible_monotonic == 0.0
    assert state.last_error is None
    assert state.is_eligible(0.0)


def test_is_exhausted_after_max_attempts() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=False)
    state = RetryState()
    now = 0.0
    for _ in range(2):
        state.record_failure(now=now, policy=policy)
        now = state.next_eligible_monotonic
        assert not state.is_exhausted(policy)
    state.record_failure(now=now, policy=policy)
    assert state.attempts == 3
    assert state.is_exhausted(policy)
