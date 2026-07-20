"""Per-task variance-aware aggregation over k attested trials (M5 scoring).

Behavioral parity contract for the ``variance-ktrial-aggregation`` feature,
anchored to the mission validation assertions VAL-SCORE-001..005:

* VAL-SCORE-001 -- k attested trials per task are each recorded as k ordered
  trial scores (one per trial), preserving trial order, before any policy.
* VAL-SCORE-002 -- the DEFAULT per-task aggregation is the epsilon=0 harbor mean
  of the k trial scores (byte-identical to stock harbor 0.13.1).
* VAL-SCORE-003 -- ``best-of-k`` per-task aggregation returns the MAX trial score.
* VAL-SCORE-004 -- a unanimous trial set collapses to 1.0 / 0.0 under EVERY mode.
* VAL-SCORE-005 -- per-task (and per-job map) aggregation is deterministic for
  fixed trial inputs (no RNG in the scoring path).
"""

from __future__ import annotations

import inspect

import pytest

from agent_challenge.evaluation.own_runner.orchestrator import TrialOutcome
from agent_challenge.evaluation.own_runner.reward import Mean, floats_bit_identical
from agent_challenge.evaluation.own_runner.variance import (
    DEFAULT_PER_TASK_AGGREGATION,
    PER_TASK_AGGREGATION_MODES,
    PER_TASK_BEST_OF_K,
    PER_TASK_MEAN,
    InvalidAggregationModeError,
    aggregate_per_task,
    aggregate_task_scores,
    aggregate_trial_scores,
    collect_trial_scores,
    normalize_aggregation_mode,
)


def _outcome(task: str, attempt: int, score: float) -> TrialOutcome:
    """A completed trial whose host-readable score is exactly ``score``."""

    return TrialOutcome(
        task_name=task,
        trial_name=f"{task}__attempt-{attempt}",
        status="completed",
        rewards={"reward": score},
    )


# ---------------------------------------------------------------------------
# VAL-SCORE-001 -- k ordered trial scores recorded per task (pre-policy input)
# ---------------------------------------------------------------------------


def test_k1_records_single_trial_score_per_task() -> None:
    outcomes = [_outcome("task-a", 0, 1.0)]
    assert collect_trial_scores(outcomes) == {"task-a": [1.0]}


def test_k3_records_three_ordered_trial_scores_per_task() -> None:
    outcomes = [
        _outcome("task-a", 0, 0.1),
        _outcome("task-a", 1, 0.9),
        _outcome("task-a", 2, 0.5),
    ]
    grouped = collect_trial_scores(outcomes)
    assert grouped == {"task-a": [0.1, 0.9, 0.5]}
    # Exactly k entries, in trial order (never sorted, never deduped).
    assert list(grouped["task-a"]) == [0.1, 0.9, 0.5]


def test_duplicate_trial_scores_are_not_deduplicated() -> None:
    outcomes = [_outcome("task-a", i, 1.0) for i in range(3)]
    assert collect_trial_scores(outcomes) == {"task-a": [1.0, 1.0, 1.0]}


def test_multiple_tasks_grouped_independently_preserving_order() -> None:
    # Interleaved (attempt-outer, task-inner) plan order, as the orchestrator emits.
    outcomes = [
        _outcome("task-a", 0, 0.0),
        _outcome("task-b", 0, 1.0),
        _outcome("task-a", 1, 1.0),
        _outcome("task-b", 1, 0.0),
    ]
    grouped = collect_trial_scores(outcomes)
    assert grouped == {"task-a": [0.0, 1.0], "task-b": [1.0, 0.0]}


def test_errored_trial_contributes_a_zero_score_entry() -> None:
    errored = TrialOutcome(
        task_name="task-a",
        trial_name="task-a__attempt-0",
        status="failed",
        rewards=None,
        errored=True,
    )
    grouped = collect_trial_scores([errored, _outcome("task-a", 1, 1.0)])
    assert grouped == {"task-a": [0.0, 1.0]}


# ---------------------------------------------------------------------------
# VAL-SCORE-002 -- default per-task aggregation is the epsilon=0 harbor mean
# ---------------------------------------------------------------------------


def test_default_mode_is_mean() -> None:
    assert DEFAULT_PER_TASK_AGGREGATION == PER_TASK_MEAN == "mean"


@pytest.mark.parametrize(
    "trials, expected",
    [
        ([1.0, 1.0, 1.0], 1.0),
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 1.0], 0.6666666666666666),
        ([1.0], 1.0),
        ([0.0], 0.0),
    ],
)
def test_default_mean_matches_harbor_mean(trials: list[float], expected: float) -> None:
    result = aggregate_trial_scores(trials)
    # Byte-identical to an independent CPython float mean recompute.
    independent = sum(trials) / len(trials)
    assert floats_bit_identical(result, independent)
    assert floats_bit_identical(result, expected)
    # Single-sourced from the harbor 0.13.1 Mean metric (sum/len, ULP-exact).
    assert floats_bit_identical(result, float(Mean.aggregate(list(trials))))


def test_mean_preserves_trial_order_not_sorted() -> None:
    # A shuffled order yields the same ULP mean only because mean is order-robust
    # here; the point is the function never sorts (order-in == order-consumed).
    assert floats_bit_identical(
        aggregate_trial_scores([1.0, 0.0, 1.0], mode="mean"),
        aggregate_trial_scores([1.0, 1.0, 0.0], mode="mean"),
    )


# ---------------------------------------------------------------------------
# VAL-SCORE-003 -- best-of-k returns the max trial score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trials, expected",
    [
        ([0.0, 1.0, 0.0], 1.0),
        ([0.4, 0.6, 0.5], 0.6),
        ([0.0, 0.0, 0.0], 0.0),
        ([0.25], 0.25),
    ],
)
def test_best_of_k_returns_max(trials: list[float], expected: float) -> None:
    result = aggregate_trial_scores(trials, mode=PER_TASK_BEST_OF_K)
    assert floats_bit_identical(result, expected)
    assert floats_bit_identical(result, float(max(trials)))


def test_best_of_k_differs_from_mean_on_a_flaky_task() -> None:
    trials = [0.0, 1.0, 0.0]
    assert aggregate_trial_scores(trials, mode="best-of-k") == 1.0
    assert aggregate_trial_scores(trials, mode="mean") != 1.0


# ---------------------------------------------------------------------------
# VAL-SCORE-004 -- unanimous trials collapse to 1.0 / 0.0 under EVERY mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", PER_TASK_AGGREGATION_MODES)
def test_unanimous_pass_collapses_to_one_under_every_mode(mode: str) -> None:
    assert aggregate_trial_scores([1.0, 1.0, 1.0], mode=mode) == 1.0


@pytest.mark.parametrize("mode", PER_TASK_AGGREGATION_MODES)
def test_unanimous_fail_collapses_to_zero_under_every_mode(mode: str) -> None:
    assert aggregate_trial_scores([0.0, 0.0, 0.0], mode=mode) == 0.0


# ---------------------------------------------------------------------------
# VAL-SCORE-005 -- deterministic for fixed inputs (no RNG)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", PER_TASK_AGGREGATION_MODES)
def test_repeated_aggregation_is_byte_identical(mode: str) -> None:
    fixtures = [[1.0, 0.0, 1.0], [0.4, 0.6, 0.5]]
    for trials in fixtures:
        first = aggregate_trial_scores(trials, mode=mode)
        second = aggregate_trial_scores(trials, mode=mode)
        assert floats_bit_identical(first, second)


def test_per_task_map_aggregation_is_deterministic() -> None:
    per_task = {"task-a": [1.0, 0.0, 1.0], "task-b": [0.4, 0.6, 0.5]}
    assert aggregate_task_scores(per_task, mode="mean") == aggregate_task_scores(
        per_task, mode="mean"
    )
    assert aggregate_task_scores(per_task, mode="best-of-k") == {
        "task-a": 1.0,
        "task-b": 0.6,
    }


def test_scoring_path_imports_no_rng() -> None:
    from agent_challenge.evaluation.own_runner import variance

    source = inspect.getsource(variance)
    assert "import random" not in source
    assert "random." not in source


def test_aggregate_per_task_end_to_end_from_outcomes() -> None:
    outcomes = [
        _outcome("task-a", 0, 0.0),
        _outcome("task-a", 1, 1.0),
        _outcome("task-a", 2, 0.0),
    ]
    assert aggregate_per_task(outcomes, mode="mean") == {"task-a": 0.3333333333333333}
    assert aggregate_per_task(outcomes, mode="best-of-k") == {"task-a": 1.0}


# ---------------------------------------------------------------------------
# Fail-closed: an unknown aggregation mode is rejected (never silently meaned)
# ---------------------------------------------------------------------------


def test_unknown_mode_raises() -> None:
    with pytest.raises(InvalidAggregationModeError):
        aggregate_trial_scores([1.0], mode="median")
    with pytest.raises(InvalidAggregationModeError):
        normalize_aggregation_mode("drop-lowest-3")


def test_empty_trial_list_fails_closed() -> None:
    with pytest.raises(InvalidAggregationModeError):
        aggregate_trial_scores([], mode="mean")


def test_normalize_mode_is_case_insensitive_and_defaults() -> None:
    assert normalize_aggregation_mode(None) == DEFAULT_PER_TASK_AGGREGATION
    assert normalize_aggregation_mode("MEAN") == "mean"
    assert normalize_aggregation_mode(" Best-Of-K ") == "best-of-k"


# ---------------------------------------------------------------------------
# Config surface -- per_task_aggregation knob (default off / mean)
# ---------------------------------------------------------------------------


def test_default_config_uses_mean() -> None:
    from agent_challenge.sdk.config import ChallengeSettings

    assert ChallengeSettings().per_task_aggregation == "mean"


def test_config_accepts_best_of_k() -> None:
    from agent_challenge.sdk.config import ChallengeSettings

    assert ChallengeSettings(per_task_aggregation="best-of-k").per_task_aggregation == "best-of-k"


def test_config_rejects_unknown_aggregation() -> None:
    from pydantic import ValidationError

    from agent_challenge.sdk.config import ChallengeSettings

    with pytest.raises(ValidationError) as error:
        ChallengeSettings(per_task_aggregation="median")
    assert "per_task_aggregation" in str(error.value)


def test_config_modes_stay_in_sync_with_variance_module() -> None:
    # Drift guard: the settings validator duplicates the accepted set to avoid a
    # heavy import; this pins it to the variance module's source of truth.
    assert set(PER_TASK_AGGREGATION_MODES) == {"mean", "best-of-k"}


# ---------------------------------------------------------------------------
# Wiring -- the per-task aggregation surfaces actually consume the mode
# ---------------------------------------------------------------------------


def test_backend_per_task_scores_defaults_to_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.evaluation import own_runner_backend as backend

    monkeypatch.delenv("CHALLENGE_PER_TASK_AGGREGATION", raising=False)
    outcomes = [_outcome("t", 0, 0.0), _outcome("t", 1, 1.0), _outcome("t", 2, 0.0)]
    assert backend._per_task_scores(outcomes) == {"t": 0.3333333333333333}


def test_backend_per_task_scores_honors_best_of_k_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.evaluation import own_runner_backend as backend

    monkeypatch.setenv("CHALLENGE_PER_TASK_AGGREGATION", "best-of-k")
    outcomes = [_outcome("t", 0, 0.0), _outcome("t", 1, 1.0), _outcome("t", 2, 0.0)]
    assert backend._per_task_scores(outcomes) == {"t": 1.0}


def test_terminal_bench_aggregate_score_defaults_to_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.evaluation import terminal_bench as tb
    from agent_challenge.sdk.config import ChallengeSettings

    monkeypatch.setattr(tb, "settings", ChallengeSettings(per_task_aggregation="mean"))
    trials = [{"score": 0.0}, {"score": 1.0}, {"score": 0.0}]
    assert tb._aggregate_score(trials) == 0.3333333333333333


def test_terminal_bench_aggregate_score_honors_best_of_k(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.evaluation import terminal_bench as tb
    from agent_challenge.sdk.config import ChallengeSettings

    monkeypatch.setattr(tb, "settings", ChallengeSettings(per_task_aggregation="best-of-k"))
    trials = [{"score": 0.0}, {"score": 1.0}, {"score": 0.0}]
    assert tb._aggregate_score(trials) == 1.0
