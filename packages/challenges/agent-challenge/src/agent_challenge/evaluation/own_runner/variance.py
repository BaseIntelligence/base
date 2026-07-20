"""Variance-aware per-task aggregation over k attested trials (architecture sec 4 C5 / sec 5).

Given the ``k`` attested trial scores a single task produced (one per trial, in
plan order), collapse them into ONE per-task score under a configurable
aggregation MODE:

* :data:`PER_TASK_MEAN` (``"mean"``, the default) -- the epsilon=0 harbor mean of
  the k trial scores. Delegated to :meth:`reward.Mean.aggregate`
  (``sum(values) / len(values)``, left-to-right, order-preserving) so ``k=1`` +
  ``mean`` reproduces legacy per-task scoring byte-identically and ``k>1`` +
  ``mean`` is the legacy ``n_attempts`` mean (ULP-identical to stock harbor
  0.13.1).
* :data:`PER_TASK_BEST_OF_K` (``"best-of-k"``) -- the MAXIMUM trial score, so a
  flaky task that passes at least one trial scores its best result.

The path is pure and DETERMINISTIC (no RNG): identical trial inputs always yield
identical task scores. The keep-good-scoring-tasks JOB policies (drop-lowest-N,
threshold-band) layer on top of the per-task scores this module produces and are
implemented separately.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agent_challenge.evaluation.own_runner.reward import Max, Mean, NumericReward

#: Default per-task aggregation over the k trials: the epsilon=0 harbor mean.
PER_TASK_MEAN = "mean"
#: best-of-k: the maximum trial score (keeps a flaky task's best trial).
PER_TASK_BEST_OF_K = "best-of-k"
#: The per-task aggregation mode used when none is configured (legacy-compatible).
DEFAULT_PER_TASK_AGGREGATION = PER_TASK_MEAN
#: All supported per-task aggregation modes (``mean`` is the default).
PER_TASK_AGGREGATION_MODES: tuple[str, ...] = (PER_TASK_MEAN, PER_TASK_BEST_OF_K)


class InvalidAggregationModeError(ValueError):
    """Raised when an unrecognized per-task aggregation mode is requested.

    Fail-closed: an unknown mode is rejected rather than silently falling back to
    the mean (which would let a misconfiguration change scores undetected).
    """


def normalize_aggregation_mode(mode: str | None) -> str:
    """Normalize/validate a per-task aggregation mode (``None`` -> the default).

    Trims surrounding whitespace and lower-cases so ``" Best-Of-K "`` resolves to
    ``"best-of-k"``. Raises :class:`InvalidAggregationModeError` for any value
    outside :data:`PER_TASK_AGGREGATION_MODES`.
    """

    if mode is None:
        return DEFAULT_PER_TASK_AGGREGATION
    normalized = mode.strip().lower()
    if normalized not in PER_TASK_AGGREGATION_MODES:
        raise InvalidAggregationModeError(
            f"unknown per-task aggregation mode {mode!r}; "
            f"expected one of {sorted(PER_TASK_AGGREGATION_MODES)}"
        )
    return normalized


def aggregate_trial_scores(
    trial_scores: Sequence[NumericReward],
    *,
    mode: str = DEFAULT_PER_TASK_AGGREGATION,
) -> float:
    """Collapse one task's ``k`` ordered trial scores into a single per-task score.

    ``mean`` is the epsilon=0 harbor mean (``sum/len``); ``best-of-k`` is the max.
    Trial ORDER is preserved (never sorted) so the mean stays ULP-identical to
    harbor. Raises :class:`InvalidAggregationModeError` for an unknown mode or an
    empty trial list (a task always contributes ``k>=1`` trials).
    """

    normalized = normalize_aggregation_mode(mode)
    values = [float(score) for score in trial_scores]
    if not values:
        raise InvalidAggregationModeError(
            "cannot aggregate an empty trial-score list (expected k>=1 trials)"
        )
    if normalized == PER_TASK_BEST_OF_K:
        return float(Max.aggregate(values))
    return float(Mean.aggregate(values))


def collect_trial_scores(
    outcomes: Iterable[Any],
) -> OrderedDict[str, list[float]]:
    """Group trial outcomes by task into ordered per-trial score lists (VAL-SCORE-001).

    Preserves the trial (plan) order the orchestrator produced and records EXACTLY
    one entry per trial (never deduped), so a task run with ``k`` trials yields a
    list of ``k`` ordered scores -- the aggregation input before any policy is
    applied. Each outcome's host-readable per-trial score is derived identically
    to the orchestrator (an errored trial contributes ``0.0``).
    """

    from agent_challenge.evaluation.own_runner.orchestrator import _trial_score

    grouped: OrderedDict[str, list[float]] = OrderedDict()
    for outcome in outcomes:
        grouped.setdefault(outcome.task_name, []).append(_trial_score(outcome.rewards))
    return grouped


def aggregate_task_scores(
    trial_scores_by_task: Mapping[str, Sequence[NumericReward]],
    *,
    mode: str = DEFAULT_PER_TASK_AGGREGATION,
) -> dict[str, float]:
    """Aggregate each task's ordered trial scores into one per-task score.

    Deterministic: iterates the mapping in its given order and applies
    :func:`aggregate_trial_scores` per task with the validated ``mode``.
    """

    normalized = normalize_aggregation_mode(mode)
    return {
        task: aggregate_trial_scores(scores, mode=normalized)
        for task, scores in trial_scores_by_task.items()
    }


def aggregate_per_task(
    outcomes: Iterable[Any],
    *,
    mode: str = DEFAULT_PER_TASK_AGGREGATION,
) -> dict[str, float]:
    """Collect the k ordered trial scores per task, then aggregate each under ``mode``.

    The end-to-end per-task aggregation surface: trial outcomes in, one score per
    task out (the ``mean`` default reproduces the legacy per-task mean exactly).
    """

    return aggregate_task_scores(collect_trial_scores(outcomes), mode=mode)


__all__ = [
    "PER_TASK_MEAN",
    "PER_TASK_BEST_OF_K",
    "DEFAULT_PER_TASK_AGGREGATION",
    "PER_TASK_AGGREGATION_MODES",
    "InvalidAggregationModeError",
    "normalize_aggregation_mode",
    "aggregate_trial_scores",
    "collect_trial_scores",
    "aggregate_task_scores",
    "aggregate_per_task",
]
