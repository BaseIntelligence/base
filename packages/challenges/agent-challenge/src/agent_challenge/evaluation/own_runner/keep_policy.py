"""Keep-good-scoring-tasks JOB policy over per-task scores (architecture sec 4 C5).

After per-task aggregation collapses each task's ``k`` attested trials into ONE
per-task score (:mod:`agent_challenge.evaluation.own_runner.variance`), the JOB
score is the mean over a KEPT subset of those per-task scores, selected by a
configurable keep-good-scoring-tasks policy:

* :data:`KEEP_POLICY_OFF` (``"off"``, the default) -- keep ALL tasks; the job
  score is the mean over every task (byte-identical legacy denominator = the full
  task count). This preserves today's scoring exactly.
* :data:`KEEP_POLICY_DROP_LOWEST_N` (``"drop-lowest-n"``) -- drop the ``N`` lowest
  per-task scores, then mean over the survivors. Clamped so at least one (the
  highest) task is always retained: ``N >= task_count`` never empties the set nor
  divides by zero.
* :data:`KEEP_POLICY_THRESHOLD_BAND` (``"threshold-band"``) -- keep only tasks
  scoring at/above an (inclusive) threshold, then mean over the kept set. When the
  kept set is empty the job score is a defined ``0.0`` (never NaN / div-by-zero).
* :data:`KEEP_POLICY_BEST_OF_K` (``"best-of-k"``) -- keep only the single
  best-scoring task (job score = the max per-task score).

Invariants:

* **Anti-gaming.** This module produces only the SCORE aggregation; it NEVER
  touches the reward-eligibility task-count gate. The caller keeps counting the
  FULL selected task set for eligibility, so excluding tasks from the score can
  never shrink the eligibility denominator.
* **Fail-closed.** An unknown policy, a negative ``N``, or a threshold outside
  ``[0, 1]`` raises :class:`InvalidKeepPolicyError` rather than silently falling
  back to a different policy.
* **Deterministic.** No RNG: identical inputs always yield an identical job score.
  ``drop-lowest-n`` preserves the survivors' original order so its ``N=0`` mean is
  byte-identical to the legacy mean-over-all-tasks.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_challenge.evaluation.own_runner.reward import Mean

#: Keep every task; mean over all (legacy, the default).
KEEP_POLICY_OFF = "off"
#: Keep only the single best-scoring task (job score = max per-task score).
KEEP_POLICY_BEST_OF_K = "best-of-k"
#: Drop the N lowest-scoring tasks; mean over the survivors.
KEEP_POLICY_DROP_LOWEST_N = "drop-lowest-n"
#: Keep tasks scoring at/above an inclusive threshold; mean over the kept set.
KEEP_POLICY_THRESHOLD_BAND = "threshold-band"

#: The keep policy used when none is configured (legacy-compatible).
DEFAULT_KEEP_POLICY = KEEP_POLICY_OFF
#: All supported keep policies (``off`` is the default).
KEEP_POLICY_MODES: tuple[str, ...] = (
    KEEP_POLICY_OFF,
    KEEP_POLICY_BEST_OF_K,
    KEEP_POLICY_DROP_LOWEST_N,
    KEEP_POLICY_THRESHOLD_BAND,
)


class InvalidKeepPolicyError(ValueError):
    """Raised when a keep policy (or its parameter) is unrecognized/out of range.

    Fail-closed: an unknown policy, a negative drop-lowest ``N``, or a threshold
    outside ``[0, 1]`` is rejected rather than silently coerced (which would let a
    misconfiguration change scores undetected).
    """


def normalize_keep_policy(policy: str | None) -> str:
    """Normalize/validate a keep policy (``None`` -> :data:`DEFAULT_KEEP_POLICY`).

    Trims surrounding whitespace and lower-cases so ``" Drop-Lowest-N "`` resolves
    to ``"drop-lowest-n"``. Raises :class:`InvalidKeepPolicyError` for any value
    outside :data:`KEEP_POLICY_MODES`.
    """

    if policy is None:
        return DEFAULT_KEEP_POLICY
    normalized = policy.strip().lower()
    if normalized not in KEEP_POLICY_MODES:
        raise InvalidKeepPolicyError(
            f"unknown keep policy {policy!r}; expected one of {sorted(KEEP_POLICY_MODES)}"
        )
    return normalized


def _validated_drop_lowest_n(drop_lowest_n: int) -> int:
    if drop_lowest_n < 0:
        raise InvalidKeepPolicyError(f"drop-lowest-n requires N >= 0, got {drop_lowest_n!r}")
    return drop_lowest_n


def _validated_threshold(threshold: float) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise InvalidKeepPolicyError(
            f"threshold-band requires a threshold in [0, 1], got {threshold!r}"
        )
    return threshold


def select_kept_scores(
    task_scores: Sequence[float],
    *,
    policy: str = DEFAULT_KEEP_POLICY,
    drop_lowest_n: int = 0,
    threshold: float = 0.0,
) -> list[float]:
    """Return the KEPT subset of per-task scores under ``policy`` (original order).

    ``off`` keeps every score; ``best-of-k`` keeps the single max; ``drop-lowest-n``
    removes the ``N`` lowest (clamped to always retain the highest one);
    ``threshold-band`` keeps scores ``>= threshold`` (inclusive). Survivors keep
    their original order so a downstream mean stays ULP-stable. Raises
    :class:`InvalidKeepPolicyError` for an unknown policy or out-of-range param.
    """

    normalized = normalize_keep_policy(policy)
    scores = [float(score) for score in task_scores]

    if normalized == KEEP_POLICY_OFF:
        return scores

    # Validate policy params BEFORE the empty-input short-circuit so a direct call
    # with a misconfigured param fails closed even on an empty score list (the
    # config-layer validators already guard the load path; this closes the
    # direct-call gap).
    if normalized == KEEP_POLICY_DROP_LOWEST_N:
        n = _validated_drop_lowest_n(drop_lowest_n)
    elif normalized == KEEP_POLICY_THRESHOLD_BAND:
        t = _validated_threshold(threshold)

    if not scores:
        return []
    if normalized == KEEP_POLICY_BEST_OF_K:
        return [max(scores)]
    if normalized == KEEP_POLICY_DROP_LOWEST_N:
        # Clamp so at least one (the highest) task always survives.
        drop_count = min(n, len(scores) - 1)
        if drop_count <= 0:
            return scores
        # Rank indices by (score, index) ascending; the lowest ``drop_count`` are
        # dropped. Ties break on original index so the drop set is deterministic;
        # the survivors are emitted in their original order.
        ranked = sorted(range(len(scores)), key=lambda i: (scores[i], i))
        dropped = set(ranked[:drop_count])
        return [score for index, score in enumerate(scores) if index not in dropped]
    # KEEP_POLICY_THRESHOLD_BAND
    return [score for score in scores if score >= t]


def keep_good_job_score(
    task_scores: Sequence[float],
    *,
    policy: str = DEFAULT_KEEP_POLICY,
    drop_lowest_n: int = 0,
    threshold: float = 0.0,
) -> float:
    """Job score = the epsilon=0 harbor mean over the KEPT per-task scores.

    Returns a defined ``0.0`` when the kept set is empty (all tasks below a
    ``threshold-band`` threshold, or no tasks at all), never NaN / div-by-zero.
    For ``off`` (and ``drop-lowest-n`` with ``N=0``) this is byte-identical to the
    legacy mean over all tasks (order preserved, ``sum(kept) / len(kept)``).
    """

    kept = select_kept_scores(
        task_scores,
        policy=policy,
        drop_lowest_n=drop_lowest_n,
        threshold=threshold,
    )
    if not kept:
        return 0.0
    return float(Mean.aggregate(kept))


__all__ = [
    "KEEP_POLICY_OFF",
    "KEEP_POLICY_BEST_OF_K",
    "KEEP_POLICY_DROP_LOWEST_N",
    "KEEP_POLICY_THRESHOLD_BAND",
    "DEFAULT_KEEP_POLICY",
    "KEEP_POLICY_MODES",
    "InvalidKeepPolicyError",
    "normalize_keep_policy",
    "select_kept_scores",
    "keep_good_job_score",
]
