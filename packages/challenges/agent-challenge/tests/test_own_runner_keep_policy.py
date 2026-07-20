"""Keep-good-scoring-tasks JOB policy over per-task scores (M5 scoring).

Behavioral contract for the ``keep-good-tasks-policy`` feature, anchored to the
mission validation assertions VAL-SCORE-006..012:

* VAL-SCORE-006 -- ``drop-lowest-N`` job score is the mean of the task scores that
  survive after dropping the N lowest (denominator = kept count).
* VAL-SCORE-007 -- ``drop-lowest-N`` with N >= task count clamps safely (retains
  the single best task; no div-by-zero, no NaN, no raise).
* VAL-SCORE-008 -- ``drop-lowest-N`` with N=0 equals the legacy mean over ALL tasks.
* VAL-SCORE-009 -- ``threshold-band`` keeps only tasks scoring at/above the
  (inclusive) threshold; job score = mean over the kept set.
* VAL-SCORE-010 -- ``threshold-band`` with every task below the threshold yields a
  DEFINED 0.0 job score (kept set empty), never NaN / div-by-zero.
* VAL-SCORE-011 -- an unknown policy / negative N / out-of-range threshold is
  rejected fail-closed (typed error), never silently coerced.
* VAL-SCORE-012 -- the keep policy affects only the SCORE aggregation, never the
  reward-eligibility task-count gate (covered end-to-end in
  ``test_finalize_keep_policy.py``).
"""

from __future__ import annotations

import inspect

import pytest

from agent_challenge.evaluation.own_runner.keep_policy import (
    DEFAULT_KEEP_POLICY,
    KEEP_POLICY_MODES,
    KEEP_POLICY_OFF,
    InvalidKeepPolicyError,
    keep_good_job_score,
    normalize_keep_policy,
    select_kept_scores,
)
from agent_challenge.evaluation.own_runner.reward import floats_bit_identical

# ---------------------------------------------------------------------------
# Defaults / normalization
# ---------------------------------------------------------------------------


def test_default_policy_is_off() -> None:
    assert DEFAULT_KEEP_POLICY == KEEP_POLICY_OFF == "off"


def test_supported_modes() -> None:
    assert set(KEEP_POLICY_MODES) == {
        "off",
        "best-of-k",
        "drop-lowest-n",
        "threshold-band",
    }


def test_normalize_is_case_insensitive_and_defaults() -> None:
    assert normalize_keep_policy(None) == KEEP_POLICY_OFF
    assert normalize_keep_policy("OFF") == "off"
    assert normalize_keep_policy(" Drop-Lowest-N ") == "drop-lowest-n"
    assert normalize_keep_policy("Threshold-Band") == "threshold-band"
    assert normalize_keep_policy("Best-Of-K") == "best-of-k"


# ---------------------------------------------------------------------------
# off (default) -- keep ALL tasks, mean over every task (legacy denominator)
# ---------------------------------------------------------------------------


def test_off_keeps_all_tasks_in_order() -> None:
    scores = [1.0, 0.0, 0.5, 1.0]
    assert select_kept_scores(scores, policy="off") == scores


def test_off_job_score_is_plain_mean_over_all() -> None:
    scores = [1.0, 0.0, 0.5, 1.0]
    result = keep_good_job_score(scores, policy="off")
    assert floats_bit_identical(result, sum(scores) / len(scores))


def test_off_empty_is_defined_zero() -> None:
    assert keep_good_job_score([], policy="off") == 0.0


# ---------------------------------------------------------------------------
# VAL-SCORE-006 -- drop-lowest-N mean over the surviving tasks
# ---------------------------------------------------------------------------


def test_drop_lowest_n1_drops_single_lowest() -> None:
    scores = [1.0, 1.0, 0.5, 0.0]
    kept = select_kept_scores(scores, policy="drop-lowest-n", drop_lowest_n=1)
    # The single lowest (0.0) is dropped; the surviving three keep original order.
    assert kept == [1.0, 1.0, 0.5]
    result = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=1)
    assert floats_bit_identical(result, 0.8333333333333334)
    assert floats_bit_identical(result, sum([1.0, 1.0, 0.5]) / 3)


def test_drop_lowest_n2_drops_two_lowest() -> None:
    scores = [1.0, 1.0, 0.5, 0.0]
    kept = select_kept_scores(scores, policy="drop-lowest-n", drop_lowest_n=2)
    assert kept == [1.0, 1.0]
    result = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=2)
    assert floats_bit_identical(result, 1.0)


def test_drop_lowest_n_drops_the_lowest_not_arbitrary() -> None:
    # Lowest two are 0.1 and 0.2; they must be the ones removed.
    scores = [0.2, 0.9, 0.1, 0.8]
    kept = select_kept_scores(scores, policy="drop-lowest-n", drop_lowest_n=2)
    assert kept == [0.9, 0.8]


# ---------------------------------------------------------------------------
# VAL-SCORE-007 -- N >= task count clamps safely (keep single best task)
# ---------------------------------------------------------------------------


def test_drop_lowest_n_equal_task_count_keeps_best() -> None:
    scores = [0.3, 0.9, 0.5]
    kept = select_kept_scores(scores, policy="drop-lowest-n", drop_lowest_n=3)
    assert kept == [0.9]
    result = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=3)
    assert floats_bit_identical(result, 0.9)


def test_drop_lowest_n_above_task_count_keeps_best() -> None:
    scores = [0.3, 0.9, 0.5]
    result = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=99)
    assert floats_bit_identical(result, 0.9)


def test_drop_lowest_n_minus_one_equals_best_task() -> None:
    scores = [0.2, 0.7, 0.4, 0.6]
    # N == tasks - 1 -> single best task retained.
    result = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=len(scores) - 1)
    assert floats_bit_identical(result, max(scores))


def test_drop_lowest_n_single_task_never_empty() -> None:
    result = keep_good_job_score([0.42], policy="drop-lowest-n", drop_lowest_n=5)
    assert floats_bit_identical(result, 0.42)


# ---------------------------------------------------------------------------
# VAL-SCORE-008 -- N=0 equals the legacy mean over all tasks
# ---------------------------------------------------------------------------


def test_drop_lowest_n0_equals_mean_over_all() -> None:
    scores = [1.0, 0.0, 0.6666666666666666, 0.5]
    result = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=0)
    # Byte-equal to the plain mean-over-all-tasks (order preserved, nothing dropped).
    assert floats_bit_identical(result, sum(scores) / len(scores))
    assert select_kept_scores(scores, policy="drop-lowest-n", drop_lowest_n=0) == scores


def test_drop_lowest_n0_matches_off_policy() -> None:
    scores = [0.3, 0.9, 0.1, 0.8, 0.6666666666666666]
    off = keep_good_job_score(scores, policy="off")
    n0 = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=0)
    assert floats_bit_identical(off, n0)


# ---------------------------------------------------------------------------
# VAL-SCORE-009 -- threshold-band keeps at/above-threshold tasks (inclusive)
# ---------------------------------------------------------------------------


def test_threshold_band_keeps_at_or_above() -> None:
    scores = [1.0, 0.4, 0.5, 0.0]
    kept = select_kept_scores(scores, policy="threshold-band", threshold=0.5)
    # 0.5 is INCLUSIVE; 0.4 and 0.0 excluded.
    assert kept == [1.0, 0.5]
    result = keep_good_job_score(scores, policy="threshold-band", threshold=0.5)
    assert floats_bit_identical(result, 0.75)


def test_threshold_band_boundary_is_inclusive() -> None:
    scores = [0.5, 0.5, 0.5]
    result = keep_good_job_score(scores, policy="threshold-band", threshold=0.5)
    assert floats_bit_identical(result, 0.5)


def test_threshold_band_zero_keeps_all() -> None:
    scores = [1.0, 0.0, 0.5]
    result = keep_good_job_score(scores, policy="threshold-band", threshold=0.0)
    assert floats_bit_identical(result, sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# VAL-SCORE-010 -- all tasks below threshold -> defined 0.0 (never NaN/raise)
# ---------------------------------------------------------------------------


def test_threshold_band_all_below_is_defined_zero() -> None:
    scores = [0.4, 0.0, 0.49]
    kept = select_kept_scores(scores, policy="threshold-band", threshold=0.5)
    assert kept == []
    result = keep_good_job_score(scores, policy="threshold-band", threshold=0.5)
    assert result == 0.0
    assert result == result  # not NaN


# ---------------------------------------------------------------------------
# best-of-k keep policy -- keep the single best task (defined, finite)
# ---------------------------------------------------------------------------


def test_best_of_k_keeps_single_best_task() -> None:
    scores = [0.3, 0.9, 0.5]
    assert select_kept_scores(scores, policy="best-of-k") == [0.9]
    result = keep_good_job_score(scores, policy="best-of-k")
    assert floats_bit_identical(result, 0.9)


def test_best_of_k_empty_is_defined_zero() -> None:
    assert keep_good_job_score([], policy="best-of-k") == 0.0


# ---------------------------------------------------------------------------
# VAL-SCORE-011 -- fail-closed on unknown policy / bad params
# ---------------------------------------------------------------------------


def test_unknown_policy_raises() -> None:
    with pytest.raises(InvalidKeepPolicyError):
        normalize_keep_policy("keep-everything")
    with pytest.raises(InvalidKeepPolicyError):
        keep_good_job_score([1.0], policy="median-band")


def test_negative_n_raises() -> None:
    with pytest.raises(InvalidKeepPolicyError):
        keep_good_job_score([1.0, 0.0], policy="drop-lowest-n", drop_lowest_n=-1)


def test_threshold_out_of_range_raises() -> None:
    with pytest.raises(InvalidKeepPolicyError):
        keep_good_job_score([1.0], policy="threshold-band", threshold=-0.1)
    with pytest.raises(InvalidKeepPolicyError):
        keep_good_job_score([1.0], policy="threshold-band", threshold=1.5)


# ---------------------------------------------------------------------------
# VAL-SCORE-011 (direct-call gap) -- params are validated BEFORE the empty-input
# short-circuit, so an empty score list with invalid params still fails closed
# (the config-layer validators already guard the load path).
# ---------------------------------------------------------------------------


def test_empty_scores_with_negative_n_raises() -> None:
    with pytest.raises(InvalidKeepPolicyError):
        select_kept_scores([], policy="drop-lowest-n", drop_lowest_n=-1)
    with pytest.raises(InvalidKeepPolicyError):
        keep_good_job_score([], policy="drop-lowest-n", drop_lowest_n=-1)


def test_empty_scores_with_out_of_range_threshold_raises() -> None:
    with pytest.raises(InvalidKeepPolicyError):
        select_kept_scores([], policy="threshold-band", threshold=1.5)
    with pytest.raises(InvalidKeepPolicyError):
        select_kept_scores([], policy="threshold-band", threshold=-0.1)
    with pytest.raises(InvalidKeepPolicyError):
        keep_good_job_score([], policy="threshold-band", threshold=1.5)


def test_empty_scores_with_valid_params_still_returns_empty() -> None:
    # Pre-validation must not change the empty-input result for VALID params.
    assert select_kept_scores([], policy="drop-lowest-n", drop_lowest_n=2) == []
    assert select_kept_scores([], policy="threshold-band", threshold=0.5) == []
    assert select_kept_scores([], policy="best-of-k") == []
    assert keep_good_job_score([], policy="drop-lowest-n", drop_lowest_n=2) == 0.0
    assert keep_good_job_score([], policy="threshold-band", threshold=0.5) == 0.0


# ---------------------------------------------------------------------------
# Determinism (VAL-SCORE-005 sibling) -- no RNG in the keep-policy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", KEEP_POLICY_MODES)
def test_repeated_scoring_is_byte_identical(policy: str) -> None:
    scores = [0.3, 0.9, 0.1, 0.8, 0.5]
    first = keep_good_job_score(scores, policy=policy, drop_lowest_n=2, threshold=0.5)
    second = keep_good_job_score(scores, policy=policy, drop_lowest_n=2, threshold=0.5)
    assert floats_bit_identical(first, second)


def test_keep_policy_path_imports_no_rng() -> None:
    from agent_challenge.evaluation.own_runner import keep_policy

    source = inspect.getsource(keep_policy)
    assert "import random" not in source
    assert "random." not in source


# ---------------------------------------------------------------------------
# Config surface -- keep-policy knobs (default off) + fail-closed validation
# ---------------------------------------------------------------------------


def test_default_config_keep_policy_off() -> None:
    from agent_challenge.sdk.config import ChallengeSettings

    settings = ChallengeSettings()
    assert settings.keep_good_tasks_policy == "off"
    assert settings.keep_good_tasks_drop_lowest == 0
    assert settings.keep_good_tasks_threshold == 0.0


@pytest.mark.parametrize("policy", ["off", "best-of-k", "drop-lowest-n", "threshold-band"])
def test_config_accepts_every_supported_policy(policy: str) -> None:
    from agent_challenge.sdk.config import ChallengeSettings

    assert ChallengeSettings(keep_good_tasks_policy=policy).keep_good_tasks_policy == policy


def test_config_normalizes_policy_case() -> None:
    from agent_challenge.sdk.config import ChallengeSettings

    assert ChallengeSettings(keep_good_tasks_policy=" Drop-Lowest-N ").keep_good_tasks_policy == (
        "drop-lowest-n"
    )


def test_config_rejects_unknown_policy() -> None:
    from pydantic import ValidationError

    from agent_challenge.sdk.config import ChallengeSettings

    with pytest.raises(ValidationError) as error:
        ChallengeSettings(keep_good_tasks_policy="keep-everything")
    assert "keep_good_tasks_policy" in str(error.value)


def test_config_rejects_negative_drop_lowest() -> None:
    from pydantic import ValidationError

    from agent_challenge.sdk.config import ChallengeSettings

    with pytest.raises(ValidationError):
        ChallengeSettings(keep_good_tasks_drop_lowest=-1)


def test_config_rejects_out_of_range_threshold() -> None:
    from pydantic import ValidationError

    from agent_challenge.sdk.config import ChallengeSettings

    with pytest.raises(ValidationError):
        ChallengeSettings(keep_good_tasks_threshold=-0.1)
    with pytest.raises(ValidationError):
        ChallengeSettings(keep_good_tasks_threshold=1.5)


def test_config_modes_stay_in_sync_with_keep_policy_module() -> None:
    # Drift guard: the settings validator duplicates the accepted set to avoid a
    # heavy import; this pins it to the keep_policy module's source of truth.
    assert set(KEEP_POLICY_MODES) == {"off", "best-of-k", "drop-lowest-n", "threshold-band"}
