"""Independent, faithful reimplementation of harbor 0.13.1 reward computation.

This module is the A3 independent runner's reward parser + scorer (Task 9). It
reproduces — **byte-for-byte / ε=0** — exactly how stock ``harbor==0.13.1`` turns
the verifier's ``/logs/verifier/reward.txt`` (or ``reward.json``, which takes
precedence) into a reward, aggregates it across metrics and trials (pass@k), and
how agent-challenge maps that into ``{status, score, resolved, total,
reason_code}``.

Authoritative spec: ``/droid/platform-v10/challenge OpenAPI / package evaluation docs`` (GATE G2,
PASS). Ground-truth fixtures executed against the real harbor wheel:
``.omo/evidence/task-2-reward-{semantics,edgecases}.txt``. Source mirrored:
``harbor/verifier/verifier.py``, ``harbor/metrics/{base,mean,max,min,sum}.py``,
``harbor/utils/pass_at_k.py``, ``harbor/models/job/result.py``, and the live
agent-challenge mapping at ``runner.py:1399-1438``.

=============================================================================
REWARD PRECISION SPEC (ε=0) — FINAL. See ``.omo/evidence/task-9-precision-spec.txt``.
=============================================================================

The parity target between this runner and stock harbor is **exact equality on
the IEEE-754 double**, NOT an epsilon tolerance. This is achievable and correct
because every arithmetic step here is identical to harbor's:

1.  Reward read uses CPython ``float(text)`` (builtin). ``0`` / ``1`` are exactly
    representable; general floats follow IEEE-754 round-to-nearest and are
    deterministic for a given input string.
2.  ``Mean`` uses ``sum(values) / len(values)`` — Python ``sum`` over a list is
    **left-to-right sequential** float addition, then one ``/``. We MUST NOT use
    ``math.fsum``, NumPy pairwise summation, or ``Decimal``: any of those would
    diverge from CPython in the last ULP and break ε=0 parity.
3.  **Trial order is preserved.** The caller's reward list order is the ``sum``
    operand order (and the pass@k task iteration order). Reordering trials would
    change the last-ULP result. Never sort the reward list.
4.  ``Max`` / ``Min`` / ``Sum`` are exact (no division).
5.  pass@k uses sequential ``product *= (n-c-i)/(n-i)`` then ``sum(...)/len`` —
    deterministic, ULP-sensitive to operand order, reproduced exactly.

FP-nondeterminism handling: ``nan`` propagates through ``sum``/``/`` to a ``nan``
metric, and ``nan == nan`` is always ``False`` in IEEE-754. Therefore the parity
comparator (:func:`reward_values_equal` / :func:`reward_parity_equal`) **special
-cases nan to compare equal to nan**, and otherwise asserts exact ``==`` on the
double. ``inf`` propagates to ``inf``/``nan`` and compares by exact value. A
strict bit-level diagnostic is available via :func:`floats_bit_identical`.

PARITY-DIFF HOOK: ``tools/parity_diff.py`` (Task 4) should import
:func:`reward_parity_equal` (exposed as :data:`PARITY_COMPARATOR`) as the
canonical reward-equality predicate. Do not reimplement comparison there.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NumericReward = float | int
RewardDict = dict[str, NumericReward]

# Standard verifier output filenames (the container's /logs/verifier/).
REWARD_TEXT_FILENAME = "reward.txt"
REWARD_JSON_FILENAME = "reward.json"


# ===========================================================================
# 1. Reward file parsing — mirrors harbor/verifier/verifier.py
# ===========================================================================
#
# harbor raises three distinct exceptions. agent-challenge classifies them into
# canonical reason codes (terminal_bench.py:55-68). NOTE: harbor's real message
# for the not-found case is "No reward file found at ..." which does NOT contain
# the substring "missing", so the message-substring normalizer would not map it.
# We therefore attach the canonical ``reason_code`` DIRECTLY to each exception
# (and on the convenience helpers) rather than relying on message matching, while
# still preserving harbor's exact message text for fidelity.


class RewardError(Exception):
    """Base class carrying the canonical agent-challenge reason code."""

    reason_code: str = "harbor_reward_parse_error"


class VerifierOutputParseError(RewardError):
    """Reward file present but could not be parsed (float/JSON failure)."""

    reason_code = "harbor_reward_parse_error"


class RewardFileNotFoundError(RewardError, FileNotFoundError):
    """No reward file (neither reward.json nor reward.txt) was found."""

    reason_code = "harbor_reward_missing"


class RewardFileEmptyError(RewardError):
    """Reward file exists but is zero bytes (st_size == 0)."""

    reason_code = "harbor_reward_empty"


def reason_code_for_error(error: BaseException) -> str | None:
    """Return the canonical reason code for a reward parsing error, else None."""

    if isinstance(error, RewardError):
        return error.reason_code
    return None


def parse_reward_text(reward_text_path: Path) -> RewardDict:
    """Parse ``reward.txt`` exactly as ``Verifier._parse_reward_text``.

    Emptiness is a **byte-size** check (``st_size == 0``), NOT ``.strip()``: a
    whitespace-only file is NOT empty and falls through to ``float("  ")`` →
    parse error. ``float()`` accepts surrounding whitespace, ``nan``, ``inf``,
    ``-1``, ``1e0``.
    """

    if reward_text_path.stat().st_size == 0:
        raise RewardFileEmptyError(f"Reward file is empty at {reward_text_path}")
    try:
        return {"reward": float(reward_text_path.read_text())}
    except (ValueError, TypeError) as error:
        raise VerifierOutputParseError(
            f"Failed to parse rewards from text file {reward_text_path}"
        ) from error


def parse_reward_json(reward_json_path: Path) -> RewardDict:
    """Parse ``reward.json`` exactly as ``Verifier._parse_reward_json``.

    The parsed object is returned **verbatim** — arbitrary metric keys are
    allowed (the multi-metric entry point). ``json.JSONDecodeError`` is a
    ``ValueError`` subclass, so malformed JSON becomes a parse error.
    """

    if reward_json_path.stat().st_size == 0:
        raise RewardFileEmptyError(f"Reward file is empty at {reward_json_path}")
    try:
        return json.loads(reward_json_path.read_text())
    except (ValueError, TypeError) as error:
        raise VerifierOutputParseError(
            f"Failed to parse rewards from JSON file {reward_json_path}"
        ) from error


def parse_reward_files(
    *,
    reward_text_path: Path,
    reward_json_path: Path,
) -> RewardDict:
    """Apply harbor's precedence: ``reward.json`` WINS over ``reward.txt``.

    Mirrors ``verifier.py:209-218`` — ``reward_json_path.exists()`` is checked
    first; only if absent is ``reward_text_path`` consulted; if neither exists a
    :class:`RewardFileNotFoundError` is raised.
    """

    if reward_json_path.exists():
        return parse_reward_json(reward_json_path)
    if reward_text_path.exists():
        return parse_reward_text(reward_text_path)
    raise RewardFileNotFoundError(
        f"No reward file found at {reward_text_path} or {reward_json_path}"
    )


def parse_verifier_dir(verifier_dir: Path) -> RewardDict:
    """Parse the standard ``reward.json``/``reward.txt`` under a verifier dir."""

    return parse_reward_files(
        reward_text_path=verifier_dir / REWARD_TEXT_FILENAME,
        reward_json_path=verifier_dir / REWARD_JSON_FILENAME,
    )


# ===========================================================================
# 2. Metric aggregation — mirrors harbor/metrics/base.py:16-37
# ===========================================================================


def aggregate_reward_dicts(
    rewards: list[RewardDict | None],
    metric_name: str,
    aggregate: Callable[[list[NumericReward]], NumericReward],
) -> RewardDict:
    """Aggregate per-trial reward dicts. **The #1 interop gotcha lives here.**

    - ``<= 1`` distinct reward key (the tbench norm): output is keyed by the
      **metric name** (e.g. ``{"mean": x}``); the reward key is discarded. A
      ``None`` trial contributes ``0``; an empty ``{}`` trial contributes ``0``
      via ``next(iter(...), 0)``.
    - ``> 1`` distinct reward keys (multi-metric via reward.json): output is
      keyed by the **reward keys** (metric name LOST); a trial missing a key
      contributes ``r.get(key, 0)`` = ``0``.

    Reward keys are ``sorted`` for deterministic multi-key output order.
    """

    reward_keys = sorted({key for reward in rewards if reward is not None for key in reward})

    if len(reward_keys) <= 1:
        values = [0 if reward is None else next(iter(reward.values()), 0) for reward in rewards]
        return {metric_name: aggregate(values)}

    return {
        key: aggregate([0 if reward is None else reward.get(key, 0) for reward in rewards])
        for key in reward_keys
    }


class BaseMetric:
    """Base metric. Subclasses define ``metric_name`` and ``aggregate``."""

    metric_name: str

    @staticmethod
    def aggregate(values: list[NumericReward]) -> NumericReward:  # pragma: no cover - abstract
        raise NotImplementedError

    def compute(self, rewards: list[RewardDict | None]) -> RewardDict:
        return aggregate_reward_dicts(rewards, self.metric_name, self.aggregate)


class Mean(BaseMetric):
    metric_name = "mean"

    @staticmethod
    def aggregate(values: list[NumericReward]) -> float:
        # CPython left-to-right list ``sum`` then a single ``/`` — ε=0 critical.
        return sum(values) / len(values)


class Max(BaseMetric):
    metric_name = "max"

    @staticmethod
    def aggregate(values: list[NumericReward]) -> NumericReward:
        return max(values)


class Min(BaseMetric):
    metric_name = "min"

    @staticmethod
    def aggregate(values: list[NumericReward]) -> NumericReward:
        return min(values)


class Sum(BaseMetric):
    metric_name = "sum"

    @staticmethod
    def aggregate(values: list[NumericReward]) -> NumericReward:
        return sum(values)


def default_metrics() -> list[BaseMetric]:
    """Default per-dataset metric list: exactly ``[Mean()]`` (job.py:456-458)."""

    return [Mean()]


def compute_metrics(
    rewards: list[RewardDict | None],
    metrics: list[BaseMetric] | None = None,
) -> list[RewardDict]:
    """Compute the ``evals[key].metrics`` list (one dict per configured metric)."""

    chosen = default_metrics() if metrics is None else metrics
    return [metric.compute(rewards) for metric in chosen]


# ===========================================================================
# 3. pass@k — mirrors harbor/utils/pass_at_k.py
# ===========================================================================


@dataclass(frozen=True)
class Trial:
    """Minimal per-trial view needed for pass@k grouping.

    Flattens the harbor ``TrialResult`` fields the algorithm reads:
    ``verifier_result.rewards``, ``task_name``, ``agent_info.name``,
    ``agent_info.model_info.name`` and ``source``.
    """

    task_name: str
    rewards: Mapping[str, NumericReward] | None
    agent_name: str = "agent"
    model_name: str | None = None
    source: str | None = None
    errored: bool = False


def format_agent_evals_key(agent_name: str, model_name: str | None, dataset_name: str) -> str:
    """``{agent}__{model}__{dataset}`` if model set, else ``{agent}__{dataset}``."""

    if model_name:
        return f"{agent_name}__{model_name}__{dataset_name}"
    return f"{agent_name}__{dataset_name}"


def pass_at_k_for_task(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator ``1 - C(n-c,k)/C(n,k)`` (1.0 when ``n-c < k``)."""

    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


def eligible_k_values(max_k: int) -> list[int]:
    """k ∈ (powers-of-2 ≥ 2) ∪ (multiples-of-5 ≥ 5), each ``<= max_k``, sorted.

    pass@1 is NEVER computed (k starts at 2).
    """

    k_values: set[int] = set()

    k = 2
    while k <= max_k:
        k_values.add(k)
        k *= 2

    k = 5
    while k <= max_k:
        k_values.add(k)
        k += 5

    return sorted(k_values)


def compute_pass_at_k_for_trials(trial_results: list[Trial]) -> dict[int, float]:
    """pass@k for one evals group. Returns ``{}`` if pass@k is disabled.

    pass@k is emitted only when EVERY trial has exactly one reward key valued
    strictly ``0`` or ``1``. A ``None`` reward counts as a failure (success 0)
    but does NOT disable pass@k. Multi-metric (``len != 1``), non-numeric, or
    fractional/other values disable pass@k for the whole group (``{}``).
    """

    task_successes: defaultdict[str, list[int]] = defaultdict(list)

    for trial_result in trial_results:
        rewards = trial_result.rewards
        if rewards is None:
            task_successes[trial_result.task_name].append(0)
            continue
        if len(rewards) != 1:
            return {}
        reward_value = next(iter(rewards.values()))
        if not isinstance(reward_value, int | float):
            return {}
        if reward_value not in (0, 1):
            return {}
        task_successes[trial_result.task_name].append(int(reward_value))

    if not task_successes:
        return {}

    min_trials_per_task = min(len(successes) for successes in task_successes.values())
    k_values = eligible_k_values(min_trials_per_task)

    return {
        k: sum(
            pass_at_k_for_task(len(successes), sum(successes), k)
            for successes in task_successes.values()
        )
        / len(task_successes)
        for k in k_values
    }


def compute_pass_at_k_by_evals(trial_results: list[Trial]) -> dict[str, dict[int, float]]:
    """Group trials by evals key and compute pass@k, dropping empty groups."""

    eval_groups: defaultdict[str, list[Trial]] = defaultdict(list)
    for trial_result in trial_results:
        evals_key = format_agent_evals_key(
            trial_result.agent_name,
            trial_result.model_name,
            trial_result.source or "adhoc",
        )
        eval_groups[evals_key].append(trial_result)

    return {
        evals_key: pass_at_k
        for evals_key, trials in eval_groups.items()
        if (pass_at_k := compute_pass_at_k_for_trials(trials))
    }


# ===========================================================================
# 4. Outcome mapping — mirrors agent-challenge runner.py:1399-1438
# ===========================================================================
#
# score = flat mean of metric values gathered across ALL evals groups and ALL
# metric entries: a dict with a "mean" key pushes float(metric["mean"]); else
# every value is pushed (multi-metric values become SEPARATE samples). status is
# driven by error count, not score. resolved = round(score * total) with Python
# banker's rounding. total = n_total_trials or n_completed + n_errored.

_MISSING_SUMMARY: dict[str, object] = {
    "status": "failed",
    "score": 0.0,
    "resolved": 0,
    "total": 0,
    "reason_code": "harbor_result_missing",
}
_MALFORMED_SUMMARY: dict[str, object] = {
    "status": "failed",
    "score": 0.0,
    "resolved": 0,
    "total": 0,
    "reason_code": "harbor_result_malformed",
}


def _flat_score(metric_dicts: Iterable[Mapping[str, Any]]) -> float:
    metric_values: list[float] = []
    for metric in metric_dicts:
        if "mean" in metric:
            metric_values.append(float(metric["mean"]))
        else:
            metric_values.extend(float(value) for value in metric.values())
    if metric_values:
        return sum(metric_values) / len(metric_values)
    return 0.0


def _build_summary(
    metric_dicts: Iterable[Mapping[str, Any]],
    *,
    n_total_trials: int,
    n_completed_trials: int,
    n_errored_trials: int,
) -> dict[str, object]:
    score = _flat_score(metric_dicts)
    return {
        "status": "completed" if n_errored_trials == 0 else "failed",
        "score": score,
        "resolved": round(score * n_total_trials),
        "total": n_total_trials or n_completed_trials + n_errored_trials,
        "reason_code": None,
    }


def derive_outcome_from_metrics(
    evals_metrics: Mapping[str, list[Mapping[str, Any]]],
    *,
    n_total_trials: int,
    n_completed_trials: int,
    n_errored_trials: int,
) -> dict[str, object]:
    """Map already-computed ``evals[key].metrics`` lists into the outcome dict.

    In-memory entry point for the verifier (Task 14) and full-parity (Task 22)
    consumers that hold metrics directly rather than a result JSON file.
    """

    metric_dicts = [metric for metrics in evals_metrics.values() for metric in metrics]
    return _build_summary(
        metric_dicts,
        n_total_trials=n_total_trials,
        n_completed_trials=n_completed_trials,
        n_errored_trials=n_errored_trials,
    )


def derive_outcome_from_result_data(data: object) -> dict[str, object]:
    """Map a parsed harbor job-result object into the outcome dict.

    Replicates the ``if result_path.exists():`` body of runner.py:1410-1436,
    including the ``except Exception`` → ``harbor_result_malformed`` guard.
    """

    try:
        stats = data.get("stats", {}) if isinstance(data, dict) else {}
        total = int(data.get("n_total_trials") or 0) if isinstance(data, dict) else 0
        completed = int(stats.get("n_completed_trials") or 0)
        errored = int(stats.get("n_errored_trials") or 0)
        evals = stats.get("evals", {})
        metric_dicts: list[Mapping[str, Any]] = []
        for eval_stats in evals.values():
            for metric in eval_stats.get("metrics", []):
                metric_dicts.append(metric)
        return _build_summary(
            metric_dicts,
            n_total_trials=total,
            n_completed_trials=completed,
            n_errored_trials=errored,
        )
    except Exception:
        return dict(_MALFORMED_SUMMARY)


def derive_outcome(result_path: Path) -> dict[str, object]:
    """Read a harbor job-result JSON file and map it to the outcome dict.

    Full replica of runner.py:1399-1438: absent file →
    ``harbor_result_missing``; unreadable / malformed JSON or aggregation error
    → ``harbor_result_malformed``; otherwise the mapped summary.
    """

    if not result_path.exists():
        return dict(_MISSING_SUMMARY)
    try:
        data = json.loads(result_path.read_text())
    except Exception:
        return dict(_MALFORMED_SUMMARY)
    return derive_outcome_from_result_data(data)


# ===========================================================================
# 5. Reward precision comparators (ε=0). PARITY-DIFF HOOK for Task 4.
# ===========================================================================


def reward_values_equal(left: object, right: object) -> bool:
    """ε=0 exact equality for a single reward/score value, nan-aware.

    ``nan`` compares equal to ``nan`` (IEEE-754 would say False); every other
    value compares by exact ``==``. ``inf`` compares by value; ``1`` and ``1.0``
    are equal (harbor reward values may be int or float).
    """

    left_nan = isinstance(left, float) and left != left
    right_nan = isinstance(right, float) and right != right
    if left_nan or right_nan:
        return left_nan and right_nan
    return left == right


def reward_parity_equal(left: object, right: object) -> bool:
    """Recursive ε=0 parity comparison over reward dicts / lists / scalars.

    This is the canonical comparator the parity tool imports. Mappings must have
    identical key sets; sequences identical length; floats/ints compared via
    :func:`reward_values_equal` (nan-aware, exact).
    """

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(reward_parity_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(reward_parity_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return reward_values_equal(left, right)
    return left == right


def floats_bit_identical(left: object, right: object) -> bool:
    """Strictest diagnostic: identical IEEE-754 64-bit pattern (nan-aware)."""

    left_value = float(left)  # type: ignore[arg-type]
    right_value = float(right)  # type: ignore[arg-type]
    if left_value != left_value or right_value != right_value:
        return left_value != left_value and right_value != right_value
    return struct.pack(">d", left_value) == struct.pack(">d", right_value)


# The reward-equality predicate the Task 4 parity tool should import:
#   from agent_challenge.evaluation.own_runner.reward import PARITY_COMPARATOR
PARITY_COMPARATOR: Callable[[object, object], bool] = reward_parity_equal
