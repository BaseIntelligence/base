"""Plausibility gate for worker-reported results (architecture.md 3.5).

The verify-only validator pipeline rejects a worker-reported run whose ``prism_run_manifest.v2``
is implausible BEFORE it is ever scored, routing it to the reject/retry path. A plausible manifest
passes through UNCHANGED (this module only READS the manifest; it never mutates the payload or the
score inputs), so a genuine result finalizes to the exact same score as the legacy path.

The checks, in order (architecture.md 3.5), each firing only when the manifest carries the data it
needs so a sparse-but-honest manifest is never spuriously rejected:

* **schema/version** -- ``schema_version`` must be ``prism_run_manifest.v2`` and ``metrics`` must be
  a well-formed object;
* **step-0 anomaly** -- an initial loss impossibly below the from-scratch baseline (< the existing
  ``STEP0_ANOMALY_FRACTION`` of ``~ln(vocab)`` nats) is the smuggled-pretrained-weights signal;
* **loss-trajectory sanity** -- a final/late online loss still sitting at (or above) the random-init
  baseline means no learning happened;
* **log/score consistency** -- the logged online-loss metrics must agree with the derived score
  block (a self-inconsistent manifest is fabricated);
* **wall-clock vs declared budget** -- a declared wall-clock grossly beyond the budget (or negative)
  is implausible.

Rejection reasons are all namespaced ``plausibility_*`` so they are DISTINCT from the proof-
verification reasons in :mod:`prism_challenge.ingestion` (VAL-PRISM-018 requires the two rejection
classes be distinguishable).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

#: Only the challenge-authored v2 re-execution manifest is scored (schemas.RUN_MANIFEST_V2_...).
MANIFEST_SCHEMA_VERSION = "prism_run_manifest.v2"

# Kept in sync with ``evaluator/container.py``'s ``STEP0_ANOMALY_FRACTION``: the step-0 forced-init
# loss must sit near the from-scratch baseline (~ln(vocab) nats); an initial loss below this
# fraction of the baseline is the smuggled-pretrained-weights anomaly.
STEP0_ANOMALY_FRACTION = 0.5
# A learner's late loss must fall meaningfully below the from-scratch baseline; a late loss at or
# above this fraction of the baseline means no learning happened (architecture.md 3.5).
NO_LEARNING_FRACTION = 0.98
# A declared wall-clock beyond this multiple of the run's wall-clock budget is grossly inconsistent
# (the run would have been stopped at the hard cap long before).
WALL_CLOCK_BUDGET_MULTIPLIER = 2.0
# Relative tolerance for the logged-vs-derived loss consistency check.
LOSS_CONSISTENCY_REL_TOL = 1e-3
LOSS_CONSISTENCY_ABS_TOL = 1e-6

REASON_SCHEMA_VERSION = "plausibility_schema_version"
REASON_METRICS_MALFORMED = "plausibility_metrics_malformed"
REASON_STEP0_ANOMALY = "plausibility_step0_anomaly"
REASON_LOSS_AT_BASELINE = "plausibility_loss_at_baseline"
REASON_LOSS_INCONSISTENT = "plausibility_loss_inconsistent"
REASON_WALLCLOCK_BUDGET = "plausibility_wallclock_budget"


class PlausibilityError(Exception):
    """A worker-reported result is implausible and must not be scored.

    ``reason`` is a stable machine code namespaced ``plausibility_*`` so a plausibility rejection is
    always distinguishable from a proof-verification rejection (VAL-PRISM-009/018).
    """

    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


def check_manifest_plausibility(
    manifest: Mapping[str, Any],
    *,
    wall_clock_budget_seconds: float | None = None,
) -> None:
    """Raise :class:`PlausibilityError` if ``manifest`` is implausible; return ``None`` otherwise.

    Read-only: the manifest is never mutated, so a plausible result passes through unchanged. Each
    numeric check runs only when its inputs are present, so a manifest that simply omits a
    trajectory/compute field is not rejected for the absence alone (only the schema/version check is
    unconditional).
    """

    if not isinstance(manifest, Mapping):
        raise PlausibilityError(REASON_METRICS_MALFORMED, "manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PlausibilityError(
            REASON_SCHEMA_VERSION,
            f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION!r}",
        )
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise PlausibilityError(REASON_METRICS_MALFORMED, "manifest metrics block is malformed")

    anti_cheat = manifest.get("anti_cheat")
    anti_cheat = anti_cheat if isinstance(anti_cheat, Mapping) else {}
    baseline = _opt_float(metrics.get("random_init_baseline_nats"))
    online_loss = _loss_series(metrics.get("online_loss"))
    step0 = _opt_float(metrics.get("step0_loss"))
    if step0 is None and online_loss:
        step0 = online_loss[0]

    _check_step0_anomaly(anti_cheat, baseline=baseline, step0=step0)
    _check_loss_trajectory(anti_cheat, baseline=baseline, online_loss=online_loss)
    _check_log_consistency(manifest, metrics, online_loss=online_loss, step0=step0)
    _check_wallclock(manifest, budget_seconds=wall_clock_budget_seconds)


def _check_step0_anomaly(
    anti_cheat: Mapping[str, Any], *, baseline: float | None, step0: float | None
) -> None:
    if anti_cheat.get("step0_anomaly") is True:
        raise PlausibilityError(REASON_STEP0_ANOMALY, "manifest flags a step-0 anomaly")
    if baseline is not None and step0 is not None and step0 < STEP0_ANOMALY_FRACTION * baseline:
        raise PlausibilityError(
            REASON_STEP0_ANOMALY,
            "step-0 loss is impossibly below the from-scratch baseline",
        )


def _check_loss_trajectory(
    anti_cheat: Mapping[str, Any], *, baseline: float | None, online_loss: list[float]
) -> None:
    if anti_cheat.get("no_learning") is True or anti_cheat.get("zero_forward") is True:
        raise PlausibilityError(REASON_LOSS_AT_BASELINE, "manifest flags a no-learning run")
    if baseline is None or not online_loss:
        return
    late = _late_loss(online_loss)
    if late >= NO_LEARNING_FRACTION * baseline:
        raise PlausibilityError(
            REASON_LOSS_AT_BASELINE,
            "final/late online loss sits at or above the random-init baseline (no learning)",
        )


def _check_log_consistency(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    online_loss: list[float],
    step0: float | None,
) -> None:
    reported_step0 = _opt_float(metrics.get("step0_loss"))
    if reported_step0 is not None and online_loss and not _close(reported_step0, online_loss[0]):
        raise PlausibilityError(
            REASON_LOSS_INCONSISTENT,
            "reported step0_loss disagrees with the logged online-loss curve",
        )
    metrics_bpb = _opt_float(metrics.get("prequential_bpb"))
    score = manifest.get("score")
    score_bpb = _opt_float(score.get("prequential_bpb")) if isinstance(score, Mapping) else None
    if metrics_bpb is not None and score_bpb is not None and not _close(metrics_bpb, score_bpb):
        raise PlausibilityError(
            REASON_LOSS_INCONSISTENT,
            "score-block bits-per-byte disagrees with the logged metrics",
        )


def _check_wallclock(manifest: Mapping[str, Any], *, budget_seconds: float | None) -> None:
    compute = manifest.get("compute")
    wall_clock = (
        _opt_float(compute.get("wall_clock_seconds")) if isinstance(compute, Mapping) else None
    )
    if wall_clock is None:
        return
    if wall_clock < 0.0:
        raise PlausibilityError(REASON_WALLCLOCK_BUDGET, "declared wall-clock is negative")
    if (
        budget_seconds is not None
        and budget_seconds > 0.0
        and wall_clock > budget_seconds * WALL_CLOCK_BUDGET_MULTIPLIER
    ):
        raise PlausibilityError(
            REASON_WALLCLOCK_BUDGET,
            "declared wall-clock is grossly inconsistent with the declared budget",
        )


def _loss_series(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    series = [_opt_float(item) for item in value]
    return [item for item in series if item is not None]


def _late_loss(online_loss: list[float]) -> float:
    """The mean of the trailing quarter of the loss curve (the run's converged/"late" loss)."""

    tail = max(1, len(online_loss) // 4)
    window = online_loss[-tail:]
    return sum(window) / len(window)


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left, right, rel_tol=LOSS_CONSISTENCY_REL_TOL, abs_tol=LOSS_CONSISTENCY_ABS_TOL
    )


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "NO_LEARNING_FRACTION",
    "REASON_LOSS_AT_BASELINE",
    "REASON_LOSS_INCONSISTENT",
    "REASON_METRICS_MALFORMED",
    "REASON_SCHEMA_VERSION",
    "REASON_STEP0_ANOMALY",
    "REASON_WALLCLOCK_BUDGET",
    "STEP0_ANOMALY_FRACTION",
    "WALL_CLOCK_BUDGET_MULTIPLIER",
    "PlausibilityError",
    "check_manifest_plausibility",
]
