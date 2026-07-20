"""Scorecard suite helpers for multimetric.v1.1 long-ctx, sample-eff, efficiency.

Challenge-owned metric constructors for Official Comparison scorecard annex
(``scorecard_id=multimetric.v1.1``). CPU unit fixtures + GPU-ready hooks.

This module **does not** sole-rank on efficiency over scientific axes: peak VRAM,
tokens/s, params, wall-clock, and 6ND FLOPs are Pareto / diagnostic only. Polar
conflict and long-ctx floors remain owned by :mod:`official_comparison`.

Assertions: VAL-SCORE-005 (long-ctx), VAL-SCORE-006 (sample-eff), VAL-SCORE-007
(efficiency annex non-overriding).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from .official_comparison import (
    OFFICIAL_LONG_CTX_CHANCE,
    OFFICIAL_LONG_CTX_FLOOR,
    OFFICIAL_LONG_CTX_RELATIVE_FLOOR,
    OFFICIAL_WALL_CLOCK_NEVER_RANKS,
    OfficialScoreRecord,
)

# --- Protocol floors relative to chance (VAL-SCORE-005 / docs §14.4) ----------------
# Seed-scale absolute macro floor remains OFFICIAL_LONG_CTX_FLOOR (0.15). Relative
# floors use (acc - chance) / (1 - chance) and require ≥ LONG_CTX_RELATIVE_FLOOR
# on needle and MQAR when the suite is enabled for public claims language.
# Chance baselines + relative floor constants are owned by official_comparison so
# multi-seed recompute and per-seed suite check stay identical.
LONG_CTX_CHANCE: dict[str, float] = dict(OFFICIAL_LONG_CTX_CHANCE)
LONG_CTX_RELATIVE_FLOOR = OFFICIAL_LONG_CTX_RELATIVE_FLOOR
LONG_CTX_SUITE_TASKS: tuple[str, ...] = (
    "needle",
    "mqar",
    "induction_copy",
    "lag_nll",
)
# Default token marks for sample-efficiency curves under the 500k pin.
DEFAULT_SAMPLE_EFF_MARKS_TOKENS: tuple[int, ...] = (50_000, 100_000, 250_000, 500_000)
# Sample-efficiency quality transform: quality = 1 / (1 + bpb_at_mark). Higher better.
DeviceHint = Literal["cpu", "cuda", "auto", "fixture"]


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def relative_to_chance(accuracy: float, chance: float) -> float:
    """Map accuracy to relative-to-chance in [0, 1] (design §3.4).

    ``relative = (acc − chance) / (1 − chance)`` clamped to [0, 1]. When chance
    is ≥ 1 (degenerate), returns 0.0 so floors fail closed rather than explode.
    """
    acc = float(accuracy)
    ch = float(chance)
    if not math.isfinite(acc) or not math.isfinite(ch):
        return 0.0
    if ch >= 1.0:
        return 0.0
    return clamp01((acc - ch) / (1.0 - ch))


def documented_floors_relative_to_chance() -> dict[str, Any]:
    """Publish floor documentation for suite outputs and scorecard annex honesty."""
    return {
        "floors_relative_to_chance": True,
        "absolute_suite_mean_floor": OFFICIAL_LONG_CTX_FLOOR,
        "relative_floor": LONG_CTX_RELATIVE_FLOOR,
        "chance_baselines": dict(LONG_CTX_CHANCE),
        "relative_floor_tasks": ["needle", "mqar"],
        "tasks": list(LONG_CTX_SUITE_TASKS),
        "note": (
            "Seed-scale long-ctx floors: absolute suite mean ≥ "
            f"{OFFICIAL_LONG_CTX_FLOOR}; relative_to_chance ≥ "
            f"{LONG_CTX_RELATIVE_FLOOR} on needle and mqar when suite enabled."
        ),
    }


# --- Long-context suite (VAL-SCORE-005) -------------------------------------------


@dataclass(frozen=True)
class LongCtxTaskScore:
    """One task score in [0, 1] accuracy units plus chance bookkeeping."""

    task: str
    accuracy: float
    chance: float
    relative: float
    trials: int = 0
    detail: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "accuracy": self.accuracy,
            "chance": self.chance,
            "relative_to_chance": self.relative,
            "trials": self.trials,
            "detail": dict(self.detail) if self.detail is not None else None,
        }


@dataclass(frozen=True)
class LongCtxSuiteResult:
    """Normalized long-ctx fields for ``OfficialScoreRecord`` / annex vector."""

    enabled: bool
    needle: float | None
    mqar: float | None
    induction_copy: float | None
    lag_nll: float | None
    suite_mean: float | None
    floor_pass: bool | None
    relative: dict[str, float]
    chance: dict[str, float]
    absolute_floor: float = OFFICIAL_LONG_CTX_FLOOR
    relative_floor: float = LONG_CTX_RELATIVE_FLOOR
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "needle": self.needle,
            "mqar": self.mqar,
            "induction_copy": self.induction_copy,
            "lag_nll": self.lag_nll,
            "suite_mean": self.suite_mean,
            "floor_pass": self.floor_pass,
            "absolute_floor": self.absolute_floor,
            "relative_floor": self.relative_floor,
            "relative_to_chance": dict(self.relative),
            "chance": dict(self.chance),
            "floors_relative_to_chance": True,
            "device": self.device,
            "notes": list(self.notes),
        }

    def task_scores(self) -> list[float]:
        vals: list[float] = []
        for v in (self.needle, self.mqar, self.induction_copy):
            if v is not None and math.isfinite(v):
                vals.append(float(v))
        return vals


def score_closed_choice_accuracy(
    correct: Sequence[bool] | Sequence[int] | Sequence[float],
    *,
    task: str,
    chance: float | None = None,
) -> LongCtxTaskScore:
    """Compute closed-choice accuracy for needle / MQAR / induction from trial outcomes."""
    if not correct:
        ch = float(LONG_CTX_CHANCE.get(task, chance if chance is not None else 0.0))
        return LongCtxTaskScore(task=task, accuracy=0.0, chance=ch, relative=0.0, trials=0)
    vals = [float(x) for x in correct]
    # Allow 0/1 floats or bools.
    acc = sum(1.0 if v >= 0.5 else 0.0 for v in vals) / len(vals)
    ch = float(chance if chance is not None else LONG_CTX_CHANCE.get(task, 0.0))
    return LongCtxTaskScore(
        task=task,
        accuracy=clamp01(acc),
        chance=ch,
        relative=relative_to_chance(acc, ch),
        trials=len(vals),
    )


def score_accuracy_value(
    accuracy: float,
    *,
    task: str,
    chance: float | None = None,
    trials: int = 0,
) -> LongCtxTaskScore:
    """Build a task score from a precomputed accuracy in [0, 1]."""
    ch = float(chance if chance is not None else LONG_CTX_CHANCE.get(task, 0.0))
    acc = clamp01(float(accuracy))
    return LongCtxTaskScore(
        task=task,
        accuracy=acc,
        chance=ch,
        relative=relative_to_chance(acc, ch),
        trials=int(trials),
    )


def lag_nll_from_bins(
    lag_nll_by_bin: Mapping[str | int, float] | Sequence[float],
    *,
    long_bin_keys: Sequence[str | int] = ("lag_ge_64", "lag_ge_256", 64, 256),
) -> float:
    """Report long-lag next-token NLL (nats proxy); lower is better, finite forced.

    Accepts either a mapping of lag-bin → NLL or a sequence (uses last third mean
    as the long-lag estimate). Missing / non-finite → ``math.inf`` so callers can
    mark suite incomplete rather than invent scores.
    """
    if isinstance(lag_nll_by_bin, Mapping):
        picks: list[float] = []
        for key in long_bin_keys:
            if key in lag_nll_by_bin:
                val = lag_nll_by_bin[key]
                if val is not None and math.isfinite(float(val)):
                    picks.append(float(val))
        if picks:
            return float(sum(picks) / len(picks))
        # Fallback: max lag key if numeric keys present.
        numeric = []
        for k, v in lag_nll_by_bin.items():
            try:
                numeric.append((float(k), float(v)))
            except (TypeError, ValueError):
                continue
        if not numeric:
            return float("inf")
        numeric.sort(key=lambda kv: kv[0])
        # Use the middle of the upper half.
        upper_pairs = numeric[len(numeric) // 2 :]
        vals = [v for _, v in upper_pairs if math.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else float("inf")
    seq = [float(x) for x in lag_nll_by_bin]
    if not seq:
        return float("inf")
    cut = max(1, len(seq) // 3)
    upper_seq = seq[-cut:]
    finite = [v for v in upper_seq if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("inf")


def aggregate_long_ctx_suite(
    *,
    needle: LongCtxTaskScore | float | None = None,
    mqar: LongCtxTaskScore | float | None = None,
    induction_copy: LongCtxTaskScore | float | None = None,
    lag_nll: float | None = None,
    enabled: bool = True,
    device: str = "fixture",
    absolute_floor: float = OFFICIAL_LONG_CTX_FLOOR,
    relative_floor: float = LONG_CTX_RELATIVE_FLOOR,
) -> LongCtxSuiteResult:
    """Aggregate needle / MQAR / induction-copy / optional lag-NLL into suite fields.

    ``suite_mean`` is the macro-mean of the **normalized accuracy** tasks
    (needle, mqar, induction_copy). ``lag_nll`` is attached as a supporting
    primary field (lower better) and does not enter the [0,1] macro mean.
    """
    notes: list[str] = []

    def _norm(task: str, raw: LongCtxTaskScore | float | None) -> LongCtxTaskScore | None:
        if raw is None:
            return None
        if isinstance(raw, LongCtxTaskScore):
            return raw
        return score_accuracy_value(float(raw), task=task)

    needle_s = _norm("needle", needle)
    mqar_s = _norm("mqar", mqar)
    ind_s = _norm("induction_copy", induction_copy)

    acc_vals: list[float] = []
    relative: dict[str, float] = {}
    chance: dict[str, float] = {}
    for sc in (needle_s, mqar_s, ind_s):
        if sc is None:
            continue
        acc_vals.append(sc.accuracy)
        relative[sc.task] = sc.relative
        chance[sc.task] = sc.chance

    suite_mean: float | None
    if enabled and acc_vals:
        suite_mean = float(sum(acc_vals) / len(acc_vals))
    elif enabled and not acc_vals:
        suite_mean = None
        notes.append("suite_enabled_but_no_accuracy_tasks")
    else:
        suite_mean = None

    floor_pass: bool | None = None
    if enabled and suite_mean is not None and math.isfinite(suite_mean):
        abs_ok = float(suite_mean) >= float(absolute_floor)
        # Relative floor required on needle + mqar when those tasks are present.
        rel_ok = True
        for task in ("needle", "mqar"):
            if task in relative:
                if float(relative[task]) < float(relative_floor):
                    rel_ok = False
                    notes.append(f"relative_floor_fail:{task}")
        floor_pass = bool(abs_ok and rel_ok)
        if not abs_ok:
            notes.append("absolute_floor_fail")

    lag_val: float | None = None
    if lag_nll is not None and math.isfinite(float(lag_nll)):
        lag_val = float(lag_nll)
    elif lag_nll is not None:
        notes.append("lag_nll_non_finite")

    return LongCtxSuiteResult(
        enabled=bool(enabled),
        needle=None if needle_s is None else needle_s.accuracy,
        mqar=None if mqar_s is None else mqar_s.accuracy,
        induction_copy=None if ind_s is None else ind_s.accuracy,
        lag_nll=lag_val,
        suite_mean=suite_mean,
        floor_pass=floor_pass,
        relative=relative,
        chance=chance,
        absolute_floor=float(absolute_floor),
        relative_floor=float(relative_floor),
        device=device,
        notes=tuple(notes),
    )


def apply_long_ctx_to_record(
    record: OfficialScoreRecord,
    suite: LongCtxSuiteResult,
) -> OfficialScoreRecord:
    """Stamp long-ctx suite fields onto an :class:`OfficialScoreRecord`."""
    return replace(
        record,
        long_ctx_enabled=bool(suite.enabled),
        long_ctx_score=suite.suite_mean,
        long_ctx_needle=suite.needle,
        long_ctx_mqar=suite.mqar,
        long_ctx_induction_copy=suite.induction_copy,
        lag_nll=suite.lag_nll,
        long_ctx_floor_pass=suite.floor_pass,
    )


# Tiny GPU-ready logits probe hooks (optional). CPU fixtures use the closed-choice path.


def probe_next_token_accuracy(
    logits_fn: Callable[[Any], Any],
    contexts: Sequence[Any],
    targets: Sequence[int],
    *,
    candidate_sets: Sequence[Sequence[int]] | None = None,
) -> list[bool]:
    """GPU/CPU-ready accuracy probe over forced / open next-token predictions.

    ``logits_fn(ctx)`` must return a 1D logits vector (vocab). When ``candidate_sets``
    is provided, prediction is restricted to that set (closed choice). This is the
    hook paper-style needle/MQAR suites call on host or remote GPU eval.
    """
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets length mismatch")
    if candidate_sets is not None and len(candidate_sets) != len(targets):
        raise ValueError("candidate_sets length mismatch")
    outcomes: list[bool] = []
    for i, ctx in enumerate(contexts):
        logits = logits_fn(ctx)
        # Support torch tensors without requiring torch at import time.
        if hasattr(logits, "detach"):
            logits = logits.detach()
        if hasattr(logits, "cpu"):
            logits = logits.cpu()
        if hasattr(logits, "tolist"):
            raw = logits.tolist()
            # Flatten last-dim if nested.
            if raw and isinstance(raw[0], (list, tuple)):
                raw = raw[-1] if isinstance(raw[-1], (list, tuple)) else raw[0]
            scores = [float(x) for x in raw]
        else:
            scores = [float(x) for x in logits]
        target = int(targets[i])
        if candidate_sets is not None:
            cands = list(candidate_sets[i])
            if not cands:
                outcomes.append(False)
                continue
            best = max(cands, key=lambda c: scores[c] if 0 <= c < len(scores) else float("-inf"))
            outcomes.append(best == target)
        else:
            if not scores:
                outcomes.append(False)
                continue
            pred = max(range(len(scores)), key=lambda j: scores[j])
            outcomes.append(pred == target)
    return outcomes


def run_long_ctx_fixture_suite(
    *,
    needle_correct: Sequence[bool] | Sequence[float],
    mqar_correct: Sequence[bool] | Sequence[float],
    induction_correct: Sequence[bool] | Sequence[float],
    lag_nll_by_bin: Mapping[str | int, float] | Sequence[float] | None = None,
    device: str = "fixture",
) -> LongCtxSuiteResult:
    """CPU unit fixture path producing normalized numeric long-ctx fields."""
    needle = score_closed_choice_accuracy(needle_correct, task="needle")
    mqar = score_closed_choice_accuracy(mqar_correct, task="mqar")
    induction = score_closed_choice_accuracy(induction_correct, task="induction_copy")
    lag = None if lag_nll_by_bin is None else lag_nll_from_bins(lag_nll_by_bin)
    return aggregate_long_ctx_suite(
        needle=needle,
        mqar=mqar,
        induction_copy=induction,
        lag_nll=lag,
        enabled=True,
        device=device,
    )


# --- Sample-efficiency curve (VAL-SCORE-006) --------------------------------------


@dataclass(frozen=True)
class SampleEffResult:
    """Quality-vs-tokens marks + AUC from challenge-owned online streams."""

    marks_tokens: tuple[int, ...]
    bpb_at_marks: tuple[float, ...]
    quality_at_marks: tuple[float, ...]
    auc: float
    covered_bytes_total: float
    method: str = "online_stream_trapezoid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "marks_tokens": list(self.marks_tokens),
            "bpb_at_marks": list(self.bpb_at_marks),
            "quality_at_marks": list(self.quality_at_marks),
            "auc": self.auc,
            "covered_bytes_total": self.covered_bytes_total,
            "method": self.method,
        }


def _bpb_at_fraction(
    online_loss: Sequence[float],
    covered_bytes_cumulative: Sequence[float] | None,
    *,
    frac: float,
) -> float:
    """Estimate prequential-style bpb from the stream prefix at progress ``frac`` ∈ (0,1]."""
    if not online_loss:
        return float("inf")
    n = len(online_loss)
    k = max(1, min(n, int(math.ceil(float(frac) * n))))
    losses = [float(x) for x in online_loss[:k]]
    # online_loss is typically nats/token (or nats/byte proxy). Convert nats→bits.
    mean_nats = sum(losses) / len(losses)
    bits = mean_nats / math.log(2.0)
    if covered_bytes_cumulative is not None and len(covered_bytes_cumulative) >= k:
        # Prefer true bit density when cumulative bytes progressive: integral proxy
        # remains mean_nats/ln2 (already length-normalized per sample).
        _ = covered_bytes_cumulative[k - 1]
    return float(bits)


def quality_from_bpb(bpb: float) -> float:
    """Monotone quality transform: higher better, finite bpb required."""
    if not math.isfinite(bpb) or bpb < 0:
        return 0.0
    return 1.0 / (1.0 + float(bpb))


def sample_efficiency_from_stream(
    online_loss: Sequence[float],
    *,
    covered_bytes_cumulative: Sequence[float] | None = None,
    token_budget: int = 500_000,
    marks_tokens: Sequence[int] = DEFAULT_SAMPLE_EFF_MARKS_TOKENS,
) -> SampleEffResult:
    """Compute sample-efficiency marks + AUC from a challenge online-loss stream.

    For each token mark m ≤ token_budget, estimate bpb at progress m/token_budget
    (stream fraction). Quality = ``1/(1+bpb)``. AUC is the trapezoidal integral of
    quality over mark fractions in [0, 1] (marks normalized by token_budget).
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    marks = tuple(int(m) for m in marks_tokens if int(m) > 0)
    if not marks:
        raise ValueError("marks_tokens must contain at least one positive mark")
    bpb_marks: list[float] = []
    q_marks: list[float] = []
    fracs: list[float] = []
    for mark in marks:
        frac = min(1.0, float(mark) / float(token_budget))
        bpb = _bpb_at_fraction(online_loss, covered_bytes_cumulative, frac=frac)
        bpb_marks.append(bpb)
        q_marks.append(quality_from_bpb(bpb))
        fracs.append(frac)
    # Trapezoidal / piecewise-constant AUC of quality vs token-progress in [0, 1].
    # From 0→first mark: flat quality_at_first (no invented early curve). Then
    # trapezoid between successive marks. No double-count across segments.
    auc = 0.0
    if fracs:
        auc += q_marks[0] * fracs[0]
        for i in range(1, len(fracs)):
            dx = fracs[i] - fracs[i - 1]
            if dx <= 0:
                continue
            auc += 0.5 * (q_marks[i] + q_marks[i - 1]) * dx
    covered_total = 0.0
    if covered_bytes_cumulative:
        covered_total = float(covered_bytes_cumulative[-1])
    return SampleEffResult(
        marks_tokens=marks,
        bpb_at_marks=tuple(bpb_marks),
        quality_at_marks=tuple(q_marks),
        auc=float(auc),
        covered_bytes_total=covered_total,
    )


def sample_efficiency_from_manifest(
    metrics: Mapping[str, Any],
    *,
    token_budget: int = 500_000,
    marks_tokens: Sequence[int] = DEFAULT_SAMPLE_EFF_MARKS_TOKENS,
) -> SampleEffResult:
    """Extract stream fields from a challenge metrics / manifest payload."""
    online = metrics.get("online_loss") or metrics.get("online_losses") or []
    if not isinstance(online, (list, tuple)):
        raise TypeError("metrics.online_loss must be a list of floats")
    cum = metrics.get("covered_bytes_cumulative")
    if cum is not None and not isinstance(cum, (list, tuple)):
        raise TypeError("metrics.covered_bytes_cumulative must be a list when present")
    return sample_efficiency_from_stream(
        [float(x) for x in online],
        covered_bytes_cumulative=(
            None if cum is None else [float(x) for x in cum]  # type: ignore[arg-type]
        ),
        token_budget=token_budget,
        marks_tokens=marks_tokens,
    )


def apply_sample_eff_to_record(
    record: OfficialScoreRecord,
    suite: SampleEffResult,
) -> OfficialScoreRecord:
    """Stamp sample-efficiency AUC + mark bpb vector onto a score record."""
    return replace(
        record,
        sample_eff_auc=float(suite.auc),
        sample_eff_marks=tuple(float(x) for x in suite.bpb_at_marks),
    )


# --- Efficiency annex (VAL-SCORE-007) ---------------------------------------------


@dataclass(frozen=True)
class EfficiencyAnnex:
    """Secondary / Pareto diagnostics. Never sole-ranks scientific axes."""

    params: int | None = None
    peak_vram_gib: float | None = None
    tokens_per_s: float | None = None
    wall_clock_seconds: float | None = None
    flops_6nd: float | None = None
    device: str = "cpu"
    wall_clock_never_ranks: bool = OFFICIAL_WALL_CLOCK_NEVER_RANKS
    sole_rank_forbidden: bool = True
    flops_diagnostic_only: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "peak_vram_gib": self.peak_vram_gib,
            "tokens_per_s": self.tokens_per_s,
            "wall_clock_seconds": self.wall_clock_seconds,
            "flops_6nd": self.flops_6nd,
            "device": self.device,
            "wall_clock_never_ranks": self.wall_clock_never_ranks,
            "sole_rank_forbidden": self.sole_rank_forbidden,
            "flops_diagnostic_only": self.flops_diagnostic_only,
            "overrides_scientific_axes": False,
            "overrides_polar_rule": False,
        }


def estimate_6nd_flops(params: int | None, tokens: int | None) -> float | None:
    """6ND diagnostic FLOPs note only (never sole rank; docs / design audit)."""
    if params is None or tokens is None:
        return None
    if params <= 0 or tokens <= 0:
        return None
    return float(6.0 * float(params) * float(tokens))


def measure_tokens_per_s(
    *,
    tokens_processed: int,
    wall_seconds: float,
) -> float | None:
    if wall_seconds <= 0 or tokens_processed <= 0:
        return None
    return float(tokens_processed) / float(wall_seconds)


def peak_vram_gib_available(
    *,
    device: DeviceHint = "auto",
    peak_allocator_bytes: int | float | None = None,
) -> tuple[float | None, str]:
    """Return (peak_vram_GiB, device_label).

    Prefer explicit ``peak_allocator_bytes`` (from a real GPU train/eval capture).
    When absent, probe torch CUDA max_memory_allocated if available; else null
    with honest device=cpu/fixture (do not invent VRAM).
    """
    if peak_allocator_bytes is not None and float(peak_allocator_bytes) > 0:
        gib = float(peak_allocator_bytes) / float(1024**3)
        label = "cuda" if device in ("cuda", "auto") else str(device)
        return gib, label
    if device in ("cpu", "fixture"):
        return None, device
    # Best-effort CUDA probe — GPU-ready hook; never invent numbers when empty.
    try:
        import torch

        if not torch.cuda.is_available():
            return None, "cpu"
        # max_memory_allocated is 0 until a real allocation peeks; still report honestly.
        peak_b = int(torch.cuda.max_memory_allocated())
        if peak_b <= 0:
            return None, "cuda"
        return float(peak_b) / float(1024**3), "cuda"
    except Exception:
        return None, "cpu"


def build_efficiency_annex(
    *,
    params: int | None = None,
    peak_vram_gib: float | None = None,
    tokens_per_s: float | None = None,
    wall_clock_seconds: float | None = None,
    tokens_processed: int | None = None,
    peak_allocator_bytes: int | float | None = None,
    device: DeviceHint = "auto",
) -> EfficiencyAnnex:
    """Assemble efficiency annex; null fields stay null (no invention)."""
    vram = peak_vram_gib
    device_label: str = device
    if vram is None:
        vram, device_label = peak_vram_gib_available(
            device=device, peak_allocator_bytes=peak_allocator_bytes
        )
    tps = tokens_per_s
    if tps is None and tokens_processed is not None and wall_clock_seconds is not None:
        tps = measure_tokens_per_s(
            tokens_processed=int(tokens_processed),
            wall_seconds=float(wall_clock_seconds),
        )
    flops = estimate_6nd_flops(params, tokens_processed)
    return EfficiencyAnnex(
        params=None if params is None else int(params),
        peak_vram_gib=None if vram is None else float(vram),
        tokens_per_s=None if tps is None else float(tps),
        wall_clock_seconds=(None if wall_clock_seconds is None else float(wall_clock_seconds)),
        flops_6nd=flops,
        device=str(device_label),
    )


def apply_efficiency_to_record(
    record: OfficialScoreRecord,
    annex: EfficiencyAnnex,
) -> OfficialScoreRecord:
    """Stamp efficiency diagnostics onto a record without changing rank fields."""
    return replace(
        record,
        params=annex.params if annex.params is not None else record.params,
        peak_vram_gib=(
            annex.peak_vram_gib if annex.peak_vram_gib is not None else record.peak_vram_gib
        ),
        tokens_per_s=(
            annex.tokens_per_s if annex.tokens_per_s is not None else record.tokens_per_s
        ),
        wall_clock_seconds=(
            annex.wall_clock_seconds
            if annex.wall_clock_seconds is not None
            else record.wall_clock_seconds
        ),
    )


def timed_tokens_probe(
    step_fn: Callable[[], int],
    *,
    steps: int = 1,
) -> tuple[int, float]:
    """GPU/CPU-ready wall-clock probe: returns (tokens, wall_seconds)."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        total_tokens += int(step_fn())
    dt = time.perf_counter() - t0
    return total_tokens, float(dt)


# --- Combined enricher ------------------------------------------------------------


def enrich_record_with_suites(
    record: OfficialScoreRecord,
    *,
    long_ctx: LongCtxSuiteResult | None = None,
    sample_eff: SampleEffResult | None = None,
    efficiency: EfficiencyAnnex | None = None,
) -> OfficialScoreRecord:
    """Apply any combination of suite results onto one score record."""
    out = record
    if long_ctx is not None:
        out = apply_long_ctx_to_record(out, long_ctx)
    if sample_eff is not None:
        out = apply_sample_eff_to_record(out, sample_eff)
    if efficiency is not None:
        out = apply_efficiency_to_record(out, efficiency)
    return out


def long_ctx_suite_schema() -> dict[str, Any]:
    """JSON-schema-ish description for suite output honesty evidence."""
    return {
        "type": "object",
        "required": [
            "enabled",
            "needle",
            "mqar",
            "induction_copy",
            "suite_mean",
            "floors_relative_to_chance",
        ],
        "properties": {
            "enabled": {"type": "boolean"},
            "needle": {"type": ["number", "null"]},
            "mqar": {"type": ["number", "null"]},
            "induction_copy": {"type": ["number", "null"]},
            "lag_nll": {"type": ["number", "null"]},
            "suite_mean": {"type": ["number", "null"]},
            "floor_pass": {"type": ["boolean", "null"]},
            "floors_relative_to_chance": {"const": True},
            "chance": {"type": "object"},
            "relative_to_chance": {"type": "object"},
        },
        "floors": documented_floors_relative_to_chance(),
    }
