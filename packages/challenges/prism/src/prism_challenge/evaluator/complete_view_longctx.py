"""Complete View quality + long-context expansions (VAL-COMPLETE-002..007).

Architecture-agnostic suite builders for Complete View v1.2. Expands the v1.1
scorecard long-ctx base (needle / MQAR / fused induction_copy / lag_nll) with:

* multi-seed absolute ``val_bpb_trained`` (and optional ``medium_free_ce``)
* multi-T long-ctx suite means (256 / 512 / 1024 or max feasible)
* needle-by-depth + lost-in-middle mid-depth panel
* MQAR N×lag accuracy grid
* unfused induction vs exact-copy probes
* lag-NLL bins + train-short eval-long CE (length extrapolate, no retrain)

CPU unit fixtures produce fully filled fields from synthetic trial outcomes.
GPU/CPU-ready hooks accept logits callables / free-CE callables for host or remote
eval on reused K=3 ``trained_state`` weights.

This module does **not** invent scientific winners: missing suites stay null + reason.
Efficiency axes remain non-ranking (owned by later Complete View efficiency feature).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .complete_view import (
    COMPLETE_VIEW_PANEL_KEYS,
    build_complete_view,
)
from .official_comparison import OfficialScoreRecord
from .scorecard_suite import (
    LONG_CTX_CHANCE,
    LongCtxSuiteResult,
    LongCtxTaskScore,
    aggregate_long_ctx_suite,
    clamp01,
    lag_nll_from_bins,
    relative_to_chance,
    run_long_ctx_fixture_suite,
    score_accuracy_value,
    score_closed_choice_accuracy,
)

# Default multi-T ladder for seed-scale Complete View (or max feasible below).
DEFAULT_LONG_CTX_TS: tuple[int, ...] = (256, 512, 1024)
# Needle depth strata; "mid" is the lost-in-middle depth (≈0.5).
DEFAULT_NEEDLE_DEPTHS: tuple[float, ...] = (0.1, 0.5, 0.9)
LOST_IN_MIDDLE_DEPTH = 0.5
# MQAR closed-choice grid defaults.
DEFAULT_MQAR_NS: tuple[int, ...] = (4, 8, 16)
DEFAULT_MQAR_LAGS: tuple[int, ...] = (16, 64, 256)
DEFAULT_MQAR_CANDIDATES = 16
# Lag bins for free-text next-token NLL.
DEFAULT_LAG_BIN_KEYS: tuple[str, ...] = (
    "lag_16",
    "lag_64",
    "lag_ge_64",
    "lag_ge_256",
    "lag_ge_512",
)
# Length-extrapolation eval Ts (train short, eval long, no retrain).
DEFAULT_LENGTH_EXTRAP_TS: tuple[int, ...] = (128, 256, 512, 1024)

DeviceHint = Literal["cpu", "cuda", "auto", "fixture"]


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None, None
    mean = float(sum(clean) / len(clean))
    if len(clean) == 1:
        return mean, 0.0
    return mean, float(statistics.pstdev(clean))


# --- VAL-COMPLETE-002: absolute multi-seed val_bpb_trained -------------------------


@dataclass(frozen=True)
class MultiSeedValBpb:
    """Absolute free-CE val_bpb_trained multi-seed summary (lower better)."""

    mean: float
    std: float
    seeds: tuple[int, ...]
    per_seed: tuple[float, ...]
    form: str = "val_bpb_trained"
    device: str = "fixture"
    covered_bytes: int | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "seeds": list(self.seeds),
            "per_seed": list(self.per_seed),
            "form": self.form,
            "device": self.device,
            "covered_bytes": self.covered_bytes,
            "notes": list(self.notes),
            "status": "filled",
        }


def multi_seed_val_bpb_trained(
    per_seed_bpb: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    device: str = "fixture",
    covered_bytes: int | None = None,
    form: str = "val_bpb_trained",
) -> MultiSeedValBpb:
    """Aggregate absolute val_bpb_trained across public seeds (K≥1 required).

    Accepts either ``{seed: bpb}`` or a sequence of ``(seed, bpb)``. Non-finite
    entries are dropped; empty after clean → raises so callers do not invent
    zero scores for Complete View.
    """
    pairs: list[tuple[int, float]]
    if isinstance(per_seed_bpb, Mapping):
        pairs = [(int(k), float(v)) for k, v in per_seed_bpb.items()]
    else:
        pairs = [(int(s), float(v)) for s, v in per_seed_bpb]
    clean = [(s, v) for s, v in pairs if math.isfinite(v)]
    if not clean:
        raise ValueError("multi_seed_val_bpb_trained requires at least one finite bpb")
    clean.sort(key=lambda kv: kv[0])
    seeds = tuple(s for s, _ in clean)
    vals = tuple(v for _, v in clean)
    mean, std = _mean_std(vals)
    assert mean is not None and std is not None
    return MultiSeedValBpb(
        mean=mean,
        std=std,
        seeds=seeds,
        per_seed=vals,
        form=form,
        device=device,
        covered_bytes=covered_bytes,
    )


def medium_free_ce_by_T(
    ce_by_t: Mapping[int | str, float],
    *,
    device: str = "fixture",
) -> dict[str, Any]:
    """Medium free CE at T=256/512 (and any provided T). Lower better."""
    out: dict[str, float] = {}
    for key, val in ce_by_t.items():
        fv = _finite(val)
        if fv is None:
            continue
        out[str(int(key))] = fv
    return {
        "by_T": out,
        "device": device,
        "status": "filled" if out else "not_run",
        "reason": None if out else "medium_free_ce_not_provided",
    }


# --- VAL-COMPLETE-003: multi-T long-context suite ---------------------------------


@dataclass(frozen=True)
class LongCtxAtT:
    """Per-T long-ctx suite slice (extends v1.1 with unfused fields)."""

    t: int
    suite_mean: float | None
    needle: float | None
    mqar: float | None
    induction: float | None
    exact_copy: float | None
    induction_copy_fused: float | None
    lag_nll: float | None
    floor_pass: bool | None
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "suite_mean": self.suite_mean,
            "needle": self.needle,
            "mqar": self.mqar,
            "induction": self.induction,
            "exact_copy": self.exact_copy,
            "induction_copy_fused": self.induction_copy_fused,
            "lag_nll": self.lag_nll,
            "floor_pass": self.floor_pass,
            "device": self.device,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MultiTLongCtxResult:
    """Multi-T long-ctx matrix + aggregate (VAL-COMPLETE-003)."""

    by_T: dict[str, LongCtxAtT]
    aggregate_suite_mean: float | None
    max_feasible_t: int | None
    requested_ts: tuple[int, ...]
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_T": {k: v.as_dict() for k, v in self.by_T.items()},
            "aggregate_suite_mean": self.aggregate_suite_mean,
            "max_feasible_t": self.max_feasible_t,
            "requested_ts": list(self.requested_ts),
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
        }


def build_long_ctx_at_t(
    *,
    t: int,
    needle: float | LongCtxTaskScore | None,
    mqar: float | LongCtxTaskScore | None,
    induction: float | LongCtxTaskScore | None = None,
    exact_copy: float | LongCtxTaskScore | None = None,
    # Optional fused legacy field (mean of induction + exact_copy when both present).
    induction_copy_fused: float | LongCtxTaskScore | None = None,
    lag_nll: float | None = None,
    device: str = "fixture",
) -> LongCtxAtT:
    """Build one T-slice. Suite mean uses unfused tasks when present, else fused."""

    def _acc(raw: float | LongCtxTaskScore | None) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, LongCtxTaskScore):
            return float(raw.accuracy)
        return clamp01(float(raw))

    needle_a = _acc(needle)
    mqar_a = _acc(mqar)
    ind_a = _acc(induction)
    copy_a = _acc(exact_copy)
    fused_a = _acc(induction_copy_fused)
    if fused_a is None and ind_a is not None and copy_a is not None:
        fused_a = float((ind_a + copy_a) / 2.0)
    elif fused_a is None and ind_a is not None:
        fused_a = ind_a
    elif fused_a is None and copy_a is not None:
        fused_a = copy_a

    # Prefer unfused (needle, mqar, induction, exact_copy) for suite_mean when ≥2 present.
    unfused_vals = [v for v in (needle_a, mqar_a, ind_a, copy_a) if v is not None]
    if len(unfused_vals) >= 2:
        suite_mean = float(sum(unfused_vals) / len(unfused_vals))
    else:
        # Fall back to v1.1 aggregator (fused induction_copy).
        base = aggregate_long_ctx_suite(
            needle=needle_a,
            mqar=mqar_a,
            induction_copy=fused_a,
            lag_nll=lag_nll,
            enabled=True,
            device=device,
        )
        return LongCtxAtT(
            t=int(t),
            suite_mean=base.suite_mean,
            needle=base.needle,
            mqar=base.mqar,
            induction=ind_a,
            exact_copy=copy_a,
            induction_copy_fused=base.induction_copy,
            lag_nll=base.lag_nll,
            floor_pass=base.floor_pass,
            device=device,
            notes=base.notes,
        )

    # Recompute floor honesty using needle+mqar relative thresholds.
    relative: dict[str, float] = {}
    if needle_a is not None:
        relative["needle"] = relative_to_chance(needle_a, LONG_CTX_CHANCE["needle"])
    if mqar_a is not None:
        relative["mqar"] = relative_to_chance(mqar_a, LONG_CTX_CHANCE["mqar"])
    from .official_comparison import OFFICIAL_LONG_CTX_FLOOR
    from .scorecard_suite import LONG_CTX_RELATIVE_FLOOR

    abs_ok = suite_mean >= float(OFFICIAL_LONG_CTX_FLOOR)
    rel_ok = True
    notes: list[str] = []
    for task, rel in relative.items():
        if rel < float(LONG_CTX_RELATIVE_FLOOR):
            rel_ok = False
            notes.append(f"relative_floor_fail:{task}")
    if not abs_ok:
        notes.append("absolute_floor_fail")
    lag_val = _finite(lag_nll)

    return LongCtxAtT(
        t=int(t),
        suite_mean=suite_mean,
        needle=needle_a,
        mqar=mqar_a,
        induction=ind_a,
        exact_copy=copy_a,
        induction_copy_fused=fused_a,
        lag_nll=lag_val,
        floor_pass=bool(abs_ok and rel_ok),
        device=device,
        notes=tuple(notes),
    )


def multi_t_long_ctx_suite(
    slice_by_t: Mapping[int, LongCtxAtT | Mapping[str, Any]],
    *,
    requested_ts: Sequence[int] = DEFAULT_LONG_CTX_TS,
    device: str = "fixture",
    max_feasible_t: int | None = None,
) -> MultiTLongCtxResult:
    """Assemble multi-T long-ctx matrix from per-T slices (VAL-COMPLETE-003)."""
    by_t: dict[str, LongCtxAtT] = {}
    for t, raw in slice_by_t.items():
        if isinstance(raw, LongCtxAtT):
            by_t[str(int(t))] = raw
        else:
            by_t[str(int(t))] = LongCtxAtT(
                t=int(t),
                suite_mean=_finite(raw.get("suite_mean")),
                needle=_finite(raw.get("needle")),
                mqar=_finite(raw.get("mqar")),
                induction=_finite(raw.get("induction")),
                exact_copy=_finite(raw.get("exact_copy")),
                induction_copy_fused=_finite(
                    raw.get("induction_copy_fused") or raw.get("induction_copy")
                ),
                lag_nll=_finite(raw.get("lag_nll")),
                floor_pass=(None if raw.get("floor_pass") is None else bool(raw.get("floor_pass"))),
                device=str(raw.get("device") or device),
                notes=tuple(raw.get("notes") or ()),
            )
    suite_vals = [
        float(s.suite_mean)
        for s in by_t.values()
        if s.suite_mean is not None and math.isfinite(s.suite_mean)
    ]
    agg = float(sum(suite_vals) / len(suite_vals)) if suite_vals else None
    if max_feasible_t is None and by_t:
        max_feasible_t = max(int(k) for k in by_t)
    notes: list[str] = []
    for req in requested_ts:
        if str(int(req)) not in by_t:
            notes.append(f"missing_T:{req}")
    return MultiTLongCtxResult(
        by_T=by_t,
        aggregate_suite_mean=agg,
        max_feasible_t=max_feasible_t,
        requested_ts=tuple(int(t) for t in requested_ts),
        device=device,
        notes=tuple(notes),
    )


def run_multi_t_long_ctx_fixture(
    outcomes_by_t: Mapping[int, Mapping[str, Sequence[bool] | Sequence[float] | float]],
    *,
    lag_nll_by_t: Mapping[int, float] | None = None,
    requested_ts: Sequence[int] | None = None,
    device: str = "fixture",
) -> MultiTLongCtxResult:
    """CPU fixture path: closed-choice outcomes per task per T → multi-T matrix.

    ``outcomes_by_t[T]`` keys: ``needle``, ``mqar``, optional ``induction``,
    ``exact_copy``, and optional fused ``induction_copy`` (legacy). Values are
    trial outcome sequences or precomputed accuracies.
    """

    def _task_score(
        tasks: Mapping[str, Sequence[bool] | Sequence[float] | float],
        name: str,
        chance_key: str | None = None,
    ) -> LongCtxTaskScore | float | None:
        raw = tasks.get(name)
        if raw is None:
            return None
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return score_accuracy_value(float(raw), task=chance_key or name)
        return score_closed_choice_accuracy(
            raw,  # type: ignore[arg-type]
            task=chance_key or name,
        )

    slices: dict[int, LongCtxAtT] = {}
    for t, tasks in outcomes_by_t.items():
        lag = None if lag_nll_by_t is None else lag_nll_by_t.get(int(t))
        slices[int(t)] = build_long_ctx_at_t(
            t=int(t),
            needle=_task_score(tasks, "needle", "needle"),
            mqar=_task_score(tasks, "mqar", "mqar"),
            induction=_task_score(tasks, "induction", "induction_copy"),
            exact_copy=_task_score(tasks, "exact_copy", "induction_copy"),
            induction_copy_fused=_task_score(tasks, "induction_copy", "induction_copy"),
            lag_nll=lag,
            device=device,
        )
    ts = (
        tuple(sorted(int(t) for t in outcomes_by_t))
        if requested_ts is None
        else tuple(int(t) for t in requested_ts)
    )
    return multi_t_long_ctx_suite(slices, requested_ts=ts, device=device)


# --- VAL-COMPLETE-004: needle-by-depth + lost-in-middle ---------------------------


@dataclass(frozen=True)
class NeedleByDepthResult:
    """Needle accuracy stratified by depth position (lost-in-middle mid panel)."""

    by_depth: dict[str, float]
    lost_in_middle: float | None
    mid_depth: float = LOST_IN_MIDDLE_DEPTH
    trials_by_depth: dict[str, int] = field(default_factory=dict)
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_depth": dict(self.by_depth),
            "lost_in_middle": self.lost_in_middle,
            "mid_depth": self.mid_depth,
            "trials_by_depth": dict(self.trials_by_depth),
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
        }


def needle_by_depth_from_outcomes(
    outcomes_by_depth: Mapping[float | str, Sequence[bool] | Sequence[float] | float],
    *,
    mid_depth: float = LOST_IN_MIDDLE_DEPTH,
    device: str = "fixture",
) -> NeedleByDepthResult:
    """Build needle-by-depth curve + lost-in-middle mid accuracy."""
    by_depth: dict[str, float] = {}
    trials: dict[str, int] = {}
    for depth, raw in outcomes_by_depth.items():
        key = f"{float(depth):g}"
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            by_depth[key] = clamp01(float(raw))
            trials[key] = 0
        else:
            sc = score_closed_choice_accuracy(raw, task="needle")  # type: ignore[arg-type]
            by_depth[key] = sc.accuracy
            trials[key] = sc.trials
    mid_key = f"{float(mid_depth):g}"
    # Explicit lost-in-middle: prefer exact mid key; else nearest depth to mid.
    lim: float | None = by_depth.get(mid_key)
    if lim is None and by_depth:
        nearest = min(by_depth.keys(), key=lambda k: abs(float(k) - float(mid_depth)))
        lim = by_depth[nearest]
    return NeedleByDepthResult(
        by_depth=by_depth,
        lost_in_middle=lim,
        mid_depth=float(mid_depth),
        trials_by_depth=trials,
        device=device,
    )


# --- VAL-COMPLETE-005: MQAR N×lag grid --------------------------------------------


@dataclass(frozen=True)
class MqarGridResult:
    """MQAR associative-recall accuracy as N×lag grid (not pooled scalar only)."""

    grid: dict[str, dict[str, float]]
    n_values: tuple[int, ...]
    lag_values: tuple[int, ...]
    macro_mean: float | None
    chance: float = LONG_CTX_CHANCE["mqar"]
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid": {n: dict(lags) for n, lags in self.grid.items()},
            "n_values": list(self.n_values),
            "lag_values": list(self.lag_values),
            "macro_mean": self.macro_mean,
            "chance": self.chance,
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
        }


def mqar_grid_from_outcomes(
    outcomes: Mapping[tuple[int, int], Sequence[bool] | Sequence[float] | float]
    | Mapping[str, Mapping[str | int, Sequence[bool] | Sequence[float] | float]],
    *,
    n_values: Sequence[int] = DEFAULT_MQAR_NS,
    lag_values: Sequence[int] = DEFAULT_MQAR_LAGS,
    device: str = "fixture",
    chance: float = LONG_CTX_CHANCE["mqar"],
) -> MqarGridResult:
    """Build MQAR accuracy grid.

    Accepts either:
    * ``{(N, lag): trials_or_acc}`` flat keys, or
    * ``{"N4": {"lag_16": ..., "64": ...}, ...}`` nested form.
    """
    flat: dict[tuple[int, int], float] = {}
    trials_note = 0

    def _score(raw: Sequence[bool] | Sequence[float] | float) -> float:
        nonlocal trials_note
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return clamp01(float(raw))
        sc = score_closed_choice_accuracy(raw, task="mqar", chance=chance)  # type: ignore[arg-type]
        trials_note += sc.trials
        return sc.accuracy

    # Detect nested vs flat.
    is_flat = bool(outcomes) and all(
        isinstance(k, tuple) and len(k) == 2
        for k in outcomes  # type: ignore[arg-type]
    )
    if is_flat:
        for key, raw in outcomes.items():  # type: ignore[assignment]
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("flat mqar outcomes keys must be (N, lag) tuples")
            n_i, lag_i = int(key[0]), int(key[1])
            flat[(n_i, lag_i)] = _score(raw)  # type: ignore[arg-type]
    else:
        for n_key, lag_map in outcomes.items():  # type: ignore[assignment]
            if isinstance(n_key, tuple):
                raise TypeError("mixed flat/nested mqar grid keys are not supported")
            n_raw = str(n_key)
            if n_raw.isdigit():
                n_i = int(n_raw)
            else:
                n_i = int(n_raw.lstrip("Nn"))
            if not isinstance(lag_map, Mapping):
                raise TypeError("nested mqar grid values must be lag→outcomes maps")
            for lag_key, raw in lag_map.items():
                lag_s = str(lag_key).replace("lag_", "").replace("lag", "")
                lag_i = int(lag_s)
                flat[(n_i, lag_i)] = _score(raw)  # type: ignore[arg-type]

    ns = tuple(int(n) for n in n_values)
    lags = tuple(int(lag) for lag in lag_values)
    # Expand axes with any extras found in flat.
    extra_ns = sorted({n for n, _ in flat if n not in ns})
    extra_lags = sorted({lag for _, lag in flat if lag not in lags})
    ns = ns + tuple(extra_ns)
    lags = lags + tuple(extra_lags)

    grid: dict[str, dict[str, float]] = {}
    all_vals: list[float] = []
    for n in ns:
        row: dict[str, float] = {}
        for lag in lags:
            if (n, lag) in flat:
                row[f"lag_{lag}"] = flat[(n, lag)]
                all_vals.append(flat[(n, lag)])
        if row:
            grid[f"N{n}"] = row
    macro = float(sum(all_vals) / len(all_vals)) if all_vals else None
    return MqarGridResult(
        grid=grid,
        n_values=ns,
        lag_values=lags,
        macro_mean=macro,
        chance=float(chance),
        device=device,
    )


# --- VAL-COMPLETE-006: induction + exact-copy unfused -----------------------------


@dataclass(frozen=True)
class UnfusedInductionCopy:
    """Separate induction-heads and exact-copy probes (no single fused field)."""

    induction_acc: float
    exact_copy_acc: float
    induction_trials: int = 0
    exact_copy_trials: int = 0
    chance_induction: float = LONG_CTX_CHANCE["induction_copy"]
    chance_exact_copy: float = LONG_CTX_CHANCE["induction_copy"]
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "induction_acc": self.induction_acc,
            "exact_copy_acc": self.exact_copy_acc,
            "induction_trials": self.induction_trials,
            "exact_copy_trials": self.exact_copy_trials,
            "chance_induction": self.chance_induction,
            "chance_exact_copy": self.chance_exact_copy,
            "relative_induction": relative_to_chance(self.induction_acc, self.chance_induction),
            "relative_exact_copy": relative_to_chance(self.exact_copy_acc, self.chance_exact_copy),
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
            "fused_only": False,
        }


def unfuse_induction_and_copy(
    *,
    induction: Sequence[bool] | Sequence[float] | float,
    exact_copy: Sequence[bool] | Sequence[float] | float,
    device: str = "fixture",
    chance_induction: float = LONG_CTX_CHANCE["induction_copy"],
    chance_exact_copy: float = LONG_CTX_CHANCE["induction_copy"],
) -> UnfusedInductionCopy:
    """Score induction and exact-copy probes as separate metrics."""

    def _score(raw: Sequence[bool] | Sequence[float] | float, chance: float) -> LongCtxTaskScore:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return score_accuracy_value(float(raw), task="induction_copy", chance=chance)
        # Sequence path (bool/float/int trial outcomes).
        seq: Sequence[bool] | Sequence[float] | Sequence[int] = raw  # type: ignore[assignment]
        return score_closed_choice_accuracy(
            seq,
            task="induction_copy",
            chance=chance,
        )

    ind = _score(induction, chance_induction)
    copy = _score(exact_copy, chance_exact_copy)
    return UnfusedInductionCopy(
        induction_acc=ind.accuracy,
        exact_copy_acc=copy.accuracy,
        induction_trials=ind.trials,
        exact_copy_trials=copy.trials,
        chance_induction=float(chance_induction),
        chance_exact_copy=float(chance_exact_copy),
        device=device,
    )


# --- VAL-COMPLETE-007: lag-NLL bins + length-extrapolation CE ---------------------


@dataclass(frozen=True)
class LagNllBinsResult:
    """Lag-binned next-token NLL/bpb on text (lower better)."""

    bins: dict[str, float]
    macro_long: float | None
    device: str = "fixture"
    unit: str = "nats"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "bins": dict(self.bins),
            "macro_long": self.macro_long,
            "device": self.device,
            "unit": self.unit,
            "notes": list(self.notes),
            "status": "filled",
        }


def lag_nll_bins_result(
    lag_nll_by_bin: Mapping[str | int, float],
    *,
    long_bin_keys: Sequence[str | int] = ("lag_ge_64", "lag_ge_256", "lag_ge_512"),
    device: str = "fixture",
    unit: str = "nats",
) -> LagNllBinsResult:
    """Normalize lag bins + macro long-lag estimate."""
    bins: dict[str, float] = {}
    for k, v in lag_nll_by_bin.items():
        fv = _finite(v)
        if fv is None:
            continue
        if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
            key = f"lag_{k}"
        else:
            key = str(k)
        bins[key] = fv
    macro = lag_nll_from_bins(
        bins,  # type: ignore[arg-type]
        long_bin_keys=long_bin_keys,
    )
    if not math.isfinite(macro):
        macro_out: float | None = None
    else:
        macro_out = float(macro)
    return LagNllBinsResult(
        bins=bins,
        macro_long=macro_out,
        device=device,
        unit=unit,
    )


@dataclass(frozen=True)
class LengthExtrapResult:
    """Train-short eval-long free CE without retrain (VAL-COMPLETE-007)."""

    train_t: int
    ce_by_t: dict[str, float]
    ratio_t_over_train: dict[str, float]
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_t": self.train_t,
            "ce_by_t": dict(self.ce_by_t),
            "ratio_t_over_train": dict(self.ratio_t_over_train),
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
            "retrain": False,
        }


def length_extrapolate_ce(
    ce_by_t: Mapping[int | str, float],
    *,
    train_t: int = 128,
    device: str = "fixture",
) -> LengthExtrapResult:
    """Compute CE@T and ratio CE@T / CE@train_t (no retrain semantics).

    Requires a finite train_t CE entry; missing train raises so callers do not
    invent stable ratios.
    """
    ce: dict[str, float] = {}
    for k, v in ce_by_t.items():
        fv = _finite(v)
        if fv is None:
            continue
        ce[str(int(k))] = fv
    train_key = str(int(train_t))
    if train_key not in ce:
        raise ValueError(f"length_extrapolate_ce requires CE at train_t={train_t}")
    base = ce[train_key]
    ratios: dict[str, float] = {}
    notes: list[str] = ["no_retrain"]
    if base <= 0 or not math.isfinite(base):
        notes.append("non_positive_train_ce")
        for k, v in ce.items():
            ratios[k] = float("inf") if not math.isfinite(v) else float(v)
    else:
        for k, v in ce.items():
            ratios[k] = float(v) / float(base)
    return LengthExtrapResult(
        train_t=int(train_t),
        ce_by_t=ce,
        ratio_t_over_train=ratios,
        device=device,
        notes=tuple(notes),
    )


# --- Combined family quality bundle + Complete View panel fill --------------------


@dataclass(frozen=True)
class FamilyLongCtxQuality:
    """One-family Complete View quality + long-ctx expansion payload."""

    val_bpb_trained: MultiSeedValBpb | None = None
    medium_free_ce: dict[str, Any] | None = None
    multi_t: MultiTLongCtxResult | None = None
    needle_by_depth: NeedleByDepthResult | None = None
    mqar_grid: MqarGridResult | None = None
    induction_copy: UnfusedInductionCopy | None = None
    lag_bins: LagNllBinsResult | None = None
    length_extrap: LengthExtrapResult | None = None
    device: str = "fixture"

    def as_dict(self) -> dict[str, Any]:
        return {
            "val_bpb_trained": (
                None if self.val_bpb_trained is None else self.val_bpb_trained.as_dict()
            ),
            "medium_free_ce": self.medium_free_ce,
            "multi_t": None if self.multi_t is None else self.multi_t.as_dict(),
            "needle_by_depth": (
                None if self.needle_by_depth is None else self.needle_by_depth.as_dict()
            ),
            "mqar_grid": None if self.mqar_grid is None else self.mqar_grid.as_dict(),
            "induction_copy": (
                None if self.induction_copy is None else self.induction_copy.as_dict()
            ),
            "lag_bins": None if self.lag_bins is None else self.lag_bins.as_dict(),
            "length_extrap": (None if self.length_extrap is None else self.length_extrap.as_dict()),
            "device": self.device,
        }


def _side_fill(
    a_val: Any,
    b_val: Any,
    *,
    status_filled: bool,
    reason_if_null: str,
) -> dict[str, Any]:
    return {
        "a": a_val,
        "b": b_val,
        "status": "filled" if status_filled else "not_run",
        "reason": None if status_filled else reason_if_null,
    }


def build_longctx_quality_panels(
    fam_a: FamilyLongCtxQuality | None,
    fam_b: FamilyLongCtxQuality | None,
    *,
    a_record: OfficialScoreRecord | None = None,
    b_record: OfficialScoreRecord | None = None,
) -> dict[str, Any]:
    """Panel overrides for P1 / P3 / P4 from family quality suites.

    Safe partial fill: only keys with measured data are set; remaining keys stay
    as Complete View null shells when suited.
    """
    a = fam_a or FamilyLongCtxQuality()
    b = fam_b or FamilyLongCtxQuality()

    # --- P1 short gen: val_bpb_trained + optional medium free CE ---
    val_a = (
        a.val_bpb_trained.as_dict()
        if a.val_bpb_trained is not None
        else (
            {
                "mean": a_record.val_bpb_trained,
                "std": None,
                "status": "filled" if a_record and a_record.val_bpb_trained is not None else "null",
            }
            if a_record is not None
            else None
        )
    )
    val_b = (
        b.val_bpb_trained.as_dict()
        if b.val_bpb_trained is not None
        else (
            {
                "mean": b_record.val_bpb_trained,
                "std": None,
                "status": "filled" if b_record and b_record.val_bpb_trained is not None else "null",
            }
            if b_record is not None
            else None
        )
    )
    both_val = (
        a.val_bpb_trained is not None
        and b.val_bpb_trained is not None
        or (
            a_record is not None
            and b_record is not None
            and a_record.val_bpb_trained is not None
            and b_record.val_bpb_trained is not None
        )
    )
    p1: dict[str, Any] = {
        "val_bpb_trained": {
            "a": val_a,
            "b": val_b,
            "status": "filled" if both_val else "null_pending_VAL-COMPLETE-002",
        }
    }
    if a.medium_free_ce is not None or b.medium_free_ce is not None:
        p1["medium_free_ce"] = {
            "a": a.medium_free_ce,
            "b": b.medium_free_ce,
            "status": "filled",
        }

    # --- P3 long ctx expansions ---
    multi_filled = a.multi_t is not None and b.multi_t is not None
    by_t_merged: dict[str, Any] = {}
    if multi_filled:
        assert a.multi_t is not None and b.multi_t is not None
        all_ts = sorted(
            set(a.multi_t.by_T) | set(b.multi_t.by_T),
            key=lambda x: int(x),
        )
        for t_key in all_ts:
            sa = a.multi_t.by_T.get(t_key)
            sb = b.multi_t.by_T.get(t_key)
            by_t_merged[t_key] = {
                "suite_mean": {
                    "a": None if sa is None else sa.suite_mean,
                    "b": None if sb is None else sb.suite_mean,
                },
                "needle": {
                    "a": None if sa is None else sa.needle,
                    "b": None if sb is None else sb.needle,
                },
                "mqar": {
                    "a": None if sa is None else sa.mqar,
                    "b": None if sb is None else sb.mqar,
                },
                "induction": {
                    "a": None if sa is None else sa.induction,
                    "b": None if sb is None else sb.induction,
                },
                "exact_copy": {
                    "a": None if sa is None else sa.exact_copy,
                    "b": None if sb is None else sb.exact_copy,
                },
                "induction_copy_fused_historical": {
                    "a": None if sa is None else sa.induction_copy_fused,
                    "b": None if sb is None else sb.induction_copy_fused,
                },
                "lag_nll": {
                    "a": None if sa is None else sa.lag_nll,
                    "b": None if sb is None else sb.lag_nll,
                },
                "floor_pass": {
                    "a": None if sa is None else sa.floor_pass,
                    "b": None if sb is None else sb.floor_pass,
                },
            }

    depth_filled = a.needle_by_depth is not None and b.needle_by_depth is not None
    mqar_filled = a.mqar_grid is not None and b.mqar_grid is not None
    unfused_filled = a.induction_copy is not None and b.induction_copy is not None
    lag_filled = a.lag_bins is not None and b.lag_bins is not None

    p3: dict[str, Any] = {
        "multi_T": _side_fill(
            None if a.multi_t is None else a.multi_t.as_dict(),
            None if b.multi_t is None else b.multi_t.as_dict(),
            status_filled=multi_filled,
            reason_if_null="multi_T_pending_VAL-COMPLETE-003",
        ),
        "long_ctx_by_T": by_t_merged if multi_filled else None,
        "aggregate_suite_mean": {
            "a": None if a.multi_t is None else a.multi_t.aggregate_suite_mean,
            "b": None if b.multi_t is None else b.multi_t.aggregate_suite_mean,
        },
        "needle_by_depth": _side_fill(
            None if a.needle_by_depth is None else a.needle_by_depth.as_dict(),
            None if b.needle_by_depth is None else b.needle_by_depth.as_dict(),
            status_filled=depth_filled,
            reason_if_null="pending_VAL-COMPLETE-004",
        ),
        "lost_in_middle": _side_fill(
            None if a.needle_by_depth is None else a.needle_by_depth.lost_in_middle,
            None if b.needle_by_depth is None else b.needle_by_depth.lost_in_middle,
            status_filled=depth_filled,
            reason_if_null="pending_VAL-COMPLETE-004",
        ),
        "mqar_grid": _side_fill(
            None if a.mqar_grid is None else a.mqar_grid.as_dict(),
            None if b.mqar_grid is None else b.mqar_grid.as_dict(),
            status_filled=mqar_filled,
            reason_if_null="pending_VAL-COMPLETE-005",
        ),
        "induction_acc": _side_fill(
            None if a.induction_copy is None else a.induction_copy.induction_acc,
            None if b.induction_copy is None else b.induction_copy.induction_acc,
            status_filled=unfused_filled,
            reason_if_null="pending_VAL-COMPLETE-006",
        ),
        "copy_acc": _side_fill(
            None if a.induction_copy is None else a.induction_copy.exact_copy_acc,
            None if b.induction_copy is None else b.induction_copy.exact_copy_acc,
            status_filled=unfused_filled,
            reason_if_null="pending_VAL-COMPLETE-006",
        ),
        "induction_and_copy_unfused": _side_fill(
            None if a.induction_copy is None else a.induction_copy.as_dict(),
            None if b.induction_copy is None else b.induction_copy.as_dict(),
            status_filled=unfused_filled,
            reason_if_null="pending_VAL-COMPLETE-006",
        ),
        "lag_nll_bins": {
            "macro": {
                "a": None if a.lag_bins is None else a.lag_bins.macro_long,
                "b": None if b.lag_bins is None else b.lag_bins.macro_long,
            },
            "binned": _side_fill(
                None if a.lag_bins is None else a.lag_bins.as_dict(),
                None if b.lag_bins is None else b.lag_bins.as_dict(),
                status_filled=lag_filled,
                reason_if_null="pending_VAL-COMPLETE-007",
            ),
        },
    }
    if multi_filled and by_t_merged:
        p3["by_T"] = by_t_merged

    # --- P4 length extrap ---
    le_filled = a.length_extrap is not None and b.length_extrap is not None
    p4: dict[str, Any] = {
        "ce_by_T": _side_fill(
            None if a.length_extrap is None else a.length_extrap.ce_by_t,
            None if b.length_extrap is None else b.length_extrap.ce_by_t,
            status_filled=le_filled,
            reason_if_null="pending_VAL-COMPLETE-007",
        ),
        "ratio_T_over_train": _side_fill(
            None if a.length_extrap is None else a.length_extrap.ratio_t_over_train,
            None if b.length_extrap is None else b.length_extrap.ratio_t_over_train,
            status_filled=le_filled,
            reason_if_null="pending_VAL-COMPLETE-007",
        ),
        "length_extrapolate": _side_fill(
            None if a.length_extrap is None else a.length_extrap.as_dict(),
            None if b.length_extrap is None else b.length_extrap.as_dict(),
            status_filled=le_filled,
            reason_if_null="pending_VAL-COMPLETE-007",
        ),
        "retrain": False,
    }

    # Only return known Complete View panel keys.
    panels: dict[str, Any] = {}
    if "P1_short_gen" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P1_short_gen"] = p1
    if "P3_long_ctx" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P3_long_ctx"] = p3
    if "P4_length_extrap" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P4_length_extrap"] = p4
    return panels


def apply_val_bpb_to_record(
    record: OfficialScoreRecord,
    val: MultiSeedValBpb,
) -> OfficialScoreRecord:
    """Stamp multi-seed mean val_bpb_trained onto an OfficialScoreRecord."""
    from dataclasses import replace

    return replace(record, val_bpb_trained=float(val.mean))


def build_complete_view_with_longctx_quality(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    fam_a: FamilyLongCtxQuality | None = None,
    fam_b: FamilyLongCtxQuality | None = None,
    score_class: str = "fixture",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build complete_view.v1.2 with quality/long-ctx panels filled when provided."""
    # Stamp records when multi-seed val present.
    rec_a = a
    rec_b = b
    if fam_a is not None and fam_a.val_bpb_trained is not None:
        rec_a = apply_val_bpb_to_record(a, fam_a.val_bpb_trained)
    if fam_b is not None and fam_b.val_bpb_trained is not None:
        rec_b = apply_val_bpb_to_record(b, fam_b.val_bpb_trained)

    overrides = build_longctx_quality_panels(fam_a, fam_b, a_record=rec_a, b_record=rec_b)
    return build_complete_view(
        rec_a,
        rec_b,
        panels_override=overrides,
        score_class=score_class,
        **kwargs,
    )


# --- GPU/CPU-ready free-CE probe hook ---------------------------------------------


def free_ce_bits_from_nll_stream(
    nll_nats: Sequence[float],
    *,
    basis: Literal["nats", "bits"] = "nats",
) -> float:
    """Mean free CE in bits from a stream of per-token NLL values."""
    clean = [float(x) for x in nll_nats if math.isfinite(float(x))]
    if not clean:
        return float("inf")
    mean = sum(clean) / len(clean)
    if basis == "bits":
        return float(mean)
    return float(mean / math.log(2.0))


def probe_free_ce(
    nll_fn: Callable[[Sequence[int]], Sequence[float] | float],
    sequences: Sequence[Sequence[int]],
    *,
    basis: Literal["nats", "bits"] = "nats",
) -> float:
    """GPU/CPU-ready free CE probe: ``nll_fn(token_ids) → nll stream or scalar``."""
    all_nll: list[float] = []
    for seq in sequences:
        raw = nll_fn(list(seq))
        if isinstance(raw, (int, float)):
            all_nll.append(float(raw))
        else:
            all_nll.extend(float(x) for x in raw)
    return free_ce_bits_from_nll_stream(all_nll, basis=basis)


def fixture_family_longctx_quality(
    *,
    val_bpb_per_seed: Mapping[int, float],
    multi_t_outcomes: Mapping[int, Mapping[str, Sequence[bool] | Sequence[float] | float]],
    needle_depth_outcomes: Mapping[float | str, Sequence[bool] | Sequence[float] | float],
    mqar_outcomes: Mapping[tuple[int, int], Sequence[bool] | Sequence[float] | float]
    | Mapping[str, Mapping[str | int, Sequence[bool] | Sequence[float] | float]],
    induction: Sequence[bool] | Sequence[float] | float,
    exact_copy: Sequence[bool] | Sequence[float] | float,
    lag_bins: Mapping[str | int, float],
    length_extrap_ce: Mapping[int | str, float],
    train_t: int = 128,
    medium_ce: Mapping[int | str, float] | None = None,
    device: str = "fixture",
) -> FamilyLongCtxQuality:
    """One-shot fixture builder covering VAL-COMPLETE-002..007 for a family."""
    return FamilyLongCtxQuality(
        val_bpb_trained=multi_seed_val_bpb_trained(val_bpb_per_seed, device=device),
        medium_free_ce=(
            None if medium_ce is None else medium_free_ce_by_T(medium_ce, device=device)
        ),
        multi_t=run_multi_t_long_ctx_fixture(multi_t_outcomes, device=device),
        needle_by_depth=needle_by_depth_from_outcomes(needle_depth_outcomes, device=device),
        mqar_grid=mqar_grid_from_outcomes(mqar_outcomes, device=device),
        induction_copy=unfuse_induction_and_copy(
            induction=induction, exact_copy=exact_copy, device=device
        ),
        lag_bins=lag_nll_bins_result(lag_bins, device=device),
        length_extrap=length_extrapolate_ce(length_extrap_ce, train_t=train_t, device=device),
        device=device,
    )


__all__ = [
    "DEFAULT_LAG_BIN_KEYS",
    "DEFAULT_LENGTH_EXTRAP_TS",
    "DEFAULT_LONG_CTX_TS",
    "DEFAULT_MQAR_LAGS",
    "DEFAULT_MQAR_NS",
    "DEFAULT_NEEDLE_DEPTHS",
    "LOST_IN_MIDDLE_DEPTH",
    "FamilyLongCtxQuality",
    "LagNllBinsResult",
    "LengthExtrapResult",
    "LongCtxAtT",
    "MultiSeedValBpb",
    "MultiTLongCtxResult",
    "MqarGridResult",
    "NeedleByDepthResult",
    "UnfusedInductionCopy",
    "apply_val_bpb_to_record",
    "build_complete_view_with_longctx_quality",
    "build_long_ctx_at_t",
    "build_longctx_quality_panels",
    "fixture_family_longctx_quality",
    "free_ce_bits_from_nll_stream",
    "lag_nll_bins_result",
    "length_extrapolate_ce",
    "medium_free_ce_by_T",
    "mqar_grid_from_outcomes",
    "multi_seed_val_bpb_trained",
    "multi_t_long_ctx_suite",
    "needle_by_depth_from_outcomes",
    "probe_free_ce",
    "run_multi_t_long_ctx_fixture",
    "unfuse_induction_and_copy",
    # re-export slight convenience for callers looking for v1.1 base
    "LongCtxSuiteResult",
    "run_long_ctx_fixture_suite",
]
