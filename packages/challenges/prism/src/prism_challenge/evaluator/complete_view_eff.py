"""Complete View efficiency, sample-eff densification, stability, robustness, nice-to-have.

Architecture-agnostic suite builders for Complete View v1.2 residual panels
(VAL-COMPLETE-008..012):

* denser sample-eff marks / curve summary from online streams (+ optional heldout@marks)
* train vs eval@T VRAM and tokens/s split + step_time_ms residual
* agnostic state / activation footprint@T
* multi-seed grad_spike_rate + nan_inf_events (filled, not schema-null)
* multi-order stream residual **or honest BLOCKED_with_reason**
* derived quality_per_param / quality_per_gib
* nice-to-have diagnostics with null + reason (no silent omission)

CPU unit fixtures fully populate filled fields from synthetic streams/captures.
GPU/CPU hooks accept peak-allocator bytes, timed step probes, and train logs.
Efficiency axes never sole-rank; scientific winners remain polar / multi-axis.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .complete_view import (
    COMPLETE_VIEW_NICE_TO_HAVE,
    COMPLETE_VIEW_PANEL_KEYS,
    build_complete_view,
)
from .official_comparison import OfficialScoreRecord
from .scorecard_suite import (
    DEFAULT_SAMPLE_EFF_MARKS_TOKENS,
    quality_from_bpb,
    sample_efficiency_from_stream,
)

# Dense sample-eff marks beyond the four-scorecard base (50k/100k/250k/500k).
# Host recompute from existing online_loss streams is free (CV-MH-09).
DENSE_SAMPLE_EFF_MARKS_TOKENS: tuple[int, ...] = (
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    250_000,
    375_000,
    500_000,
)
DEFAULT_EVAL_TS: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_STATE_FOOTPRINT_TS: tuple[int, ...] = (128, 256, 512, 1024)

DeviceHint = Literal["cpu", "cuda", "auto", "fixture"]
QualityProxy = Literal["heldout_delta", "quality_auc", "inv_val_bpb", "inv_bpb"]


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


def _percentile(sorted_vals: Sequence[float], q: float) -> float | None:
    """Linear interpolation percentile for q in [0, 1]; empty → None."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


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


# --- VAL-COMPLETE-008: dense sample-efficiency ------------------------------------


@dataclass(frozen=True)
class DenseSampleEffResult:
    """Denser marks + curve summary from challenge online streams (CV-MH-09/10)."""

    marks_tokens: tuple[int, ...]
    bpb_at_marks: tuple[float, ...]
    quality_at_marks: tuple[float, ...]
    auc: float
    curve_summary: dict[str, float | None]
    heldout_at_marks: dict[str, Any] | None = None
    covered_bytes_total: float = 0.0
    method: str = "online_stream_trapezoid_dense"
    base_marks_tokens: tuple[int, ...] = DEFAULT_SAMPLE_EFF_MARKS_TOKENS
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "marks_tokens": list(self.marks_tokens),
            "bpb_at_marks": list(self.bpb_at_marks),
            "quality_at_marks": list(self.quality_at_marks),
            "auc": self.auc,
            "curve_summary": dict(self.curve_summary),
            "heldout_at_marks": self.heldout_at_marks,
            "covered_bytes_total": self.covered_bytes_total,
            "method": self.method,
            "base_marks_tokens": list(self.base_marks_tokens),
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
        }


def online_bpb_curve_summary(
    online_loss: Sequence[float],
    *,
    basis: Literal["nats", "bits"] = "nats",
) -> dict[str, float | None]:
    """p10 / median / p90 / mean of online stream interpreted as bpb.

    When ``basis='nats'``, converts mean-compatible per-sample nats→bits via /ln2.
    Empty stream yields an honest all-null summary (never invent).
    """
    clean = [float(x) for x in online_loss if math.isfinite(float(x))]
    if not clean:
        return {
            "p10_bpb": None,
            "median_bpb": None,
            "p90_bpb": None,
            "mean_bpb": None,
            "min_bpb": None,
            "max_bpb": None,
            "n_samples": 0.0,
        }
    scale = 1.0 if basis == "bits" else (1.0 / math.log(2.0))
    bits = sorted(float(x) * scale for x in clean)
    mean_b = float(sum(bits) / len(bits))
    return {
        "p10_bpb": _percentile(bits, 0.10),
        "median_bpb": _percentile(bits, 0.50),
        "p90_bpb": _percentile(bits, 0.90),
        "mean_bpb": mean_b,
        "min_bpb": float(bits[0]),
        "max_bpb": float(bits[-1]),
        "n_samples": float(len(bits)),
    }


def dense_sample_efficiency_from_stream(
    online_loss: Sequence[float],
    *,
    covered_bytes_cumulative: Sequence[float] | None = None,
    token_budget: int = 500_000,
    marks_tokens: Sequence[int] = DENSE_SAMPLE_EFF_MARKS_TOKENS,
    heldout_at_marks: Mapping[int | str, float] | None = None,
    online_loss_basis: Literal["nats", "bits"] = "nats",
    device: str = "fixture",
) -> DenseSampleEffResult:
    """Compute dense marks + curve summary from an online-loss stream.

    Host-side densification of scorecard 4-mark sample_eff without retrain.
    Optional heldout@token-marks included when provided; else heldout_at_marks
    payload documents host-stream marks fully (reason not-run for checkpoint curve).
    """
    base = sample_efficiency_from_stream(
        online_loss,
        covered_bytes_cumulative=covered_bytes_cumulative,
        token_budget=token_budget,
        marks_tokens=marks_tokens,
    )
    curve = online_bpb_curve_summary(online_loss, basis=online_loss_basis)
    notes: list[str] = ["host_stream_marks_dense"]
    heldout_payload: dict[str, Any] | None
    if heldout_at_marks is not None:
        entries: dict[str, float] = {}
        for k, v in heldout_at_marks.items():
            fv = _finite(v)
            if fv is None:
                continue
            entries[str(int(k))] = fv
        if entries:
            heldout_payload = {
                "status": "filled",
                "marks": entries,
                "method": "checkpoint_or_host_reeval",
                "reason": None,
            }
            notes.append("heldout_at_marks_filled")
        else:
            heldout_payload = {
                "status": "not_run",
                "marks": None,
                "method": None,
                "reason": "heldout_at_marks_empty_or_nonfinite",
            }
            notes.append("heldout_at_marks_empty")
    else:
        heldout_payload = {
            "status": "not_run",
            "marks": None,
            "method": None,
            "reason": (
                "heldout_checkpoint_curve_not_available; "
                "host_stream dense marks fully documented in bpb_at_marks"
            ),
        }
        notes.append("heldout_at_marks_not_run_host_stream_documented")
    return DenseSampleEffResult(
        marks_tokens=base.marks_tokens,
        bpb_at_marks=base.bpb_at_marks,
        quality_at_marks=base.quality_at_marks,
        auc=float(base.auc),
        curve_summary=curve,
        heldout_at_marks=heldout_payload,
        covered_bytes_total=float(base.covered_bytes_total),
        method="online_stream_trapezoid_dense",
        base_marks_tokens=DEFAULT_SAMPLE_EFF_MARKS_TOKENS,
        device=device,
        notes=tuple(notes),
    )


def apply_dense_sample_eff_to_record(
    record: OfficialScoreRecord,
    suite: DenseSampleEffResult,
) -> OfficialScoreRecord:
    """Stamp dense sample-eff AUC + bpb marks onto a score record."""
    return replace(
        record,
        sample_eff_auc=float(suite.auc),
        sample_eff_marks=tuple(float(x) for x in suite.bpb_at_marks),
    )


# --- VAL-COMPLETE-009: efficiency train vs eval@T + state footprint ---------------


@dataclass(frozen=True)
class TrainEvalEfficiency:
    """Train peak VRAM/tok/s separated from eval@T diagnostics (CV-MH-11..13)."""

    params: int | None = None
    peak_vram_train_gib: float | None = None
    tokens_per_s_train: float | None = None
    wall_clock_train_s: float | None = None
    peak_vram_eval_by_T: dict[str, float] = field(default_factory=dict)
    tokens_per_s_eval_by_T: dict[str, float] = field(default_factory=dict)
    step_time_ms: dict[str, float | None] = field(default_factory=dict)
    device: str = "fixture"
    sole_rank_forbidden: bool = True
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "peak_vram_train_gib": self.peak_vram_train_gib,
            "tokens_per_s_train": self.tokens_per_s_train,
            "wall_clock_train_s": self.wall_clock_train_s,
            "peak_vram_eval_by_T": dict(self.peak_vram_eval_by_T),
            "tokens_per_s_eval_by_T": dict(self.tokens_per_s_eval_by_T),
            "step_time_ms": dict(self.step_time_ms),
            "device": self.device,
            "sole_rank_forbidden": self.sole_rank_forbidden,
            "overrides_scientific_axes": False,
            "overrides_polar_rule": False,
            "notes": list(self.notes),
            "status": "filled",
        }


def build_train_eval_efficiency(
    *,
    params: int | None = None,
    peak_vram_train_gib: float | None = None,
    tokens_per_s_train: float | None = None,
    wall_clock_train_s: float | None = None,
    peak_vram_eval_by_T: Mapping[int | str, float] | None = None,
    tokens_per_s_eval_by_T: Mapping[int | str, float] | None = None,
    step_time_ms_mean: float | None = None,
    step_time_ms_p99: float | None = None,
    device: str = "fixture",
    notes: Sequence[str] = (),
) -> TrainEvalEfficiency:
    """Assemble train vs eval@T efficiency annex (null fields stay null; no invention)."""
    vram_eval: dict[str, float] = {}
    if peak_vram_eval_by_T:
        for k, v in peak_vram_eval_by_T.items():
            fv = _finite(v)
            if fv is None:
                continue
            vram_eval[str(int(k))] = fv
    tps_eval: dict[str, float] = {}
    if tokens_per_s_eval_by_T:
        for k, v in tokens_per_s_eval_by_T.items():
            fv = _finite(v)
            if fv is None:
                continue
            tps_eval[str(int(k))] = fv
    step: dict[str, float | None] = {
        "mean": _finite(step_time_ms_mean),
        "p99": _finite(step_time_ms_p99),
    }
    note_list = list(notes)
    if not vram_eval and not tps_eval:
        note_list.append("eval_by_T_partial_or_empty")
    return TrainEvalEfficiency(
        params=None if params is None else int(params),
        peak_vram_train_gib=_finite(peak_vram_train_gib),
        tokens_per_s_train=_finite(tokens_per_s_train),
        wall_clock_train_s=_finite(wall_clock_train_s),
        peak_vram_eval_by_T=vram_eval,
        tokens_per_s_eval_by_T=tps_eval,
        step_time_ms=step,
        device=device,
        notes=tuple(note_list),
    )


def apply_train_eval_efficiency_to_record(
    record: OfficialScoreRecord,
    annex: TrainEvalEfficiency,
) -> OfficialScoreRecord:
    """Stamp train-side efficiency fields onto a score record (eval@T lives in panels)."""
    return replace(
        record,
        params=annex.params if annex.params is not None else record.params,
        peak_vram_gib=(
            annex.peak_vram_train_gib
            if annex.peak_vram_train_gib is not None
            else record.peak_vram_gib
        ),
        tokens_per_s=(
            annex.tokens_per_s_train
            if annex.tokens_per_s_train is not None
            else record.tokens_per_s
        ),
        wall_clock_seconds=(
            annex.wall_clock_train_s
            if annex.wall_clock_train_s is not None
            else record.wall_clock_seconds
        ),
    )


@dataclass(frozen=True)
class StateFootprintResult:
    """Architecture-agnostic state/activation footprint@T (CV-MH-14)."""

    state_bytes_by_T: dict[str, float]
    activation_peak_bytes_by_T: dict[str, float]
    param_bytes: float | None = None
    method: str = "agnostic_activation_state_probe"
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_bytes_by_T": dict(self.state_bytes_by_T),
            "activation_peak_bytes_by_T": dict(self.activation_peak_bytes_by_T),
            "param_bytes": self.param_bytes,
            "method": self.method,
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
        }


def build_state_footprint(
    *,
    state_bytes_by_T: Mapping[int | str, float] | None = None,
    activation_peak_bytes_by_T: Mapping[int | str, float] | None = None,
    param_bytes: float | int | None = None,
    params: int | None = None,
    bytes_per_param: float = 4.0,
    device: str = "fixture",
    notes: Sequence[str] = (),
) -> StateFootprintResult:
    """Build state/activation footprint maps (architecture-agnostic bytes)."""
    state: dict[str, float] = {}
    if state_bytes_by_T:
        for k, v in state_bytes_by_T.items():
            fv = _finite(v)
            if fv is None:
                continue
            state[str(int(k))] = fv
    act: dict[str, float] = {}
    if activation_peak_bytes_by_T:
        for k, v in activation_peak_bytes_by_T.items():
            fv = _finite(v)
            if fv is None:
                continue
            act[str(int(k))] = fv
    pbytes = _finite(param_bytes)
    if pbytes is None and params is not None and params > 0:
        pbytes = float(params) * float(bytes_per_param)
    if not state and not act:
        raise ValueError(
            "state_footprint requires at least one of state_bytes_by_T / "
            "activation_peak_bytes_by_T with finite values"
        )
    return StateFootprintResult(
        state_bytes_by_T=state,
        activation_peak_bytes_by_T=act,
        param_bytes=pbytes,
        device=device,
        notes=tuple(notes),
    )


# --- VAL-COMPLETE-010: stability grad/nan multi-seed ------------------------------


@dataclass(frozen=True)
class StabilityMultiSeed:
    """Multi-seed grad spike + nan/inf residual (CV-MH-15). Filled, not schema-null."""

    grad_spike_rate_mean: float
    grad_spike_rate_std: float
    nan_inf_events_total: int
    nan_inf_events_mean: float
    per_seed_grad_spike_rate: tuple[float, ...]
    per_seed_nan_inf_events: tuple[int, ...]
    seeds: tuple[int, ...]
    instability_flag: bool
    step0_anomaly: bool = False
    seed_std_bpb: float | None = None
    seed_std_heldout: float | None = None
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "grad_spike_rate": {
                "mean": self.grad_spike_rate_mean,
                "std": self.grad_spike_rate_std,
                "per_seed": list(self.per_seed_grad_spike_rate),
            },
            "nan_inf_events": {
                "total": self.nan_inf_events_total,
                "mean": self.nan_inf_events_mean,
                "per_seed": list(self.per_seed_nan_inf_events),
            },
            "seeds": list(self.seeds),
            "instability_flag": self.instability_flag,
            "step0_anomaly": self.step0_anomaly,
            "seed_std_bpb": self.seed_std_bpb,
            "seed_std_heldout": self.seed_std_heldout,
            "device": self.device,
            "notes": list(self.notes),
            "status": "filled",
        }


def multi_seed_stability(
    per_seed: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
    spike_threshold: float = 0.0,
    step0_anomaly: bool = False,
    seed_std_bpb: float | None = None,
    seed_std_heldout: float | None = None,
    device: str = "fixture",
) -> StabilityMultiSeed:
    """Aggregate multi-seed grad_spike_rate + nan_inf_events (not schema-null).

    Each entry must expose ``grad_spike_rate`` (float in [0,1] or rate) and
    ``nan_inf_events`` (non-neg int). Missing keys fail closed rather than invent 0.
    """
    ordered_seeds: list[int]
    rows: list[Mapping[str, Any]]
    if isinstance(per_seed, Mapping):
        ordered_seeds = list(seeds) if seeds is not None else sorted(int(k) for k in per_seed)
        rows = []
        for s in ordered_seeds:
            if s not in per_seed and int(s) not in per_seed:
                raise KeyError(f"multi_seed_stability missing seed {s}")
            rows.append(per_seed[s] if s in per_seed else per_seed[int(s)])
    else:
        rows = list(per_seed)
        if not rows:
            raise ValueError("multi_seed_stability requires at least one seed row")
        if seeds is not None:
            ordered_seeds = [int(s) for s in seeds]
            if len(ordered_seeds) != len(rows):
                raise ValueError("seeds length must match per_seed sequence length")
        else:
            ordered_seeds = list(range(len(rows)))

    rates: list[float] = []
    nans: list[int] = []
    notes: list[str] = []
    for i, row in enumerate(rows):
        if "grad_spike_rate" not in row:
            raise ValueError(f"seed row {ordered_seeds[i]} missing grad_spike_rate")
        if "nan_inf_events" not in row:
            raise ValueError(f"seed row {ordered_seeds[i]} missing nan_inf_events")
        rate = _finite(row["grad_spike_rate"])
        if rate is None:
            raise ValueError(f"seed row {ordered_seeds[i]} non-finite grad_spike_rate")
        try:
            ne = int(row["nan_inf_events"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"seed row {ordered_seeds[i]} nan_inf_events must be int") from exc
        if ne < 0:
            raise ValueError(f"seed row {ordered_seeds[i]} nan_inf_events must be >= 0")
        rates.append(float(rate))
        nans.append(int(ne))

    mean_r, std_r = _mean_std(rates)
    assert mean_r is not None and std_r is not None
    total_nan = int(sum(nans))
    mean_nan = float(total_nan) / float(len(nans))
    instability = bool(
        step0_anomaly
        or total_nan > 0
        or any(r > float(spike_threshold) for r in rates if spike_threshold > 0)
        or any(r > 0.0 for r in rates)
        and spike_threshold == 0.0
        and total_nan > 0
    )
    # Prefer explicit instability if any rate > 0 when threshold is 0 and any nan,
    # but also flag pure high rates purely when threshold set. Additionally mark
    # instability if any nan OR any rate strictly above 0 when that is caller intent
    # via residual flag on row.
    if any(bool(row.get("instability_flag")) for row in rows):
        instability = True
        notes.append("per_seed_instability_flag")
    if total_nan > 0:
        instability = True
    # High mean rate alone does not force instability unless threshold exceeded.
    thr = float(spike_threshold)
    if thr > 0.0 and mean_r > thr:
        instability = True
        notes.append("mean_grad_spike_above_threshold")
    notes.append("multi_seed_grad_nan_filled")

    return StabilityMultiSeed(
        grad_spike_rate_mean=float(mean_r),
        grad_spike_rate_std=float(std_r),
        nan_inf_events_total=total_nan,
        nan_inf_events_mean=mean_nan,
        per_seed_grad_spike_rate=tuple(rates),
        per_seed_nan_inf_events=tuple(nans),
        seeds=tuple(int(s) for s in ordered_seeds),
        instability_flag=bool(instability),
        step0_anomaly=bool(step0_anomaly),
        seed_std_bpb=_finite(seed_std_bpb),
        seed_std_heldout=_finite(seed_std_heldout),
        device=device,
        notes=tuple(notes),
    )


def apply_stability_to_record(
    record: OfficialScoreRecord,
    suite: StabilityMultiSeed,
) -> OfficialScoreRecord:
    """Stamp multi-seed stability residuals onto OfficialScoreRecord."""
    return replace(
        record,
        grad_spike_rate=float(suite.grad_spike_rate_mean),
        nan_inf_events=int(suite.nan_inf_events_total),
        instability_flag=bool(suite.instability_flag or record.instability_flag),
        step0_anomaly=bool(suite.step0_anomaly or record.step0_anomaly),
        bpb_std=(suite.seed_std_bpb if suite.seed_std_bpb is not None else record.bpb_std),
        heldout_std=(
            suite.seed_std_heldout if suite.seed_std_heldout is not None else record.heldout_std
        ),
    )


# --- VAL-COMPLETE-011: multi-order residual + derived quality ratios ---------------


@dataclass(frozen=True)
class MultiOrderRobustness:
    """Alternate train-order / reshuffle residual (CV-MH-16) or BLOCKED honesty."""

    status: Literal["filled", "BLOCKED"]
    delta_primary: float | None = None
    order_a_primary: float | None = None
    order_b_primary: float | None = None
    primary_form: str = "heldout_delta"
    orders: tuple[str, ...] = ()
    reason: str | None = None
    device: str = "fixture"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "delta_primary": self.delta_primary,
            "order_a_primary": self.order_a_primary,
            "order_b_primary": self.order_b_primary,
            "primary_form": self.primary_form,
            "orders": list(self.orders),
            "reason": self.reason,
            "device": self.device,
            "notes": list(self.notes),
        }


def multi_order_residual(
    *,
    order_a_primary: float | None = None,
    order_b_primary: float | None = None,
    primary_form: str = "heldout_delta",
    orders: Sequence[str] = ("stream_order_0", "stream_order_1"),
    blocked: bool = False,
    blocked_reason: str | None = None,
    device: str = "fixture",
) -> MultiOrderRobustness:
    """Build multi-order residual or honest BLOCKED_with_reason (CV-MH-16)."""
    if blocked:
        reason = blocked_reason or (
            "BLOCKED_with_reason: multi_order stream residual infeasible "
            "(no alternate train-order / reshuffle capture for this remesure)"
        )
        return MultiOrderRobustness(
            status="BLOCKED",
            delta_primary=None,
            order_a_primary=None,
            order_b_primary=None,
            primary_form=primary_form,
            orders=tuple(str(o) for o in orders),
            reason=reason,
            device=device,
            notes=("honest_blocked",),
        )
    a = _finite(order_a_primary)
    b = _finite(order_b_primary)
    if a is None or b is None:
        return MultiOrderRobustness(
            status="BLOCKED",
            delta_primary=None,
            order_a_primary=a,
            order_b_primary=b,
            primary_form=primary_form,
            orders=tuple(str(o) for o in orders),
            reason=(
                blocked_reason or "BLOCKED_with_reason: multi_order primaries missing/non-finite"
            ),
            device=device,
            notes=("honest_blocked_missing_primaries",),
        )
    return MultiOrderRobustness(
        status="filled",
        delta_primary=float(abs(a - b)),
        order_a_primary=float(a),
        order_b_primary=float(b),
        primary_form=primary_form,
        orders=tuple(str(o) for o in orders),
        reason=None,
        device=device,
        notes=("multi_order_filled",),
    )


@dataclass(frozen=True)
class QualityEfficiencyRatios:
    """Derived quality_per_param + quality_per_gib (CV-MH-17); pure host derive."""

    quality_proxy: float
    quality_proxy_name: str
    params: int | None
    peak_vram_gib: float | None
    quality_per_param: float | None
    quality_per_gib: float | None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality_proxy": self.quality_proxy,
            "quality_proxy_name": self.quality_proxy_name,
            "params": self.params,
            "peak_vram_gib": self.peak_vram_gib,
            "quality_per_param": self.quality_per_param,
            "quality_per_gib": self.quality_per_gib,
            "notes": list(self.notes),
            "status": "filled",
            "sole_rank_forbidden": True,
        }


def derive_quality_efficiency_ratios(
    *,
    heldout_delta: float | None = None,
    sample_eff_auc: float | None = None,
    val_bpb_trained: float | None = None,
    bpb: float | None = None,
    params: int | None = None,
    peak_vram_gib: float | None = None,
    prefer: QualityProxy = "heldout_delta",
) -> QualityEfficiencyRatios:
    """Derive quality_per_param / quality_per_gib without inventing denominators."""
    notes: list[str] = []
    proxy: float | None = None
    proxy_name = str(prefer)

    def _inv_pos(x: float | None, name: str) -> float | None:
        fx = _finite(x)
        if fx is None or fx <= 0:
            return None
        return 1.0 / fx

    candidates: list[tuple[str, float | None]] = [
        ("heldout_delta", _finite(heldout_delta)),
        ("quality_auc", _finite(sample_eff_auc)),
        ("inv_val_bpb", _inv_pos(val_bpb_trained, "val_bpb")),
        ("inv_bpb", _inv_pos(bpb, "bpb")),
    ]
    by_name = {n: v for n, v in candidates}
    if prefer in by_name and by_name[prefer] is not None:
        proxy = by_name[prefer]
        proxy_name = prefer
    else:
        for name, val in candidates:
            if val is not None:
                proxy = val
                proxy_name = name
                notes.append(f"fell_back_from_{prefer}_to_{name}")
                break
    if proxy is None:
        raise ValueError("derive_quality_efficiency_ratios requires a finite quality proxy")

    qpp: float | None = None
    if params is not None and int(params) > 0:
        qpp = float(proxy) / float(params)
    else:
        notes.append("quality_per_param_null_missing_params")

    qpg: float | None = None
    vram = _finite(peak_vram_gib)
    if vram is not None and vram > 0:
        qpg = float(proxy) / float(vram)
    else:
        notes.append("quality_per_gib_null_missing_vram")

    notes.append("derived_host_side")
    return QualityEfficiencyRatios(
        quality_proxy=float(proxy),
        quality_proxy_name=proxy_name,
        params=None if params is None else int(params),
        peak_vram_gib=vram,
        quality_per_param=qpp,
        quality_per_gib=qpg,
        notes=tuple(notes),
    )


# --- VAL-COMPLETE-012: nice-to-have residual panel --------------------------------


def build_nice_to_have_panel(
    *,
    filled: Mapping[str, Mapping[str, Any] | None] | None = None,
    default_reason: str = "not_run_nice_to_have",
) -> dict[str, Any]:
    """Build P8 nice-to-have entries with explicit null+reason (no silent omission).

    ``filled`` maps metric ``key`` → optional side-bearing mapping
    ``{"a": ..., "b": ..., "reason": optional}``. Unlisted keys stay null + reason.
    """
    filled_map = dict(filled or {})
    entries: list[dict[str, Any]] = []
    for row in COMPLETE_VIEW_NICE_TO_HAVE:
        key = str(row["key"])
        payload = filled_map.get(key)
        if payload is None:
            entries.append(
                {
                    "matrix_id": row["matrix_id"],
                    "key": key,
                    "a": None,
                    "b": None,
                    "status": "not_run",
                    "reason": default_reason,
                }
            )
            continue
        a_val = payload.get("a") if isinstance(payload, Mapping) else None
        b_val = payload.get("b") if isinstance(payload, Mapping) else None
        reason = None
        status = "filled"
        if isinstance(payload, Mapping):
            reason = payload.get("reason")
            if payload.get("status") is not None:
                status = str(payload.get("status"))
            elif a_val is None and b_val is None:
                status = "not_run"
                reason = reason or default_reason
        entries.append(
            {
                "matrix_id": row["matrix_id"],
                "key": key,
                "a": a_val,
                "b": b_val,
                "status": status,
                "reason": reason,
            }
        )
    # Guard: every catalogue nice-to-have key appears (no silent omission).
    cataloque_keys = {str(r["key"]) for r in COMPLETE_VIEW_NICE_TO_HAVE}
    present = {e["key"] for e in entries}
    if present != cataloque_keys:
        missing = sorted(cataloque_keys - present)
        raise RuntimeError(f"nice_to_have silent omission of keys: {missing}")
    return {
        "status": "nice_to_have",
        "no_silent_omission": True,
        "entries": entries,
    }


def rapid_decay_flag_from_online(
    online_loss: Sequence[float],
    *,
    rebound_frac: float = 0.10,
    min_samples: int = 16,
) -> dict[str, Any]:
    """Cheap rapid-decay residual from dense online curve (CV-NH-04 fixture/hook)."""
    clean = [float(x) for x in online_loss if math.isfinite(float(x))]
    if len(clean) < min_samples:
        return {
            "flag": None,
            "status": "not_run",
            "reason": "online_stream_too_short_for_rapid_decay",
            "min_loss": None,
            "end_loss": None,
        }
    min_loss = min(clean)
    end_loss = clean[-1]
    # Rebound if late loss risen above min by rebound_frac relative.
    denom = max(abs(min_loss), 1e-9)
    rebound = (end_loss - min_loss) / denom
    flag = bool(rebound >= float(rebound_frac) and end_loss > min_loss)
    return {
        "flag": flag,
        "status": "filled",
        "reason": None,
        "min_loss": float(min_loss),
        "end_loss": float(end_loss),
        "rebound_frac": float(rebound),
        "threshold": float(rebound_frac),
    }


# --- Combined family efficiency / residual bundle ---------------------------------


@dataclass(frozen=True)
class FamilyEffStability:
    """One-family Complete View efficiency + stability + robustness payload."""

    sample_eff: DenseSampleEffResult | None = None
    efficiency: TrainEvalEfficiency | None = None
    state_footprint: StateFootprintResult | None = None
    stability: StabilityMultiSeed | None = None
    multi_order: MultiOrderRobustness | None = None
    quality_ratios: QualityEfficiencyRatios | None = None
    nice_to_have_side: Mapping[str, Any] | None = None
    device: str = "fixture"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_eff": None if self.sample_eff is None else self.sample_eff.as_dict(),
            "efficiency": None if self.efficiency is None else self.efficiency.as_dict(),
            "state_footprint": (
                None if self.state_footprint is None else self.state_footprint.as_dict()
            ),
            "stability": None if self.stability is None else self.stability.as_dict(),
            "multi_order": None if self.multi_order is None else self.multi_order.as_dict(),
            "quality_ratios": (
                None if self.quality_ratios is None else self.quality_ratios.as_dict()
            ),
            "nice_to_have_side": (
                None if self.nice_to_have_side is None else dict(self.nice_to_have_side)
            ),
            "device": self.device,
        }


def build_eff_stability_panels(
    fam_a: FamilyEffStability | None,
    fam_b: FamilyEffStability | None,
    *,
    a_record: OfficialScoreRecord | None = None,
    b_record: OfficialScoreRecord | None = None,
    nice_to_have_filled: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Panel overrides for P2 / P5 / P6 / P7 / P8 from family efficiency suites."""
    a = fam_a or FamilyEffStability()
    b = fam_b or FamilyEffStability()

    # --- P2 sample efficiency dense ---
    se_filled = a.sample_eff is not None and b.sample_eff is not None
    marks_union: list[int] = list(DENSE_SAMPLE_EFF_MARKS_TOKENS)
    if se_filled:
        assert a.sample_eff is not None and b.sample_eff is not None
        marks_union = sorted(set(a.sample_eff.marks_tokens) | set(b.sample_eff.marks_tokens))
    # Prefer dense AUC when present, fall back to record sample_eff_auc.
    quality_auc = {
        "a": (
            a.sample_eff.auc
            if a.sample_eff is not None
            else (None if a_record is None else a_record.sample_eff_auc)
        ),
        "b": (
            b.sample_eff.auc
            if b.sample_eff is not None
            else (None if b_record is None else b_record.sample_eff_auc)
        ),
    }
    bpb_at_marks = {
        "a": (
            list(a.sample_eff.bpb_at_marks)
            if a.sample_eff is not None
            else (
                list(a_record.sample_eff_marks)
                if a_record is not None and a_record.sample_eff_marks
                else None
            )
        ),
        "b": (
            list(b.sample_eff.bpb_at_marks)
            if b.sample_eff is not None
            else (
                list(b_record.sample_eff_marks)
                if b_record is not None and b_record.sample_eff_marks
                else None
            )
        ),
    }
    heldout_filled = False
    held_a = None if a.sample_eff is None else a.sample_eff.heldout_at_marks
    held_b = None if b.sample_eff is None else b.sample_eff.heldout_at_marks
    if (
        isinstance(held_a, Mapping)
        and isinstance(held_b, Mapping)
        and held_a.get("status") == "filled"
        and held_b.get("status") == "filled"
    ):
        heldout_filled = True

    p2: dict[str, Any] = {
        "marks_tokens": marks_union,
        "quality_auc": quality_auc,
        "bpb_at_marks": bpb_at_marks,
        "dense_marks": _side_fill(
            None if a.sample_eff is None else a.sample_eff.as_dict(),
            None if b.sample_eff is None else b.sample_eff.as_dict(),
            status_filled=se_filled,
            reason_if_null="dense_sample_eff_pending_VAL-COMPLETE-008",
        ),
        "curve_summary": _side_fill(
            None if a.sample_eff is None else a.sample_eff.curve_summary,
            None if b.sample_eff is None else b.sample_eff.curve_summary,
            status_filled=se_filled,
            reason_if_null="dense_sample_eff_pending_VAL-COMPLETE-008",
        ),
        "heldout_at_marks": _side_fill(
            held_a,
            held_b,
            status_filled=heldout_filled,
            reason_if_null=(
                "heldout_checkpoint_curve_not_available; host_stream marks documented"
                if se_filled
                else "pending_VAL-COMPLETE-008"
            ),
        ),
        "sample_eff_dense": _side_fill(
            None if a.sample_eff is None else a.sample_eff.as_dict(),
            None if b.sample_eff is None else b.sample_eff.as_dict(),
            status_filled=se_filled,
            reason_if_null="dense_sample_eff_pending_VAL-COMPLETE-008",
        ),
    }

    # --- P5 efficiency + derived ratios ---
    eff_filled = a.efficiency is not None and b.efficiency is not None
    q_filled = a.quality_ratios is not None and b.quality_ratios is not None
    # Auto-derive ratios from efficiency + records when not provided.
    ratios_a = a.quality_ratios
    ratios_b = b.quality_ratios
    if ratios_a is None and (a.efficiency is not None or a_record is not None):
        try:
            ratios_a = derive_quality_efficiency_ratios(
                heldout_delta=None if a_record is None else a_record.heldout_delta,
                sample_eff_auc=(
                    a.sample_eff.auc
                    if a.sample_eff is not None
                    else (None if a_record is None else a_record.sample_eff_auc)
                ),
                val_bpb_trained=None if a_record is None else a_record.val_bpb_trained,
                bpb=None if a_record is None else a_record.bpb,
                params=(
                    a.efficiency.params
                    if a.efficiency is not None and a.efficiency.params is not None
                    else (None if a_record is None else a_record.params)
                ),
                peak_vram_gib=(
                    a.efficiency.peak_vram_train_gib
                    if a.efficiency is not None
                    else (None if a_record is None else a_record.peak_vram_gib)
                ),
            )
        except ValueError:
            ratios_a = None
    if ratios_b is None and (b.efficiency is not None or b_record is not None):
        try:
            ratios_b = derive_quality_efficiency_ratios(
                heldout_delta=None if b_record is None else b_record.heldout_delta,
                sample_eff_auc=(
                    b.sample_eff.auc
                    if b.sample_eff is not None
                    else (None if b_record is None else b_record.sample_eff_auc)
                ),
                val_bpb_trained=None if b_record is None else b_record.val_bpb_trained,
                bpb=None if b_record is None else b_record.bpb,
                params=(
                    b.efficiency.params
                    if b.efficiency is not None and b.efficiency.params is not None
                    else (None if b_record is None else b_record.params)
                ),
                peak_vram_gib=(
                    b.efficiency.peak_vram_train_gib
                    if b.efficiency is not None
                    else (None if b_record is None else b_record.peak_vram_gib)
                ),
            )
        except ValueError:
            ratios_b = None
    q_filled = ratios_a is not None and ratios_b is not None

    params_side = {
        "a": (
            a.efficiency.params
            if a.efficiency is not None and a.efficiency.params is not None
            else (None if a_record is None else a_record.params)
        ),
        "b": (
            b.efficiency.params
            if b.efficiency is not None and b.efficiency.params is not None
            else (None if b_record is None else b_record.params)
        ),
    }
    p5: dict[str, Any] = {
        "params": params_side,
        "peak_vram_train_gib": {
            "a": (
                a.efficiency.peak_vram_train_gib
                if a.efficiency is not None
                else (None if a_record is None else a_record.peak_vram_gib)
            ),
            "b": (
                b.efficiency.peak_vram_train_gib
                if b.efficiency is not None
                else (None if b_record is None else b_record.peak_vram_gib)
            ),
        },
        "tokens_per_s_train": {
            "a": (
                a.efficiency.tokens_per_s_train
                if a.efficiency is not None
                else (None if a_record is None else a_record.tokens_per_s)
            ),
            "b": (
                b.efficiency.tokens_per_s_train
                if b.efficiency is not None
                else (None if b_record is None else b_record.tokens_per_s)
            ),
        },
        "peak_vram_eval_by_T": _side_fill(
            None if a.efficiency is None else a.efficiency.peak_vram_eval_by_T,
            None if b.efficiency is None else b.efficiency.peak_vram_eval_by_T,
            status_filled=eff_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
        "tokens_per_s_eval_by_T": _side_fill(
            None if a.efficiency is None else a.efficiency.tokens_per_s_eval_by_T,
            None if b.efficiency is None else b.efficiency.tokens_per_s_eval_by_T,
            status_filled=eff_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
        "step_time_ms": _side_fill(
            None if a.efficiency is None else a.efficiency.step_time_ms,
            None if b.efficiency is None else b.efficiency.step_time_ms,
            status_filled=eff_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
        "train_eval_efficiency": _side_fill(
            None if a.efficiency is None else a.efficiency.as_dict(),
            None if b.efficiency is None else b.efficiency.as_dict(),
            status_filled=eff_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
        "quality_per_param": _side_fill(
            None if ratios_a is None else ratios_a.quality_per_param,
            None if ratios_b is None else ratios_b.quality_per_param,
            status_filled=q_filled,
            reason_if_null="pending_VAL-COMPLETE-011",
        ),
        "quality_per_gib": _side_fill(
            None if ratios_a is None else ratios_a.quality_per_gib,
            None if ratios_b is None else ratios_b.quality_per_gib,
            status_filled=q_filled,
            reason_if_null="pending_VAL-COMPLETE-011",
        ),
        "quality_efficiency_ratios": _side_fill(
            None if ratios_a is None else ratios_a.as_dict(),
            None if ratios_b is None else ratios_b.as_dict(),
            status_filled=q_filled,
            reason_if_null="pending_VAL-COMPLETE-011",
        ),
        "sole_rank_forbidden": True,
        "overrides_polar_rule": False,
    }

    # --- P6 memory / state footprint ---
    sp_filled = a.state_footprint is not None and b.state_footprint is not None
    p6: dict[str, Any] = {
        "state_footprint_bytes_by_T": _side_fill(
            None if a.state_footprint is None else a.state_footprint.state_bytes_by_T,
            None if b.state_footprint is None else b.state_footprint.state_bytes_by_T,
            status_filled=sp_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
        "activation_peak_bytes_by_T": _side_fill(
            None if a.state_footprint is None else a.state_footprint.activation_peak_bytes_by_T,
            None if b.state_footprint is None else b.state_footprint.activation_peak_bytes_by_T,
            status_filled=sp_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
        "state_footprint": _side_fill(
            None if a.state_footprint is None else a.state_footprint.as_dict(),
            None if b.state_footprint is None else b.state_footprint.as_dict(),
            status_filled=sp_filled,
            reason_if_null="pending_VAL-COMPLETE-009",
        ),
    }

    # --- P7 stability + multi-order ---
    stab_filled = a.stability is not None and b.stability is not None
    # multi_order present when both sides have an object (filled OR BLOCKED honesty).
    mo_present = a.multi_order is not None and b.multi_order is not None

    grad_a: Any
    grad_b: Any
    nan_a: Any
    nan_b: Any
    if a.stability is not None:
        grad_a = {
            "mean": a.stability.grad_spike_rate_mean,
            "std": a.stability.grad_spike_rate_std,
            "per_seed": list(a.stability.per_seed_grad_spike_rate),
        }
        nan_a = {
            "total": a.stability.nan_inf_events_total,
            "mean": a.stability.nan_inf_events_mean,
            "per_seed": list(a.stability.per_seed_nan_inf_events),
        }
    else:
        grad_a = None if a_record is None else a_record.grad_spike_rate
        nan_a = None if a_record is None else a_record.nan_inf_events
    if b.stability is not None:
        grad_b = {
            "mean": b.stability.grad_spike_rate_mean,
            "std": b.stability.grad_spike_rate_std,
            "per_seed": list(b.stability.per_seed_grad_spike_rate),
        }
        nan_b = {
            "total": b.stability.nan_inf_events_total,
            "mean": b.stability.nan_inf_events_mean,
            "per_seed": list(b.stability.per_seed_nan_inf_events),
        }
    else:
        grad_b = None if b_record is None else b_record.grad_spike_rate
        nan_b = None if b_record is None else b_record.nan_inf_events

    # Both multi-order BLOCKED is still a valid residual (not pending null shell).
    mo_status: str = "not_run"
    mo_reason: str | None = "pending_VAL-COMPLETE-011"
    if mo_present:
        assert a.multi_order is not None and b.multi_order is not None
        if a.multi_order.status == "filled" and b.multi_order.status == "filled":
            mo_status = "filled"
            mo_reason = None
        elif a.multi_order.status == "BLOCKED" or b.multi_order.status == "BLOCKED":
            mo_status = "BLOCKED"
            # Prefer first explicit reason.
            mo_reason = a.multi_order.reason or b.multi_order.reason
        else:
            mo_status = a.multi_order.status
            mo_reason = a.multi_order.reason or b.multi_order.reason

    p7: dict[str, Any] = {
        "grad_spike_rate": {
            "a": grad_a,
            "b": grad_b,
            "status": "filled" if stab_filled else "pending_VAL-COMPLETE-010",
        },
        "nan_inf_events": {
            "a": nan_a,
            "b": nan_b,
            "status": "filled" if stab_filled else "pending_VAL-COMPLETE-010",
        },
        "instability_flag": {
            "a": (
                a.stability.instability_flag
                if a.stability is not None
                else (None if a_record is None else a_record.instability_flag)
            ),
            "b": (
                b.stability.instability_flag
                if b.stability is not None
                else (None if b_record is None else b_record.instability_flag)
            ),
        },
        "step0_anomaly": {
            "a": (
                a.stability.step0_anomaly
                if a.stability is not None
                else (None if a_record is None else a_record.step0_anomaly)
            ),
            "b": (
                b.stability.step0_anomaly
                if b.stability is not None
                else (None if b_record is None else b_record.step0_anomaly)
            ),
        },
        "seed_std_bpb": {
            "a": (
                a.stability.seed_std_bpb
                if a.stability is not None and a.stability.seed_std_bpb is not None
                else (None if a_record is None else a_record.bpb_std)
            ),
            "b": (
                b.stability.seed_std_bpb
                if b.stability is not None and b.stability.seed_std_bpb is not None
                else (None if b_record is None else b_record.bpb_std)
            ),
        },
        "seed_std_heldout": {
            "a": (
                a.stability.seed_std_heldout
                if a.stability is not None and a.stability.seed_std_heldout is not None
                else (None if a_record is None else a_record.heldout_std)
            ),
            "b": (
                b.stability.seed_std_heldout
                if b.stability is not None and b.stability.seed_std_heldout is not None
                else (None if b_record is None else b_record.heldout_std)
            ),
        },
        "multi_order_delta": {
            "a": None if a.multi_order is None else a.multi_order.as_dict(),
            "b": None if b.multi_order is None else b.multi_order.as_dict(),
            "status": mo_status,
            "reason": mo_reason,
        },
        "stability_multi_seed": _side_fill(
            None if a.stability is None else a.stability.as_dict(),
            None if b.stability is None else b.stability.as_dict(),
            status_filled=stab_filled,
            reason_if_null="pending_VAL-COMPLETE-010",
        ),
    }

    # --- P8 nice-to-have ---
    # Merge side-specific snapshot keys from families into filled map.
    filled_nh: dict[str, Mapping[str, Any] | None] = dict(nice_to_have_filled or {})
    if a.nice_to_have_side or b.nice_to_have_side:
        side_a = dict(a.nice_to_have_side or {})
        side_b = dict(b.nice_to_have_side or {})
        for key in set(side_a) | set(side_b):
            if key in filled_nh and filled_nh[key] is not None:
                continue
            filled_nh[key] = {
                "a": side_a.get(key),
                "b": side_b.get(key),
                "status": (
                    "filled"
                    if (side_a.get(key) is not None or side_b.get(key) is not None)
                    else "not_run"
                ),
                "reason": (
                    None
                    if (side_a.get(key) is not None or side_b.get(key) is not None)
                    else "not_run_nice_to_have"
                ),
            }
    p8 = build_nice_to_have_panel(filled=filled_nh)

    panels: dict[str, Any] = {}
    if "P2_sample_efficiency" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P2_sample_efficiency"] = p2
    if "P5_efficiency" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P5_efficiency"] = p5
    if "P6_memory_state" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P6_memory_state"] = p6
    if "P7_stability_robustness" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P7_stability_robustness"] = p7
    if "P8_calibration_entropy_optional" in COMPLETE_VIEW_PANEL_KEYS:
        panels["P8_calibration_entropy_optional"] = p8
    return panels


def apply_eff_stability_to_record(
    record: OfficialScoreRecord,
    fam: FamilyEffStability,
) -> OfficialScoreRecord:
    """Stamp densified sample-eff + train efficiency + stability onto a record."""
    out = record
    if fam.sample_eff is not None:
        out = apply_dense_sample_eff_to_record(out, fam.sample_eff)
    if fam.efficiency is not None:
        out = apply_train_eval_efficiency_to_record(out, fam.efficiency)
    if fam.stability is not None:
        out = apply_stability_to_record(out, fam.stability)
    return out


def build_complete_view_with_eff_stability(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    fam_a: FamilyEffStability | None = None,
    fam_b: FamilyEffStability | None = None,
    nice_to_have_filled: Mapping[str, Mapping[str, Any] | None] | None = None,
    panels_override: Mapping[str, Any] | None = None,
    score_class: str = "fixture",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build complete_view.v1.2 with efficiency/stability/robustness/nice panels filled."""
    rec_a = a if fam_a is None else apply_eff_stability_to_record(a, fam_a)
    rec_b = b if fam_b is None else apply_eff_stability_to_record(b, fam_b)
    overrides = build_eff_stability_panels(
        fam_a,
        fam_b,
        a_record=rec_a,
        b_record=rec_b,
        nice_to_have_filled=nice_to_have_filled,
    )
    if panels_override:
        for key, value in panels_override.items():
            if key not in COMPLETE_VIEW_PANEL_KEYS:
                raise ValueError(f"unknown complete_view panel key: {key}")
            if isinstance(value, Mapping) and isinstance(overrides.get(key), dict):
                overrides[key] = {**overrides[key], **dict(value)}
            else:
                overrides[key] = value
    return build_complete_view(
        rec_a,
        rec_b,
        panels_override=overrides,
        score_class=score_class,
        **kwargs,
    )


def fixture_family_eff_stability(
    *,
    online_loss: Sequence[float],
    params: int,
    peak_vram_train_gib: float,
    tokens_per_s_train: float,
    peak_vram_eval_by_T: Mapping[int | str, float],
    tokens_per_s_eval_by_T: Mapping[int | str, float],
    state_bytes_by_T: Mapping[int | str, float],
    activation_peak_bytes_by_T: Mapping[int | str, float],
    stability_per_seed: Mapping[int, Mapping[str, Any]],
    heldout_delta: float,
    sample_eff_auc_hint: float | None = None,
    heldout_at_marks: Mapping[int | str, float] | None = None,
    multi_order_blocked: bool = False,
    multi_order_a: float | None = None,
    multi_order_b: float | None = None,
    step_time_ms_mean: float | None = 10.0,
    step_time_ms_p99: float | None = 20.0,
    seed_std_bpb: float | None = 0.01,
    seed_std_heldout: float | None = 0.02,
    wall_clock_train_s: float | None = 15.0,
    token_budget: int = 500_000,
    nice_side: Mapping[str, Any] | None = None,
    device: str = "fixture",
) -> FamilyEffStability:
    """One-shot fixture builder covering VAL-COMPLETE-008..012 for a family."""
    sample = dense_sample_efficiency_from_stream(
        online_loss,
        token_budget=token_budget,
        heldout_at_marks=heldout_at_marks,
        device=device,
    )
    # Allow callers to force a known AUC only if they pass explicit hint by
    # rebuilding is unnecessary; stream drives AUC (host densify honesty).
    _ = sample_eff_auc_hint
    efficiency = build_train_eval_efficiency(
        params=params,
        peak_vram_train_gib=peak_vram_train_gib,
        tokens_per_s_train=tokens_per_s_train,
        wall_clock_train_s=wall_clock_train_s,
        peak_vram_eval_by_T=peak_vram_eval_by_T,
        tokens_per_s_eval_by_T=tokens_per_s_eval_by_T,
        step_time_ms_mean=step_time_ms_mean,
        step_time_ms_p99=step_time_ms_p99,
        device=device,
    )
    footprint = build_state_footprint(
        state_bytes_by_T=state_bytes_by_T,
        activation_peak_bytes_by_T=activation_peak_bytes_by_T,
        params=params,
        device=device,
    )
    stability = multi_seed_stability(
        stability_per_seed,
        seed_std_bpb=seed_std_bpb,
        seed_std_heldout=seed_std_heldout,
        device=device,
    )
    if multi_order_blocked:
        multi = multi_order_residual(blocked=True, device=device)
    else:
        multi = multi_order_residual(
            order_a_primary=multi_order_a if multi_order_a is not None else heldout_delta,
            order_b_primary=(
                multi_order_b if multi_order_b is not None else float(heldout_delta) * 0.98
            ),
            device=device,
        )
    ratios = derive_quality_efficiency_ratios(
        heldout_delta=heldout_delta,
        sample_eff_auc=sample.auc,
        params=params,
        peak_vram_gib=peak_vram_train_gib,
    )
    side_nh: dict[str, Any] = dict(nice_side or {})
    # Always publish a cheap rapid-decay residual from the same stream (optionally).
    if "rapid_decay_flag" not in side_nh:
        side_nh["rapid_decay_flag"] = rapid_decay_flag_from_online(online_loss)
    return FamilyEffStability(
        sample_eff=sample,
        efficiency=efficiency,
        state_footprint=footprint,
        stability=stability,
        multi_order=multi,
        quality_ratios=ratios,
        nice_to_have_side=side_nh,
        device=device,
    )


__all__ = [
    "DENSE_SAMPLE_EFF_MARKS_TOKENS",
    "DEFAULT_EVAL_TS",
    "DEFAULT_STATE_FOOTPRINT_TS",
    "DenseSampleEffResult",
    "FamilyEffStability",
    "MultiOrderRobustness",
    "QualityEfficiencyRatios",
    "StateFootprintResult",
    "StabilityMultiSeed",
    "TrainEvalEfficiency",
    "apply_dense_sample_eff_to_record",
    "apply_eff_stability_to_record",
    "apply_stability_to_record",
    "apply_train_eval_efficiency_to_record",
    "build_complete_view_with_eff_stability",
    "build_eff_stability_panels",
    "build_nice_to_have_panel",
    "build_state_footprint",
    "build_train_eval_efficiency",
    "dense_sample_efficiency_from_stream",
    "derive_quality_efficiency_ratios",
    "fixture_family_eff_stability",
    "multi_order_residual",
    "multi_seed_stability",
    "online_bpb_curve_summary",
    "quality_from_bpb",
    "rapid_decay_flag_from_online",
]
