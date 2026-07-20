"""Complete View P10 reasoning/logic panel fill from synthetic probe scores.

Wires challenge-owned logic suite metrics (VAL-REASON-002..011) into the
``P10_reasoning_logic`` panel shell defined by ``complete_view.reasoning_panel_shell``.

This module:

* scores MUST probes via fixture outcomes or host densify hooks
* dual-channels closed accuracy + forced CE with published chance baselines
* fills per-probe fields on a one-family or dual-family panel
* aggregates suite_mean (macro) with relative-to-chance floors (VAL-REASON-009)
* expands multi-axis comparison with ``reasoning`` lead + TIE_POLAR vs short_gen
* nice-to-have residuals filled OR explicit null+reason (VAL-REASON-011)

Architecture-agnostic. Prefer pure-torch CPU unit paths. No lm-eval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .benchmarks.logic_suite import (
    DEFAULT_LOGIC_SUITE_SEED,
    DEFAULT_TRIALS_PER_PROBE,
    LOGIC_CHANCE,
    LOGIC_PROBE_KEYS,
    LogicTaskScore,
    documented_logic_suite,
    score_suite_fixture,
)
from .complete_view import (
    COMPLETE_VIEW_NICE_TO_HAVE,
    COMPLETE_VIEW_REASONING_CHANCE_TABLE,
    REASONING_REL_FLOOR,
    REASONING_SUITE_ID,
    CompleteAxisScore,
    build_complete_view,
    compare_complete_multi_axis,
    default_axis_scores_from_records,
    reasoning_panel_shell,
)
from .official_comparison import OfficialScoreRecord
from .scorecard_suite import relative_to_chance

DeviceHint = Literal["cpu", "cuda", "auto", "fixture"]

# Nice-to-have P10 residual keys (VAL-REASON-011 catalogue).
REASONING_NICE_KEYS: tuple[str, ...] = tuple(
    str(row["key"])
    for row in COMPLETE_VIEW_NICE_TO_HAVE
    if str(row.get("panel")) == "P10_reasoning_logic"
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


@dataclass(frozen=True)
class FamilyReasoningScores:
    """One-family P10 probe metric bundle (acc + relative + forced CE)."""

    probes: dict[str, LogicTaskScore]
    device: str = "fixture"
    seeds: tuple[int, ...] | None = None
    suite_id: str = REASONING_SUITE_ID

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "device": self.device,
            "seeds": None if self.seeds is None else list(self.seeds),
            "probes": {k: v.as_dict() for k, v in self.probes.items()},
        }

    def accuracy_map(self) -> dict[str, float]:
        return {k: float(v.accuracy) for k, v in self.probes.items()}

    def relative_map(self) -> dict[str, float]:
        return {k: float(v.relative) for k, v in self.probes.items()}

    def forced_ce_map(self) -> dict[str, float | None]:
        return {k: v.forced_ce for k, v in self.probes.items()}


def family_reasoning_from_scores(
    scores: Mapping[str, LogicTaskScore],
    *,
    device: str = "fixture",
    seeds: Sequence[int] | None = None,
) -> FamilyReasoningScores:
    """Pack an existing probe→score map into a family payload."""
    probes: dict[str, LogicTaskScore] = {}
    for key in LOGIC_PROBE_KEYS:
        if key not in scores:
            raise KeyError(f"missing MUST probe score for {key}")
        probes[key] = scores[key]
    return FamilyReasoningScores(
        probes=probes,
        device=device,
        seeds=None if seeds is None else tuple(int(s) for s in seeds),
    )


def family_reasoning_fixture(
    *,
    accuracy_by_probe: Mapping[str, float] | None = None,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    device: str = "fixture",
    seeds: Sequence[int] | None = None,
) -> FamilyReasoningScores:
    """CPU fixture path for one family (dual channel metrics for all MUST probes)."""
    scores = score_suite_fixture(
        n_trials=n_trials,
        suite_seed=suite_seed,
        accuracy_by_probe=accuracy_by_probe,
        device=device,
    )
    return family_reasoning_from_scores(scores, device=device, seeds=seeds)


def _probe_entry_from_score(
    score: LogicTaskScore, *, side_status: str = "filled"
) -> dict[str, Any]:
    return {
        "accuracy": score.accuracy,
        "relative_to_chance": score.relative,
        "forced_ce": score.forced_ce,
        "chance": score.chance,
        "trials": score.trials,
        "device": score.device,
        "status": side_status,
        "detail": dict(score.detail) if score.detail is not None else None,
    }


# --- Suite aggregation (VAL-REASON-009) -------------------------------------------


@dataclass(frozen=True)
class ReasonSuiteAggregate:
    """Macro suite_mean + relative floors for one family (logic_synthetic.v1)."""

    suite_mean: float | None
    logic_acc_macro: float | None
    logic_rel_macro: float | None
    logic_ce_macro: float | None
    logic_below_chance_count: int | None
    logic_floor_pass: bool | None
    n_probes: int = 0
    rel_floor: float = REASONING_REL_FLOOR
    status: str = "not_run"
    reason: str | None = "not_run_until_probe_suite_aggregate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_mean": self.suite_mean,
            "logic_acc_macro": self.logic_acc_macro,
            "logic_rel_macro": self.logic_rel_macro,
            "logic_ce_macro": self.logic_ce_macro,
            "logic_below_chance_count": self.logic_below_chance_count,
            "logic_floor_pass": self.logic_floor_pass,
            "n_probes": self.n_probes,
            "rel_floor": self.rel_floor,
            "status": self.status,
            "reason": self.reason,
        }


def aggregate_reasoning_suite(
    family: FamilyReasoningScores | Mapping[str, LogicTaskScore] | None,
    *,
    rel_floor: float = REASONING_REL_FLOOR,
) -> ReasonSuiteAggregate:
    """Macro-mean over MUST probes with relative-to-chance floor discipline.

    * ``suite_mean`` / ``logic_acc_macro`` = mean closed accuracy (all present probes)
    * ``logic_rel_macro`` = mean relative_to_chance (higher = better; multi-axis scalar)
    * ``logic_ce_macro`` = mean forced CE when present (lower better)
    * ``logic_floor_pass`` = True when every probe relative ≥ rel_floor and macros exist
    """
    if family is None:
        return ReasonSuiteAggregate(
            suite_mean=None,
            logic_acc_macro=None,
            logic_rel_macro=None,
            logic_ce_macro=None,
            logic_below_chance_count=None,
            logic_floor_pass=None,
            n_probes=0,
            rel_floor=float(rel_floor),
            status="not_run",
            reason="not_run_until_probe_suite_aggregate",
        )
    probes: Mapping[str, LogicTaskScore]
    if isinstance(family, FamilyReasoningScores):
        probes = family.probes
    else:
        probes = family

    accs: list[float] = []
    rels: list[float] = []
    ces: list[float] = []
    below = 0
    for key in LOGIC_PROBE_KEYS:
        score = probes.get(key)
        if score is None:
            continue
        acc = _finite(score.accuracy)
        rel = _finite(score.relative)
        if acc is None or rel is None:
            continue
        accs.append(acc)
        rels.append(rel)
        if acc < float(score.chance) - 1e-12:
            below += 1
        ce = _finite(score.forced_ce)
        if ce is not None:
            ces.append(ce)

    if not accs:
        return ReasonSuiteAggregate(
            suite_mean=None,
            logic_acc_macro=None,
            logic_rel_macro=None,
            logic_ce_macro=None,
            logic_below_chance_count=None,
            logic_floor_pass=None,
            n_probes=0,
            rel_floor=float(rel_floor),
            status="not_run",
            reason="no_probe_scores_to_aggregate",
        )

    acc_macro = float(sum(accs) / len(accs))
    rel_macro = float(sum(rels) / len(rels))
    ce_macro = float(sum(ces) / len(ces)) if ces else None
    # Floor discipline: every probe relative ≥ soft floor, and all MUST present.
    all_must = all(k in probes for k in LOGIC_PROBE_KEYS)
    floor_ok = all_must and all(r >= float(rel_floor) for r in rels)
    return ReasonSuiteAggregate(
        suite_mean=acc_macro,
        logic_acc_macro=acc_macro,
        logic_rel_macro=rel_macro,
        logic_ce_macro=ce_macro,
        logic_below_chance_count=below,
        logic_floor_pass=floor_ok,
        n_probes=len(accs),
        rel_floor=float(rel_floor),
        status="filled",
        reason=None,
    )


def suite_aggregate_side_map(
    a: FamilyReasoningScores | None,
    b: FamilyReasoningScores | None,
    *,
    rel_floor: float = REASONING_REL_FLOOR,
) -> dict[str, Any]:
    """Dual-side aggregates object for P10.aggregates (VAL-REASON-009)."""
    ag_a = aggregate_reasoning_suite(a, rel_floor=rel_floor)
    ag_b = aggregate_reasoning_suite(b, rel_floor=rel_floor)
    any_filled = ag_a.status == "filled" or ag_b.status == "filled"
    both_filled = ag_a.status == "filled" and ag_b.status == "filled"
    if both_filled:
        status = "filled"
        reason: str | None = None
    elif any_filled:
        status = "partial"
        reason = "suite_mean_partial_one_family_only"
    else:
        status = "not_run"
        reason = "not_run_until_probe_suite_aggregate"
    return {
        "logic_acc_macro": {"a": ag_a.logic_acc_macro, "b": ag_b.logic_acc_macro},
        "logic_rel_macro": {"a": ag_a.logic_rel_macro, "b": ag_b.logic_rel_macro},
        "logic_ce_macro": {"a": ag_a.logic_ce_macro, "b": ag_b.logic_ce_macro},
        "logic_below_chance_count": {
            "a": ag_a.logic_below_chance_count,
            "b": ag_b.logic_below_chance_count,
        },
        "logic_floor_pass": {"a": ag_a.logic_floor_pass, "b": ag_b.logic_floor_pass},
        "suite_mean": {"a": ag_a.suite_mean, "b": ag_b.suite_mean},
        "n_probes": {"a": ag_a.n_probes, "b": ag_b.n_probes},
        "rel_floor": float(rel_floor),
        "floors_relative_to_chance": True,
        "status": status,
        "reason": reason,
        "side_a": ag_a.as_dict(),
        "side_b": ag_b.as_dict(),
    }


# --- Nice-to-have residuals (VAL-REASON-011) --------------------------------------


def _expected_calibration_error(
    confidences: Sequence[float],
    corrects: Sequence[bool] | Sequence[int],
    *,
    n_bins: int = 10,
) -> float | None:
    """Naive equal-width ECE over [0,1] confidence bins (fixture/host densify)."""
    if not confidences or len(confidences) != len(corrects):
        return None
    pairs: list[tuple[float, float]] = []
    for c, ok in zip(confidences, corrects, strict=True):
        cf = _finite(c)
        if cf is None:
            continue
        pairs.append((max(0.0, min(1.0, cf)), 1.0 if bool(ok) else 0.0))
    if not pairs:
        return None
    bins: list[list[tuple[float, float]]] = [[] for _ in range(max(1, int(n_bins)))]
    for conf, y in pairs:
        idx = min(len(bins) - 1, int(conf * len(bins)))
        if conf >= 1.0:
            idx = len(bins) - 1
        bins[idx].append((conf, y))
    ece = 0.0
    n = len(pairs)
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_acc - avg_conf)
    return float(ece)


def logic_ece_from_probe_details(
    family: FamilyReasoningScores | None,
    *,
    n_bins: int = 10,
) -> float | None:
    """Aggregate ECE from per-probe ``detail`` confidence/correct sequences if present."""
    if family is None:
        return None
    confs: list[float] = []
    corrects: list[bool] = []
    for score in family.probes.values():
        detail = score.detail or {}
        conf_seq = detail.get("confidences") or detail.get("confidence")
        ok_seq = detail.get("correct") or detail.get("corrects")
        if not isinstance(conf_seq, Sequence) or not isinstance(ok_seq, Sequence):
            continue
        if len(conf_seq) != len(ok_seq):
            continue
        for c, ok in zip(conf_seq, ok_seq, strict=False):
            cf = _finite(c)
            if cf is None:
                continue
            confs.append(cf)
            corrects.append(bool(ok))
    return _expected_calibration_error(confs, corrects, n_bins=n_bins)


def logic_ece_fixture_proxy(
    family: FamilyReasoningScores | None,
    *,
    n_bins: int = 10,
) -> float | None:
    """Fixture ECE proxy when logits confidences are absent.

    Treats closed accuracy as calibrated confidence mass: synthetic bernoulli from
    accuracy with confidence=accuracy (overconfident near-chance models get ECE~0;
    this is a diagnostic residual only, never a MUST floor).
    """
    if family is None:
        return None
    confs: list[float] = []
    corrects: list[bool] = []
    for score in family.probes.values():
        acc = _finite(score.accuracy)
        if acc is None or score.trials <= 0:
            continue
        # Expand synthetic trials from accuracy (stable, seed-free residual).
        n_ok = int(round(acc * score.trials))
        n_ok = max(0, min(score.trials, n_ok))
        for _ in range(n_ok):
            confs.append(acc)
            corrects.append(True)
        for _ in range(score.trials - n_ok):
            confs.append(acc)
            corrects.append(False)
    if not confs:
        return None
    # Prefer real confidences when probes carried them.
    real = logic_ece_from_probe_details(family, n_bins=n_bins)
    if real is not None:
        return real
    return _expected_calibration_error(confs, corrects, n_bins=n_bins)


def cot_free_gen_collapse_fixture_proxy(
    family: FamilyReasoningScores | None,
) -> float | None:
    """Free-gen / CoT collapse proxy from dual-channel gap (0..1 higher = more collapse).

    When open free-gen fails while forced CE stays near chance (or accuracy near chance),
    proxy ≡ ``max(0, chance_gap_from_forced)`` style residual using:
    ``collapse ≈ clamp01(1 - relative_to_chance(acc, chance))`` macro mean.
    """
    if family is None:
        return None
    vals: list[float] = []
    for score in family.probes.values():
        rel = _finite(score.relative)
        if rel is None:
            continue
        # Low relative → high collapse proxy (answers near random / free-gen fail).
        vals.append(max(0.0, min(1.0, 1.0 - rel)))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_reasoning_nice_entries(
    *,
    a: FamilyReasoningScores | None = None,
    b: FamilyReasoningScores | None = None,
    filled: Mapping[str, Mapping[str, Any] | None] | None = None,
    compute_fixture_proxies: bool = True,
) -> dict[str, Any]:
    """P10 nice-to-have panel: filled values OR explicit null+reason (no silent omit).

    Covers ``cot_free_gen_collapse``, ``logic_ece``, ``poly_vs_exp_length``
    (VAL-REASON-011). Fixture proxies may densify ECE / CoT collapse from must probe
    dual-channel metrics when no densify payload is supplied.
    """
    filled_map: dict[str, Mapping[str, Any] | None] = dict(filled or {})

    if compute_fixture_proxies:
        # logic_ece
        if "logic_ece" not in filled_map:
            ece_a = logic_ece_fixture_proxy(a)
            ece_b = logic_ece_fixture_proxy(b)
            if ece_a is not None or ece_b is not None:
                filled_map["logic_ece"] = {
                    "a": ece_a,
                    "b": ece_b,
                    "status": "filled",
                    "reason": None,
                    "method": "fixture_confidence_proxy_or_probe_detail",
                }
            else:
                filled_map["logic_ece"] = {
                    "a": None,
                    "b": None,
                    "status": "not_run",
                    "reason": "logic_ece_no_confidences_or_probe_scores",
                }
        # cot_free_gen_collapse
        if "cot_free_gen_collapse" not in filled_map:
            cot_a = cot_free_gen_collapse_fixture_proxy(a)
            cot_b = cot_free_gen_collapse_fixture_proxy(b)
            if cot_a is not None or cot_b is not None:
                filled_map["cot_free_gen_collapse"] = {
                    "a": cot_a,
                    "b": cot_b,
                    "status": "filled",
                    "reason": None,
                    "method": "relative_gap_collapse_proxy",
                }
            else:
                filled_map["cot_free_gen_collapse"] = {
                    "a": None,
                    "b": None,
                    "status": "not_run",
                    "reason": "cot_free_gen_collapse_unavailable_without_probe_scores",
                }
        # poly_vs_exp_length stays host-densify / retrain residual by default
        if "poly_vs_exp_length" not in filled_map:
            filled_map["poly_vs_exp_length"] = {
                "a": None,
                "b": None,
                "status": "not_run",
                "reason": "poly_vs_exp_length_not_run_host_or_retrain_residual",
            }

    entries: list[dict[str, Any]] = []
    catalogue_rows = [
        row for row in COMPLETE_VIEW_NICE_TO_HAVE if str(row.get("panel")) == "P10_reasoning_logic"
    ]
    for row in catalogue_rows:
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
                    "reason": "not_run_nice_to_have_reasoning",
                }
            )
            continue
        a_val = payload.get("a") if isinstance(payload, Mapping) else None
        b_val = payload.get("b") if isinstance(payload, Mapping) else None
        status = str(
            payload.get("status")
            or ("filled" if (a_val is not None or b_val is not None) else "not_run")
        )
        reason = payload.get("reason")
        if status == "filled":
            reason = None if reason in (None, "") else reason
        elif reason in (None, ""):
            reason = "not_run_nice_to_have_reasoning"
        entry: dict[str, Any] = {
            "matrix_id": row["matrix_id"],
            "key": key,
            "a": a_val,
            "b": b_val,
            "status": status,
            "reason": reason,
        }
        if isinstance(payload, Mapping) and payload.get("method") is not None:
            entry["method"] = payload.get("method")
        # Preserve extra diagnostic fields if callers supplied them.
        for extra_key in ("detail", "notes", "by_length"):
            if isinstance(payload, Mapping) and extra_key in payload:
                entry[extra_key] = payload[extra_key]
        entries.append(entry)

    present = {str(e["key"]) for e in entries}
    missing = [k for k in REASONING_NICE_KEYS if k not in present]
    if missing:
        raise RuntimeError(f"P10 nice-to-have silent omission of keys: {missing}")
    any_filled = any(e.get("status") == "filled" for e in entries)
    return {
        "status": "nice_to_have",
        "no_silent_omission": True,
        "any_filled": any_filled,
        "entries": entries,
    }


def fill_reasoning_panel(
    panel: dict[str, Any] | None = None,
    *,
    a: FamilyReasoningScores | None = None,
    b: FamilyReasoningScores | None = None,
    status_if_partial: str = "partial",
    nice_filled: Mapping[str, Mapping[str, Any] | None] | None = None,
    compute_aggregates: bool = True,
    compute_nice_proxies: bool = True,
    rel_floor: float = REASONING_REL_FLOOR,
) -> dict[str, Any]:
    """Fill (or create) a P10 panel from one/both family score bundles.

    * Per-probe dual-score metrics (VAL-REASON-002..008)
    * suite_mean macro + relative floors (VAL-REASON-009)
    * nice-to-have residuals filled OR null+reason (VAL-REASON-011)
    """
    out = reasoning_panel_shell(status="not_run") if panel is None else dict(panel)
    # deep-copy probes map lightly
    probes_in = out.get("probes")
    probes: dict[str, Any] = {}
    if isinstance(probes_in, Mapping):
        for k, v in probes_in.items():
            probes[str(k)] = dict(v) if isinstance(v, Mapping) else v
    else:
        probes = {}

    filled_any = False
    for key in LOGIC_PROBE_KEYS:
        entry = dict(probes.get(key) or {})
        entry.setdefault("matrix_id", None)
        entry.setdefault("chance", COMPLETE_VIEW_REASONING_CHANCE_TABLE.get(key))
        entry.setdefault("acc", {"a": None, "b": None})
        entry.setdefault("rel_to_chance", {"a": None, "b": None})
        entry.setdefault("forced_ce", {"a": None, "b": None})
        entry.setdefault("by_length", {})
        entry.setdefault("trials", 0)
        if not isinstance(entry["acc"], dict):
            entry["acc"] = {"a": None, "b": None}
        if not isinstance(entry["rel_to_chance"], dict):
            entry["rel_to_chance"] = {"a": None, "b": None}
        if not isinstance(entry["forced_ce"], dict):
            entry["forced_ce"] = {"a": None, "b": None}

        trials = 0
        if a is not None and key in a.probes:
            sa = a.probes[key]
            entry["acc"]["a"] = sa.accuracy
            entry["rel_to_chance"]["a"] = sa.relative
            entry["forced_ce"]["a"] = sa.forced_ce
            entry["chance"] = sa.chance
            entry["side_a"] = _probe_entry_from_score(sa)
            trials = max(trials, int(sa.trials))
            filled_any = True
        if b is not None and key in b.probes:
            sb = b.probes[key]
            entry["acc"]["b"] = sb.accuracy
            entry["rel_to_chance"]["b"] = sb.relative
            entry["forced_ce"]["b"] = sb.forced_ce
            entry["chance"] = sb.chance
            entry["side_b"] = _probe_entry_from_score(sb)
            trials = max(trials, int(sb.trials))
            filled_any = True

        both = (
            entry["acc"].get("a") is not None
            and entry["acc"].get("b") is not None
            and math.isfinite(float(entry["acc"]["a"]))
            and math.isfinite(float(entry["acc"]["b"]))
        )
        one = entry["acc"].get("a") is not None or entry["acc"].get("b") is not None
        if both:
            entry["status"] = "filled"
            entry["reason"] = None
        elif one:
            entry["status"] = status_if_partial
            entry["reason"] = "partial_one_family_only"
        # else keep not_run shell
        entry["trials"] = trials
        probes[key] = entry

    out["probes"] = probes
    out["chance_table"] = dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE)
    out["suite_id"] = REASONING_SUITE_ID
    out["rel_floor"] = REASONING_REL_FLOOR
    out["scoring"] = {
        "closed_choice_accuracy": True,
        "forced_ce": True,
        "chance_baselines": True,
        "architecture_agnostic_logits_only": True,
    }
    seeds_a = None if a is None else a.seeds
    seeds_b = None if b is None else b.seeds
    if seeds_a is not None or seeds_b is not None:
        out["seeds"] = {
            "a": None if seeds_a is None else list(seeds_a),
            "b": None if seeds_b is None else list(seeds_b),
        }
    devices = []
    if a is not None:
        devices.append(a.device)
    if b is not None:
        devices.append(b.device)
    if devices and len(set(devices)) == 1:
        out["device"] = devices[0]
    else:
        out["device"] = (devices or ["fixture"])[0]

    # Panel-level status: filled only when every MUST probe has dual sides; else partial/not_run.
    all_both = all(
        isinstance(probes.get(k), Mapping) and probes[k].get("status") == "filled"
        for k in LOGIC_PROBE_KEYS
    )
    any_probe = any(
        isinstance(probes.get(k), Mapping)
        and probes[k].get("status") in ("filled", "partial", status_if_partial)
        for k in LOGIC_PROBE_KEYS
    )
    if all_both:
        out["status"] = "filled"
        out["reason"] = None
    elif any_probe or filled_any:
        out["status"] = status_if_partial
        out["reason"] = "probe_suite_partial_or_one_family"

    # Aggregates: suite_mean + relative floors (VAL-REASON-009).
    if compute_aggregates:
        out["aggregates"] = suite_aggregate_side_map(a, b, rel_floor=rel_floor)
    else:
        ag = dict(out.get("aggregates") or {})
        ag.setdefault("logic_acc_macro", {"a": None, "b": None})
        ag.setdefault("logic_rel_macro", {"a": None, "b": None})
        ag.setdefault("logic_ce_macro", {"a": None, "b": None})
        ag.setdefault("logic_below_chance_count", {"a": None, "b": None})
        ag.setdefault("logic_floor_pass", {"a": None, "b": None})
        ag.setdefault("suite_mean", {"a": None, "b": None})
        ag["status"] = "not_run"
        ag["reason"] = "not_run_until_probe_suite_aggregate"
        out["aggregates"] = ag

    # Nice-to-have residuals: never silent omit (VAL-REASON-011).
    out["nice"] = build_reasoning_nice_entries(
        a=a,
        b=b,
        filled=nice_filled,
        compute_fixture_proxies=compute_nice_proxies,
    )

    out["meta"] = {
        "suite_doc": documented_logic_suite(),
        "chance_table": dict(LOGIC_CHANCE),
        "host_densify_trained_state": True,
        "no_lm_eval": True,
        "suite_mean_macro": True,
        "floors_relative_to_chance": True,
        "rel_floor": float(rel_floor),
    }
    return out


def reasoning_axis_scores(
    a: OfficialScoreRecord | FamilyReasoningScores | None,
    b: OfficialScoreRecord | FamilyReasoningScores | None,
    *,
    family_a: FamilyReasoningScores | None = None,
    family_b: FamilyReasoningScores | None = None,
    rel_floor: float = REASONING_REL_FLOOR,
) -> tuple[CompleteAxisScore, CompleteAxisScore]:
    """Build multi-axis ``reasoning`` side scores from suite aggregates.

    Prefers ``logic_rel_macro`` (higher better). When floor fails on both sides,
    still publish filled scalars so polar honesty can fire without inventing crowns.
    """
    fa = family_a
    fb = family_b
    if fa is None and isinstance(a, FamilyReasoningScores):
        fa = a
    if fb is None and isinstance(b, FamilyReasoningScores):
        fb = b
    ag_a = aggregate_reasoning_suite(fa, rel_floor=rel_floor)
    ag_b = aggregate_reasoning_suite(fb, rel_floor=rel_floor)

    def _score(ag: ReasonSuiteAggregate) -> CompleteAxisScore:
        if ag.logic_rel_macro is None or not math.isfinite(ag.logic_rel_macro):
            return CompleteAxisScore(
                None,
                "logic_rel_macro",
                "higher",
                reason_if_null=ag.reason or "reasoning_suite_not_run",
            )
        return CompleteAxisScore(
            float(ag.logic_rel_macro),
            "logic_rel_macro",
            "higher",
            reason_if_null=None,
        )

    return _score(ag_a), _score(ag_b)


def axis_scores_with_reasoning(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    family_a: FamilyReasoningScores | None = None,
    family_b: FamilyReasoningScores | None = None,
    base_axis_scores: Mapping[str, tuple[CompleteAxisScore, CompleteAxisScore]] | None = None,
    rel_floor: float = REASONING_REL_FLOOR,
) -> dict[str, tuple[CompleteAxisScore, CompleteAxisScore]]:
    """Overlay reasoning suite scalars onto complete-view multi-axis scores."""
    scores = dict(base_axis_scores or default_axis_scores_from_records(a, b))
    scores["reasoning"] = reasoning_axis_scores(
        a,
        b,
        family_a=family_a,
        family_b=family_b,
        rel_floor=rel_floor,
    )
    return scores


def build_complete_view_with_reasoning(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    family_a: FamilyReasoningScores | None = None,
    family_b: FamilyReasoningScores | None = None,
    score_class: str = "fixture",
    nice_filled: Mapping[str, Mapping[str, Any] | None] | None = None,
    compute_nice_proxies: bool = True,
    rel_floor: float = REASONING_REL_FLOOR,
    axis_scores: Mapping[str, tuple[CompleteAxisScore, CompleteAxisScore]] | None = None,
    panels_override: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build complete_view.v1.3 with P10 suite_mean + multi-axis reasoning (v1.3).

    Expands ``comparison.per_axis_leads.reasoning`` and can force TIE_POLAR when
    short_gen and reasoning disagree beyond ε (VAL-REASON-009). Nice residuals
    never silently omit (VAL-REASON-011).
    """
    scores = axis_scores_with_reasoning(
        a,
        b,
        family_a=family_a,
        family_b=family_b,
        base_axis_scores=axis_scores,
        rel_floor=rel_floor,
    )
    comparison = compare_complete_multi_axis(a, b, axis_scores=scores)
    p10 = fill_reasoning_panel(
        a=family_a,
        b=family_b,
        nice_filled=nice_filled,
        compute_aggregates=True,
        compute_nice_proxies=compute_nice_proxies,
        rel_floor=rel_floor,
    )
    overrides: dict[str, Any] = {"P10_reasoning_logic": p10}
    if panels_override:
        for key, value in panels_override.items():
            if key == "P10_reasoning_logic" and isinstance(value, Mapping):
                overrides["P10_reasoning_logic"] = {**p10, **dict(value)}
            else:
                overrides[key] = value
    doc = build_complete_view(
        a,
        b,
        score_class=score_class,
        comparison=comparison,
        panels_override=overrides,
        **kwargs,
    )
    # Keep P0 reasoning_lead consistent with comparison after merge.
    panels = dict(doc.get("panels") or {})
    p0 = dict(panels.get("P0_rank_overlay") or {})
    p0["reasoning_lead"] = comparison.per_axis_leads.get("reasoning", "missing")
    p0["winner"] = comparison.winner
    p0["reason"] = comparison.reason
    p0["tie_polar"] = comparison.tie_polar
    p0["crown_allowed"] = comparison.crown_allowed
    p0["authoritative_claim"] = "TIE_POLAR" if comparison.tie_polar else comparison.reason
    panels["P0_rank_overlay"] = p0
    doc["panels"] = panels
    doc["comparison"] = comparison.as_dict()
    return doc


@dataclass(frozen=True)
class DualFamilyReasoning:
    """Convenience dual-side fixture holder for VAL-REASON unit tests."""

    a: FamilyReasoningScores
    b: FamilyReasoningScores
    notes: tuple[str, ...] = field(default_factory=tuple)

    def panel(self) -> dict[str, Any]:
        return fill_reasoning_panel(a=self.a, b=self.b)


def dual_family_reasoning_fixture(
    *,
    a_acc: Mapping[str, float] | None = None,
    b_acc: Mapping[str, float] | None = None,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed_a: int = DEFAULT_LOGIC_SUITE_SEED,
    suite_seed_b: int = DEFAULT_LOGIC_SUITE_SEED + 7,
    device: str = "fixture",
    seeds: Sequence[int] = (1337, 2027, 4242),
) -> DualFamilyReasoning:
    """Symmetric dual-family synthetic accuracies for unit fixtures."""
    fa = family_reasoning_fixture(
        accuracy_by_probe=a_acc,
        n_trials=n_trials,
        suite_seed=suite_seed_a,
        device=device,
        seeds=seeds,
    )
    fb = family_reasoning_fixture(
        accuracy_by_probe=b_acc,
        n_trials=n_trials,
        suite_seed=suite_seed_b,
        device=device,
        seeds=seeds,
    )
    return DualFamilyReasoning(
        a=fa,
        b=fb,
        notes=(
            "synthetic_fixture_not_trained_state",
            "PROVIDER_TRUST",
            "LAB-GPU_or_fixture_lab_only",
            "not_gsm8k_mmlu",
        ),
    )


def probe_metric_bundle(
    scores_a: Mapping[str, LogicTaskScore] | FamilyReasoningScores,
    scores_b: Mapping[str, LogicTaskScore] | FamilyReasoningScores,
) -> dict[str, dict[str, Any]]:
    """Compact dual-side metrics dict (acc/rel/ce) per probe for tests & reports."""

    def _as_map(
        s: Mapping[str, LogicTaskScore] | FamilyReasoningScores,
    ) -> Mapping[str, LogicTaskScore]:
        if isinstance(s, FamilyReasoningScores):
            return s.probes
        return s

    ma, mb = _as_map(scores_a), _as_map(scores_b)
    out: dict[str, dict[str, Any]] = {}
    for key in LOGIC_PROBE_KEYS:
        sa, sb = ma[key], mb[key]
        out[key] = {
            "chance": sa.chance,
            "acc": {"a": sa.accuracy, "b": sb.accuracy},
            "rel_to_chance": {"a": sa.relative, "b": sb.relative},
            "forced_ce": {"a": sa.forced_ce, "b": sb.forced_ce},
            "trials": {"a": sa.trials, "b": sb.trials},
        }
    return out


def assert_probe_dual_channel(score: LogicTaskScore) -> None:
    """Raise AssertionError if a MUST probe score lacks dual channel fields."""
    if score.probe not in LOGIC_PROBE_KEYS:
        raise AssertionError(f"unknown probe {score.probe}")
    if not math.isfinite(score.accuracy) or not (0.0 <= score.accuracy <= 1.0):
        raise AssertionError(f"{score.probe}: accuracy out of range: {score.accuracy}")
    ch = LOGIC_CHANCE[score.probe]
    if abs(score.chance - ch) > 1e-9:
        raise AssertionError(f"{score.probe}: chance {score.chance} != table {ch}")
    expected_rel = relative_to_chance(score.accuracy, score.chance)
    if abs(score.relative - expected_rel) > 1e-9:
        raise AssertionError(
            f"{score.probe}: relative_to_chance {score.relative} != {expected_rel}"
        )
    if score.forced_ce is None or not math.isfinite(float(score.forced_ce)):
        raise AssertionError(f"{score.probe}: forced_ce missing/non-finite")
    if score.trials <= 0:
        raise AssertionError(f"{score.probe}: trials must be > 0")


__all__ = [
    "DualFamilyReasoning",
    "FamilyReasoningScores",
    "REASONING_NICE_KEYS",
    "ReasonSuiteAggregate",
    "aggregate_reasoning_suite",
    "assert_probe_dual_channel",
    "axis_scores_with_reasoning",
    "build_complete_view_with_reasoning",
    "build_reasoning_nice_entries",
    "cot_free_gen_collapse_fixture_proxy",
    "dual_family_reasoning_fixture",
    "family_reasoning_fixture",
    "family_reasoning_from_scores",
    "fill_reasoning_panel",
    "logic_ece_fixture_proxy",
    "logic_ece_from_probe_details",
    "probe_metric_bundle",
    "reasoning_axis_scores",
    "suite_aggregate_side_map",
]
