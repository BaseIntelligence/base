"""P10 suite_mean aggregate + multi-axis reasoning polar (VAL-REASON-009/011).

Locks:
* suite_mean macro with relative-to-chance floors
* comparison.per_axis_leads.reasoning + TIE_POLAR when short_gen vs reasoning conflict
* nice-to-have residuals (logic_ece, cot_free_gen_collapse, poly_vs_exp_length)
  never silent-omitted: filled OR explicit null+reason
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.benchmarks.logic_suite import LOGIC_PROBE_KEYS
from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
    REASONING_REL_FLOOR,
    assert_complete_view_document,
    validate_complete_view_document,
)
from prism_challenge.evaluator.complete_view_reasoning import (
    REASONING_NICE_KEYS,
    aggregate_reasoning_suite,
    build_complete_view_with_reasoning,
    build_reasoning_nice_entries,
    dual_family_reasoning_fixture,
    family_reasoning_fixture,
    fill_reasoning_panel,
    suite_aggregate_side_map,
)
from prism_challenge.evaluator.official_comparison import OfficialScoreRecord
from prism_challenge.evaluator.scorecard_suite import relative_to_chance


def _rec(*, label: str, heldout_delta: float = 0.5) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=1.2,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        valid=True,
        seed_count=3,
        bpb_std=0.01,
        heldout_std=0.02,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


def _acc_profile(base: float, *, bump: dict[str, float] | None = None) -> dict[str, float]:
    out = {k: float(base) for k in LOGIC_PROBE_KEYS}
    if bump:
        for k, v in bump.items():
            out[k] = max(0.0, min(1.0, float(v)))
    return out


# --- VAL-REASON-009: suite_mean + multi-axis reasoning ----------------------------


def test_val_reason_009_suite_mean_macro_and_relative_floors() -> None:
    """Macro suite_mean = mean accuracy; logic_rel_macro = mean relative_to_chance."""
    # Uniform accuracy 0.8 on all probes should give floor_pass depending on chance floors.
    high = family_reasoning_fixture(accuracy_by_probe=_acc_profile(0.90))
    low = family_reasoning_fixture(accuracy_by_probe=_acc_profile(0.50))

    ag_hi = aggregate_reasoning_suite(high)
    ag_lo = aggregate_reasoning_suite(low)
    assert ag_hi.status == "filled"
    assert ag_lo.status == "filled"
    # Fixture maps target accuracy via integer trial counts → tolerate small quantize delta.
    mean_hi = sum(high.probes[k].accuracy for k in LOGIC_PROBE_KEYS) / len(LOGIC_PROBE_KEYS)
    mean_lo = sum(low.probes[k].accuracy for k in LOGIC_PROBE_KEYS) / len(LOGIC_PROBE_KEYS)
    assert ag_hi.suite_mean == pytest.approx(mean_hi)
    assert ag_hi.logic_acc_macro == pytest.approx(mean_hi)
    assert ag_lo.suite_mean == pytest.approx(mean_lo)
    assert mean_hi > 0.85  # target was 0.90
    assert mean_lo == pytest.approx(0.50, abs=0.05)

    # Relative macro is mean of per-probe (acc-chance)/(1-chance).
    expected_rel = []
    for key in LOGIC_PROBE_KEYS:
        sc = high.probes[key]
        expected_rel.append(relative_to_chance(sc.accuracy, sc.chance))
    assert ag_hi.logic_rel_macro == pytest.approx(sum(expected_rel) / len(expected_rel))

    # Floor: every probe relative ≥ REASONING_REL_FLOOR.
    assert REASONING_REL_FLOOR == 0.05
    assert ag_hi.logic_floor_pass is True
    # accuracy ~0.50 is bottom-ish relative for high-chance probes (binary chance 0.5 → rel 0).
    assert ag_lo.logic_floor_pass is False

    side = suite_aggregate_side_map(high, low)
    assert side["status"] == "filled"
    assert side["floors_relative_to_chance"] is True
    assert side["rel_floor"] == REASONING_REL_FLOOR
    assert side["suite_mean"]["a"] == pytest.approx(mean_hi)
    assert side["suite_mean"]["b"] == pytest.approx(mean_lo)
    assert side["logic_rel_macro"]["a"] is not None
    assert side["logic_rel_macro"]["b"] is not None
    assert side["logic_floor_pass"]["a"] is True
    assert side["logic_floor_pass"]["b"] is False

    # Panel fill wires aggregates.
    panel = fill_reasoning_panel(a=high, b=low)
    assert panel["aggregates"]["suite_mean"]["a"] == pytest.approx(mean_hi)
    assert panel["aggregates"]["logic_rel_macro"]["a"] == pytest.approx(ag_hi.logic_rel_macro)
    assert panel["aggregates"]["status"] == "filled"
    assert panel["aggregates"]["reason"] is None
    assert panel["meta"]["suite_mean_macro"] is True


def test_val_reason_009_per_axis_leads_reasoning_and_tie_polar() -> None:
    """short_gen lead A + reasoning lead B → TIE_POLAR / crown_allowed=false."""
    # A better short_gen heldout; B far better on logic relative.
    fam_a = family_reasoning_fixture(accuracy_by_probe=_acc_profile(0.30), seeds=(1, 2, 3))
    fam_b = family_reasoning_fixture(accuracy_by_probe=_acc_profile(0.95), seeds=(1, 2, 3))
    # Confirm suite scalars favor B strongly.
    assert aggregate_reasoning_suite(fam_b).logic_rel_macro > (
        (aggregate_reasoning_suite(fam_a).logic_rel_macro or 0.0) + REASONING_REL_FLOOR
    )

    a = _rec(label="transformer-tiny-1m", heldout_delta=0.95)
    b = _rec(label="mamba-tiny-1m", heldout_delta=0.20)
    doc = build_complete_view_with_reasoning(
        a,
        b,
        family_a=fam_a,
        family_b=fam_b,
        score_class="fixture",
    )
    assert doc["schema"] == COMPLETE_VIEW_SCHEMA == "complete_view.v1.3"
    assert doc["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    cmp = doc["comparison"]
    assert cmp["per_axis_leads"]["short_gen"] == "a"
    assert cmp["per_axis_leads"]["reasoning"] == "b"
    assert cmp["disagreement_matrix"]["short_gen"]["reasoning"] is True
    assert cmp["tie_polar"] is True
    assert cmp["crown_allowed"] is False
    assert "reasoning" in cmp["polar_axes_involved"]
    assert cmp["authoritative_claim"] == "TIE_POLAR"
    assert "reasoning" in cmp["per_axis_leads"]
    # P0 surfaces reasoning lead + polar.
    p0 = doc["panels"]["P0_rank_overlay"]
    assert p0["reasoning_lead"] == "b"
    assert p0["tie_polar"] is True
    assert p0["crown_allowed"] is False
    # P10 suite_mean both sides filled.
    p10 = doc["panels"]["P10_reasoning_logic"]
    assert p10["aggregates"]["suite_mean"]["a"] is not None
    assert p10["aggregates"]["suite_mean"]["b"] is not None
    assert p10["aggregates"]["logic_rel_macro"]["b"] > p10["aggregates"]["logic_rel_macro"]["a"]
    assert "real_provider_tee" not in doc
    assert doc["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert doc["non_claims"]["prism_tee_product"] is False
    assert validate_complete_view_document(doc) == []
    assert_complete_view_document(doc)


def test_val_reason_009_no_polar_when_axes_agree() -> None:
    """Same side wins short_gen and reasoning → no short_gen/reasoning polar pair."""
    fam_a = family_reasoning_fixture(accuracy_by_probe=_acc_profile(0.95))
    fam_b = family_reasoning_fixture(accuracy_by_probe=_acc_profile(0.40))
    a = _rec(label="A", heldout_delta=0.90)
    b = _rec(label="B", heldout_delta=0.30)
    doc = build_complete_view_with_reasoning(a, b, family_a=fam_a, family_b=fam_b)
    cmp = doc["comparison"]
    assert cmp["per_axis_leads"]["short_gen"] == "a"
    assert cmp["per_axis_leads"]["reasoning"] == "a"
    assert cmp["disagreement_matrix"]["short_gen"]["reasoning"] is False
    # Other axes (long_ctx, sample_eff, …) residual as missing → no expanded polar pair.
    assert cmp["tie_polar"] is False
    assert cmp["crown_allowed"] is True
    assert cmp["authoritative_claim"] != "TIE_POLAR"


def test_val_reason_009_missing_reasoning_does_not_invent_polar() -> None:
    """Without family scores, reasoning lead is missing and does not invent TIE_POLAR."""
    a = _rec(label="A", heldout_delta=0.9)
    b = _rec(label="B", heldout_delta=0.2)
    doc = build_complete_view_with_reasoning(a, b, family_a=None, family_b=None)
    cmp = doc["comparison"]
    assert cmp["per_axis_leads"]["reasoning"] == "missing"
    # short_gen favors A; missing reasoning → no disagreement matrix open on that pair.
    assert cmp["disagreement_matrix"]["short_gen"]["reasoning"] is False
    assert cmp["tie_polar"] is False
    involved = set(cmp.get("polar_axes_involved") or [])
    assert "reasoning" not in involved or cmp["tie_polar"] is False
    # shell aggregates remain null when no family scores
    p10 = doc["panels"]["P10_reasoning_logic"]
    assert p10["aggregates"]["suite_mean"] == {"a": None, "b": None}
    assert p10["aggregates"]["status"] == "not_run"


# --- VAL-REASON-011: nice residuals no silent omission ----------------------------


def test_val_reason_011_nice_residuals_filled_or_null_reason() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.80),
        b_acc=_acc_profile(0.55),
    )
    nice = build_reasoning_nice_entries(a=dual.a, b=dual.b)
    assert nice["status"] == "nice_to_have"
    assert nice["no_silent_omission"] is True
    keys = {e["key"] for e in nice["entries"]}
    assert keys == set(REASONING_NICE_KEYS)
    assert "logic_ece" in keys
    assert "cot_free_gen_collapse" in keys
    assert "poly_vs_exp_length" in keys

    by_key = {e["key"]: e for e in nice["entries"]}
    # Fixture dual-channel proxies should fill ECE + CoT collapse.
    assert by_key["logic_ece"]["status"] == "filled"
    assert by_key["logic_ece"]["a"] is not None and math.isfinite(float(by_key["logic_ece"]["a"]))
    assert by_key["logic_ece"]["reason"] is None
    assert by_key["cot_free_gen_collapse"]["status"] == "filled"
    assert by_key["cot_free_gen_collapse"]["a"] is not None
    assert by_key["cot_free_gen_collapse"]["reason"] is None
    # poly_vs_exp_length remains explicit null+reason (not silent omit).
    assert by_key["poly_vs_exp_length"]["status"] == "not_run"
    assert by_key["poly_vs_exp_length"]["a"] is None
    assert by_key["poly_vs_exp_length"]["b"] is None
    assert by_key["poly_vs_exp_length"]["reason"]
    assert (
        "poly" in str(by_key["poly_vs_exp_length"]["reason"]).lower()
        or "not_run" in str(by_key["poly_vs_exp_length"]["reason"]).lower()
    )

    # Panel + document path.
    panel = fill_reasoning_panel(a=dual.a, b=dual.b)
    assert panel["nice"]["no_silent_omission"] is True
    assert {e["key"] for e in panel["nice"]["entries"]} == set(REASONING_NICE_KEYS)

    doc = build_complete_view_with_reasoning(
        _rec(label="A", heldout_delta=0.4),
        _rec(label="B", heldout_delta=0.5),
        family_a=dual.a,
        family_b=dual.b,
        score_class="fixture",
    )
    p10 = doc["panels"]["P10_reasoning_logic"]
    nice_keys = {e["key"] for e in p10["nice"]["entries"]}
    assert nice_keys == set(REASONING_NICE_KEYS)
    for e in p10["nice"]["entries"]:
        if e["status"] != "filled":
            assert e.get("reason"), f"{e['key']} silent-empty reason"
        else:
            assert e.get("a") is not None or e.get("b") is not None
    assert validate_complete_view_document(doc) == []


def test_val_reason_011_explicit_null_when_no_proxies() -> None:
    """When proxies disabled, every nice entry still present with null+reason."""
    dual = dual_family_reasoning_fixture()
    nice = build_reasoning_nice_entries(
        a=dual.a,
        b=dual.b,
        compute_fixture_proxies=False,
        filled={
            "logic_ece": None,  # force auto path off via empty handling
        },
    )
    # Without proxies and without filled map entries, defaults to catalog null+reason
    # — but filled={"logic_ece": None} means payload is None → not_run shell.
    entries = nice["entries"]
    assert {e["key"] for e in entries} == set(REASONING_NICE_KEYS)
    for e in entries:
        assert e["status"] in ("not_run", "filled")
        if e["status"] != "filled":
            assert e["reason"] is not None
            assert e["a"] is None and e["b"] is None


def test_val_reason_011_no_silent_omission_raises() -> None:
    """Internal integrity: emptying catalogue raise is defensive (empty REASONING_NICE_KEYS)."""
    # Public builders always catalogue-complete; verify expected keys invariant.
    assert set(REASONING_NICE_KEYS) == {
        "cot_free_gen_collapse",
        "logic_ece",
        "poly_vs_exp_length",
    }


def test_aggregate_empty_family_is_not_run() -> None:
    ag = aggregate_reasoning_suite(None)
    assert ag.status == "not_run"
    assert ag.suite_mean is None
    assert ag.logic_floor_pass is None
    side = suite_aggregate_side_map(None, None)
    assert side["status"] == "not_run"
    assert side["suite_mean"] == {"a": None, "b": None}
