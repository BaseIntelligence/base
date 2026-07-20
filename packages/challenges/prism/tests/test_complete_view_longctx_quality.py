"""Complete View quality + long-ctx expansions (VAL-COMPLETE-002..007).

CPU unit fixtures only: no NVIDIA, no Lium, no REAL TEE claim. Prefer
reused multi-seed val_bpb + synthetic closed-choice outcomes; GPU/CPU hooks
are exercised via lightweight callables.
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
    assert_complete_view_document,
    validate_complete_view_document,
)
from prism_challenge.evaluator.complete_view_longctx import (
    DEFAULT_LONG_CTX_TS,
    FamilyLongCtxQuality,
    apply_val_bpb_to_record,
    build_complete_view_with_longctx_quality,
    build_longctx_quality_panels,
    fixture_family_longctx_quality,
    free_ce_bits_from_nll_stream,
    lag_nll_bins_result,
    length_extrapolate_ce,
    medium_free_ce_by_T,
    mqar_grid_from_outcomes,
    multi_seed_val_bpb_trained,
    multi_t_long_ctx_suite,
    needle_by_depth_from_outcomes,
    probe_free_ce,
    run_multi_t_long_ctx_fixture,
    unfuse_induction_and_copy,
)
from prism_challenge.evaluator.official_comparison import OfficialScoreRecord
from prism_challenge.evaluator.scorecard_suite import run_long_ctx_fixture_suite


def _rec(
    *,
    label: str,
    heldout_delta: float = 0.5,
    bpb: float = 1.2,
    val_bpb_trained: float | None = None,
    seed_count: int = 3,
    long_ctx_enabled: bool = True,
    long_ctx_score: float | None = 0.2,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        val_bpb_trained=val_bpb_trained,
        memorization_flag=False,
        train_heldout_gap=0.1,
        step0_anomaly=False,
        valid=True,
        seed_count=seed_count,
        bpb_std=0.01,
        heldout_std=0.02,
        long_ctx_enabled=long_ctx_enabled,
        long_ctx_score=long_ctx_score,
        long_ctx_floor_pass=(None if long_ctx_score is None else float(long_ctx_score) >= 0.15),
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
        params=1_000_000,
    )


def _family_fixture(
    *,
    val_mean_shift: float = 0.0,
    multi_t_needle: float = 0.75,
) -> FamilyLongCtxQuality:
    """Synthetic dual-side convertible family fixture (deterministic)."""
    # Distinct per-seed absolute bpb around a mean.
    base = 2.5 + val_mean_shift
    per_seed = {
        1337: base + 0.02,
        2027: base - 0.01,
        4242: base + 0.0,
    }
    # Multi-T outcomes: needle/mqar high; induction/copy unfused trials.
    multi_t = {
        256: {
            "needle": [True, True, True, False],  # 0.75
            "mqar": [True, True, False, True],  # 0.75
            "induction": [True, True, False, True],  # 0.75
            "exact_copy": [True, True, True, True],  # 1.0
        },
        512: {
            "needle": multi_t_needle,
            "mqar": 0.70,
            "induction": 0.55,
            "exact_copy": 0.80,
        },
        1024: {
            "needle": [True, False, True, True, True, False],  # 4/6≈0.667
            "mqar": 0.60,
            "induction": 0.40,
            "exact_copy": 0.70,
        },
    }
    needle_depth = {
        0.1: [True, True, True, False],  # 0.75
        0.5: [True, False, True, False],  # 0.50 mid / lost-in-middle
        0.9: [True, True, False, True],  # 0.75
    }
    mqar = {
        (4, 16): 0.80,
        (4, 64): 0.70,
        (4, 256): 0.55,
        (8, 16): 0.65,
        (8, 64): 0.50,
        (8, 256): 0.40,
        (16, 16): 0.45,
        (16, 64): 0.30,
        (16, 256): 0.20,
    }
    lag = {
        "lag_16": 2.0,
        "lag_64": 2.5,
        "lag_ge_64": 2.6,
        "lag_ge_256": 3.0,
        "lag_ge_512": 3.4,
    }
    length_ce = {128: 1.0, 256: 1.15, 512: 1.40, 1024: 1.80}
    return fixture_family_longctx_quality(
        val_bpb_per_seed=per_seed,
        multi_t_outcomes=multi_t,
        needle_depth_outcomes=needle_depth,
        mqar_outcomes=mqar,
        induction=[True, True, False, True, True],  # 0.8
        exact_copy=[True, True, True, True, False],  # 0.8
        lag_bins=lag,
        length_extrap_ce=length_ce,
        train_t=128,
        medium_ce={256: 1.2, 512: 1.5},
        device="fixture",
    )


# --- VAL-COMPLETE-002 ------------------------------------------------------------


def test_val_complete_002_multi_seed_val_bpb_trained_both_sides() -> None:
    fam_a = _family_fixture(val_mean_shift=0.0)
    fam_b = _family_fixture(val_mean_shift=-0.3)
    assert fam_a.val_bpb_trained is not None
    assert fam_b.val_bpb_trained is not None
    assert fam_a.val_bpb_trained.mean is not None
    assert fam_b.val_bpb_trained.mean is not None
    assert math.isfinite(fam_a.val_bpb_trained.mean)
    assert math.isfinite(fam_b.val_bpb_trained.mean)
    assert fam_a.val_bpb_trained.std is not None
    assert len(fam_a.val_bpb_trained.seeds) == 3
    assert len(fam_a.val_bpb_trained.per_seed) == 3
    # B better (lower free CE).
    assert fam_b.val_bpb_trained.mean < fam_a.val_bpb_trained.mean

    a = _rec(label="transformer-tiny-1m")
    b = _rec(label="mamba-tiny-1m")
    doc = build_complete_view_with_longctx_quality(a, b, fam_a=fam_a, fam_b=fam_b)
    assert_complete_view_document(doc)
    val = doc["panels"]["P1_short_gen"]["val_bpb_trained"]
    assert val["status"] == "filled"
    assert val["a"] is not None and val["b"] is not None
    assert val["a"]["mean"] == pytest.approx(fam_a.val_bpb_trained.mean)
    assert val["b"]["mean"] == pytest.approx(fam_b.val_bpb_trained.mean)
    assert val["a"].get("std") is not None
    assert val["b"].get("std") is not None
    # Record stamp also non-null after apply.
    stamped = apply_val_bpb_to_record(a, fam_a.val_bpb_trained)
    assert stamped.val_bpb_trained == pytest.approx(fam_a.val_bpb_trained.mean)


def test_val_complete_002_rejects_all_nonfinite() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        multi_seed_val_bpb_trained({1: float("nan"), 2: float("inf")})


def test_val_complete_002_medium_free_ce_optional() -> None:
    out = medium_free_ce_by_T({256: 1.1, 512: 1.4})
    assert out["status"] == "filled"
    assert out["by_T"]["256"] == pytest.approx(1.1)
    assert out["by_T"]["512"] == pytest.approx(1.4)


# --- VAL-COMPLETE-003 ------------------------------------------------------------


def test_val_complete_003_multi_t_long_ctx_matrix() -> None:
    fam = _family_fixture()
    assert fam.multi_t is not None
    payload = fam.multi_t.as_dict()
    assert payload["status"] == "filled"
    # At least 256/512/1024 (or max feasible = those three).
    assert set(payload["by_T"]) >= {"256", "512", "1024"}
    assert payload["aggregate_suite_mean"] is not None
    assert math.isfinite(payload["aggregate_suite_mean"])
    assert payload["max_feasible_t"] == 1024
    assert tuple(payload["requested_ts"]) == DEFAULT_LONG_CTX_TS
    for t in ("256", "512", "1024"):
        slice_t = payload["by_T"][t]
        assert slice_t["suite_mean"] is not None
        assert slice_t["needle"] is not None
        assert slice_t["mqar"] is not None

    a = _rec(label="A", long_ctx_score=0.3)
    b = _rec(label="B", long_ctx_score=0.2)
    fam_a = _family_fixture(multi_t_needle=0.80)
    fam_b = _family_fixture(multi_t_needle=0.40)
    doc = build_complete_view_with_longctx_quality(a, b, fam_a=fam_a, fam_b=fam_b)
    p3 = doc["panels"]["P3_long_ctx"]
    assert p3["multi_T"]["status"] == "filled"
    assert p3["long_ctx_by_T"] is not None
    assert "256" in p3["long_ctx_by_T"]
    assert "512" in p3["long_ctx_by_T"]
    assert "1024" in p3["long_ctx_by_T"]
    assert p3["aggregate_suite_mean"]["a"] is not None
    assert p3["aggregate_suite_mean"]["b"] is not None
    # Per-T suite scores both sides.
    for t in ("256", "512", "1024"):
        assert p3["long_ctx_by_T"][t]["suite_mean"]["a"] is not None
        assert p3["long_ctx_by_T"][t]["suite_mean"]["b"] is not None


def test_val_complete_003_run_multi_t_fixture_and_legacy_v11_base() -> None:
    # Compatibility: v1.1 fixture suite still works as base.
    v11 = run_long_ctx_fixture_suite(
        needle_correct=[True, True, False, True],
        mqar_correct=[True, False, True, True],
        induction_correct=[True, True, True, False],
        lag_nll_by_bin={"lag_ge_64": 2.0, "lag_ge_256": 3.0},
    )
    assert v11.suite_mean is not None
    # Multi-T builder across explicit slice dicts.
    multi = run_multi_t_long_ctx_fixture(
        {
            256: {"needle": 0.5, "mqar": 0.5, "induction": 0.4, "exact_copy": 0.6},
            512: {"needle": 0.4, "mqar": 0.3, "induction": 0.3, "exact_copy": 0.5},
        }
    )
    assert multi.aggregate_suite_mean is not None
    # multi_t_long_ctx_suite accepts mapping of LongCtxAtT-ready dicts.
    rebuilt = multi_t_long_ctx_suite(
        {
            256: multi.by_T["256"],
            512: multi.by_T["512"],
        },
        requested_ts=(256, 512, 1024),
    )
    assert any("missing_T:1024" in n for n in rebuilt.notes)
    assert rebuilt.max_feasible_t == 512


# --- VAL-COMPLETE-004 ------------------------------------------------------------


def test_val_complete_004_needle_by_depth_and_lost_in_middle() -> None:
    depth = needle_by_depth_from_outcomes(
        {
            0.1: [True, True, True, True],  # 1.0
            0.5: [True, False, False, False],  # 0.25 mid
            0.9: [True, True, False, True],  # 0.75
        }
    )
    payload = depth.as_dict()
    assert payload["status"] == "filled"
    assert payload["by_depth"]["0.1"] == pytest.approx(1.0)
    assert payload["by_depth"]["0.5"] == pytest.approx(0.25)
    assert payload["by_depth"]["0.9"] == pytest.approx(0.75)
    assert payload["lost_in_middle"] == pytest.approx(0.25)
    assert payload["mid_depth"] == pytest.approx(0.5)

    fam_a = _family_fixture()
    fam_b = _family_fixture()
    a = _rec(label="A")
    b = _rec(label="B")
    doc = build_complete_view_with_longctx_quality(a, b, fam_a=fam_a, fam_b=fam_b)
    p3 = doc["panels"]["P3_long_ctx"]
    assert p3["needle_by_depth"]["status"] == "filled"
    assert p3["needle_by_depth"]["a"]["by_depth"]
    assert p3["needle_by_depth"]["b"]["by_depth"]
    assert p3["lost_in_middle"]["status"] == "filled"
    assert p3["lost_in_middle"]["a"] is not None
    assert p3["lost_in_middle"]["b"] is not None


# --- VAL-COMPLETE-005 ------------------------------------------------------------


def test_val_complete_005_mqar_scale_grid() -> None:
    grid = mqar_grid_from_outcomes(
        {
            (4, 16): [True, True, True, False],
            (4, 64): 0.5,
            (8, 16): 0.4,
            (8, 64): 0.25,
            (16, 16): 0.2,
            (16, 64): 0.1,
        }
    )
    payload = grid.as_dict()
    assert payload["status"] == "filled"
    assert "N4" in payload["grid"]
    assert "N8" in payload["grid"]
    assert "N16" in payload["grid"]
    assert "lag_16" in payload["grid"]["N4"]
    assert payload["macro_mean"] is not None
    # Nested form accepted.
    nested = mqar_grid_from_outcomes(
        {
            "N4": {"lag_16": 0.9, "lag_64": 0.8},
            "N8": {"16": 0.7, "64": 0.6},
        }
    )
    assert nested.grid["N4"]["lag_16"] == pytest.approx(0.9)
    assert nested.grid["N8"]["lag_16"] == pytest.approx(0.7)

    fam_a = _family_fixture()
    fam_b = _family_fixture()
    doc = build_complete_view_with_longctx_quality(
        _rec(label="A"), _rec(label="B"), fam_a=fam_a, fam_b=fam_b
    )
    mq = doc["panels"]["P3_long_ctx"]["mqar_grid"]
    assert mq["status"] == "filled"
    assert mq["a"]["grid"] and mq["b"]["grid"]
    assert "N4" in mq["a"]["grid"]


# --- VAL-COMPLETE-006 ------------------------------------------------------------


def test_val_complete_006_induction_and_exact_copy_unfused() -> None:
    unfused = unfuse_induction_and_copy(
        induction=[True, True, False, True],  # 0.75
        exact_copy=[True, True, True, True, True],  # 1.0
    )
    payload = unfused.as_dict()
    assert payload["status"] == "filled"
    assert payload["fused_only"] is False
    assert payload["induction_acc"] == pytest.approx(0.75)
    assert payload["exact_copy_acc"] == pytest.approx(1.0)
    assert "induction_acc" in payload and "exact_copy_acc" in payload
    # Separate fields are not a single fused scalar.
    assert payload["induction_acc"] != payload["exact_copy_acc"]

    fam_a = _family_fixture()
    fam_b = _family_fixture()
    doc = build_complete_view_with_longctx_quality(
        _rec(label="A"), _rec(label="B"), fam_a=fam_a, fam_b=fam_b
    )
    p3 = doc["panels"]["P3_long_ctx"]
    assert p3["induction_acc"]["status"] == "filled"
    assert p3["copy_acc"]["status"] == "filled"
    assert p3["induction_acc"]["a"] is not None
    assert p3["copy_acc"]["a"] is not None
    assert p3["induction_and_copy_unfused"]["status"] == "filled"
    assert p3["induction_and_copy_unfused"]["a"]["fused_only"] is False


# --- VAL-COMPLETE-007 ------------------------------------------------------------


def test_val_complete_007_lag_bins_and_length_extrapolate() -> None:
    lag = lag_nll_bins_result(
        {
            "lag_16": 1.5,
            "lag_64": 2.0,
            "lag_ge_64": 2.2,
            "lag_ge_256": 2.8,
            "lag_ge_512": 3.1,
        }
    )
    assert lag.as_dict()["status"] == "filled"
    assert lag.bins["lag_16"] == pytest.approx(1.5)
    assert lag.macro_long is not None
    assert math.isfinite(lag.macro_long)

    le = length_extrapolate_ce(
        {128: 1.0, 256: 1.2, 512: 1.5, 1024: 2.0},
        train_t=128,
    )
    payload = le.as_dict()
    assert payload["status"] == "filled"
    assert payload["retrain"] is False
    assert payload["train_t"] == 128
    assert payload["ce_by_t"]["128"] == pytest.approx(1.0)
    assert payload["ratio_t_over_train"]["128"] == pytest.approx(1.0)
    assert payload["ratio_t_over_train"]["256"] == pytest.approx(1.2)
    assert payload["ratio_t_over_train"]["1024"] == pytest.approx(2.0)

    fam_a = _family_fixture()
    fam_b = _family_fixture(val_mean_shift=-0.1)
    doc = build_complete_view_with_longctx_quality(
        _rec(label="A"), _rec(label="B"), fam_a=fam_a, fam_b=fam_b
    )
    p3 = doc["panels"]["P3_long_ctx"]
    assert p3["lag_nll_bins"]["binned"]["status"] == "filled"
    assert p3["lag_nll_bins"]["binned"]["a"]["bins"]
    assert p3["lag_nll_bins"]["macro"]["a"] is not None

    p4 = doc["panels"]["P4_length_extrap"]
    assert p4["ce_by_T"]["status"] == "filled"
    assert p4["ratio_T_over_train"]["status"] == "filled"
    assert p4["length_extrapolate"]["status"] == "filled"
    assert p4["retrain"] is False
    assert p4["length_extrapolate"]["a"]["retrain"] is False


def test_val_complete_007_length_extrap_requires_train_t() -> None:
    with pytest.raises(ValueError, match="train_t"):
        length_extrapolate_ce({256: 1.2, 512: 1.5}, train_t=128)


# --- Combined document fill + hooks ----------------------------------------------


def test_complete_view_longctx_fills_002_to_007_non_null() -> None:
    """End-to-end: both sides filled for all VAL-COMPLETE-002..007 evidence keys."""
    fam_a = _family_fixture(val_mean_shift=0.0)
    fam_b = _family_fixture(val_mean_shift=-0.2)
    doc = build_complete_view_with_longctx_quality(
        _rec(label="transformer-tiny-1m", heldout_delta=3.46, bpb=0.122),
        _rec(label="mamba-tiny-1m", heldout_delta=4.70, bpb=0.118),
        fam_a=fam_a,
        fam_b=fam_b,
        score_class="fixture",
        pin={
            "token_budget": 500_000,
            "seq_len_train": 128,
            "seeds": [1337, 2027, 4242],
        },
    )
    assert doc["schema"] == COMPLETE_VIEW_SCHEMA
    assert doc["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert validate_complete_view_document(doc) == []

    p1 = doc["panels"]["P1_short_gen"]
    p3 = doc["panels"]["P3_long_ctx"]
    p4 = doc["panels"]["P4_length_extrap"]

    # 002
    assert p1["val_bpb_trained"]["status"] == "filled"
    assert p1["val_bpb_trained"]["a"]["mean"] is not None
    assert p1["val_bpb_trained"]["b"]["mean"] is not None
    # 003
    assert p3["multi_T"]["status"] == "filled"
    assert p3["long_ctx_by_T"] and set(p3["long_ctx_by_T"]) >= {"256", "512", "1024"}
    # 004
    assert p3["needle_by_depth"]["status"] == "filled"
    assert p3["lost_in_middle"]["status"] == "filled"
    # 005
    assert p3["mqar_grid"]["status"] == "filled"
    # 006
    assert p3["induction_acc"]["status"] == "filled"
    assert p3["copy_acc"]["status"] == "filled"
    # 007
    assert p3["lag_nll_bins"]["binned"]["status"] == "filled"
    assert p4["length_extrapolate"]["status"] == "filled"
    assert p4["ce_by_T"]["a"] is not None


def test_gpu_ready_free_ce_probe_hook() -> None:
    # Scalar NLL stream → bits.
    bits = free_ce_bits_from_nll_stream([math.log(2.0), math.log(2.0)], basis="nats")
    assert bits == pytest.approx(1.0)

    def nll_fn(seq: list[int]) -> list[float]:
        return [math.log(2.0)] * max(1, len(seq) - 1)

    ce = probe_free_ce(nll_fn, [[1, 2, 3, 4], [5, 6, 7]])
    assert ce == pytest.approx(1.0)


def test_build_longctx_quality_panels_partial_is_honest() -> None:
    """Partial suite: only val filled leaves long-ctx reasons intact."""
    val_only = FamilyLongCtxQuality(
        val_bpb_trained=multi_seed_val_bpb_trained({1337: 2.0, 2027: 2.1, 4242: 1.9})
    )
    panels = build_longctx_quality_panels(val_only, val_only)
    assert panels["P1_short_gen"]["val_bpb_trained"]["status"] == "filled"
    assert panels["P3_long_ctx"]["multi_T"]["status"] == "not_run"
    assert "VAL-COMPLETE-003" in (panels["P3_long_ctx"]["multi_T"]["reason"] or "")
    assert panels["P4_length_extrap"]["length_extrapolate"]["status"] == "not_run"
