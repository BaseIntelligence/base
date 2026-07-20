"""Complete View efficiency / stability / robustness / nice-to-have (VAL-COMPLETE-008..012).

CPU unit fixtures only: no NVIDIA, no Lium, no REAL TEE claim. Streams and
captures are synthetic; densified sample-eff reuses scorecard trapezoid path.
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_NICE_TO_HAVE,
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
    assert_complete_view_document,
    validate_complete_view_document,
)
from prism_challenge.evaluator.complete_view_eff import (
    DENSE_SAMPLE_EFF_MARKS_TOKENS,
    FamilyEffStability,
    apply_dense_sample_eff_to_record,
    apply_stability_to_record,
    build_complete_view_with_eff_stability,
    build_nice_to_have_panel,
    build_state_footprint,
    build_train_eval_efficiency,
    dense_sample_efficiency_from_stream,
    derive_quality_efficiency_ratios,
    fixture_family_eff_stability,
    multi_order_residual,
    multi_seed_stability,
    online_bpb_curve_summary,
    rapid_decay_flag_from_online,
)
from prism_challenge.evaluator.official_comparison import OfficialScoreRecord
from prism_challenge.evaluator.scorecard_suite import DEFAULT_SAMPLE_EFF_MARKS_TOKENS


def _rec(
    *,
    label: str,
    heldout_delta: float = 3.5,
    bpb: float = 0.12,
    sample_eff_auc: float | None = 0.55,
    params: int | None = 7_000_000,
    peak_vram_gib: float | None = 0.55,
    tokens_per_s: float | None = 30_000.0,
    seed_count: int = 3,
    grad_spike_rate: float | None = None,
    nan_inf_events: int | None = None,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        val_bpb_trained=None,
        memorization_flag=False,
        train_heldout_gap=0.6,
        step0_anomaly=False,
        valid=True,
        seed_count=seed_count,
        bpb_std=0.001,
        heldout_std=0.02,
        sample_eff_auc=sample_eff_auc,
        params=params,
        peak_vram_gib=peak_vram_gib,
        tokens_per_s=tokens_per_s,
        grad_spike_rate=grad_spike_rate,
        nan_inf_events=nan_inf_events,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


def _decreasing_online_loss(n: int = 200, start: float = 4.0, end: float = 0.8) -> list[float]:
    """Synthetic nats/token stream declining roughly linearly then flat."""
    out: list[float] = []
    for i in range(n):
        t = i / max(1, n - 1)
        out.append(float(start + (end - start) * t))
    return out


def _family(
    *,
    start_loss: float = 4.0,
    end_loss: float = 0.8,
    params: int = 6_963_840,
    peak_vram: float = 0.55,
    tok_s: float = 35_000.0,
    heldout: float = 3.46,
    multi_order_blocked: bool = False,
    with_heldout_marks: bool = False,
    grad_rates: tuple[float, float, float] = (0.0, 0.0, 0.01),
    nan_events: tuple[int, int, int] = (0, 0, 0),
) -> FamilyEffStability:
    online = _decreasing_online_loss(n=240, start=start_loss, end=end_loss)
    held_marks = None
    if with_heldout_marks:
        held_marks = {50_000: 2.5, 100_000: 1.8, 250_000: 1.2, 500_000: 0.9}
    stability = {
        1337: {"grad_spike_rate": grad_rates[0], "nan_inf_events": nan_events[0]},
        2027: {"grad_spike_rate": grad_rates[1], "nan_inf_events": nan_events[1]},
        4242: {"grad_spike_rate": grad_rates[2], "nan_inf_events": nan_events[2]},
    }
    return fixture_family_eff_stability(
        online_loss=online,
        params=params,
        peak_vram_train_gib=peak_vram,
        tokens_per_s_train=tok_s,
        peak_vram_eval_by_T={
            128: peak_vram * 0.4,
            256: peak_vram * 0.5,
            512: peak_vram * 0.7,
            1024: peak_vram,
        },
        tokens_per_s_eval_by_T={
            128: tok_s * 1.2,
            256: tok_s,
            512: tok_s * 0.8,
            1024: tok_s * 0.5,
        },
        state_bytes_by_T={128: 1e6, 256: 2e6, 512: 4e6, 1024: 8e6},
        activation_peak_bytes_by_T={128: 2e6, 256: 4e6, 512: 8e6, 1024: 1.6e7},
        stability_per_seed=stability,
        heldout_delta=heldout,
        heldout_at_marks=held_marks,
        multi_order_blocked=multi_order_blocked,
        multi_order_a=heldout,
        multi_order_b=heldout * 0.97,
        device="fixture",
    )


# --- VAL-COMPLETE-008 ------------------------------------------------------------


def test_val_complete_008_dense_sample_eff_object() -> None:
    online = _decreasing_online_loss()
    dense = dense_sample_efficiency_from_stream(online, token_budget=500_000)
    payload = dense.as_dict()
    assert payload["status"] == "filled"
    # Denser than base 4 marks.
    assert len(payload["marks_tokens"]) > len(DEFAULT_SAMPLE_EFF_MARKS_TOKENS)
    assert set(DENSE_SAMPLE_EFF_MARKS_TOKENS).issubset(set(payload["marks_tokens"]))
    assert len(payload["bpb_at_marks"]) == len(payload["marks_tokens"])
    assert len(payload["quality_at_marks"]) == len(payload["marks_tokens"])
    assert math.isfinite(payload["auc"])
    # Curve summary p10/median/p90 present.
    cs = payload["curve_summary"]
    assert cs["p10_bpb"] is not None
    assert cs["median_bpb"] is not None
    assert cs["p90_bpb"] is not None
    assert cs["mean_bpb"] is not None
    assert cs["n_samples"] == float(len(online))
    # Host-stream documented; heldout checkpoint not artificially invented.
    held = payload["heldout_at_marks"]
    assert held is not None
    assert held["status"] == "not_run"
    assert "host_stream" in (held.get("reason") or "")

    fam_a = _family(start_loss=4.0, end_loss=0.9, heldout=3.46)
    fam_b = _family(start_loss=3.8, end_loss=0.7, heldout=4.70, params=6_672_256)
    a = _rec(label="transformer-tiny-1m", heldout_delta=3.46)
    b = _rec(label="mamba-tiny-1m", heldout_delta=4.70)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam_a, fam_b=fam_b)
    assert_complete_view_document(doc)
    p2 = doc["panels"]["P2_sample_efficiency"]
    assert p2["dense_marks"]["status"] == "filled"
    assert p2["curve_summary"]["status"] == "filled"
    assert p2["sample_eff_dense"]["status"] == "filled"
    assert p2["dense_marks"]["a"]["marks_tokens"]
    assert len(p2["marks_tokens"]) > 4
    assert p2["quality_auc"]["a"] is not None
    assert p2["quality_auc"]["b"] is not None


def test_val_complete_008_heldout_at_marks_when_available() -> None:
    fam = _family(with_heldout_marks=True)
    assert fam.sample_eff is not None
    held = fam.sample_eff.heldout_at_marks
    assert held is not None
    assert held["status"] == "filled"
    assert held["marks"]["50000"] == pytest.approx(2.5)

    a = _rec(label="A")
    b = _rec(label="B")
    fam_b = _family(with_heldout_marks=True, heldout=4.0, tok_s=3000.0)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam, fam_b=fam_b)
    p2 = doc["panels"]["P2_sample_efficiency"]
    assert p2["heldout_at_marks"]["status"] == "filled"
    assert p2["heldout_at_marks"]["a"]["marks"] is not None
    assert p2["heldout_at_marks"]["b"]["marks"] is not None


def test_val_complete_008_online_curve_summary_empty_honest() -> None:
    empty = online_bpb_curve_summary([])
    assert empty["n_samples"] == 0.0
    assert empty["median_bpb"] is None


def test_val_complete_008_stamp_record_sample_eff() -> None:
    dense = dense_sample_efficiency_from_stream(_decreasing_online_loss())
    stamped = apply_dense_sample_eff_to_record(_rec(label="A", sample_eff_auc=None), dense)
    assert stamped.sample_eff_auc == pytest.approx(dense.auc)
    assert stamped.sample_eff_marks is not None
    assert len(stamped.sample_eff_marks) == len(dense.bpb_at_marks)


# --- VAL-COMPLETE-009 ------------------------------------------------------------


def test_val_complete_009_efficiency_train_eval_and_footprint() -> None:
    fam_a = _family(params=6_963_840, peak_vram=0.55, tok_s=35_000.0)
    fam_b = _family(params=6_672_256, peak_vram=0.56, tok_s=3_280.0, heldout=4.7)
    assert fam_a.efficiency is not None
    eff = fam_a.efficiency.as_dict()
    assert eff["status"] == "filled"
    assert eff["peak_vram_train_gib"] == pytest.approx(0.55)
    assert "128" in eff["peak_vram_eval_by_T"]
    assert "1024" in eff["tokens_per_s_eval_by_T"]
    assert eff["step_time_ms"]["mean"] is not None
    assert eff["step_time_ms"]["p99"] is not None
    assert eff["sole_rank_forbidden"] is True

    assert fam_a.state_footprint is not None
    sp = fam_a.state_footprint.as_dict()
    assert sp["status"] == "filled"
    assert "512" in sp["state_bytes_by_T"]
    assert "512" in sp["activation_peak_bytes_by_T"]
    assert sp["param_bytes"] is not None

    a = _rec(label="transformer-tiny-1m", params=6_963_840, peak_vram_gib=0.55)
    b = _rec(label="mamba-tiny-1m", params=6_672_256, peak_vram_gib=0.56, tokens_per_s=3280.0)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam_a, fam_b=fam_b)
    assert_complete_view_document(doc)
    p5 = doc["panels"]["P5_efficiency"]
    assert p5["peak_vram_eval_by_T"]["status"] == "filled"
    assert p5["tokens_per_s_eval_by_T"]["status"] == "filled"
    assert p5["step_time_ms"]["status"] == "filled"
    assert p5["train_eval_efficiency"]["status"] == "filled"
    assert p5["tokens_per_s_train"]["a"] == pytest.approx(35_000.0)
    assert p5["tokens_per_s_train"]["b"] == pytest.approx(3_280.0)
    assert p5["sole_rank_forbidden"] is True

    p6 = doc["panels"]["P6_memory_state"]
    assert p6["state_footprint_bytes_by_T"]["status"] == "filled"
    assert p6["activation_peak_bytes_by_T"]["status"] == "filled"
    assert p6["state_footprint"]["status"] == "filled"
    assert p6["state_footprint"]["a"]["state_bytes_by_T"]["256"] == pytest.approx(2e6)


def test_val_complete_009_builders_reject_empty_footprint() -> None:
    with pytest.raises(ValueError, match="state_footprint requires"):
        build_state_footprint()


def test_val_complete_009_train_eval_builder_partial_ok() -> None:
    # Train-only is allowed; eval maps may be empty with honest note.
    annex = build_train_eval_efficiency(
        params=1000,
        peak_vram_train_gib=0.5,
        tokens_per_s_train=1000.0,
    )
    assert annex.peak_vram_train_gib == pytest.approx(0.5)
    assert annex.peak_vram_eval_by_T == {}
    assert any("eval_by_T" in n for n in annex.notes)


# --- VAL-COMPLETE-010 ------------------------------------------------------------


def test_val_complete_010_stability_grad_nan_multi_seed() -> None:
    stab = multi_seed_stability(
        {
            1337: {"grad_spike_rate": 0.0, "nan_inf_events": 0},
            2027: {"grad_spike_rate": 0.02, "nan_inf_events": 0},
            4242: {"grad_spike_rate": 0.01, "nan_inf_events": 1},
        },
        seed_std_bpb=0.001,
        seed_std_heldout=0.02,
    )
    payload = stab.as_dict()
    assert payload["status"] == "filled"
    assert payload["grad_spike_rate"]["mean"] == pytest.approx((0.0 + 0.02 + 0.01) / 3)
    assert payload["grad_spike_rate"]["std"] is not None
    assert len(payload["grad_spike_rate"]["per_seed"]) == 3
    assert payload["nan_inf_events"]["total"] == 1
    assert payload["nan_inf_events"]["mean"] == pytest.approx(1.0 / 3)
    assert payload["instability_flag"] is True  # nan total > 0
    assert list(payload["seeds"]) == [1337, 2027, 4242]

    fam_a = _family(grad_rates=(0.0, 0.0, 0.0), nan_events=(0, 0, 0))
    fam_b = _family(
        heldout=4.7,
        tok_s=3280.0,
        params=6_672_256,
        grad_rates=(0.01, 0.0, 0.02),
        nan_events=(0, 1, 0),
    )
    a = _rec(label="transformer-tiny-1m")
    b = _rec(label="mamba-tiny-1m", heldout_delta=4.7)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam_a, fam_b=fam_b)
    p7 = doc["panels"]["P7_stability_robustness"]
    assert p7["grad_spike_rate"]["status"] == "filled"
    assert p7["nan_inf_events"]["status"] == "filled"
    # Not schema-null: both sides present as multi-seed objects.
    assert p7["grad_spike_rate"]["a"]["mean"] is not None
    assert p7["grad_spike_rate"]["b"]["mean"] is not None
    assert p7["nan_inf_events"]["a"]["total"] == 0
    assert p7["nan_inf_events"]["b"]["total"] == 1
    assert p7["stability_multi_seed"]["status"] == "filled"
    # Record stamp.
    assert fam_a.stability is not None
    stamped = apply_stability_to_record(a, fam_a.stability)
    assert stamped.grad_spike_rate is not None
    assert stamped.nan_inf_events is not None


def test_val_complete_010_rejects_missing_grad_fields() -> None:
    with pytest.raises(ValueError, match="missing grad_spike_rate"):
        multi_seed_stability({1: {"nan_inf_events": 0}})
    with pytest.raises(ValueError, match="missing nan_inf_events"):
        multi_seed_stability({1: {"grad_spike_rate": 0.0}})


# --- VAL-COMPLETE-011 ------------------------------------------------------------


def test_val_complete_011_multi_order_filled_and_derived_ratios() -> None:
    mo = multi_order_residual(order_a_primary=3.5, order_b_primary=3.4)
    assert mo.status == "filled"
    assert mo.delta_primary == pytest.approx(0.1)
    assert mo.reason is None

    fam_a = _family(multi_order_blocked=False, heldout=3.46)
    fam_b = _family(multi_order_blocked=False, heldout=4.70, tok_s=3280.0, params=6_672_256)
    a = _rec(label="A", heldout_delta=3.46, params=6_963_840, peak_vram_gib=0.55)
    b = _rec(label="B", heldout_delta=4.70, params=6_672_256, peak_vram_gib=0.56)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam_a, fam_b=fam_b)
    p7 = doc["panels"]["P7_stability_robustness"]
    assert p7["multi_order_delta"]["status"] == "filled"
    assert p7["multi_order_delta"]["a"]["delta_primary"] is not None
    assert p7["multi_order_delta"]["b"]["delta_primary"] is not None

    p5 = doc["panels"]["P5_efficiency"]
    assert p5["quality_per_param"]["status"] == "filled"
    assert p5["quality_per_gib"]["status"] == "filled"
    assert p5["quality_per_param"]["a"] is not None
    assert p5["quality_per_param"]["b"] is not None
    assert p5["quality_per_gib"]["a"] is not None
    assert p5["quality_per_gib"]["b"] is not None
    # Derived pure: quality_per_param = heldout / params.
    assert p5["quality_per_param"]["a"] == pytest.approx(3.46 / 6_963_840)
    assert p5["quality_per_gib"]["a"] == pytest.approx(3.46 / 0.55)


def test_val_complete_011_multi_order_blocked_with_reason() -> None:
    blocked = multi_order_residual(blocked=True)
    assert blocked.status == "BLOCKED"
    assert blocked.reason is not None
    assert "BLOCKED" in blocked.reason

    fam_a = _family(multi_order_blocked=True)
    fam_b = _family(multi_order_blocked=True, heldout=4.7, tok_s=3280.0)
    a = _rec(label="A")
    b = _rec(label="B", heldout_delta=4.7)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam_a, fam_b=fam_b)
    p7 = doc["panels"]["P7_stability_robustness"]
    assert p7["multi_order_delta"]["status"] == "BLOCKED"
    assert p7["multi_order_delta"]["reason"] is not None
    assert "BLOCKED" in p7["multi_order_delta"]["reason"]
    # Not a silent null shell: objects present with honesty reason.
    assert p7["multi_order_delta"]["a"]["status"] == "BLOCKED"
    assert p7["multi_order_delta"]["b"]["status"] == "BLOCKED"


def test_val_complete_011_derive_quality_ratios_fallback() -> None:
    ratios = derive_quality_efficiency_ratios(
        heldout_delta=None,
        sample_eff_auc=0.5,
        params=1000,
        peak_vram_gib=2.0,
        prefer="heldout_delta",
    )
    assert ratios.quality_proxy_name == "quality_auc"
    assert ratios.quality_per_param == pytest.approx(0.5 / 1000)
    assert ratios.quality_per_gib == pytest.approx(0.5 / 2.0)
    with pytest.raises(ValueError, match="requires a finite quality proxy"):
        derive_quality_efficiency_ratios(params=1000, peak_vram_gib=1.0)


# --- VAL-COMPLETE-012 ------------------------------------------------------------


def test_val_complete_012_nice_to_have_no_silent_omission() -> None:
    panel = build_nice_to_have_panel()
    assert panel["status"] == "nice_to_have"
    assert panel["no_silent_omission"] is True
    keys = {e["key"] for e in panel["entries"]}
    expected = {str(r["key"]) for r in COMPLETE_VIEW_NICE_TO_HAVE}
    assert keys == expected
    for entry in panel["entries"]:
        # Explicit null + reason when not run.
        if entry["status"] == "not_run":
            assert entry["a"] is None
            assert entry["b"] is None
            assert entry["reason"]

    # Partial fill does not omit remaining catalogue keys.
    filled = {
        "rapid_decay_flag": {
            "a": {"flag": False, "status": "filled"},
            "b": {"flag": True, "status": "filled"},
            "status": "filled",
        },
        "ece_entropy_calibration": {
            "a": None,
            "b": None,
            "status": "not_run",
            "reason": "calibration_probe_not_run",
        },
    }
    panel2 = build_nice_to_have_panel(filled=filled)
    by_key = {e["key"]: e for e in panel2["entries"]}
    assert by_key["rapid_decay_flag"]["status"] == "filled"
    assert by_key["rapid_decay_flag"]["a"] is not None
    assert by_key["ece_entropy_calibration"]["reason"] == "calibration_probe_not_run"
    assert len(panel2["entries"]) == len(COMPLETE_VIEW_NICE_TO_HAVE)

    fam_a = _family()
    fam_b = _family(heldout=4.7, tok_s=3280.0, params=6_672_256)
    a = _rec(label="A")
    b = _rec(label="B", heldout_delta=4.7)
    doc = build_complete_view_with_eff_stability(
        a,
        b,
        fam_a=fam_a,
        fam_b=fam_b,
        nice_to_have_filled={
            "multi_budget_slope": {
                "a": None,
                "b": None,
                "status": "not_run",
                "reason": "multi_budget_scaling_not_run_single_500k_pin",
            },
            "free_gen_loop_collapse": {
                "a": None,
                "b": None,
                "status": "not_run",
                "reason": "ar_gen_proxy_not_run",
            },
        },
    )
    assert_complete_view_document(doc)
    p8 = doc["panels"]["P8_calibration_entropy_optional"]
    assert p8["no_silent_omission"] is True
    assert len(p8["entries"]) == len(COMPLETE_VIEW_NICE_TO_HAVE)
    by_key2 = {e["key"]: e for e in p8["entries"]}
    # Rapid-decay was auto-derived from online stream via fixture side.
    assert by_key2["rapid_decay_flag"]["status"] == "filled"
    assert by_key2["multi_budget_slope"]["reason"] is not None
    assert by_key2["free_gen_loop_collapse"]["reason"] is not None


def test_val_complete_012_rapid_decay_detector() -> None:
    improving = _decreasing_online_loss(n=64, start=3.0, end=0.5)
    flag_ok = rapid_decay_flag_from_online(improving, rebound_frac=0.10)
    assert flag_ok["status"] == "filled"
    assert flag_ok["flag"] is False

    rebound = list(improving) + [float(improving[-1]) * 2.0] * 16
    flag_bad = rapid_decay_flag_from_online(rebound, rebound_frac=0.10)
    assert flag_bad["status"] == "filled"
    assert flag_bad["flag"] is True

    short = rapid_decay_flag_from_online([1.0, 0.9], min_samples=16)
    assert short["status"] == "not_run"
    assert short["reason"]


# --- Integration / identity -------------------------------------------------------


def test_complete_view_eff_document_identity_and_schema() -> None:
    fam_a = _family()
    fam_b = _family(heldout=4.7, tok_s=3280.0, params=6_672_256, multi_order_blocked=True)
    a = _rec(label="transformer-tiny-1m")
    b = _rec(label="mamba-tiny-1m", heldout_delta=4.7, tokens_per_s=3280.0)
    doc = build_complete_view_with_eff_stability(a, b, fam_a=fam_a, fam_b=fam_b)
    assert doc["schema"] == COMPLETE_VIEW_SCHEMA
    assert doc["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert "real_provider_tee" not in doc
    assert doc["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert doc["non_claims"]["prism_tee_product"] is False
    errors = validate_complete_view_document(doc)
    assert errors == []
    # Panels present and filled residual keys not pending.
    assert doc["panels"]["P2_sample_efficiency"]["dense_marks"]["status"] == "filled"
    assert doc["panels"]["P5_efficiency"]["peak_vram_eval_by_T"]["status"] == "filled"
    assert doc["panels"]["P6_memory_state"]["state_footprint"]["status"] == "filled"
    assert doc["panels"]["P7_stability_robustness"]["grad_spike_rate"]["status"] == "filled"
    assert doc["panels"]["P7_stability_robustness"]["multi_order_delta"]["status"] == "BLOCKED"
    assert doc["panels"]["P5_efficiency"]["quality_per_param"]["status"] == "filled"
    assert doc["panels"]["P8_calibration_entropy_optional"]["no_silent_omission"] is True
    # Efficiency never sole-ranks.
    assert doc["comparison"]["efficiency_sole_rank_forbidden"] is True
    assert doc["panels"]["P5_efficiency"]["sole_rank_forbidden"] is True
    assert doc["panels"]["P5_efficiency"]["overrides_polar_rule"] is False
