"""Official Comparison multimetric scorecard annex v1.1 core (VAL-SCORE-002..004,008..010).

Additive on prism_official_compare.v1. Synthetic fixtures only: no NVIDIA, no Lium,
no real TEE, no emission leaderboard rewrite.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from prism_challenge.evaluator.official_compare_harness import (
    DEFAULT_MAMBA_PROFILE,
    DEFAULT_TRANSFORMER_PROFILE,
    REPORT_SCHEMA,
    build_compare_report,
    default_protocol_pin,
    package_unknown_style_pair,
    run_dual_family_official_compare,
)
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_EPS_HELDOUT,
    OFFICIAL_EPS_LONG_CTX,
    OFFICIAL_LONG_CTX_CHANCE,
    OFFICIAL_LONG_CTX_FLOOR,
    OFFICIAL_LONG_CTX_RELATIVE_FLOOR,
    OFFICIAL_MIN_PUBLIC_SEEDS,
    PROTOCOL_ID,
    SCORECARD_ID,
    SCORECARD_PROVISIONAL_HONESTY_NOTE,
    OfficialScoreRecord,
    aggregate_official_records,
    attach_scorecard_to_report,
    build_scorecard_annex,
    compare_official,
    compare_official_scorecard,
    detect_polar_conflict,
    evaluate_pair_validity,
    evaluate_validity_gates,
    protocol_budget_constants,
)


def _rec(
    *,
    label: str,
    bpb: float = 1.5,
    heldout_delta: float | None = 0.5,
    seed_count: int = 3,
    bpb_std: float | None = 0.01,
    heldout_std: float | None = 0.02,
    memorization_flag: bool = False,
    train_heldout_gap: float | None = 0.15,
    step0_anomaly: bool = False,
    valid: bool = True,
    overfit_rate: float = 0.0,
    long_ctx_score: float | None = None,
    long_ctx_enabled: bool = False,
    long_ctx_needle: float | None = None,
    long_ctx_mqar: float | None = None,
    long_ctx_induction_copy: float | None = None,
    lag_nll: float | None = None,
    stop_token_budget: bool | None = True,
    finite_bpb: bool | None = True,
    param_cap_ok: bool | None = True,
    matched_pin: bool | None = True,
    challenge_authored: bool = True,
    force_instrument: bool | None = True,
    sample_eff_auc: float | None = None,
    nan_inf_events: int | None = 0,
    grad_spike_rate: float | None = 0.0,
    instability_flag: bool = False,
    params: int | None = 1_000_000,
) -> OfficialScoreRecord:
    floor_pass: bool | None = None
    if long_ctx_enabled and long_ctx_score is not None and math.isfinite(long_ctx_score):
        floor_pass = float(long_ctx_score) >= OFFICIAL_LONG_CTX_FLOOR
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        memorization_flag=memorization_flag,
        train_heldout_gap=train_heldout_gap,
        step0_anomaly=step0_anomaly,
        valid=valid,
        seed_count=seed_count,
        bpb_std=bpb_std,
        overfit_rate=overfit_rate,
        heldout_std=heldout_std,
        stop_token_budget=stop_token_budget,
        finite_bpb=finite_bpb,
        param_cap_ok=param_cap_ok,
        matched_pin=matched_pin,
        challenge_authored=challenge_authored,
        force_instrument=force_instrument,
        long_ctx_score=long_ctx_score,
        long_ctx_needle=long_ctx_needle,
        long_ctx_mqar=long_ctx_mqar,
        long_ctx_induction_copy=long_ctx_induction_copy,
        lag_nll=lag_nll,
        long_ctx_enabled=long_ctx_enabled,
        long_ctx_floor_pass=floor_pass,
        sample_eff_auc=sample_eff_auc,
        params=params,
        nan_inf_events=nan_inf_events,
        grad_spike_rate=grad_spike_rate,
        instability_flag=instability_flag,
    )


# --- VAL-SCORE-002: Default v1 rank preserved when no polar conflict -------------


def test_val_score_002_v1_winner_stable_when_long_ctx_disabled() -> None:
    """When long-ctx suite is disabled / not filled, scorecard preserves v1 winner."""
    a = _rec(label="A", heldout_delta=0.95, bpb=1.80, long_ctx_enabled=False)
    b = _rec(label="B", heldout_delta=0.25, bpb=1.10, long_ctx_enabled=False)

    v1 = compare_official(a, b)
    sc = compare_official_scorecard(a, b)

    assert v1.winner == "a"
    assert v1.reason == "primary_heldout"
    assert sc.winner == v1.winner
    assert sc.reason == v1.reason
    assert sc.tie_polar is False
    assert sc.crown_allowed is True
    assert sc.scorecard_id == SCORECARD_ID

    polar = detect_polar_conflict(a, b)
    assert polar.tie_polar is False
    assert polar.long_ctx_lead == "missing"


def test_val_score_002_v1_winner_stable_when_long_ctx_agrees() -> None:
    """When long-ctx agrees with short-gen (no reverse beyond ε), v1 winner stays."""
    a = _rec(
        label="A",
        heldout_delta=0.90,
        bpb=1.5,
        long_ctx_enabled=True,
        long_ctx_score=0.55,
        long_ctx_needle=0.6,
        long_ctx_mqar=0.5,
    )
    b = _rec(
        label="B",
        heldout_delta=0.20,
        bpb=1.2,
        long_ctx_enabled=True,
        long_ctx_score=0.40,  # A still better on long too
        long_ctx_needle=0.4,
        long_ctx_mqar=0.4,
    )
    v1 = compare_official(a, b)
    sc = compare_official_scorecard(a, b)
    assert v1.winner == "a"
    assert sc.winner == "a"
    assert sc.reason == "primary_heldout"
    assert sc.tie_polar is False
    assert sc.crown_allowed is True


def test_val_score_002_fixture_dual_family_preserves_heldout_primary(tmp_path) -> None:
    """End-to-end harness: default dual-family fixture still yields primary_heldout A."""
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=DEFAULT_TRANSFORMER_PROFILE,
        side_b_profile=DEFAULT_MAMBA_PROFILE,
        device_class="fixture",
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["protocol_id"] == PROTOCOL_ID
    assert report["scorecard_id"] == SCORECARD_ID
    assert report["ranking"]["winner"] == "a"
    assert report["ranking"]["reason"] == "primary_heldout"
    assert report["ranking"]["tie_polar"] is False
    assert report["ranking"]["crown_allowed"] is True
    assert report["ranking"]["default_v1_preserved_when_no_polar_conflict"] is True
    assert report["scorecard"]["polar"]["tie_polar"] is False


# --- VAL-SCORE-003: TIE_POLAR when short-gen and long-ctx disagree ----------------


def test_val_score_003_tie_polar_when_axes_disagree() -> None:
    """A better short heldout, B better long_ctx by >ε → TIE_POLAR, crown_allowed=false."""
    a = _rec(
        label="short_winner",
        heldout_delta=0.90,
        bpb=1.5,
        long_ctx_enabled=True,
        long_ctx_score=0.30,  # weaker long
        long_ctx_needle=0.25,
        long_ctx_mqar=0.30,
    )
    b = _rec(
        label="long_winner",
        heldout_delta=0.20,
        bpb=1.4,
        long_ctx_enabled=True,
        long_ctx_score=0.70,  # stronger long by >> ε_l
        long_ctx_needle=0.75,
        long_ctx_mqar=0.65,
    )
    # Pure v1 would crown A on short-gen.
    assert compare_official(a, b).winner == "a"
    polar = detect_polar_conflict(a, b)
    assert polar.tie_polar is True
    assert polar.crown_allowed is False
    assert polar.short_gen_lead == "a"
    assert polar.long_ctx_lead == "b"

    sc = compare_official_scorecard(a, b)
    assert sc.winner == "tie"
    assert sc.reason == "tie_polar"
    assert sc.tie_polar is True
    assert sc.crown_allowed is False
    assert sc.scorecard_id == SCORECARD_ID


def test_val_score_003_tie_polar_floor_veto_asymmetric() -> None:
    """A fails long_ctx floor, B passes, short still favors A → TIE_POLAR."""
    a = _rec(
        label="A_short",
        heldout_delta=1.0,
        long_ctx_enabled=True,
        long_ctx_score=0.05,  # below floor
    )
    b = _rec(
        label="B_long",
        heldout_delta=0.2,
        long_ctx_enabled=True,
        long_ctx_score=0.40,  # above floor
    )
    assert a.long_ctx_floor_pass is False
    assert b.long_ctx_floor_pass is True
    assert compare_official(a, b).winner == "a"
    sc = compare_official_scorecard(a, b)
    assert sc.tie_polar is True
    assert sc.crown_allowed is False
    assert sc.winner == "tie"
    assert sc.reason == "tie_polar"
    annex = build_scorecard_annex(a, b, compare=sc)
    assert annex["polar"]["floor_veto_a"] is True
    assert annex["polar"]["floor_veto_b"] is False
    assert annex["ranking_overlay"]["authoritative_claim"] == "TIE_POLAR"


def test_val_score_003_no_solitary_supremacy_field_on_polar(tmp_path) -> None:
    """Scorecard vector shows both axes; ranking has no solitary arch crown on polar."""
    a = _rec(
        label="transformer-short",
        heldout_delta=0.8,
        long_ctx_enabled=True,
        long_ctx_score=0.25,
        seed_count=3,
    )
    b = _rec(
        label="mamba-long",
        heldout_delta=0.2,
        long_ctx_enabled=True,
        long_ctx_score=0.65,
        seed_count=3,
    )
    sc = compare_official_scorecard(a, b)
    pin = default_protocol_pin()
    packed = package_unknown_style_pair(tmp_path / "pkgs")
    report = build_compare_report(
        pin=pin,
        side_a=a,
        side_b=b,
        packed=packed,
        result=sc,
        device_class="fixture",
    )
    assert report["ranking"]["winner"] == "tie"
    assert report["ranking"]["reason"] == "tie_polar"
    assert report["ranking"]["crown_allowed"] is False
    assert report["ranking"]["outcome_label"]["winner_label"] == "TIE_POLAR"
    assert report["scorecard"]["vector"]["a"]["short_gen"]["heldout_delta"] == 0.8
    assert report["scorecard"]["vector"]["b"]["long_ctx"]["suite_mean"] == 0.65
    assert "architecture_supremacy" not in report
    assert "sole_crown" not in report


# --- VAL-SCORE-004: Validity gates recorded (V tier) -----------------------------


def test_val_score_004_validity_gates_on_clean_record() -> None:
    rec = _rec(
        label="clean",
        seed_count=3,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        challenge_authored=True,
        force_instrument=True,
        step0_anomaly=False,
    )
    gates = evaluate_validity_gates(rec)
    assert gates.stop_token_budget is True
    assert gates.finite_bpb is True
    assert gates.step0_clean is True
    assert gates.param_cap is True
    assert gates.matched_pin is True
    assert gates.multi_seed_K == 3
    assert gates.multi_seed_public is True
    assert gates.multi_seed_provisional is False
    assert gates.challenge_authored is True
    assert gates.force_instrument is True
    assert gates.ok is True


def test_val_score_004_validity_gates_fail_step0_and_infinite() -> None:
    bad = _rec(
        label="bad",
        bpb=float("inf"),
        finite_bpb=False,
        step0_anomaly=True,
        valid=False,
        seed_count=1,
    )
    gates = evaluate_validity_gates(bad)
    assert gates.finite_bpb is False
    assert gates.step0_clean is False
    assert gates.ok is False
    assert "finite_bpb" in gates.reasons
    assert "step0_clean" in gates.reasons


def test_val_score_004_pair_validity_in_scorecard_annex() -> None:
    a = _rec(label="A", seed_count=3)
    b = _rec(label="B", seed_count=3)
    pair = evaluate_pair_validity(a, b, matched_pin=True)
    assert pair["stop_token_budget"] is True
    assert pair["finite_bpb"] is True
    assert pair["step0_clean"] is True
    assert pair["param_cap"] is True
    assert pair["matched_pin"] is True
    assert pair["multi_seed_K"] == 3
    assert pair["ok"] is True
    annex = build_scorecard_annex(a, b)
    v = annex["validity"]
    for key in (
        "stop_token_budget",
        "finite_bpb",
        "step0_clean",
        "param_cap",
        "matched_pin",
        "multi_seed_K",
        "challenge_authored",
    ):
        assert key in v
    assert v["multi_seed_K"] == 3


# --- VAL-SCORE-008: Multi-seed residual (K≥3 public) -----------------------------


def test_val_score_008_k1_is_provisional() -> None:
    k1 = _rec(label="k1", seed_count=1, heldout_delta=0.9, bpb_std=None)
    gates = evaluate_validity_gates(k1)
    assert gates.multi_seed_K == 1
    assert gates.multi_seed_public is False
    assert gates.multi_seed_provisional is True
    assert k1.multi_seed_provisional is True
    assert k1.is_public_multi_seed is False
    annex = build_scorecard_annex(k1, _rec(label="also_k1", seed_count=1))
    assert annex["multi_seed"]["K"] == 1
    assert annex["multi_seed"]["provisional"] is True
    assert annex["multi_seed"]["public"] is False


def test_val_score_008_k3_aggregates_public_and_std() -> None:
    seeds_a = [
        _rec(label="a0", heldout_delta=0.9, bpb=1.8, seed_count=1),
        _rec(label="a1", heldout_delta=1.0, bpb=1.7, seed_count=1),
        _rec(label="a2", heldout_delta=0.8, bpb=1.9, seed_count=1),
    ]
    # Clear per-seed marks so aggregate uses its own K from the list length.
    seeds_a = [replace(r, seed_count=1) for r in seeds_a]
    agg = aggregate_official_records(seeds_a, label="A")
    assert agg.seed_count == 3
    assert agg.seed_count >= OFFICIAL_MIN_PUBLIC_SEEDS
    assert agg.is_public_multi_seed is True
    assert agg.multi_seed_provisional is False
    assert agg.bpb_std is not None and agg.bpb_std > 0.0
    assert agg.heldout_std is not None and agg.heldout_std > 0.0
    gates = evaluate_validity_gates(agg)
    assert gates.multi_seed_public is True
    annex = build_scorecard_annex(agg, replace(agg, label="B", heldout_delta=0.1))
    assert annex["multi_seed"]["public"] is True
    assert annex["multi_seed"]["provisional"] is False
    assert annex["stability"]["bpb_std"]["a"] == pytest.approx(agg.bpb_std)
    assert annex["stability"]["heldout_std"]["a"] == pytest.approx(agg.heldout_std)


def test_multi_seed_long_ctx_floor_pass_retains_relative_to_chance() -> None:
    """Multi-seed recompute must not drop relative floors (absolute-only regression).

    Suite mean can sit above OFFICIAL_LONG_CTX_FLOOR while needle/mqar remain near
    chance (relative < OFFICIAL_LONG_CTX_RELATIVE_FLOOR). Aggregated floor_pass must
    stay False so multimetric.v1.1 publish retains relative-to-chance honesty.
    """
    # Per-seed: high induction lifts suite mean across absolute 0.15; needle/mqar at
    # chance so relative_to_chance ≈ 0 (well below relative floor 0.05).
    chance_needle = OFFICIAL_LONG_CTX_CHANCE["needle"]
    chance_mqar = OFFICIAL_LONG_CTX_CHANCE["mqar"]
    seeds = []
    for i, ind in enumerate((0.40, 0.42, 0.38)):
        needle = chance_needle  # relative 0
        mqar = chance_mqar  # relative 0
        suite_mean = (needle + mqar + ind) / 3.0
        assert suite_mean >= OFFICIAL_LONG_CTX_FLOOR
        # Explicit seed floor False (relative fail) even though absolute mean clears.
        seeds.append(
            replace(
                _rec(
                    label=f"s{i}",
                    seed_count=1,
                    long_ctx_enabled=True,
                    long_ctx_score=suite_mean,
                    long_ctx_needle=needle,
                    long_ctx_mqar=mqar,
                    long_ctx_induction_copy=ind,
                ),
                long_ctx_floor_pass=False,
            )
        )
    agg = aggregate_official_records(seeds, label="agg")
    assert agg.long_ctx_enabled is True
    assert agg.long_ctx_score is not None
    assert float(agg.long_ctx_score) >= OFFICIAL_LONG_CTX_FLOOR
    assert agg.long_ctx_needle == pytest.approx(chance_needle)
    assert agg.long_ctx_mqar == pytest.approx(chance_mqar)
    # Absolute-only would incorrectly pass; relative retention requires fail.
    assert agg.long_ctx_floor_pass is False
    annex = build_scorecard_annex(agg, replace(agg, label="other"))
    assert annex["long_ctx"]["floor_pass"]["a"] is False
    assert annex["long_ctx"]["floors_relative_to_chance"] is True
    assert annex["long_ctx"]["floors"]["relative_floor"] == OFFICIAL_LONG_CTX_RELATIVE_FLOOR


def test_multi_seed_long_ctx_floor_pass_true_when_relative_and_absolute_clear() -> None:
    """Aggregated floors pass only when absolute mean and relative needle/mqar clear."""
    seeds = []
    for i in range(3):
        needle, mqar, ind = 0.80, 0.75, 0.70
        suite_mean = (needle + mqar + ind) / 3.0
        seeds.append(
            replace(
                _rec(
                    label=f"ok{i}",
                    seed_count=1,
                    long_ctx_enabled=True,
                    long_ctx_score=suite_mean,
                    long_ctx_needle=needle,
                    long_ctx_mqar=mqar,
                    long_ctx_induction_copy=ind,
                ),
                long_ctx_floor_pass=True,
            )
        )
    agg = aggregate_official_records(seeds, label="agg_ok")
    assert agg.long_ctx_floor_pass is True
    assert float(agg.long_ctx_score or 0.0) >= OFFICIAL_LONG_CTX_FLOOR


# --- VAL-SCORE-009: Stability / memorization still scorecarded -------------------


def test_val_score_009_memo_step0_stability_on_scorecard() -> None:
    clean = _rec(
        label="clean",
        memorization_flag=False,
        train_heldout_gap=0.1,
        step0_anomaly=False,
        nan_inf_events=0,
        grad_spike_rate=0.0,
        instability_flag=False,
    )
    memo = _rec(
        label="memo",
        heldout_delta=0.99,  # high heldout might look strong
        memorization_flag=True,
        train_heldout_gap=1.5,
        overfit_rate=1.0,
        step0_anomaly=False,
        nan_inf_events=2,
        grad_spike_rate=0.4,
        instability_flag=True,
    )
    annex = build_scorecard_annex(clean, memo)
    mem_block = annex["memorization"]
    assert mem_block["memorization_flag_a"] is False
    assert mem_block["memorization_flag_b"] is True
    assert mem_block["memo_gap_b"] == pytest.approx(1.5)
    stab = annex["stability"]
    assert stab["nan_inf_events"]["b"] == 2
    assert stab["instability_flag"]["b"] is True
    assert stab["step0_anomaly"]["a"] is False
    # Memorizer does not silent-win under v1 either when memo residual applies on near-tie.
    near = (
        _rec(label="c", heldout_delta=0.5, bpb=1.5, memorization_flag=False),
        _rec(
            label="m",
            heldout_delta=0.5,
            bpb=1.5,
            memorization_flag=True,
            overfit_rate=1.0,
        ),
    )
    # Near-equal primary/secondary → anti_overfit residual prefers clean.
    r = compare_official(near[0], near[1])
    assert r.winner == "a"
    assert r.reason == "anti_overfit"


def test_val_score_009_step0_surface_disqualifies() -> None:
    good = _rec(label="good", heldout_delta=0.2)
    bad = _rec(
        label="step0",
        heldout_delta=5.0,
        step0_anomaly=True,
        valid=False,
    )
    r = compare_official(good, bad)
    assert r.winner == "a"
    assert r.reason == "step0_anomaly"
    annex = build_scorecard_annex(good, bad)
    assert annex["stability"]["step0_anomaly"]["b"] is True
    assert annex["validity"]["sides"]["b"]["step0_clean"] is False


# --- VAL-SCORE-010: Scorecard report schema emitted for dual-family --------------


def test_val_score_010_scorecard_annex_schema_keys() -> None:
    a = _rec(label="transformer", heldout_delta=0.9, seed_count=3, sample_eff_auc=None)
    b = _rec(label="mamba", heldout_delta=0.3, seed_count=3, sample_eff_auc=None)
    annex = build_scorecard_annex(a, b)
    assert annex["scorecard_id"] == SCORECARD_ID
    assert annex["tiers"] == ["V", "P", "S", "R"]
    required = {
        "scorecard_id",
        "scorecard_schema",
        "tiers",
        "multi_seed",
        "validity",
        "short_gen",
        "long_ctx",
        "sample_efficiency",
        "memorization",
        "efficiency",
        "stability",
        "polar",
        "vector",
        "ranking_overlay",
        "honesty_note",
    }
    assert required.issubset(annex.keys())
    assert annex["honesty_note"] == SCORECARD_PROVISIONAL_HONESTY_NOTE
    assert annex["polar"]["crown_allowed"] is True
    assert annex["long_ctx"]["enabled"] is False
    assert annex["long_ctx"]["suite_mean"]["a"] is None
    assert annex["sample_efficiency"]["a"]["auc"] is None
    assert annex["efficiency"]["wall_clock_never_ranks"] is True


def test_val_score_010_report_emits_scorecard_annex(tmp_path) -> None:
    report = run_dual_family_official_compare(tmp_path)
    assert report["schema"] == REPORT_SCHEMA
    assert report["scorecard_id"] == SCORECARD_ID
    assert "scorecard" in report
    sc = report["scorecard"]
    assert sc["scorecard_id"] == SCORECARD_ID
    assert "polar" in sc
    assert "vector" in sc
    assert sc["vector"]["a"]["label"]
    assert sc["vector"]["b"]["label"]
    assert report["honesty_note"] == SCORECARD_PROVISIONAL_HONESTY_NOTE
    ranking = report["ranking"]
    assert "crown_allowed" in ranking
    assert "tie_polar" in ranking
    assert ranking["default_v1_preserved_when_no_polar_conflict"] is True
    # attach path is idempotent-safe (re-apply does not drop keys)
    a = _rec(label="A", heldout_delta=0.9)
    b = _rec(label="B", heldout_delta=0.1)
    again = attach_scorecard_to_report(dict(report), a, b)
    assert again["scorecard_id"] == SCORECARD_ID
    assert again["scorecard"]["polar"] is not None


def test_protocol_budget_exposes_scorecard_constants() -> None:
    c = protocol_budget_constants()
    assert c["scorecard_id"] == SCORECARD_ID
    assert c["eps_long_ctx"] == OFFICIAL_EPS_LONG_CTX
    assert c["eps_heldout"] == OFFICIAL_EPS_HELDOUT
    assert c["long_ctx_floor"] == OFFICIAL_LONG_CTX_FLOOR
    assert c["min_public_seeds"] == OFFICIAL_MIN_PUBLIC_SEEDS
