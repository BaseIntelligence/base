"""Scorecard long-ctx / sample-efficiency / efficiency suite (VAL-SCORE-005..007).

CPU unit fixtures + GPU-ready hooks. No Lium, no real TEE, no emission rewrite.
Efficiency annex may be present but never sole-overrides scientific axes / polar rule.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from prism_challenge.evaluator.benchmarks import long_context as long_context_mod
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_LONG_CTX_FLOOR,
    OfficialScoreRecord,
    build_scorecard_annex,
    compare_official,
    compare_official_scorecard,
    detect_polar_conflict,
)
from prism_challenge.evaluator.scorecard_suite import (
    DEFAULT_SAMPLE_EFF_MARKS_TOKENS,
    LONG_CTX_CHANCE,
    LONG_CTX_RELATIVE_FLOOR,
    EfficiencyAnnex,
    aggregate_long_ctx_suite,
    apply_efficiency_to_record,
    apply_long_ctx_to_record,
    apply_sample_eff_to_record,
    build_efficiency_annex,
    documented_floors_relative_to_chance,
    enrich_record_with_suites,
    estimate_6nd_flops,
    lag_nll_from_bins,
    long_ctx_suite_schema,
    probe_next_token_accuracy,
    quality_from_bpb,
    relative_to_chance,
    run_long_ctx_fixture_suite,
    sample_efficiency_from_manifest,
    sample_efficiency_from_stream,
    score_closed_choice_accuracy,
    timed_tokens_probe,
)


def _base(
    *,
    label: str,
    heldout_delta: float = 0.5,
    bpb: float = 1.5,
    seed_count: int = 3,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        memorization_flag=False,
        train_heldout_gap=0.1,
        step0_anomaly=False,
        valid=True,
        seed_count=seed_count,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        challenge_authored=True,
        force_instrument=True,
        params=1_000_000,
    )


# --- VAL-SCORE-005: Long-context suite metrics defined and computable ------------


def test_val_score_005_floors_relative_to_chance_documented() -> None:
    floors = documented_floors_relative_to_chance()
    assert floors["floors_relative_to_chance"] is True
    assert floors["absolute_suite_mean_floor"] == OFFICIAL_LONG_CTX_FLOOR
    assert floors["relative_floor"] == LONG_CTX_RELATIVE_FLOOR
    assert floors["chance_baselines"]["needle"] == pytest.approx(0.25)
    assert floors["chance_baselines"]["mqar"] == pytest.approx(1.0 / 16.0)
    assert "needle" in floors["relative_floor_tasks"]
    assert "mqar" in floors["relative_floor_tasks"]
    schema = long_ctx_suite_schema()
    assert schema["floors"]["floors_relative_to_chance"] is True
    assert "suite_mean" in schema["required"]


def test_val_score_005_relative_to_chance_math() -> None:
    # chance=0.25, acc=0.25 → relative 0; acc=1 → 1; acc=0.625 → 0.5
    assert relative_to_chance(0.25, 0.25) == pytest.approx(0.0)
    assert relative_to_chance(1.0, 0.25) == pytest.approx(1.0)
    assert relative_to_chance(0.625, 0.25) == pytest.approx(0.5)
    assert relative_to_chance(0.0, 0.25) == pytest.approx(0.0)


def test_val_score_005_fixture_suite_emits_normalized_numeric_fields() -> None:
    """Needle / MQAR / induction-copy / lag-NLL produce numeric scorecard fields."""
    suite = run_long_ctx_fixture_suite(
        # Above chance for 4-way needle (0.25) and closed-16 mqar (~0.0625).
        needle_correct=[True, True, True, False],  # 0.75
        mqar_correct=[True, True, False, True],  # 0.75
        induction_correct=[True, True, True, True, False],  # 0.8
        lag_nll_by_bin={"lag_ge_64": 2.5, "lag_ge_256": 3.0, "lag_lt_16": 1.0},
        device="fixture",
    )
    payload = suite.as_dict()
    assert payload["enabled"] is True
    assert payload["floors_relative_to_chance"] is True
    assert payload["needle"] == pytest.approx(0.75)
    assert payload["mqar"] == pytest.approx(0.75)
    assert payload["induction_copy"] == pytest.approx(0.8)
    assert payload["suite_mean"] == pytest.approx((0.75 + 0.75 + 0.8) / 3.0)
    assert payload["lag_nll"] == pytest.approx(2.75)  # mean of long bins
    assert payload["floor_pass"] is True
    assert payload["relative_to_chance"]["needle"] >= LONG_CTX_RELATIVE_FLOOR
    assert payload["relative_to_chance"]["mqar"] >= LONG_CTX_RELATIVE_FLOOR
    assert payload["chance"]["needle"] == LONG_CTX_CHANCE["needle"]

    rec = apply_long_ctx_to_record(_base(label="A"), suite)
    assert rec.long_ctx_enabled is True
    assert rec.long_ctx_needle == pytest.approx(0.75)
    assert rec.long_ctx_mqar == pytest.approx(0.75)
    assert rec.long_ctx_induction_copy == pytest.approx(0.8)
    assert rec.long_ctx_score == pytest.approx(payload["suite_mean"])
    assert rec.lag_nll == pytest.approx(2.75)
    assert rec.long_ctx_floor_pass is True

    annex = build_scorecard_annex(rec, _base(label="B"))
    lc = annex["long_ctx"]
    assert lc["enabled"] is True
    assert lc["needle"]["a"] == pytest.approx(0.75)
    assert lc["mqar"]["a"] == pytest.approx(0.75)
    assert lc["induction_copy"]["a"] == pytest.approx(0.8)
    assert lc["lag_nll"]["a"] == pytest.approx(2.75)
    assert lc["suite_mean"]["a"] == pytest.approx(payload["suite_mean"])
    assert lc["floors_relative_to_chance"] is True
    assert "floors" in lc
    assert lc["floors"]["absolute_suite_mean_floor"] == OFFICIAL_LONG_CTX_FLOOR


def test_val_score_005_floor_fail_when_below_chance_relative() -> None:
    """Near-chance accuracies fail relative floor even if absolute mean edges past 0.15."""
    # Accuracy exactly at chance → relative 0 < 0.05.
    suite = aggregate_long_ctx_suite(
        needle=score_closed_choice_accuracy([False, False, False, True], task="needle"),  # 0.25
        mqar=score_closed_choice_accuracy([False] * 16, task="mqar"),  # 0.0 below chance
        induction_copy=0.30,
        enabled=True,
    )
    assert suite.suite_mean is not None
    # Absolute may be around (0.25+0+0.3)/3 ~ 0.18 > 0.15, but relative fails.
    assert suite.floor_pass is False
    assert any("relative_floor_fail" in n for n in suite.notes)


def test_val_score_005_gpu_ready_probe_hook_cpu() -> None:
    """probe_next_token_accuracy works as a CPU logits-fn hook (GPU-ready)."""

    def logits_fn(ctx: int) -> list[float]:
        # Vocabulary of 8; put mass on ctx % 8 for a toys success pattern.
        scores = [0.0] * 8
        scores[ctx % 8] = 5.0
        return scores

    contexts = [0, 1, 2, 3]
    targets = [0, 1, 2, 7]  # last one wrong under our logits
    outcomes = probe_next_token_accuracy(logits_fn, contexts, targets)
    assert outcomes == [True, True, True, False]
    closed = probe_next_token_accuracy(
        logits_fn,
        contexts,
        targets=[0, 1, 2, 3],
        candidate_sets=[[0, 1], [0, 1], [2, 5], [3, 4]],
    )
    assert all(isinstance(x, bool) for x in closed)
    # Closed set forces argmax within candidates — ctx3 maps candidate 3 max among [3,4].
    assert closed[3] is True


def test_val_score_005_benchmarks_long_context_module_usable() -> None:
    """Restored helper produces scores and re-exports suite API."""
    result = long_context_mod.long_context_from_length_accuracies({128: 0.5, 256: 0.4, 512: 0.3})
    assert 0.0 <= result.score <= 1.0
    assert result.collapse_penalty == pytest.approx(0.2)
    assert long_context_mod.LONG_CTX_CHANCE["needle"] == 0.25
    suite = long_context_mod.run_long_ctx_fixture_suite(
        needle_correct=[True, True],
        mqar_correct=[True, True],
        induction_correct=[True, True],
    )
    assert suite.suite_mean == pytest.approx(1.0)


def test_val_score_005_lag_nll_from_bins() -> None:
    assert lag_nll_from_bins({"lag_ge_64": 1.0, "lag_ge_256": 3.0}) == pytest.approx(2.0)
    assert math.isfinite(lag_nll_from_bins([4.0, 3.0, 2.0, 1.0, 0.5]))
    assert math.isinf(lag_nll_from_bins({}))


# --- VAL-SCORE-006: Sample-efficiency curve metric --------------------------------


def _descending_online_stream(n: int = 100, start: float = 4.0, end: float = 1.0) -> list[float]:
    """Synthetic online nats/token stream decreasing over steps (learning)."""
    if n < 2:
        return [start]
    return [start + (end - start) * (i / (n - 1)) for i in range(n)]


def test_val_score_006_sample_eff_from_fixture_stream() -> None:
    stream = _descending_online_stream(100, start=3.0, end=0.8)
    cum = [float(i + 1) * 64.0 for i in range(len(stream))]
    result = sample_efficiency_from_stream(
        stream,
        covered_bytes_cumulative=cum,
        token_budget=500_000,
        marks_tokens=DEFAULT_SAMPLE_EFF_MARKS_TOKENS,
    )
    payload = result.as_dict()
    assert payload["marks_tokens"] == list(DEFAULT_SAMPLE_EFF_MARKS_TOKENS)
    assert len(payload["bpb_at_marks"]) == 4
    assert len(payload["quality_at_marks"]) == 4
    assert all(math.isfinite(x) for x in payload["bpb_at_marks"])
    assert all(0.0 <= q <= 1.0 for q in payload["quality_at_marks"])
    assert math.isfinite(payload["auc"]) and payload["auc"] > 0.0
    # Learning stream: later marks ≤ early marks on bpb (monotone loss drop).
    assert payload["bpb_at_marks"][-1] <= payload["bpb_at_marks"][0] + 1e-9
    assert result.covered_bytes_total == pytest.approx(cum[-1])


def test_val_score_006_sample_eff_from_manifest_metrics() -> None:
    # Worse stream (flat high loss) should have lower AUC than a descent stream.
    good = sample_efficiency_from_manifest(
        {"online_loss": _descending_online_stream(80, 3.0, 0.5)},
        token_budget=500_000,
    )
    bad = sample_efficiency_from_manifest(
        {"online_loss": [3.0] * 80},
        token_budget=500_000,
    )
    assert good.auc > bad.auc
    assert quality_from_bpb(1.0) > quality_from_bpb(2.0)

    rec = apply_sample_eff_to_record(_base(label="fast"), good)
    assert rec.sample_eff_auc == pytest.approx(good.auc)
    assert rec.sample_eff_marks is not None
    assert len(rec.sample_eff_marks) == len(DEFAULT_SAMPLE_EFF_MARKS_TOKENS)

    annex = build_scorecard_annex(rec, apply_sample_eff_to_record(_base(label="slow"), bad))
    se = annex["sample_efficiency"]
    assert se["a"]["auc"] == pytest.approx(good.auc)
    assert se["b"]["auc"] == pytest.approx(bad.auc)
    assert se["a"]["marks"] is not None
    assert len(se["a"]["marks"]) == 4


def test_val_score_006_matched_pin_marks_both_sides() -> None:
    stream_a = _descending_online_stream(60, 2.5, 0.9)
    stream_b = _descending_online_stream(60, 2.5, 1.2)
    a_se = sample_efficiency_from_stream(stream_a, token_budget=500_000)
    b_se = sample_efficiency_from_stream(stream_b, token_budget=500_000)
    a = apply_sample_eff_to_record(_base(label="A", heldout_delta=0.9), a_se)
    b = apply_sample_eff_to_record(_base(label="B", heldout_delta=0.2), b_se)
    annex = build_scorecard_annex(a, b)
    assert annex["sample_efficiency"]["a"]["marks"] is not None
    assert annex["sample_efficiency"]["b"]["marks"] is not None
    assert annex["vector"]["a"]["sample_efficiency"]["auc"] == pytest.approx(a_se.auc)


# --- VAL-SCORE-007: Efficiency annex (VRAM, tokens/s, params) ---------------------


def test_val_score_007_efficiency_fields_present_and_non_overriding() -> None:
    annex_eff = build_efficiency_annex(
        params=6_960_000,
        peak_vram_gib=1.25,
        tokens_per_s=12_000.0,
        wall_clock_seconds=42.0,
        tokens_processed=500_000,
        device="cuda",
    )
    payload = annex_eff.as_dict()
    assert payload["params"] == 6_960_000
    assert payload["peak_vram_gib"] == pytest.approx(1.25)
    assert payload["tokens_per_s"] == pytest.approx(12_000.0)
    assert payload["wall_clock_never_ranks"] is True
    assert payload["sole_rank_forbidden"] is True
    assert payload["flops_diagnostic_only"] is True
    assert payload["overrides_scientific_axes"] is False
    assert payload["overrides_polar_rule"] is False
    assert payload["flops_6nd"] == pytest.approx(estimate_6nd_flops(6_960_000, 500_000) or 0.0)

    # Scientific axes still decide: A stronger heldout despite worse efficiency.
    a = apply_efficiency_to_record(
        _base(label="slow_but_better", heldout_delta=0.95, bpb=1.8),
        EfficiencyAnnex(
            params=10_000_000,
            peak_vram_gib=8.0,
            tokens_per_s=100.0,
            wall_clock_seconds=999.0,
            device="cuda",
        ),
    )
    b = apply_efficiency_to_record(
        _base(label="fast_worse_gen", heldout_delta=0.20, bpb=1.1),
        EfficiencyAnnex(
            params=1_000_000,
            peak_vram_gib=0.5,
            tokens_per_s=50_000.0,
            wall_clock_seconds=5.0,
            device="cuda",
        ),
    )
    v1 = compare_official(a, b)
    sc = compare_official_scorecard(a, b)
    assert v1.winner == "a"
    assert v1.reason == "primary_heldout"
    assert sc.winner == "a"
    assert sc.tie_polar is False
    assert sc.crown_allowed is True
    # Efficiency remains annex-only; rank key ignores wall_clock sole.
    assert a.wall_clock_seconds == pytest.approx(999.0)
    assert b.tokens_per_s == pytest.approx(50_000.0)

    report_annex = build_scorecard_annex(a, b)
    eff = report_annex["efficiency"]
    assert eff["params"]["a"] == 10_000_000
    assert eff["params"]["b"] == 1_000_000
    assert eff["peak_vram_gib"]["a"] == pytest.approx(8.0)
    assert eff["tokens_per_s"]["b"] == pytest.approx(50_000.0)
    assert eff["wall_clock_never_ranks"] is True
    assert eff["sole_rank_forbidden"] is True
    assert eff["overrides_scientific_axes"] is False
    assert eff["overrides_polar_rule"] is False


def test_val_score_007_efficiency_does_not_override_polar_rule() -> None:
    """Even with extreme efficiency, polar conflict still forces TIE_POLAR."""
    a = enrich_record_with_suites(
        _base(label="short_winner", heldout_delta=0.9),
        long_ctx=aggregate_long_ctx_suite(
            needle=0.30, mqar=0.30, induction_copy=0.30, enabled=True
        ),
        efficiency=build_efficiency_annex(
            params=100, peak_vram_gib=0.01, tokens_per_s=1e6, device="cpu"
        ),
    )
    b = enrich_record_with_suites(
        _base(label="long_winner", heldout_delta=0.2),
        long_ctx=aggregate_long_ctx_suite(
            needle=0.80, mqar=0.80, induction_copy=0.80, enabled=True
        ),
        efficiency=build_efficiency_annex(
            params=50_000_000,
            peak_vram_gib=20.0,
            tokens_per_s=10.0,
            device="cuda",
        ),
    )
    polar = detect_polar_conflict(a, b)
    assert polar.tie_polar is True
    sc = compare_official_scorecard(a, b)
    assert sc.winner == "tie"
    assert sc.reason == "tie_polar"
    assert sc.crown_allowed is False
    # Efficiency footprint is published but not the decision identity.
    annex = build_scorecard_annex(a, b, compare=sc)
    assert annex["efficiency"]["tokens_per_s"]["a"] == pytest.approx(1e6)
    assert annex["ranking_overlay"]["authoritative_claim"] == "TIE_POLAR"


def test_val_score_007_cpu_fixture_null_vram_honest() -> None:
    """When GPU absent / no allocator peak, VRAM stays null rather than invented."""
    annex = build_efficiency_annex(params=1_000_000, device="cpu", wall_clock_seconds=10.0)
    assert annex.params == 1_000_000
    assert annex.peak_vram_gib is None
    # GPU-ready timing probe still works on CPU.
    tokens, wall = timed_tokens_probe(lambda: 128, steps=3)
    assert tokens == 128 * 3
    assert wall >= 0.0


def test_enrich_applies_all_suites() -> None:
    long_ctx = run_long_ctx_fixture_suite(
        needle_correct=[True] * 4,
        mqar_correct=[True] * 4,
        induction_correct=[True] * 4,
        lag_nll_by_bin={"lag_ge_64": 1.2},
    )
    se = sample_efficiency_from_stream(_descending_online_stream(40))
    eff = build_efficiency_annex(params=2_000_000, tokens_per_s=1000.0, device="fixture")
    rec = enrich_record_with_suites(
        _base(label="full"),
        long_ctx=long_ctx,
        sample_eff=se,
        efficiency=eff,
    )
    assert rec.long_ctx_enabled is True
    assert rec.sample_eff_auc == pytest.approx(se.auc)
    assert rec.params == 2_000_000
    assert rec.tokens_per_s == pytest.approx(1000.0)
    # Polar postures still computed from scientific axes, not efficiency.
    other = replace(rec, label="other", heldout_delta=0.1, long_ctx_score=0.2)
    other = replace(other, long_ctx_enabled=True, long_ctx_floor_pass=True)
    annex = build_scorecard_annex(rec, other)
    assert annex["long_ctx"]["suite_mean"]["a"] is not None
    assert annex["sample_efficiency"]["a"]["auc"] is not None
    assert annex["efficiency"]["params"]["a"] == 2_000_000
