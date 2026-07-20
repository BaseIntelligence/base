"""VAL-RESLAB-006 / VAL-RESLAB-007: emission rank is held-out primary, bpb secondary.

Official-like invert on the production emission crown path (final_score / emission_rank_score)
consumed by leaderboard, q_arch_best, and q_recipe. Pure short-train lower-bpb alone cannot beat
a better held-out winner inside the epsilon rules.
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.scoring import (
    BPB_SECONDARY_TIE_BREAK_WEIGHT,
    EMISSION_HELDOUT_PRIMARY_OFFSET,
    HELDOUT_DELTA_BPB_EPSILON,
    LeaderboardRow,
    bpb_to_final_score,
    heldout_to_primary_score,
    rank_leaderboard,
    score_prequential_bpb,
)


def _manifest(
    *,
    bpb: float,
    covered_bytes: int = 1000,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    train_heldout_gap: float | None = None,
    step0_anomaly: bool = False,
) -> dict:
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    metrics: dict = {
        "online_loss": [2.0, 1.5, 1.0],
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "covered_bytes": covered_bytes,
        "predicted_tokens": 100,
        "step0_loss": 2.0,
        "consumed_batches": 3,
        "random_init_baseline_nats": math.log(256),
        "nan_inf_batches": 0,
    }
    if heldout_delta is not None:
        metrics["heldout_delta"] = heldout_delta
        metrics["held_out_delta"] = heldout_delta
    if val_bpb_trained is not None:
        metrics["val_bpb_trained"] = val_bpb_trained
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": metrics,
        "anti_cheat": {
            "step0_anomaly": step0_anomaly,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


# --- VAL-RESLAB-006: held-out primary beats pure lower bpb ---------------------------------------


def test_emission_primary_better_heldout_beats_better_bpb() -> None:
    """Pure short-train lower-bpb alone cannot beat a better held-out winner (epsilon rules)."""
    generalizer = score_prequential_bpb(_manifest(bpb=1.80, heldout_delta=1.20))
    compressor = score_prequential_bpb(_manifest(bpb=1.00, heldout_delta=0.10))
    assert generalizer.heldout_delta is not None
    assert compressor.heldout_delta is not None
    assert generalizer.heldout_delta > compressor.heldout_delta + HELDOUT_DELTA_BPB_EPSILON
    assert compressor.bpb < generalizer.bpb
    assert generalizer.final_score > compressor.final_score
    assert generalizer.emission_rank_score == pytest.approx(generalizer.final_score)
    assert generalizer.emission_crown_eligible is True
    assert compressor.emission_crown_eligible is True


def test_emission_secondary_bpb_breaks_near_tie_on_heldout() -> None:
    """Within held-out near-tie epsilon, lower prequential bpb wins (secondary)."""
    tighter = score_prequential_bpb(_manifest(bpb=1.00, heldout_delta=0.50))
    looser = score_prequential_bpb(
        _manifest(bpb=1.00 + 10 * HELDOUT_DELTA_BPB_EPSILON, heldout_delta=0.50)
    )
    assert tighter.heldout_delta == pytest.approx(looser.heldout_delta)
    assert tighter.bpb < looser.bpb
    assert tighter.final_score > looser.final_score


def test_emission_clear_heldout_gap_not_overturned_by_secondary() -> None:
    """Outside the held-out epsilon band, primary order is strict regardless of bpb."""
    strong = score_prequential_bpb(
        _manifest(bpb=2.5, heldout_delta=1.0 + 10 * HELDOUT_DELTA_BPB_EPSILON)
    )
    weak = score_prequential_bpb(_manifest(bpb=0.1, heldout_delta=0.0))
    assert strong.final_score > weak.final_score


def test_emission_primary_band_above_degraded_secondary_only() -> None:
    """Any honest held-out primary (even mild negative delta) outranks bpb-only degraded band."""
    with_heldout = score_prequential_bpb(_manifest(bpb=5.0, heldout_delta=-0.5))
    degraded = score_prequential_bpb(_manifest(bpb=0.01))  # missing held-out
    assert with_heldout.final_score > degraded.final_score
    assert with_heldout.final_score > EMISSION_HELDOUT_PRIMARY_OFFSET - 10.0
    assert degraded.final_score <= 1.0 + 1e-9
    assert degraded.emission_crown_eligible is False
    assert with_heldout.emission_crown_eligible is True


def test_skip_heldout_degrades_and_cannot_crown() -> None:
    """Worker-plane skip_heldout: ignore forged held-out, dirty flags, crown ineligible."""
    forged = _manifest(bpb=0.5, heldout_delta=9.9)
    score = score_prequential_bpb(forged, skip_heldout=True)
    assert score.heldout_delta is None
    assert score.emission_crown_eligible is False
    assert "heldout_skipped" in score.flags
    assert score.final_score == pytest.approx(bpb_to_final_score(0.5))
    # Compare against honest held-out peer: cannot outrank.
    honest = score_prequential_bpb(_manifest(bpb=1.5, heldout_delta=0.2))
    assert honest.final_score > score.final_score
    assert honest.emission_crown_eligible is True


def test_emission_payload_and_manifest_declare_heldout_primary() -> None:
    score = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.4))
    payload = score.metrics_payload()
    block = score.manifest_score_block()
    assert payload["primary_metric"] == "heldout_delta"
    assert payload["secondary_metric"] == "prequential_bpb"
    assert payload["emission_ranking"] == "heldout_primary_bpb_secondary"
    assert payload["emission_rank_score"] == pytest.approx(score.final_score)
    assert payload["emission_crown_eligible"] is True
    assert block["primary_metric"] == "heldout_delta"
    assert block["tie_breaker"] == "prequential_bpb"
    assert block["schema"] == "prism_score.v2"
    # Primary band baseline
    assert score.final_score == pytest.approx(
        heldout_to_primary_score(0.4) + score.final_score - heldout_to_primary_score(0.4),
        rel=0,
        abs=1.0,
    )
    assert score.final_score > heldout_to_primary_score(0.4) - 1e-6


def test_emission_step0_still_zeroes() -> None:
    score = score_prequential_bpb(_manifest(bpb=0.01, heldout_delta=5.0, step0_anomaly=True))
    assert score.anomaly is True
    assert score.final_score == 0.0
    assert score.emission_crown_eligible is False


def test_emission_memorization_penalty_ranks_below_benign() -> None:
    memorizer = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.5, train_heldout_gap=2.5))
    benign = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.5, train_heldout_gap=0.1))
    assert memorizer.memorization_flag is True
    assert benign.memorization_flag is False
    assert memorizer.final_score < benign.final_score


def test_emission_leaderboard_orders_heldout_primary() -> None:
    gen = score_prequential_bpb(_manifest(bpb=1.8, heldout_delta=1.0))
    comp = score_prequential_bpb(_manifest(bpb=0.5, heldout_delta=0.1))
    rows = [
        LeaderboardRow("comp", "hk-b", comp.final_score, "2024-01-01T00:00:00+00:00"),
        LeaderboardRow("gen", "hk-a", gen.final_score, "2024-01-02T00:00:00+00:00"),
    ]
    ranked = rank_leaderboard(rows)
    assert [r.submission_id for r in ranked] == ["gen", "comp"]


def test_bpb_secondary_term_is_bounded() -> None:
    """Secondary contribution stays within the documented weight bound."""
    a = score_prequential_bpb(_manifest(bpb=0.1, heldout_delta=0.5))
    b = score_prequential_bpb(_manifest(bpb=3.0, heldout_delta=0.5))
    # Same primary held-out; only secondary differs — delta of finals below the secondary cap.
    assert a.final_score > b.final_score
    assert abs(a.final_score - b.final_score) <= BPB_SECONDARY_TIE_BREAK_WEIGHT + 1e-9
