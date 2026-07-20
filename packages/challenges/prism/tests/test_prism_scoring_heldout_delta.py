from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.scoring import (
    MEMORIZATION_PENALTY_FACTOR,
    bpb_to_final_score,
    score_prequential_bpb,
)


def _manifest(
    *,
    bpb: float,
    covered_bytes: int = 1000,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    val_bpb_random_init: float | None = None,
    train_heldout_gap: float | None = None,
    memorization_flag: bool | None = None,
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
    if val_bpb_random_init is not None:
        metrics["val_bpb_random_init"] = val_bpb_random_init
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    if memorization_flag is not None:
        metrics["memorization_flag"] = memorization_flag
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


def test_scoring_heldout_delta_recorded_in_payload_and_score_block() -> None:
    # VAL-SCORE-007 / VAL-HARNESS-009: held-out delta + the two component bpb values are present,
    # numeric, and not pinned to 1.0.
    score = score_prequential_bpb(
        _manifest(
            bpb=1.0,
            heldout_delta=0.4,
            val_bpb_trained=2.6,
            val_bpb_random_init=3.0,
            train_heldout_gap=0.2,
        )
    )
    assert score.heldout_delta == pytest.approx(0.4)
    assert score.val_bpb_trained == pytest.approx(2.6)
    assert score.val_bpb_random_init == pytest.approx(3.0)
    assert score.train_heldout_gap == pytest.approx(0.2)

    payload = score.metrics_payload()
    assert payload["heldout_delta"] == pytest.approx(0.4)
    assert payload["held_out_delta"] == pytest.approx(0.4)
    assert payload["val_bpb_trained"] == pytest.approx(2.6)
    assert payload["val_bpb_random_init"] == pytest.approx(3.0)
    assert payload["train_heldout_gap"] == pytest.approx(0.2)
    assert payload["heldout_delta"] != 1.0

    block = score.manifest_score_block()
    assert block["heldout_delta"] == pytest.approx(0.4)
    assert block["held_out_delta"] == pytest.approx(0.4)
    assert block["val_bpb_trained"] == pytest.approx(2.6)
    assert block["val_bpb_random_init"] == pytest.approx(3.0)
    # Emission invert (VAL-RESLAB-006): held-out is primary; bpb is the secondary tie-breaker.
    assert block["primary_metric"] == "heldout_delta"
    assert block["tie_breaker"] == "prequential_bpb"


def test_scoring_larger_heldout_delta_ranks_better_on_near_tie() -> None:
    # VAL-RESLAB-006 / primary axis: equal secondary bpb -> the LARGER held-out delta wins.
    bigger = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.8))
    smaller = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.1))
    assert bigger.bpb == pytest.approx(smaller.bpb)
    assert bigger.final_score > smaller.final_score


def test_scoring_heldout_primary_overrides_clear_bpb() -> None:
    # VAL-RESLAB-006: clearer held-out primary beats pure lower bpb (invert of legacy bpb-primary).
    lower_bpb = score_prequential_bpb(_manifest(bpb=0.50, heldout_delta=0.0))
    higher_bpb = score_prequential_bpb(_manifest(bpb=0.60, heldout_delta=1.0))
    assert lower_bpb.bpb < higher_bpb.bpb
    assert higher_bpb.heldout_delta is not None and higher_bpb.heldout_delta > (
        lower_bpb.heldout_delta or 0.0
    )
    assert higher_bpb.final_score > lower_bpb.final_score


def test_scoring_excessive_memorization_gap_flagged_and_penalized() -> None:
    # VAL-SCORE-009 / VAL-HARNESS-016: an excessive train-vs-held-out gap is flagged and the score
    # degraded relative to a same-bpb, same-delta non-memorizing run.
    memorizer = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.3, train_heldout_gap=2.5))
    benign = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.3, train_heldout_gap=0.1))
    assert memorizer.memorization_flag is True
    assert memorizer.memorization_penalty == pytest.approx(MEMORIZATION_PENALTY_FACTOR)
    assert "memorization_gap" in memorizer.flags
    assert benign.memorization_flag is False
    assert benign.memorization_penalty == pytest.approx(1.0)
    assert memorizer.final_score < benign.final_score


def test_scoring_worse_than_random_ranks_below_baseline() -> None:
    # VAL-SCORE-018: an anti-learner (bpb HIGHER than baseline, NEGATIVE held-out delta) ranks
    # strictly below an honest no-op baseline (bpb == baseline, delta ~ 0).
    baseline = score_prequential_bpb(_manifest(bpb=3.0, heldout_delta=0.0))
    anti_learner = score_prequential_bpb(_manifest(bpb=4.0, heldout_delta=-0.5))
    assert anti_learner.bpb > baseline.bpb
    assert anti_learner.heldout_delta is not None and anti_learner.heldout_delta < 0.0
    assert anti_learner.final_score < baseline.final_score


def test_scoring_absent_heldout_is_degraded_not_crowned() -> None:
    # No secret val split scored -> held-out missing: delta None, degraded secondary-only band,
    # emission_crown_eligible=False (cannot crown emission without held-out; VAL-RESLAB-006).
    score = score_prequential_bpb(_manifest(bpb=0.5))
    assert score.heldout_delta is None
    assert score.train_heldout_gap is None
    assert score.memorization_flag is False
    assert score.memorization_penalty == pytest.approx(1.0)
    assert score.final_score == pytest.approx(bpb_to_final_score(0.5))
    assert score.emission_crown_eligible is False
    assert "heldout_missing" in score.flags


def test_scoring_step0_anomaly_zeroes_score_even_with_positive_delta() -> None:
    # A smuggled-weights step-0 anomaly zeroes the anti-cheat multiplier; a positive held-out delta
    # cannot rescue it.
    score = score_prequential_bpb(_manifest(bpb=0.01, heldout_delta=0.9, step0_anomaly=True))
    assert score.anomaly is True
    assert score.final_score == 0.0
