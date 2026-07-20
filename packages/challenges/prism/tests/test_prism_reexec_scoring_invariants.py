from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.scoring import (
    HELDOUT_DELTA_BPB_EPSILON,
    HELDOUT_DELTA_TIE_BREAK_WEIGHT,
    MEMORIZATION_GAP_THRESHOLD_BPB,
    MEMORIZATION_PENALTY_FACTOR,
    ScoreValidationError,
    score_prequential_bpb,
)


def _manifest(
    *,
    bpb: float,
    covered_bytes: int = 2048,
    online_loss: list[float] | None = None,
    step0_anomaly: bool = False,
    nan_inf: bool = False,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    train_heldout_gap: float | None = None,
    train_bpb_basis: str = "bytes",
) -> dict:
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    losses = online_loss if online_loss is not None else [3.0, 2.5, 2.0]
    metrics: dict = {
        "online_loss": losses,
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "covered_bytes": covered_bytes,
        "predicted_tokens": 96,
        "step0_loss": losses[0] if losses else None,
        "consumed_batches": len(losses),
        "random_init_baseline_nats": math.log(128),
        "nan_inf_batches": 1 if nan_inf else 0,
        "train_bpb_basis": train_bpb_basis,
    }
    if heldout_delta is not None:
        metrics["heldout_delta"] = heldout_delta
    if val_bpb_trained is not None:
        metrics["val_bpb_trained"] = val_bpb_trained
        metrics["val_bpb_basis"] = "bytes"
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "sub-inv",
        "metrics": metrics,
        "anti_cheat": {
            "step0_anomaly": step0_anomaly,
            "nan_inf_detected": nan_inf,
        },
    }


# --- VAL-PRISM-007: step-0 / smuggled-pretrained-weights anomaly zeroes the score ----------------


def test_step0_anomaly_zeroes_score_regardless_of_bpb():
    manifest = _manifest(bpb=0.2, step0_anomaly=True)
    score = score_prequential_bpb(manifest)
    assert score.anti_cheat_multiplier == 0.0
    assert score.final_score == 0.0
    assert "step0_anomaly" in score.flags


def test_no_anomaly_is_positive_ranking_score():
    score = score_prequential_bpb(_manifest(bpb=0.2, step0_anomaly=False))
    assert score.anti_cheat_multiplier == 1.0
    assert score.final_score > 0.0


# --- VAL-PRISM-009: degenerate runs raise ScoreValidationError (never a fabricated zero) ----------


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest(bpb=0.5, covered_bytes=0),
        _manifest(bpb=0.5, online_loss=[]),
    ],
)
def test_degenerate_manifest_raises(manifest):
    with pytest.raises(ScoreValidationError):
        score_prequential_bpb(manifest)


def test_non_positive_bpb_raises():
    with pytest.raises(ScoreValidationError):
        score_prequential_bpb(_manifest(bpb=0.0))


# --- VAL-RESLAB-006: held-out is primary; bpb is bounded secondary -------------------------------


def test_strictly_better_heldout_outranks_despite_worse_bpb():
    # Emission invert: a run with a clearly larger held-out delta outranks a strictly lower-bpb
    # short-train compressor outside the near-tie epsilon band.
    low_bpb = score_prequential_bpb(_manifest(bpb=0.30, heldout_delta=0.0, val_bpb_trained=0.30))
    high_bpb_big_delta = score_prequential_bpb(
        _manifest(
            bpb=0.30 + 10 * HELDOUT_DELTA_BPB_EPSILON, heldout_delta=5.0, val_bpb_trained=0.30
        )
    )
    assert high_bpb_big_delta.final_score > low_bpb.final_score


def test_within_epsilon_band_lower_bpb_wins_but_secondary_bounded():
    # Same held-out primary; secondary bpb breaks the near-tie and stays bounded.
    base_delta = 0.30
    worse_bpb = score_prequential_bpb(
        _manifest(bpb=0.50, heldout_delta=base_delta, val_bpb_trained=base_delta)
    )
    better_bpb = score_prequential_bpb(
        _manifest(
            bpb=0.50 - 0.2 * HELDOUT_DELTA_BPB_EPSILON,
            heldout_delta=base_delta,
            val_bpb_trained=base_delta,
        )
    )
    assert better_bpb.final_score >= worse_bpb.final_score
    spread = abs(better_bpb.final_score - worse_bpb.final_score)
    assert spread <= HELDOUT_DELTA_TIE_BREAK_WEIGHT + 1e-9


# --- VAL-PRISM-012: excessive train-vs-held-out gap applies the memorization penalty --------------


def test_memorization_gap_halves_score():
    gap = MEMORIZATION_GAP_THRESHOLD_BPB + 0.5
    memorizer = score_prequential_bpb(
        _manifest(bpb=0.30, val_bpb_trained=0.30 + gap, train_heldout_gap=gap)
    )
    clean = score_prequential_bpb(_manifest(bpb=0.30, val_bpb_trained=0.30, train_heldout_gap=0.0))
    assert memorizer.memorization_flag is True
    assert memorizer.memorization_penalty == MEMORIZATION_PENALTY_FACTOR
    assert clean.memorization_flag is False
    # The memorizer ranks strictly below the equivalent non-memorizing run.
    assert memorizer.final_score < clean.final_score


def test_cross_basis_gap_is_not_false_flagged():
    # A benign tokenizer-basis learner whose apparent train-vs-byte-val gap is large must NOT be
    # flagged: the bases are not comparable, so the gap is suppressed.
    score = score_prequential_bpb(
        _manifest(
            bpb=0.30,
            val_bpb_trained=0.30 + MEMORIZATION_GAP_THRESHOLD_BPB + 1.0,
            train_bpb_basis="tokenizer",
        )
    )
    assert score.memorization_flag is False
