from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.scoring import score_prequential_bpb


def _manifest(
    *,
    bpb: float,
    covered_bytes: int = 1000,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    val_bpb_random_init: float | None = None,
    train_heldout_gap: float | None = None,
    train_bpb_basis: str | None = None,
    val_bpb_basis: str | None = None,
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
    if val_bpb_trained is not None:
        metrics["val_bpb_trained"] = val_bpb_trained
    if val_bpb_random_init is not None:
        metrics["val_bpb_random_init"] = val_bpb_random_init
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    if train_bpb_basis is not None:
        metrics["train_bpb_basis"] = train_bpb_basis
    if val_bpb_basis is not None:
        metrics["val_bpb_basis"] = val_bpb_basis
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": metrics,
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


def test_scoring_tokenizer_basis_mismatch_not_memorization_flagged() -> None:
    # DEFECT 2 / VAL-SCORE-009 + VAL-SCORE-004: a benign tokenizer-using learner whose TRAIN bpb was
    # measured on a native-tokenizer basis must NOT be memorization-flagged just because the host
    # measured the val bpb on the byte basis (the cross-basis "gap" is not comparable).
    score = score_prequential_bpb(
        _manifest(
            bpb=1.0,
            heldout_delta=0.3,
            val_bpb_trained=8.0,  # huge ONLY because of the byte-vs-tokenizer basis mismatch
            train_bpb_basis="gpt2",
            val_bpb_basis="bytes",
        )
    )
    assert score.memorization_flag is False
    assert "memorization_gap" not in score.flags
    assert score.memorization_penalty == pytest.approx(1.0)
    assert score.train_heldout_gap is None


def test_scoring_same_basis_still_flags_excessive_memorization_gap() -> None:
    # Like-for-like (both byte basis): an excessive gap is STILL flagged + penalized (no change).
    score = score_prequential_bpb(
        _manifest(
            bpb=1.0,
            heldout_delta=0.3,
            val_bpb_trained=4.0,
            train_bpb_basis="bytes",
            val_bpb_basis="bytes",
        )
    )
    assert score.memorization_flag is True
    assert "memorization_gap" in score.flags


def test_scoring_heldout_primary_reorders_despite_clearly_different_bpb() -> None:
    # VAL-RESLAB-006: clearer held-out primary keeps the better rank even when the weaker held-out
    # run carries a clearly better (lower) prequential bpb. Pure short-train bpb alone cannot
    # overturn a better held-out winner inside the rules.
    lower_bpb = score_prequential_bpb(_manifest(bpb=5.0, heldout_delta=-1.0))
    higher_bpb = score_prequential_bpb(_manifest(bpb=5.03, heldout_delta=1.0))
    assert lower_bpb.bpb < higher_bpb.bpb
    assert higher_bpb.final_score > lower_bpb.final_score


def test_scoring_delta_still_breaks_a_true_near_tie() -> None:
    # Equal secondary bpb: held-out primary orders by the larger delta (VAL-RESLAB-006).
    bigger = score_prequential_bpb(_manifest(bpb=2.0, heldout_delta=0.8))
    smaller = score_prequential_bpb(_manifest(bpb=2.0, heldout_delta=0.1))
    assert bigger.bpb == pytest.approx(smaller.bpb)
    assert bigger.final_score > smaller.final_score
