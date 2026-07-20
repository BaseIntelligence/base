from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.scoring import (
    MEMORIZATION_GAP_THRESHOLD_BPB,
    MEMORIZATION_PENALTY_FACTOR,
    score_prequential_bpb,
)


def _manifest(
    *,
    bpb: float,
    covered_bytes: int = 1000,
    val_bpb_trained: float | None = None,
    val_bpb_random_init: float | None = None,
    heldout_delta: float | None = None,
    train_heldout_gap: float | None = None,
    train_bpb_converged: float | None = None,
    gap_basis: str | None = None,
    train_bpb_basis: str | None = None,
    val_bpb_basis: str | None = None,
    memorization_flag: bool | None = None,
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
    for key, value in (
        ("val_bpb_trained", val_bpb_trained),
        ("val_bpb_random_init", val_bpb_random_init),
        ("heldout_delta", heldout_delta),
        ("train_heldout_gap", train_heldout_gap),
        ("train_bpb_converged", train_bpb_converged),
        ("gap_basis", gap_basis),
        ("train_bpb_basis", train_bpb_basis),
        ("val_bpb_basis", val_bpb_basis),
        ("memorization_flag", memorization_flag),
    ):
        if value is not None:
            metrics[key] = value
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


def test_converged_gap_recomputed_from_converged_reference_flags_memorizer() -> None:
    # VAL-CHEAT-009 / VAL-SCORE-009: when the manifest carries a CONVERGED (final-checkpoint) train
    # bpb, the scorer derives the gap from THAT converged reference (not the inflated prequential
    # AUC). A converged train bpb far below the held-out val bpb is an excessive gap => flagged.
    score = score_prequential_bpb(
        _manifest(
            bpb=2.5,  # prequential (curve-averaged) AUC train bpb -- inflated by early steps
            val_bpb_trained=3.0,
            val_bpb_random_init=3.2,
            heldout_delta=0.2,
            train_bpb_converged=0.2,  # converged model fits seen train data tightly
            gap_basis="converged",
        )
    )
    assert score.train_heldout_gap == pytest.approx(3.0 - 0.2)
    assert score.train_heldout_gap > MEMORIZATION_GAP_THRESHOLD_BPB
    assert score.memorization_flag is True
    assert "memorization_gap" in score.flags
    assert score.memorization_penalty == pytest.approx(MEMORIZATION_PENALTY_FACTOR)


def test_converged_gap_bypasses_tokenizer_basis_gating() -> None:
    # The converged gap is measured byte-level on BOTH sides (host trained-model on train vs val),
    # so it is like-for-like by construction. Even when the run's prequential TRAIN basis was a
    # native tokenizer, a converged gap still flags a real memorizer (the m3 basis gating only
    # applies to the prequential-reference fallback, not the converged reference).
    score = score_prequential_bpb(
        _manifest(
            bpb=1.0,
            val_bpb_trained=3.0,
            train_bpb_converged=0.3,
            gap_basis="converged",
            train_bpb_basis="gpt2",
            val_bpb_basis="bytes",
        )
    )
    assert score.train_heldout_gap == pytest.approx(3.0 - 0.3)
    assert score.memorization_flag is True
    assert "memorization_gap" in score.flags


def test_converged_gap_benign_learner_not_flagged() -> None:
    # VAL-SCORE-009 (no flag on benign): a generalizing learner has converged train bpb ~ val bpb,
    # so the converged gap is small and the run is NOT flagged/penalized.
    score = score_prequential_bpb(
        _manifest(
            bpb=1.5,
            val_bpb_trained=1.4,
            val_bpb_random_init=2.6,
            heldout_delta=1.2,
            train_bpb_converged=1.2,
            gap_basis="converged",
        )
    )
    assert score.train_heldout_gap == pytest.approx(1.4 - 1.2)
    assert score.train_heldout_gap < MEMORIZATION_GAP_THRESHOLD_BPB
    assert score.memorization_flag is False
    assert score.memorization_penalty == pytest.approx(1.0)


def test_prequential_fallback_remains_basis_gated() -> None:
    # Regression guard: WITHOUT a converged reference the prequential fallback keeps the m3
    # tokenizer-basis gating (a benign tokenizer learner is never false-flagged on a cross-basis
    # gap).
    score = score_prequential_bpb(
        _manifest(
            bpb=1.0,
            val_bpb_trained=8.0,
            train_bpb_basis="gpt2",
            val_bpb_basis="bytes",
        )
    )
    assert score.memorization_flag is False
    assert score.train_heldout_gap is None
    assert "memorization_gap" not in score.flags


def test_converged_train_bpb_surfaced_in_manifest_blocks() -> None:
    # Observability: the converged reference + gap basis are exposed in both the metrics payload and
    # the challenge-authored score block so manifest-inspect can see how the gap was measured.
    score = score_prequential_bpb(
        _manifest(
            bpb=1.0,
            val_bpb_trained=3.0,
            train_bpb_converged=0.4,
            gap_basis="converged",
        )
    )
    payload = score.metrics_payload()
    block = score.manifest_score_block()
    assert payload["train_bpb_converged"] == pytest.approx(0.4)
    assert payload["gap_basis"] == "converged"
    assert block["train_bpb_converged"] == pytest.approx(0.4)
    assert block["gap_basis"] == "converged"
