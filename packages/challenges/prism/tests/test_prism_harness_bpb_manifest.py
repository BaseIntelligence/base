from __future__ import annotations

import json
import math

import pytest
from test_prism_harness_online_loss import (
    ARCH_LM,
    TRAIN_LEARN,
    _read_manifest,
    _run_runner,
)

from prism_challenge.evaluator.scoring import score_prequential_bpb


def test_harness_manifest_carries_computed_bpb_score_block(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="bpb", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    # VAL-HARNESS-008: schema v2 manifest exists + parses.
    assert manifest["schema_version"] == "prism_run_manifest.v2"
    metrics = manifest["metrics"]
    score = manifest["score"]
    # VAL-HARNESS-009: COMPUTED score fields, not fabricated 1.0 diagnostics.
    bpb = score["prequential_bpb"]
    assert isinstance(bpb, float)
    assert math.isfinite(bpb) and bpb > 0.0
    assert bpb != 1.0
    # Emission lock (VAL-RESLAB-006): primary heldout_delta; bpb secondary.
    assert score.get("primary_metric") in {"heldout_delta", "prequential_bpb"}
    assert score.get("secondary_metric", "prequential_bpb") == "prequential_bpb"
    # VAL-SCORE-003 / VAL-HARNESS-017: bpb == sum(-log2 p) / covered_bytes (BYTE denominator).
    covered_bytes = metrics["covered_bytes"]
    assert covered_bytes > 0
    bits = metrics["sum_neg_log_likelihood_nats"] / math.log(2.0)
    assert metrics["sum_neg_log2_likelihood_bits"] == bits
    assert bpb == bits / covered_bytes
    assert metrics["total_bytes_covered"] == covered_bytes
    # VAL-HARNESS-013 / VAL-SCORE-010: compute-normalized (tokens/bytes), NO wall-clock term.
    assert score["compute_normalization"] == "tokens_bytes"
    assert score["wall_clock_term"] is False
    assert "wall_clock" not in {k.lower() for k in score} or score["wall_clock_term"] is False
    assert metrics["predicted_tokens"] > 0


def test_harness_manifest_bpb_matches_scoring_module(tmp_path):
    # The score recorded in the manifest equals scoring.score_prequential_bpb over the SAME
    # challenge-owned manifest (single source of truth; miner-reported numbers ignored).
    proc, artifacts = _run_runner(
        tmp_path, run_name="bpb-consistency", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    recomputed = score_prequential_bpb(manifest)
    assert recomputed.bpb == manifest["score"]["prequential_bpb"]
    # Without host held-out investigate, recomputed is degraded secondary-only; matches runner
    # provisional bpb display final_score. With held-out attached later, emission rank differs.
    assert recomputed.bpb == pytest.approx(manifest["score"]["prequential_bpb"])
    assert recomputed.covered_bytes == manifest["metrics"]["covered_bytes"]
    if recomputed.heldout_delta is None:
        assert recomputed.final_score == pytest.approx(manifest["score"]["final_score"])
        assert recomputed.emission_crown_eligible is False
    assert manifest["miner_reported_ignored"] is True


def test_harness_bpb_is_area_under_curve_not_single_point(tmp_path):
    # The numerator integrates the WHOLE single-pass online-loss stream (token-weighted), so a
    # late single low loss cannot dominate: cumulative codelength == sum over per-batch segments.
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="bpb-auc",
        arch_code=ARCH_LM,
        train_code=TRAIN_LEARN,
        data_files={
            "train-00000.jsonl": "".join(
                json.dumps(
                    {
                        "id": f"doc-{i}",
                        "text": f"the locked fineweb edu sample line number {i} here",
                    }
                )
                + "\n"
                for i in range(60)
            )
        },
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    # More than one online-loss segment contributed to the integral (not a lone checkpoint).
    assert len(metrics["online_loss"]) >= 2
    assert metrics["cumulative_codelength_bits"] > 0.0
    # The recorded codelength equals nats->bits of the token-weighted online NLL integral.
    expected_bits = metrics["sum_neg_log_likelihood_nats"] / math.log(2.0)
    assert metrics["cumulative_codelength_bits"] == expected_bits
