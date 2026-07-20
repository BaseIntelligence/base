from __future__ import annotations

import json
import math

import pytest
from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.scoring import (
    NATS_TO_BITS,
    PrequentialBpbScore,
    ScoreValidationError,
    bpb_to_final_score,
    score_prequential_bpb,
)

SCORING_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

SCORING_TRAIN = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
"""


def _v2_manifest_payload(submission_id: str, *, sum_nll_nats: float, covered_bytes: int) -> dict:
    bits = sum_nll_nats / math.log(2.0)
    bpb = bits / covered_bytes
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": submission_id,
        "run_id": "prism-reexec-" + submission_id,
        "mode": "gpu_proxy_eval",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": [2.5, 2.0, 1.5],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "sum_neg_log2_likelihood_bits": bits,
            "cumulative_codelength_bits": bits,
            "covered_bytes": covered_bytes,
            "total_bytes_covered": covered_bytes,
            "predicted_tokens": 96,
            "tokens_seen": 96,
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "step0_loss": 2.5,
            "consumed_batches": 3,
            "random_init_baseline_nats": math.log(128),
            "nan_inf_batches": 0,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


def _submit(client: TestClient, nonce: str) -> str:
    payload = {
        "code": two_script_bundle(arch_code=SCORING_ARCH, train_code=SCORING_TRAIN),
        "filename": "project.zip",
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body, nonce=nonce), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _manifest(
    *,
    sum_nll_nats: float,
    covered_bytes: int,
    online_loss: list[float] | None = None,
    predicted_tokens: int = 64,
    step0_anomaly: bool = False,
    nan_inf: bool = False,
    step0_loss: float | None = 2.0,
) -> dict:
    losses = online_loss if online_loss is not None else [2.5, 2.0, 1.5, 1.0]
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "sub-bpb",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": losses,
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "covered_bytes": covered_bytes,
            "predicted_tokens": predicted_tokens,
            "step0_loss": step0_loss,
            "consumed_batches": len(losses),
            "random_init_baseline_nats": math.log(128),
            "nan_inf_batches": 1 if nan_inf else 0,
        },
        "anti_cheat": {
            "step0_anomaly": step0_anomaly,
            "nan_inf_detected": nan_inf,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


def test_scoring_bpb_is_bits_over_bytes_from_challenge_manifest() -> None:
    # sum_nll_nats chosen so total bits == 100 -> bpb == 100/200 == 0.5 (denominator = BYTES).
    manifest = _manifest(sum_nll_nats=100.0 * math.log(2.0), covered_bytes=200)
    score = score_prequential_bpb(manifest)
    assert isinstance(score, PrequentialBpbScore)
    assert score.covered_bytes == 200
    assert score.sum_neg_log2_likelihood_bits == pytest.approx(100.0)
    assert score.bpb == pytest.approx(0.5)
    # Without held-out the path is degraded secondary-only (bpb display); no emission crown.
    assert score.final_score == pytest.approx(bpb_to_final_score(0.5))
    assert score.final_score == pytest.approx(1.0 / 1.5)
    assert score.emission_crown_eligible is False


def test_scoring_bpb_matches_sum_neg_log2_over_covered_bytes() -> None:
    manifest = _manifest(sum_nll_nats=37.0, covered_bytes=512)
    score = score_prequential_bpb(manifest)
    expected_bits = 37.0 * NATS_TO_BITS
    assert score.bpb == pytest.approx(expected_bits / 512)
    assert math.isfinite(score.bpb)
    assert score.bpb > 0.0


def test_scoring_lower_bpb_ranks_better_on_degraded_secondary_path() -> None:
    good = score_prequential_bpb(_manifest(sum_nll_nats=20.0, covered_bytes=400))
    bad = score_prequential_bpb(_manifest(sum_nll_nats=200.0, covered_bytes=400))
    assert good.bpb < bad.bpb
    # Without held-out (degraded), lower bpb still ranks higher among secondary-only peers.
    # Crown remains ineligible without held-out primary (VAL-RESLAB-006).
    assert good.final_score > bad.final_score
    assert good.emission_crown_eligible is False
    assert bad.emission_crown_eligible is False


def test_scoring_denominator_is_bytes_not_tokens() -> None:
    # Two runs with identical code-length but different tokenizations (token counts) yet the SAME
    # raw UTF-8 bytes covered get the SAME bpb -> tokenizer-agnostic by construction.
    coarse = _manifest(sum_nll_nats=64.0, covered_bytes=300, predicted_tokens=50)
    fine = _manifest(sum_nll_nats=64.0, covered_bytes=300, predicted_tokens=400)
    assert score_prequential_bpb(coarse).bpb == pytest.approx(score_prequential_bpb(fine).bpb)


def test_scoring_no_wallclock_term_and_no_legacy_raw_loss_term() -> None:
    score = score_prequential_bpb(_manifest(sum_nll_nats=80.0, covered_bytes=256))
    payload = score.metrics_payload()
    block = score.manifest_score_block()
    assert "standardized_lm_quality" not in payload
    assert "wall_clock" not in str(payload).lower()
    assert block["wall_clock_term"] is False
    assert block["compute_normalization"] == "tokens_bytes"
    assert payload["prequential_bpb"] == pytest.approx(score.bpb)
    assert payload["total_bytes_covered"] == pytest.approx(256.0)


def test_scoring_step0_anomaly_is_flagged_not_rewarded() -> None:
    # An impossibly-low step-0 loss (smuggled pretrained weights) is flagged: even a tiny bpb must
    # not top the board.
    score = score_prequential_bpb(
        _manifest(sum_nll_nats=1.0, covered_bytes=10_000, step0_anomaly=True)
    )
    assert score.anomaly is True
    assert "step0_anomaly" in score.flags
    assert score.anti_cheat_multiplier == 0.0
    assert score.final_score == 0.0


def test_scoring_zero_coverage_raises_no_fabricated_score() -> None:
    with pytest.raises(ScoreValidationError):
        score_prequential_bpb(_manifest(sum_nll_nats=10.0, covered_bytes=0))


def test_scoring_non_finite_codelength_raises() -> None:
    with pytest.raises(ScoreValidationError):
        score_prequential_bpb(_manifest(sum_nll_nats=float("inf"), covered_bytes=100))


def test_scoring_requires_online_loss_stream() -> None:
    manifest = _manifest(sum_nll_nats=10.0, covered_bytes=100, online_loss=[])
    with pytest.raises(ScoreValidationError):
        score_prequential_bpb(manifest)


def test_scoring_bpb_finite_positive_in_sane_band() -> None:
    score = score_prequential_bpb(_manifest(sum_nll_nats=50.0, covered_bytes=128))
    assert math.isfinite(score.bpb)
    assert score.bpb > 0.0
    assert "bpb_out_of_band" not in score.flags


def test_container_path_scores_prequential_bpb_from_challenge_manifest(tmp_path, monkeypatch):
    # bits == 100, covered_bytes == 250 -> bpb == 0.4 ; final_score == 1/(1+0.4).
    sum_nll_nats = 100.0 * math.log(2.0)
    covered_bytes = 250
    expected_bpb = 100.0 / covered_bytes
    expected_final = bpb_to_final_score(expected_bpb)

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_dir = spec.mounts[1].source
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = _v2_manifest_payload(
            str(payload["submission_id"]),
            sum_nll_nats=sum_nll_nats,
            covered_bytes=covered_bytes,
        )
        (artifact_dir / RUN_MANIFEST_V2_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"covered_bytes":250.0}\n',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "bpb.sqlite3"
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
        plagiarism_enabled=False,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, "bpb-int")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        # Without held-out this path is degraded secondary-only (no emission crown primary).
        # final_score is the bpb display transform, NOT a NAS q_arch/q_recipe blend.
        assert status["final_score"] == pytest.approx(expected_final)
        assert status["q_arch"] == pytest.approx(expected_final)
        assert status["q_recipe"] == pytest.approx(0.0)

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT metrics, final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    metrics = json.loads(row[0])
    # Score is computed from the challenge manifest; no legacy raw-loss / NAS term feeds it.
    assert metrics["prequential_bpb"] == pytest.approx(expected_bpb)
    assert metrics["total_bytes_covered"] == pytest.approx(float(covered_bytes))
    assert "standardized_lm_quality" not in metrics
    assert "wall_clock" not in str(metrics).lower()
    assert row[1] == pytest.approx(expected_final)
