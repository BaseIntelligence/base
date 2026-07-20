from __future__ import annotations

import json
import math
import sqlite3

from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import _ensure_compute_block
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME, ComputeBlock
from prism_challenge.evaluator.scoring import score_prequential_bpb

COMPUTE_ARCH = """
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

COMPUTE_TRAIN = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
"""


def _v2_manifest(submission_id: str, *, sum_nll_nats: float, covered_bytes: int) -> dict:
    bits = sum_nll_nats / math.log(2.0)
    bpb = bits / covered_bytes
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": submission_id,
        "run_id": "prism-reexec-" + submission_id,
        "mode": "gpu_proxy_eval",
        "run": {
            "seed": 1337,
            "forced_init": True,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": "cuda:0",
            "nproc_per_node": 1,
        },
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


def test_ensure_compute_block_uses_leased_gpu_count(tmp_path) -> None:
    manifest = _v2_manifest("sub-host", sum_nll_nats=100.0 * math.log(2.0), covered_bytes=250)
    (tmp_path / RUN_MANIFEST_V2_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    before = score_prequential_bpb(manifest).final_score

    _ensure_compute_block(manifest, {"actual_gpu_count": 1, "max_gpu_count": 8}, tmp_path)

    compute = manifest["compute"]
    parsed = ComputeBlock.model_validate(compute)
    # gpu_count == the GPUs actually leased (== DB actual_gpu_count), == run.world_size / nproc.
    assert parsed.gpu_count == 1
    assert compute["gpu_count"] == manifest["run"]["world_size"] == 1
    assert compute["nproc_per_node"] == manifest["run"]["nproc_per_node"] == 1
    assert compute["device"] == manifest["run"]["device"]
    assert compute["max_gpu_count"] == 8
    # Persisted to the on-disk artifact (where manifest-inspect / VAL-GPU-005 reads it).
    on_disk = json.loads((tmp_path / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["compute"]["gpu_count"] == 1
    # The compute block does not change the challenge-computed final_score.
    assert score_prequential_bpb(manifest).final_score == before


def test_ensure_compute_block_falls_back_without_run_block(tmp_path) -> None:
    manifest = _v2_manifest("sub-norun", sum_nll_nats=50.0, covered_bytes=128)
    del manifest["run"]
    _ensure_compute_block(manifest, {"actual_gpu_count": 1}, tmp_path)
    compute = manifest["compute"]
    assert compute["gpu_count"] == 1
    assert compute["world_size"] == 1
    assert compute["nproc_per_node"] == 1
    assert compute["device"]


def _submit(client: TestClient, nonce: str) -> str:
    payload = {
        "code": two_script_bundle(arch_code=COMPUTE_ARCH, train_code=COMPUTE_TRAIN),
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


def test_scored_pipeline_manifest_compute_gpu_count_matches_db(tmp_path, monkeypatch) -> None:
    sum_nll_nats = 100.0 * math.log(2.0)
    covered_bytes = 250

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_dir = spec.mounts[1].source
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = _v2_manifest(
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
    db_path = tmp_path / "compute.sqlite3"
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
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, "compute-int")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        assert client.get(f"/v1/submissions/{submission_id}").json()["status"] == "completed"

    # The challenge-authored on-disk manifest carries the typed compute block (VAL-GPU-005).
    manifest_path = tmp_path / "artifacts" / submission_id / "attempt-1" / RUN_MANIFEST_V2_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compute = manifest["compute"]
    run = manifest["run"]
    assert compute["gpu_count"] == 1
    assert compute["gpu_count"] == run["world_size"] == run["nproc_per_node"]

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT actual_gpu_count FROM eval_jobs WHERE submission_id=? AND level!='l1'",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    # compute.gpu_count is internally consistent with the DB actual_gpu_count for the same eval.
    assert int(row[0]) == compute["gpu_count"] == 1
