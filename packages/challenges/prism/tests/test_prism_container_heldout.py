from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest
from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.scoring import score_prequential_bpb

HELDOUT_ARCH = """
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

HELDOUT_TRAIN = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
"""

VAL_LINE = '{"id": "val-%d", "text": "the locked fineweb edu held out sample sentence %d"}\n'


def _stage_val(root: Path, lines: int = 40) -> Path:
    val_dir = root / "val-data"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val-00000.jsonl").write_text(
        "".join(VAL_LINE % (i, i) for i in range(lines)), encoding="utf-8"
    )
    return val_dir


def _trained_state_bytes(vocab: int, val_dir: Path) -> dict:
    """A lightly-trained TinyLM state_dict so the held-out twin/trained models differ on val."""
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class TinyLM(nn.Module):
        def __init__(self, vocab: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, 8)
            self.head = nn.Linear(8, vocab)

        def forward(self, tokens):  # type: ignore[no-untyped-def]
            return self.head(self.emb(tokens))

    torch.manual_seed(1337)
    model = TinyLM(vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    text = (val_dir / "val-00000.jsonl").read_text(encoding="utf-8")
    byte_ids = [b % vocab for line in text.splitlines() for b in json.loads(line)["text"].encode()]
    seq = 16
    for _epoch in range(3):
        for start in range(0, len(byte_ids) - seq - 1, seq):
            chunk = byte_ids[start : start + seq + 1]
            tokens = torch.tensor(chunk[:-1]).view(1, -1)
            targets = torch.tensor(chunk[1:]).view(-1)
            opt.zero_grad()
            logits = model(tokens)
            loss = functional.cross_entropy(logits.reshape(-1, vocab), targets)
            loss.backward()
            opt.step()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _manifest_payload(submission_id: str, vocab: int) -> dict:
    covered_bytes = 1200
    bpb = 6.0
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    bits = sum_nll_nats / math.log(2.0)
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": submission_id,
        "run_id": "prism-reexec-" + submission_id,
        "mode": "gpu_proxy_eval",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": [4.5, 4.0, 3.5],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "sum_neg_log2_likelihood_bits": bits,
            "cumulative_codelength_bits": bits,
            "covered_bytes": covered_bytes,
            "total_bytes_covered": covered_bytes,
            "predicted_tokens": 1100,
            "tokens_seen": 1100,
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "step0_loss": 4.5,
            "consumed_batches": 3,
            "random_init_baseline_nats": math.log(vocab),
            "nan_inf_batches": 0,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "score": {
            "schema": "prism_score.v2",
            "primary_metric": "heldout_delta",
            "secondary_metric": "prequential_bpb",
            "emission_ranking": "heldout_primary_bpb_secondary",
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "final_score": 1.0 / (1.0 + bpb),
            "lower_is_better": True,
            "tie_breaker": "prequential_bpb",
        },
        # The host scorer reads trained_state.pt ONLY when the CHALLENGE manifest records it for
        # THIS run (a bare is_file() on the miner-writable artifacts_dir is an RCE sink).
        "artifacts": {"trained_state": "trained_state.pt"},
        "miner_reported_ignored": True,
    }


def _submit(client: TestClient, nonce: str) -> str:
    payload = {
        "code": two_script_bundle(arch_code=HELDOUT_ARCH, train_code=HELDOUT_TRAIN),
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


def test_evaluate_augments_manifest_and_scores_with_heldout_delta(tmp_path, monkeypatch):
    val_dir = _stage_val(tmp_path)
    artifact_root = tmp_path / "artifacts"

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        vocab = int(payload["context"]["vocab_size"])
        artifact_dir = spec.mounts[1].source
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = _manifest_payload(str(payload["submission_id"]), vocab)
        (artifact_dir / RUN_MANIFEST_V2_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
        import torch

        torch.save(_trained_state_bytes(vocab, val_dir), artifact_dir / "trained_state.pt")
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"covered_bytes":1200.0}\n',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "heldout.sqlite3"
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_artifact_root=artifact_root,
        base_eval_val_data_dir=str(val_dir),
        plagiarism_enabled=False,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, "heldout-int")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"

    # The on-disk challenge manifest carries the host-computed held-out delta in BOTH blocks.
    manifest_path = artifact_root / submission_id / "attempt-1" / RUN_MANIFEST_V2_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = manifest["metrics"]
    score_block = manifest["score"]
    for key in ("heldout_delta", "held_out_delta", "val_bpb_trained", "val_bpb_random_init"):
        assert key in metrics, f"missing {key} in metrics"
        assert isinstance(metrics[key], float) and math.isfinite(metrics[key])
        assert key in score_block, f"missing {key} in score block"
    # VAL-HARNESS-009: held-out delta is numeric and not a fabricated 1.0 constant.
    assert metrics["heldout_delta"] != 1.0
    assert metrics["heldout_delta"] == pytest.approx(
        metrics["val_bpb_random_init"] - metrics["val_bpb_trained"]
    )

    # The scores row + final_score derive from the (held-out-augmented) challenge manifest.
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT metrics, final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    score_metrics = json.loads(row[0])
    assert "heldout_delta" in score_metrics
    assert score_metrics["heldout_delta"] == pytest.approx(metrics["heldout_delta"])
    recomputed = score_prequential_bpb(manifest)
    assert row[1] == pytest.approx(recomputed.final_score)
    assert math.isfinite(row[1]) and row[1] > 0.0
