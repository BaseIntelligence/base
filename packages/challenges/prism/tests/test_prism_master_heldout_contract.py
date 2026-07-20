"""Master/host-side held-out scoring contract (VAL-PRISM-027/028/029).

The validator's network=none eval container reports only the online-loss stream + ``trained_state``;
the MASTER/host scorer computes the held-out result from the SECRET val split, which never enters
the container. These tests lock that boundary down end to end:

- VAL-PRISM-027: the host scorer folds a finite ``heldout_delta = bpb(random twin) - bpb(trained)``
  (never a fabricated constant) into the score, computed from a REAL CPU re-execution's persisted
  ``trained_state`` over the secret val split.
- VAL-PRISM-028: the secret val split never reaches the eval container -- mounts are exactly the
  workspace + writable artifacts, and the miner-visible payload ``context`` / env carry only the
  train ``data_dir`` (the host scorer alone reads val).
- VAL-PRISM-029: when val is absent the held-out is gracefully skipped and the submission still
  completes, scored on prequential bpb alone (no failure, no fabricated delta, no penalty).
"""

from __future__ import annotations

import json
import math
import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest
from base.challenge_sdk.executor import DockerRunResult, DockerRunSpec
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.scoring import bpb_to_final_score, score_prequential_bpb
from prism_challenge.evaluator.source_similarity import SourceFile

# A tiny byte-level CPU two-script bundle: deterministic under the forced seed, no GPU/tokenizer.
TINY_ARCH = """
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

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

_TRAIN_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)
_VAL_LINE = '{{"id": "val-{i}", "text": "secret held out fineweb edu validation sentence {i}"}}\n'


def _stage_train(root: Path, *, lines: int = 64) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_TRAIN_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return data_dir


def _stage_val(root: Path, *, lines: int = 40) -> Path:
    val_dir = root / "val-data"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val-00000.jsonl").write_text(
        "".join(_VAL_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return val_dir


def _source_files(arch: str, train: str) -> tuple[SourceFile, ...]:
    return (
        SourceFile("architecture.py", arch, sha256(arch.encode()).hexdigest()),
        SourceFile("training.py", train, sha256(train.encode()).hexdigest()),
    )


def _evaluator(
    tmp_path: Path, *, val_dir: Path | None = None, train_host_dir: Path | None = None
) -> PrismContainerEvaluator:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'heldout.sqlite3'}",
        shared_token="secret",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
        base_eval_val_data_dir=str(val_dir) if val_dir is not None else "",
        base_eval_train_data_dir=str(train_host_dir) if train_host_dir is not None else "",
        base_eval_heldout_timeout_seconds=180.0,
    )
    # Tiny CPU model: small vocab/seq + a short step budget keep the deterministic re-exec fast.
    ctx = PrismContext(vocab_size=64, sequence_length=16, seed=1234, step_budget=24)
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def _workspace_source(spec: DockerRunSpec) -> Path:
    for mount in spec.mounts:
        if mount.target == "/workspace":
            return mount.source
    raise AssertionError("captured run spec is missing the /workspace mount")


# --- VAL-PRISM-027: master host scorer folds a finite held-out delta from a real re-exec ----------


def test_master_scorer_folds_finite_heldout_delta_from_real_reexec(tmp_path, monkeypatch):
    train_dir = _stage_train(tmp_path)
    val_dir = _stage_val(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=train_dir),
    )
    evaluator = _evaluator(tmp_path, val_dir=val_dir, train_host_dir=train_dir)
    files = _source_files(TINY_ARCH, TINY_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-heldout-027",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
    )

    manifest = result.run_manifest
    assert manifest is not None
    metrics = manifest["metrics"]
    block = manifest["score"]
    for key in ("heldout_delta", "val_bpb_trained", "val_bpb_random_init"):
        assert key in metrics, f"missing {key} in metrics"
        assert isinstance(metrics[key], float) and math.isfinite(metrics[key])
        assert key in block, f"missing {key} in score block"
    # The delta is the random-twin-minus-trained tie-breaker, finite and NOT a fabricated 1.0.
    assert metrics["heldout_delta"] != 1.0
    assert metrics["heldout_delta"] == pytest.approx(
        metrics["val_bpb_random_init"] - metrics["val_bpb_trained"]
    )
    # The recomputed authoritative score reflects the held-out-augmented manifest and stays finite.
    recomputed = score_prequential_bpb(manifest)
    assert recomputed.heldout_delta == pytest.approx(metrics["heldout_delta"])
    assert math.isfinite(recomputed.final_score) and recomputed.final_score > 0.0


# --- VAL-PRISM-028: the secret val split never reaches the eval container -------------------------


def test_secret_val_never_reaches_eval_container(tmp_path, monkeypatch):
    train_dir = _stage_train(tmp_path)
    val_dir = _stage_val(tmp_path)
    captured: dict[str, object] = {}
    delegate = cpu_reexec_run(train_data_dir=train_dir)

    def capturing_run(self, spec: DockerRunSpec, timeout_seconds: float) -> DockerRunResult:
        captured["mounts"] = tuple(
            (mount.target, str(mount.source), mount.read_only) for mount in spec.mounts
        )
        captured["env"] = dict(spec.env)
        captured["network"] = spec.limits.network
        payload = json.loads((_workspace_source(spec) / "payload.json").read_text(encoding="utf-8"))
        captured["context"] = payload["context"]
        return delegate(self, spec, timeout_seconds)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", capturing_run)
    # val IS configured on the host so the held-out runs there; the container must get nothing.
    evaluator = _evaluator(tmp_path, val_dir=val_dir, train_host_dir=train_dir)
    files = _source_files(TINY_ARCH, TINY_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-iso-028",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
    )

    host_val = str(val_dir)
    # network=none and mounts are EXACTLY the read-only workspace + the writable artifacts dir.
    assert captured["network"] == "none"
    mounts = captured["mounts"]
    assert {target for target, _src, _ro in mounts} == {"/workspace", "/artifacts"}
    assert {target for target, _src, ro in mounts if not ro} == {"/artifacts"}
    assert not any("val" in target or "test" in target for target, _src, _ro in mounts)
    assert all(host_val not in src for _target, src, _ro in mounts)

    # The miner-visible payload context exposes ONLY the train data_dir -- no val/test key or value.
    context = captured["context"]
    assert context["data_dir"] == evaluator.settings.base_eval_data_dir
    assert context["artifacts_dir"] == "/artifacts"
    for forbidden in (
        "val_dir",
        "val_data_dir",
        "val_split",
        "test_dir",
        "test_data_dir",
        "heldout_dir",
    ):
        assert forbidden not in context
    assert all(host_val not in str(value) for value in context.values())

    # The eval container env carries no val/test split path either.
    env = captured["env"]
    assert all(host_val not in str(value) for value in env.values())
    assert not any("val" in key.lower() or "test" in key.lower() for key in env)

    # Proof the boundary is real: the host scorer DID read val (delta folded) while the container
    # never saw it.
    assert result.run_manifest is not None
    assert "heldout_delta" in result.run_manifest["metrics"]


# --- VAL-PRISM-029: held-out gracefully skipped when val is absent (still scores on bpb) ----------

_HELDOUT_ARCH = """
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

_HELDOUT_TRAIN = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
"""


def _manifest_payload(submission_id: str, vocab: int, *, bpb: float) -> dict:
    covered_bytes = 1200
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    bits = sum_nll_nats / math.log(2.0)
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": submission_id,
        "run_id": "prism-reexec-" + submission_id,
        "mode": "gpu_proxy_eval",
        "run": {"device": "cuda", "world_size": 1, "nproc_per_node": 1},
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
        # trained_state IS recorded for THIS run: the skip is driven purely by the ABSENT val split.
        "artifacts": {"trained_state": "trained_state.pt"},
        "miner_reported_ignored": True,
    }


def _submit(client: TestClient, nonce: str) -> str:
    payload = {
        "code": two_script_bundle(arch_code=_HELDOUT_ARCH, train_code=_HELDOUT_TRAIN),
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


def test_submission_completes_scored_on_bpb_alone_when_val_absent(tmp_path, monkeypatch):
    bpb = 6.0
    artifact_root = tmp_path / "artifacts"

    def fake_run(self, spec: DockerRunSpec, timeout_seconds: float) -> DockerRunResult:
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        vocab = int(payload["context"]["vocab_size"])
        artifact_dir = spec.mounts[1].source
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = _manifest_payload(str(payload["submission_id"]), vocab, bpb=bpb)
        (artifact_dir / RUN_MANIFEST_V2_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
        import torch

        torch.save({"emb.weight": torch.zeros(vocab, 8)}, artifact_dir / "trained_state.pt")
        return DockerRunResult(container_name="prism-eval", stdout="", stderr="", returncode=0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "noval.sqlite3"
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
        # The secret val split is ABSENT (path does not exist) -> held-out gracefully skipped.
        base_eval_val_data_dir=str(tmp_path / "does-not-exist"),
        plagiarism_enabled=False,
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, "noval-int")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"

    # The on-disk challenge manifest carries NO secret-val-derived held-out fields.
    manifest_path = artifact_root / submission_id / "attempt-1" / RUN_MANIFEST_V2_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("heldout_delta", "held_out_delta", "val_bpb_trained", "val_bpb_random_init"):
        assert key not in manifest["metrics"]
        assert key not in manifest.get("score", {})

    # The persisted score row exists and is the pure prequential-bpb transform: no held-out
    # tie-break, no memorization penalty, no fabricated delta.
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT metrics, final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    score_metrics = json.loads(row[0])
    assert "heldout_delta" not in score_metrics
    recomputed = score_prequential_bpb(manifest)
    assert recomputed.heldout_delta is None
    assert recomputed.memorization_penalty == pytest.approx(1.0)
    assert row[1] == pytest.approx(recomputed.final_score)
    assert row[1] == pytest.approx(bpb_to_final_score(recomputed.bpb))
    assert math.isfinite(row[1]) and row[1] > 0.0
