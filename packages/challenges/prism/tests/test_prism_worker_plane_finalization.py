"""Worker-plane LIGHT finalization: score from the forwarded manifest, no re-execution.

Covers the mission finalization assertions (architecture.md 4):

* VAL-FINAL-001 - with the worker plane ON, ingesting a verified+plausible worker result finalizes
  the submission from the FORWARDED ``prism_run_manifest.v2`` (prequential bpb + the deterministic
  source-static tail); the heavy evaluator (``_evaluate_within_wall_time`` / ``evaluator.evaluate``)
  and the GPU lease are NEVER invoked during ingest finalization.
* VAL-FINAL-002 - the worker-plane score is bpb-only (held-out delta absent/None even when the
  forwarded manifest fabricates one) and the master-only secret val split
  (``base_eval_val_data_dir``) is never read/required (no evaluator is constructed).
* VAL-FINAL-003 - with the worker plane OFF, ingestion falls back to the legacy
  ``process_submission`` re-execution path and produces a byte-identical score.
* VAL-FINAL-004 - the canonical manifest round-trip is stable (on-disk bytes hash == the
  ingestion-recomputed canonical hash), so a genuine manifest is never falsely rejected.

Offline, no GPU: worker-plane finalization scores from the forwarded manifest directly and the
flag-off regression drives the CPU re-exec seam the other coordination tests use. Proofs are signed
with real sr25519 worker keys.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.scoring import score_prequential_bpb
from prism_challenge.ingestion import ResultIngestionError, ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    canonical_manifest_json,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.queue import PrismWorker

WORKER_KEY = "//WorkerFinalize"

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

_SHARD_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def _stage_train(root: Path, *, lines: int = 64) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return data_dir


def _bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _settings(
    tmp_path: Path,
    *,
    worker_plane: WorkerPlaneConfig,
    base_eval_val_data_dir: str | None = None,
) -> PrismSettings:
    overrides: dict[str, Any] = {}
    if base_eval_val_data_dir is not None:
        overrides["base_eval_val_data_dir"] = base_eval_val_data_dir
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'coord.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
        worker_plane=worker_plane,
        **overrides,
    )


def _scoreable_manifest(
    marker: str = "v2", *, heldout_delta: float | None = None
) -> dict[str, Any]:
    """A plausible AND scoreable v2 manifest (positive finite bpb, decreasing loss)."""

    covered_bytes = 4096
    sum_nll_nats = 900.0
    baseline = math.log(50257)
    online_loss = [10.0, 6.0, 3.0, 2.0]
    metrics: dict[str, Any] = {
        "online_loss": online_loss,
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "covered_bytes": covered_bytes,
        "predicted_tokens": 96,
        "step0_loss": online_loss[0],
        "consumed_batches": len(online_loss),
        "random_init_baseline_nats": baseline,
        "marker": marker,
    }
    if heldout_delta is not None:
        # A worker CANNOT compute a legitimate held-out delta (no secret val split); a
        # fabricated one must never move the finalized score.
        metrics["heldout_delta"] = heldout_delta
        metrics["val_bpb_trained"] = 0.20
        metrics["val_bpb_random_init"] = 0.20 + heldout_delta
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
    }


def _proof_dict(signer, unit_id: str, manifest: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(signer=signer, manifest_sha256=digest, unit_id=unit_id)
    payload = proof.model_dump(mode="json")
    payload.update(overrides)
    return payload


def _result(proof_dict: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof_dict,
    }
    if manifest is not None:
        result[MANIFEST_PAYLOAD_KEY] = manifest
    return result


async def _make_app(settings: PrismSettings):
    app = create_app(settings)
    await app.state.database.init()
    return app


async def _seed(app, hotkey: str = "hk-owner") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    return sub.id


def _score_row(db_path: Path, submission_id: str) -> tuple[float, dict[str, Any]] | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score, metrics FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    metrics = json.loads(row[1]) if row[1] else {}
    return float(row[0]), metrics


def _final_score(db_path: Path, submission_id: str) -> float | None:
    row = _score_row(db_path, submission_id)
    return row[0] if row is not None else None


# --- VAL-FINAL-001: finalize from the forwarded manifest, no re-execution ------------------------


async def test_worker_plane_finalizes_from_manifest_without_reexecution(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    # The heavy evaluator seam and the GPU lease scheduler must NOT be touched during finalization.
    def _boom_eval(*_a: Any, **_k: Any):
        raise AssertionError("evaluator was invoked during worker-plane ingest finalization")

    def _boom_lease(*_a: Any, **_k: Any):
        raise AssertionError("a GPU lease was taken during worker-plane ingest finalization")

    monkeypatch.setattr(PrismWorker, "_evaluate_within_wall_time", _boom_eval)
    monkeypatch.setattr("prism_challenge.queue.GpuLeaseScheduler", _boom_lease)

    submission_id = await _seed(app)
    manifest = _scoreable_manifest()
    proof = _proof_dict(signer, submission_id, manifest)

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert await app.state.repository.submission_status(submission_id) == "completed"

    # The persisted score equals the score computed independently from the forwarded manifest
    # (anti-cheat multiplier is 1.0 for a first submission with no prior codes).
    stored = _final_score(db_path, submission_id)
    expected = score_prequential_bpb(manifest).final_score
    assert stored is not None
    assert stored == pytest.approx(expected)


async def test_worker_plane_requires_forwarded_manifest(tmp_path):
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    submission_id = await _seed(app)
    manifest = _scoreable_manifest()
    # A proof-only delivery (no manifest to score from) cannot be finalized without re-executing.
    proof = _proof_dict(signer, submission_id, manifest)
    with pytest.raises(ResultIngestionError) as exc:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref="hk-owner",
            result=_result(proof, None),
        )
    assert exc.value.reason == "manifest_missing"
    assert _final_score(db_path, submission_id) is None
    assert await app.state.repository.submission_status(submission_id) == "pending"


# --- VAL-FINAL-002: held-out delta skipped; secret split never read ------------------------------


async def test_worker_plane_skips_heldout_and_never_reads_secret_split(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path,
        worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY),
        base_eval_val_data_dir="/nonexistent/secret/val-split",
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    # If ANY code path constructs the evaluator (the only reader of base_eval_val_data_dir, via
    # _augment_with_heldout) or takes a GPU lease, fail loudly.
    def _boom_eval(*_a: Any, **_k: Any):
        raise AssertionError("evaluator constructed during worker-plane finalization")

    def _boom_lease(*_a: Any, **_k: Any):
        raise AssertionError("a GPU lease was taken during worker-plane finalization")

    monkeypatch.setattr(PrismWorker, "_evaluate_within_wall_time", _boom_eval)
    monkeypatch.setattr("prism_challenge.queue.GpuLeaseScheduler", _boom_lease)

    # A manifest that FABRICATES a held-out delta must be scored degraded (skip_heldout) and
    # cannot crown emission (VAL-RESLAB-006).
    forged = await _seed(app)
    forged_manifest = _scoreable_manifest("forged", heldout_delta=5.0)
    forged_proof = _proof_dict(signer, forged, forged_manifest)
    await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=forged,
        submission_ref="hk-owner",
        result=_result(forged_proof, forged_manifest),
    )
    forged_row = _score_row(db_path, forged)
    assert forged_row is not None
    forged_score, forged_metrics = forged_row

    # Degraded secondary-only: equals score_prequential_bpb with held-out skipped ...
    bpb_only = score_prequential_bpb(forged_manifest, skip_heldout=True)
    assert forged_score == pytest.approx(bpb_only.final_score)
    assert bpb_only.emission_crown_eligible is False
    # ... and is NOT the held-out-primary emission rank score.
    assert forged_score != pytest.approx(score_prequential_bpb(forged_manifest).final_score)
    # No held-out contribution is recorded in the finalized metrics; crown ineligible.
    assert "heldout_delta" not in forged_metrics
    assert "held_out_delta" not in forged_metrics
    assert forged_metrics.get("emission_crown_eligible") is False

    # A submission WITHOUT any held-out field scores identically to the fabricated-delta one.
    plain = await _seed(app)
    plain_manifest = _scoreable_manifest("plain")
    plain_proof = _proof_dict(signer, plain, plain_manifest)
    await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=plain,
        submission_ref="hk-owner",
        result=_result(plain_proof, plain_manifest),
    )
    assert _final_score(db_path, plain) == pytest.approx(forged_score)


# --- VAL-FINAL-003: flag OFF => legacy re-execution finalization, byte-identical -----------------


async def test_flag_off_uses_legacy_reexecution_finalization(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )

    # With the flag OFF the worker-plane finalizer must never be used.
    def _boom_finalize(*_a: Any, **_k: Any):
        raise AssertionError("worker-plane finalizer used while the flag was OFF")

    monkeypatch.setattr(PrismWorker, "finalize_worker_result", _boom_finalize)

    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=False, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    # Path A: ingest with the flag OFF -> legacy in-process re-execution finalization.
    ingest_id = await _seed(app)
    manifest = _scoreable_manifest()
    proof = _proof_dict(signer, ingest_id, manifest)
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=ingest_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert outcome.finalized is True
    assert await app.state.repository.submission_status(ingest_id) == "completed"
    ingest_score = _final_score(db_path, ingest_id)
    assert ingest_score is not None

    # Path B: the legacy path directly on an identical submission -> byte-identical score.
    legacy_id = await _seed(app)
    await app.state.worker.process_submission(legacy_id)
    legacy_score = _final_score(db_path, legacy_id)
    assert legacy_score is not None
    assert ingest_score == pytest.approx(legacy_score)


# --- VAL-FINAL-004: canonical manifest round-trip is stable --------------------------------------


def test_manifest_canonicalization_round_trip_stable(tmp_path):
    manifest = _scoreable_manifest()
    path = tmp_path / RUN_MANIFEST_V2_FILENAME
    # Emission persists the canonical form (sort_keys=True, indent=2).
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    on_disk_bytes = path.read_bytes()
    on_disk_sha = hashlib.sha256(on_disk_bytes).hexdigest()

    # Read -> parse -> re-serialize canonically -> identical bytes/sha.
    parsed = json.loads(on_disk_bytes.decode("utf-8"))
    reserialized = json.dumps(parsed, sort_keys=True, indent=2).encode("utf-8")
    assert hashlib.sha256(reserialized).hexdigest() == on_disk_sha

    # The ingestion-side recompute agrees with the emission-side on-disk hash ...
    assert compute_manifest_sha256(parsed) == on_disk_sha
    # ... and is key-order-insensitive (a manifest built in a different order hashes identically).
    shuffled = dict(reversed(list(manifest.items())))
    assert compute_manifest_sha256(shuffled) == on_disk_sha


async def test_genuine_manifest_never_falsely_rejected_as_tampered(tmp_path):
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    submission_id = await _seed(app)
    manifest = _scoreable_manifest()
    # Emission-side: sign over the canonical hash, persist to disk exactly as the runner/host would.
    path = tmp_path / RUN_MANIFEST_V2_FILENAME
    path.write_text(canonical_manifest_json(manifest), encoding="utf-8")
    # Ingestion-side: read + parse the on-disk manifest, then verify + finalize.
    parsed = json.loads(path.read_text(encoding="utf-8"))
    proof = _proof_dict(signer, submission_id, parsed)
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, parsed),
    )
    # A genuine manifest is accepted, not falsely flagged manifest_tampered.
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert _final_score(db_path, submission_id) is not None
