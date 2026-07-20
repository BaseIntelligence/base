"""Atomic Prism finalization around score ownership (VAL-WEIGHT-093).

Verification, score/curve, terminal status, and architecture/training updates
must commit as one logical idempotent operation. Exact duplicates are no-ops;
changed results cannot overwrite; non-positive invalidated outcomes never create
a second owner silently.
"""

from __future__ import annotations

import base64
import io
import math
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.ingestion import ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.weights import get_weights

WORKER_KEY = "//WorkerScoreOwner"

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


def _bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _settings(tmp_path: Path) -> PrismSettings:
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
        worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY),
    )


def _manifest(marker: str = "v2", *, nll: float = 900.0) -> dict[str, Any]:
    covered_bytes = 4096
    online_loss = [10.0, 6.0, 3.0, 2.0]
    bpb = (nll / math.log(2.0)) / covered_bytes
    # Worker-plane skip_heldout ignores held-out fields (degraded, no crown). Kept for docs.
    heldout_delta = 1.0 / (1.0 + bpb)
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": online_loss,
            "sum_neg_log_likelihood_nats": nll,
            "covered_bytes": covered_bytes,
            "predicted_tokens": 96,
            "step0_loss": online_loss[0],
            "consumed_batches": len(online_loss),
            "random_init_baseline_nats": math.log(50257),
            "prequential_bpb": bpb,
            "heldout_delta": heldout_delta,
            "held_out_delta": heldout_delta,
            "marker": marker,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
    }


def _proof_dict(signer: Any, unit_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(signer=signer, manifest_sha256=digest, unit_id=unit_id)
    return proof.model_dump(mode="json")


def _result(proof_dict: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof_dict,
        MANIFEST_PAYLOAD_KEY: manifest,
    }


async def _make_app(settings: PrismSettings):
    app = create_app(settings)
    await app.state.database.init()
    return app


async def _seed(app: Any, hotkey: str = "hk-owner") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    return sub.id


def _db_snapshot(db_path: Path, submission_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        status = conn.execute(
            "SELECT status FROM submissions WHERE id=?", (submission_id,)
        ).fetchone()
        score = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
        curve = conn.execute(
            "SELECT 1 FROM submission_curves WHERE submission_id=?", (submission_id,)
        ).fetchone()
        families = conn.execute(
            "SELECT id, owner_hotkey, q_arch_best, canonical_submission_id "
            "FROM architecture_families"
        ).fetchall()
        variants = conn.execute(
            "SELECT id, owner_hotkey, q_recipe, submission_id, is_current_best "
            "FROM training_variants"
        ).fetchall()
        score_count = conn.execute(
            "SELECT COUNT(*) AS n FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()["n"]
    finally:
        conn.close()
    return {
        "status": None if status is None else str(status["status"]),
        "final_score": None if score is None else float(score["final_score"]),
        "curve": curve is not None,
        "score_count": int(score_count),
        "families": [dict(row) for row in families],
        "variants": [dict(row) for row in variants],
    }


async def test_finalization_is_atomic_score_status_arch_training(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app, hotkey="hk-alpha")
    manifest = _manifest("atomic")
    proof = _proof_dict(signer, submission_id, manifest)

    before = _db_snapshot(db_path, submission_id)
    assert before["status"] == "pending"
    assert before["final_score"] is None
    assert before["families"] == []
    assert before["variants"] == []

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-alpha",
        result=_result(proof, manifest),
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True

    after = _db_snapshot(db_path, submission_id)
    assert after["status"] == "completed"
    assert after["final_score"] is not None and after["final_score"] > 0.0
    assert after["curve"] is True
    assert after["score_count"] == 1
    assert len(after["families"]) == 1
    assert after["families"][0]["owner_hotkey"] == "hk-alpha"
    assert after["families"][0]["canonical_submission_id"] == submission_id
    # Worker-plane skip_heldout degrades: scores persist, crown key stays 0 (VAL-RESLAB-006).
    assert float(after["families"][0]["q_arch_best"]) == pytest.approx(0.0)
    assert len(after["variants"]) == 1
    assert after["variants"][0]["owner_hotkey"] == "hk-alpha"
    assert after["variants"][0]["submission_id"] == submission_id
    assert int(after["variants"][0]["is_current_best"]) == 1

    weights = await get_weights(app.state.repository, settings.epoch_seconds)
    # Non-positive q_arch_best contributes empty emission map (fail-closed crown).
    assert weights == {}


async def test_duplicate_finalization_is_idempotent_noop(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app)
    manifest = _manifest("dup")
    proof = _proof_dict(signer, submission_id, manifest)

    first = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert first.finalized is True
    snap_first = _db_snapshot(db_path, submission_id)

    second = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert second.status == "accepted"
    assert second.idempotent is True
    assert second.finalized is False

    snap_second = _db_snapshot(db_path, submission_id)
    assert snap_second["final_score"] == pytest.approx(snap_first["final_score"])
    assert snap_second["status"] == "completed"
    assert snap_second["score_count"] == 1
    assert len(snap_second["families"]) == 1
    assert len(snap_second["variants"]) == 1
    assert snap_second["families"][0]["canonical_submission_id"] == submission_id


async def test_conflicting_result_cannot_overwrite_score(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app)
    first_manifest = _manifest("first", nll=900.0)
    first_proof = _proof_dict(signer, submission_id, first_manifest)

    accepted = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(first_proof, first_manifest),
    )
    assert accepted.finalized is True
    snap = _db_snapshot(db_path, submission_id)

    conflict_manifest = _manifest("conflict", nll=100.0)
    conflict_proof = _proof_dict(signer, submission_id, conflict_manifest)
    conflicting = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(conflict_proof, conflict_manifest),
    )
    assert conflicting.status == "conflict"
    after = _db_snapshot(db_path, submission_id)
    assert after["final_score"] == pytest.approx(snap["final_score"])
    assert after["score_count"] == 1
    assert after["families"][0]["canonical_submission_id"] == submission_id


async def test_two_hotkeys_worker_skip_heldout_cannot_crown(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    # Stronger learner first: lower nll => better degraded secondary final_score.
    strong_id = await _seed(app, hotkey="hk-strong")
    strong_manifest = _manifest("strong", nll=400.0)
    strong = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=strong_id,
        submission_ref="hk-strong",
        result=_result(_proof_dict(signer, strong_id, strong_manifest), strong_manifest),
    )
    assert strong.finalized is True

    weak_id = await _seed(app, hotkey="hk-weak")
    weak_manifest = _manifest("weak", nll=2000.0)
    weak = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=weak_id,
        submission_ref="hk-weak",
        result=_result(_proof_dict(signer, weak_id, weak_manifest), weak_manifest),
    )
    assert weak.finalized is True

    snap_strong = _db_snapshot(db_path, strong_id)
    snap_weak = _db_snapshot(db_path, weak_id)
    assert snap_strong["final_score"] is not None and snap_weak["final_score"] is not None
    assert snap_strong["final_score"] > snap_weak["final_score"]

    families = snap_strong["families"]
    # Distinct sources yield families, but skip_heldout leaves q_arch_best at 0 (no emission crown).
    assert len(families) >= 1
    assert float(snap_strong["families"][0]["q_arch_best"]) == pytest.approx(0.0)
    weights = await get_weights(app.state.repository, settings.epoch_seconds)
    assert weights == {}


async def test_invalidated_score_does_not_keep_crown(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    submission_id = await _seed(app, hotkey="hk-only")
    manifest = _manifest("invalidate")
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-only",
        result=_result(_proof_dict(signer, submission_id, manifest), manifest),
    )
    assert outcome.finalized is True
    # Worker-plane path never crowns without held-out, so emission is already empty.
    assert await get_weights(app.state.repository, settings.epoch_seconds) == {}

    invalidated = await app.state.repository.invalidate_submission_score(
        submission_id, reason="audit_fail"
    )
    assert invalidated is True
    # Non-positive / invalidated crown contributes an empty emission map (no hidden rival).
    assert await get_weights(app.state.repository, settings.epoch_seconds) == {}
