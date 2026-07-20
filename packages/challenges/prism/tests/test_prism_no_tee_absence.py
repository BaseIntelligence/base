"""Prism NO TEE residual: package absence + score finalize without tee (VAL-NOTEE-001..008)."""

from __future__ import annotations

import base64
import importlib
import io
import math
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.app import create_app
from prism_challenge.audit import effective_tier
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.ingestion import ResultIngestionError, ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    ExecutionProof,
    ProviderInfo,
    WorkerSignature,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerNoTee"
PINNED = "sha256:" + ("ab" * 32)
OTHER = "sha256:" + ("cd" * 32)

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


def test_tee_package_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("prism_challenge.tee")


def test_no_test_tee_modules_on_disk() -> None:
    root = Path(__file__).resolve().parent
    leftover = sorted(p.name for p in root.glob("test_tee_*.py"))
    assert leftover == []


def test_tee_package_directory_absent() -> None:
    pkg = Path(__file__).resolve().parents[1] / "src" / "prism_challenge" / "tee"
    assert not pkg.exists()


def test_config_has_no_tee_block_or_capability() -> None:
    settings = PrismSettings(
        shared_token="tok",
        docker_backend="cli",
        database_url="sqlite+aiosqlite:////tmp/prism-notee-cfg.sqlite3",
    )
    # Nested TeeConfig / settings.tee / PRISM_TEE gone; capability not advertised.
    # Base ChallengeSettings may still expose an inert tee_verification_enabled flag.
    assert not hasattr(settings, "tee")
    assert "challenge.tee_verification" not in settings.capabilities
    config_mod = __import__("prism_challenge.config", fromlist=["*"])
    assert not hasattr(config_mod, "TeeConfig")
    assert "tee" not in type(settings).model_fields


def test_max_effective_tier_is_one_never_two() -> None:
    proof_t2 = ExecutionProof(
        version=1,
        tier=2,
        manifest_sha256="c" * 64,
        image_digest=PINNED,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
        attestation={
            "version": 1,
            "provider": "local_fixture",
            "evidence_type": "prism.tee.v1",
            "tdx_quote_b64": "QUOTE",
            "gpu_eat_jwt": "JWT",
        },
    )
    proof_t1 = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256="c" * 64,
        image_digest=PINNED,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
    )
    assert effective_tier(proof_t2, pinned_image_digest=PINNED) == 0
    assert effective_tier(proof_t1, pinned_image_digest=PINNED) == 1
    assert effective_tier(proof_t1, pinned_image_digest=OTHER) == 0


def _bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _settings(tmp_path: Path, *, pinned: str | None = PINNED) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'notee.sqlite3'}",
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
        worker_plane=WorkerPlaneConfig(
            enabled=True,
            signing_key=WORKER_KEY,
            pinned_image_digest=pinned,
        ),
    )


def _manifest(marker: str = "v2") -> dict[str, Any]:
    covered_bytes = 4096
    online_loss = [10.0, 6.0, 3.0, 2.0]
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": online_loss,
            "sum_neg_log_likelihood_nats": 900.0,
            "covered_bytes": covered_bytes,
            "predicted_tokens": 96,
            "step0_loss": online_loss[0],
            "consumed_batches": len(online_loss),
            "random_init_baseline_nats": math.log(50257),
            "prequential_bpb": 1.23,
            "marker": marker,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
    }


def _proof_payload(
    signer: Any,
    unit_id: str,
    manifest: dict[str, Any],
    *,
    tier: int = 1,
    image_digest: str | None = PINNED,
    attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id=unit_id,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        image_digest=image_digest,
        attestation=attestation,
        tier=tier,  # type: ignore[arg-type]
    )
    return proof.model_dump(mode="json")


def _result(proof_dict: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof_dict,
        MANIFEST_PAYLOAD_KEY: manifest,
        "replication": 2,
    }


async def _make_app(settings: PrismSettings):
    app = create_app(settings)
    await app.state.database.init()
    return app


async def _seed(app, hotkey: str = "hk-notee") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    return sub.id


def _score(db_path: Path, submission_id: str):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


@pytest.mark.asyncio
async def test_score_finalize_works_without_tee_package(tmp_path: Path) -> None:
    """Worker-plane finalize succeeds with no tee package / no tee_required (VAL-NOTEE-003)."""
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    submission_id = await _seed(app)
    manifest = _manifest()
    proof = _proof_payload(signer, submission_id, manifest, tier=1, image_digest=PINNED)
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-notee",
        result=_result(proof, manifest),
        pinned_image_digest=PINNED,
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert outcome.effective_tier == 1
    assert outcome.claimed_tier == 1
    assert outcome.tier_downgraded is False
    assert outcome.reason is None
    assert _score(tmp_path / "notee.sqlite3", submission_id) is not None


@pytest.mark.asyncio
async def test_pin_mismatch_downgrades_but_still_finalizes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    submission_id = await _seed(app, hotkey="hk-mismatch")
    manifest = _manifest("mismatch")
    proof = _proof_payload(signer, submission_id, manifest, tier=1, image_digest=OTHER)
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-mismatch",
        result=_result(proof, manifest),
        pinned_image_digest=PINNED,
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert outcome.claimed_tier == 1
    assert outcome.effective_tier == 0
    assert outcome.tier_downgraded is True
    assert _score(tmp_path / "notee.sqlite3", submission_id) is not None


@pytest.mark.asyncio
async def test_ingestion_never_raises_tee_required(tmp_path: Path) -> None:
    """Attestation-claiming tier-2 proof finalizes without tee_required (max effective=0)."""
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    submission_id = await _seed(app, hotkey="hk-t2")
    manifest = _manifest("t2")
    attestation = {
        "version": 1,
        "provider": "local_fixture",
        "evidence_type": "prism.tee.v1",
        "tdx_quote_b64": "QUJDRA==",
        "gpu_eat_jwt": "aaa.bbb.ccc",
    }
    proof = _proof_payload(
        signer,
        submission_id,
        manifest,
        tier=2,
        image_digest=PINNED,
        attestation=attestation,
    )
    try:
        outcome = await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref="hk-t2",
            result=_result(proof, manifest),
            pinned_image_digest=PINNED,
        )
    except ResultIngestionError as exc:
        assert exc.reason != "tee_required"
        raise
    assert outcome.finalized is True
    assert outcome.effective_tier == 0
    assert outcome.tier_downgraded is True
    assert _score(tmp_path / "notee.sqlite3", submission_id) is not None
