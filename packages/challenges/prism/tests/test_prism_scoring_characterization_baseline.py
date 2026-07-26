"""Post-fail-closed scoring contract (todo 18 baseline rewritten by todo 22).

Todo 18 pinned PRE-change reality (pin mismatch still scored). Todo 22 deliberately
broke that contract (P1: no valid constation bundle ⇒ no score row). This file now
documents the NEW contract — tests are rewritten, not deleted.
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
from prism_challenge.audit import effective_tier
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.constation import CheckOutcome, ConstationBundle
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    ProviderInfo,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerCharBaseline"
PINNED = "sha256:" + ("aa" * 32)
OTHER = "sha256:" + ("bb" * 32)
DIGEST = "sha256:" + ("11" * 32)

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


def _tier1_proof_dict(signer, unit_id: str, manifest: dict[str, Any], *, image_digest: str):
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=unit_id,
        image_digest=image_digest,
        constation_digest=image_digest,
        provider=ProviderInfo(name="lium", pod_id="pod-char-1"),
        tier=1,  # type: ignore[arg-type]
    )
    return proof.model_dump(mode="json")


def _result(proof_dict: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof_dict,
        MANIFEST_PAYLOAD_KEY: manifest,
    }


def _constation_bundle(digest: str = DIGEST) -> ConstationBundle:
    man = {"h.py": "a" * 64}
    return ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest=digest,
        work_unit_id="wu",
        miner_hotkey="hk",
        pod_id="pod-char-1",
        nonce="n",
        signed_attestation={"s": "1"},
        expected_sealed_manifest_hashes=dict(man),
        reported_sealed_manifest_hashes=dict(man),
        lium_declared_digest=digest,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
    )


def _ok_checkers():
    def allow(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def nonce(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def sig(_s: object) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    return allow, nonce, sig


async def _make_app(settings: PrismSettings):
    app = create_app(settings)
    await app.state.database.init()
    return app


async def _seed(app, hotkey: str = "hk-owner") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    return sub.id


def _final_score(db_path: Path, submission_id: str) -> float | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else float(row[0])


async def test_pin_mismatch_without_constation_writes_no_final_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW (todo 22): IMAGE_PIN mismatch / missing constation ⇒ no score row (P1)."""
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    submission_id = await _seed(app)
    manifest = _manifest("pin-mismatch")
    proof = _tier1_proof_dict(signer, submission_id, manifest, image_digest=OTHER)

    from prism_challenge.proof import ExecutionProof

    assert effective_tier(ExecutionProof.model_validate(proof), pinned_image_digest=PINNED) == 0

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
        pinned_image_digest=PINNED,
        constation_bundle=None,
    )

    assert outcome.status == "rejected"
    assert outcome.finalized is False
    assert outcome.score_written is False
    assert outcome.reason == "miner_fault:missing_constation_bundle"
    assert _final_score(db_path, submission_id) is None


async def test_constation_ok_writes_score_and_sets_effective_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW: valid constation bundle ⇒ score written; effective_tier follows constation_ok."""
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    allow, nonce, sig = _ok_checkers()

    submission_id = await _seed(app, hotkey="hk-ok")
    manifest = _manifest("constation-ok")
    proof = _tier1_proof_dict(signer, submission_id, manifest, image_digest=DIGEST)
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-ok",
        result=_result(proof, manifest),
        pinned_image_digest=DIGEST,
        constation_bundle=_constation_bundle(DIGEST),
        check_allowlist=allow,
        check_nonce=nonce,
        verify_constation_signature=sig,
    )
    assert outcome.status == "accepted"
    assert outcome.effective_tier == 1
    assert outcome.finalized is True
    score = _final_score(db_path, submission_id)
    assert score is not None and score > 0.0
