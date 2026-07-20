"""base->prism result ingestion: proof verification + idempotent finalization.

Covers VAL-PRISM-007 (tampered manifest / forged digest rejected), VAL-PRISM-018 (missing/malformed
ExecutionProof rejected before scoring) and VAL-PRISM-017 (duplicate/late delivery never mutates a
finalized submission). Offline, no GPU: finalization runs through the CPU re-exec seam the other
prism coordination tests use, and proofs are signed with real sr25519 worker keys.
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
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import (
    ResultIngestionError,
    ingest_work_unit_result,
    parse_execution_proof,
    verify_proof_integrity,
)
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerIngest"

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


def _settings(tmp_path: Path, *, worker_plane: WorkerPlaneConfig) -> PrismSettings:
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
    )


def _manifest(marker: str = "v2") -> dict[str, Any]:
    """A plausible AND scoreable v2 manifest (worker-plane finalization scores from it directly)."""

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


def _score(db_path: Path, submission_id: str):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


# --- VAL-PRISM-018: missing / malformed ExecutionProof rejected before scoring -------------------


def test_parse_rejects_missing_and_malformed_proof() -> None:
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    good = _proof_dict(signer, "unit-1", manifest)

    # (a) envelope entirely absent
    with pytest.raises(ResultIngestionError) as absent:
        parse_execution_proof({})
    assert absent.value.reason == "proof_missing"

    # (b) version != 1
    with pytest.raises(ResultIngestionError) as version:
        parse_execution_proof(_result({**good, "version": 2}, manifest))
    assert version.value.reason == "proof_bad_version"

    # (c) manifest_sha256 not 64-char lowercase hex (uppercase, too short, non-hex)
    for bad_hash in (good["manifest_sha256"].upper(), "abc", "z" * 64):
        with pytest.raises(ResultIngestionError) as bad:
            parse_execution_proof(_result({**good, "manifest_sha256": bad_hash}, manifest))
        assert bad.value.reason == "proof_bad_manifest_hash"

    # (d) worker_signature missing worker_pubkey or sig
    pubkey = good["worker_signature"]["worker_pubkey"]
    for signature in ({"worker_pubkey": pubkey}, {"sig": "0xab"}):
        with pytest.raises(ResultIngestionError) as sig:
            parse_execution_proof(_result({**good, "worker_signature": signature}, manifest))
        assert sig.value.reason == "proof_missing_signature"

    # A well-formed control parses.
    assert parse_execution_proof(_result(good, manifest)).version == 1


async def test_ingestion_rejects_malformed_proof_before_scoring(tmp_path, monkeypatch) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    # Each malformed variant is rejected with a distinguishable reason and scores nothing.
    variants = {
        "proof_missing": {"executed": 1},
        "proof_bad_version": None,  # filled below
        "proof_bad_manifest_hash": None,
        "proof_missing_signature": None,
    }
    for reason in list(variants):
        submission_id = await _seed(app)
        manifest = _manifest()
        good = _proof_dict(signer, submission_id, manifest)
        if reason == "proof_missing":
            result = {"executed": 1}
        elif reason == "proof_bad_version":
            result = _result({**good, "version": 3}, manifest)
        elif reason == "proof_bad_manifest_hash":
            result = _result({**good, "manifest_sha256": "nothex"}, manifest)
        else:
            result = _result({**good, "worker_signature": {"worker_pubkey": "x"}}, manifest)

        with pytest.raises(ResultIngestionError) as exc:
            await ingest_work_unit_result(
                worker=app.state.worker,
                work_unit_id=submission_id,
                submission_ref="hk-owner",
                result=result,
            )
        assert exc.value.reason == reason
        assert _score(db_path, submission_id) is None
        assert await app.state.repository.submission_status(submission_id) == "pending"

    # A well-formed control from the same fixture IS accepted and finalizes a score.
    submission_id = await _seed(app)
    manifest = _manifest()
    good = _proof_dict(signer, submission_id, manifest)
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(good, manifest),
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert await app.state.repository.submission_status(submission_id) == "completed"
    assert _score(db_path, submission_id) is not None


# --- VAL-PRISM-007: tampered manifest / forged digest rejected ----------------------------------


def test_verify_rejects_tamper_and_forgery() -> None:
    signer = worker_signer_from_key(WORKER_KEY)
    unit_id = "unit-7"
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer, manifest_sha256=compute_manifest_sha256(manifest), unit_id=unit_id
    )

    # A genuine (manifest, proof) pair verifies.
    verify_proof_integrity(proof, unit_id=unit_id, manifest=manifest)

    # (a) a manifest mutated after signing no longer hashes to manifest_sha256
    tampered_manifest = {**manifest, "metrics": {**manifest["metrics"], "prequential_bpb": 0.01}}
    with pytest.raises(ResultIngestionError) as tamper:
        verify_proof_integrity(proof, unit_id=unit_id, manifest=tampered_manifest)
    assert tamper.value.reason == "manifest_tampered"

    # (b) manifest_sha256 rewritten to match the tampered manifest, signature NOT re-issued
    forged = proof.model_copy(
        update={"manifest_sha256": compute_manifest_sha256(tampered_manifest)}
    )
    with pytest.raises(ResultIngestionError) as forged_hash:
        verify_proof_integrity(forged, unit_id=unit_id, manifest=tampered_manifest)
    assert forged_hash.value.reason == "signature_invalid"

    # (c) corrupted signature bytes
    corrupt = proof.model_copy(
        update={"worker_signature": proof.worker_signature.model_copy(update={"sig": "0x00"})}
    )
    with pytest.raises(ResultIngestionError) as corrupt_sig:
        verify_proof_integrity(corrupt, unit_id=unit_id, manifest=manifest)
    assert corrupt_sig.value.reason == "signature_invalid"


async def test_ingestion_rejects_tampered_result_without_scoring(tmp_path, monkeypatch) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    for kind in ("tamper", "forged_digest", "corrupt_sig"):
        submission_id = await _seed(app)
        manifest = _manifest()
        proof = _proof_dict(signer, submission_id, manifest)
        if kind == "tamper":
            result = _result(proof, {**manifest, "metrics": {"prequential_bpb": 9.9}})
            expected = "manifest_tampered"
        elif kind == "forged_digest":
            tampered = {**manifest, "metrics": {"prequential_bpb": 9.9}}
            forged = {**proof, "manifest_sha256": compute_manifest_sha256(tampered)}
            result = _result(forged, tampered)
            expected = "signature_invalid"
        else:
            corrupt = {**proof, "worker_signature": {**proof["worker_signature"], "sig": "0x00"}}
            result = _result(corrupt, manifest)
            expected = "signature_invalid"

        with pytest.raises(ResultIngestionError) as exc:
            await ingest_work_unit_result(
                worker=app.state.worker,
                work_unit_id=submission_id,
                submission_ref="hk-owner",
                result=result,
            )
        assert exc.value.reason == expected
        assert _score(db_path, submission_id) is None
        assert await app.state.repository.submission_status(submission_id) == "pending"


# --- VAL-PRISM-017: duplicate / late delivery never mutates a finalized submission ---------------


async def test_duplicate_and_conflicting_delivery_never_mutates_score(
    tmp_path, monkeypatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    submission_id = await _seed(app)
    manifest = _manifest()
    proof = _proof_dict(signer, submission_id, manifest)

    # First delivery finalizes.
    first = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert first.status == "accepted" and first.finalized is True
    score_after_first = _score(db_path, submission_id)
    assert score_after_first is not None
    jobs_after_first = _eval_job_count(db_path, submission_id)

    # (a) a SECOND delivery of the SAME result is an idempotent no-op: no re-score, no new eval job.
    second = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert second.status == "accepted" and second.idempotent is True and second.finalized is False
    assert _score(db_path, submission_id) == pytest.approx(score_after_first)
    assert _eval_job_count(db_path, submission_id) == jobs_after_first

    # (b) a CONFLICTING delivery (different, validly-signed manifest) is refused, not applied.
    conflict_manifest = _manifest("conflict")
    conflict_proof = _proof_dict(signer, submission_id, conflict_manifest)
    conflicting = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(conflict_proof, conflict_manifest),
    )
    assert conflicting.status == "conflict" and conflicting.reason == "manifest_conflict"
    assert _score(db_path, submission_id) == pytest.approx(score_after_first)
    assert _eval_job_count(db_path, submission_id) == jobs_after_first

    # (c) the legacy in-process equivalent: process_submission on a finalized submission is a no-op.
    assert await app.state.worker.process_submission(submission_id) is None
    assert _score(db_path, submission_id) == pytest.approx(score_after_first)


def _eval_job_count(db_path: Path, submission_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM eval_jobs WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


# --- VAL-PRISM-019: unverifiable tier claims are downgraded + recorded at ingestion --------------


async def test_ingestion_records_downgraded_effective_tier(tmp_path, monkeypatch) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    submission_id = await _seed(app)
    manifest = _manifest()
    # A tier-1 claim whose image_digest does not match the pinned digest is
    # unverifiable under SDK-strict proof shape -> effective tier 0.
    proof = _proof_dict(
        signer,
        submission_id,
        manifest,
        tier=1,
        image_digest="sha256:" + ("ab" * 32),
        attestation=None,
    )
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
        pinned_image_digest="sha256:" + ("cd" * 32),
    )
    assert outcome.claimed_tier == 1
    assert outcome.effective_tier == 0
    assert outcome.tier_downgraded is True

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT claimed_tier, effective_tier, tier_downgraded FROM work_unit_results "
            "WHERE work_unit_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row == (1, 0, 1)


# --- HTTP route body contract + status codes (VAL-PRISM-017/018) ---------------------------------


def _envelope(
    *,
    work_unit_id: str,
    submission_ref: str,
    proof: dict[str, Any],
    manifest: dict[str, Any] | None,
    challenge_slug: str = "prism",
    assignment_id: str | None = None,
) -> dict[str, Any]:
    """Build the canonical ExternalResultEnvelope body for HTTP tests."""

    return {
        "api_version": "1.0",
        "work_unit_id": work_unit_id,
        "assignment_id": assignment_id or work_unit_id,
        "submission_ref": submission_ref,
        "challenge_slug": challenge_slug,
        "result": _result(proof, manifest),
        "proof": proof,
    }


def test_result_route_body_contract(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    signer = worker_signer_from_key(WORKER_KEY)
    headers = {"Authorization": "Bearer secret"}

    with TestClient(create_app(settings)) as client:
        seed = client.post(
            "/internal/v1/bridge/submissions",
            content=base64.b64decode(_bundle()),
            headers={
                "Authorization": "Bearer secret",
                "X-Base-Verified-Hotkey": "hk-owner",
                "X-Submission-Filename": "project.zip",
                "Content-Type": "application/octet-stream",
            },
        )
        assert seed.status_code == 200, seed.text
        submission_id = seed.json()["id"]

        manifest = _manifest()
        proof = _proof_dict(signer, submission_id, manifest)
        body = _envelope(
            work_unit_id=submission_id,
            submission_ref="hk-owner",
            proof=proof,
            manifest=manifest,
        )

        # Unauthenticated is rejected.
        assert client.post("/internal/v1/work_units/result", json=body).status_code == 401

        # A well-formed ExternalResultEnvelope is accepted (canonical base body).
        accept = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        assert accept.status_code == 200, accept.text
        assert accept.json()["status"] == "accepted"

        # Legacy reduced body without api_version/proof bindings is rejected before score.
        legacy = {
            "work_unit_id": submission_id,
            "submission_ref": "hk-owner",
            "result": _result(proof, manifest),
        }
        legacy_rejected = client.post(
            "/internal/v1/work_units/result", json=legacy, headers=headers
        )
        assert legacy_rejected.status_code == 422
        assert legacy_rejected.json()["detail"]["code"] == "result_envelope_invalid"

        # Challenge binding mismatch is rejected before scoring/persistence.
        wrong_challenge = _envelope(
            work_unit_id=submission_id,
            submission_ref="hk-owner",
            proof=proof,
            manifest=manifest,
            challenge_slug="other-challenge",
        )
        mismatched = client.post(
            "/internal/v1/work_units/result", json=wrong_challenge, headers=headers
        )
        assert mismatched.status_code == 422
        assert mismatched.json()["detail"]["code"] == "result_challenge_mismatch"

        # A malformed proof is rejected 422 with a distinguishable reason code.
        bad = _envelope(
            work_unit_id=submission_id,
            submission_ref="hk-owner",
            proof={**proof, "version": 5},
            manifest=manifest,
        )
        rejected = client.post("/internal/v1/work_units/result", json=bad, headers=headers)
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["code"] == "result_envelope_invalid"

        # A conflicting redelivery for the finalized unit is refused 409.
        conflict_manifest = _manifest("conflict")
        conflict_proof = _proof_dict(signer, submission_id, conflict_manifest)
        conflict = client.post(
            "/internal/v1/work_units/result",
            json=_envelope(
                work_unit_id=submission_id,
                submission_ref="hk-owner",
                proof=conflict_proof,
                manifest=conflict_manifest,
            ),
            headers=headers,
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "manifest_conflict"


def test_result_route_disabled_when_worker_plane_off(tmp_path) -> None:
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path, worker_plane=WorkerPlaneConfig(enabled=False))
    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/internal/v1/work_units/result",
            json=_envelope(
                work_unit_id="s1",
                submission_ref="hk",
                proof={
                    "version": 1,
                    "tier": 0,
                    "manifest_sha256": "0" * 64,
                    "worker_signature": {"worker_pubkey": "wk", "sig": "0x01"},
                },
                manifest=None,
            ),
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 404
