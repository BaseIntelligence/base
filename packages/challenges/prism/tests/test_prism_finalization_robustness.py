"""Prism finalization robustness follow-ups (scrutiny prism-finalization).

Two small robustness/efficiency guards on the sealed prism-finalization milestone:

* Fix 1 -- ``PrismWorker.finalize_worker_result`` no longer masquerades an internal
  SOURCE-derivation failure as a clean finalize. A derivation error (snapshot / component review /
  anti-cheat) reverts the claimed submission to ``pending`` (retryable) and raises
  ``WorkerFinalizationError``; ingestion surfaces it as a distinct ``finalization_failed`` outcome
  (``finalized=False``, nothing recorded, the forwarded result can be retried) instead of
  ``finalized=True`` with ``status=failed``. The happy path and every VAL-FINAL assertion hold.
* Fix 2 -- ``run_validator_audit_cycle`` claims each pending audit under a lightweight per-audit
  lease so, in a MULTI-validator deployment, each pending audit is replayed by at most one validator
  (idempotent but wasteful redundant GPU/CPU replays are avoided). Single-validator behaviour
  (harness + live e2e) and VAL-FINAL-005 are unchanged.

Offline, no GPU: worker-plane finalization scores from the forwarded manifest and audits use an
injected deterministic replay; proofs are signed with real sr25519 worker keys.
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
from prism_challenge.audit import (
    AUDIT_STATUS_PASSED,
    AUDIT_STATUS_PENDING,
    AuditSampler,
    audit_unit_id_for,
)
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.ingestion import ResultIngestionError, ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.queue import PrismWorker, WorkerFinalizationError
from prism_challenge.validator_executor import run_validator_audit_cycle

WORKER_KEY = "//WorkerRobustness"

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


def _settings(tmp_path: Path, *, worker_plane: WorkerPlaneConfig) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'robust.sqlite3'}",
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


def _always() -> AuditSampler:
    return AuditSampler(audit_rate_tier0=1.0, audit_rate_tier1=1.0, audit_rate_tier2=1.0)


async def _finalize_and_sample(app, signer, *, hotkey: str = "hk-owner") -> tuple[str, dict, Any]:
    submission_id = await _seed(app, hotkey)
    manifest = _manifest()
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref=hotkey,
        result=_result(_proof_dict(signer, submission_id, manifest), manifest),
        audit_sampler=_always(),
    )
    return submission_id, manifest, outcome


def _score(db_path: Path, submission_id: str):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


# --- Fix 1: derivation failure is retryable, not a silent terminal finalize ----------------------


async def test_finalize_worker_result_raises_on_derivation_failure(tmp_path, monkeypatch) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    repository = app.state.repository

    def _boom(self, snapshot):  # noqa: ANN001, ANN202
        raise RuntimeError("transient derivation glitch")

    monkeypatch.setattr(PrismWorker, "_component_review", _boom)

    submission_id = await _seed(app)
    with pytest.raises(WorkerFinalizationError):
        await app.state.worker.finalize_worker_result(submission_id, _manifest())

    # The claimed submission is reverted to pending (retryable), NOT terminally failed.
    assert await repository.submission_status(submission_id) == "pending"


async def test_ingestion_reports_derivation_failure_as_retryable(tmp_path, monkeypatch) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository
    db_path = tmp_path / "robust.sqlite3"

    real_review = PrismWorker._component_review
    calls = {"n": 0}

    def _flaky(self, snapshot):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient derivation glitch")
        return real_review(self, snapshot)

    monkeypatch.setattr(PrismWorker, "_component_review", _flaky)

    submission_id = await _seed(app)
    manifest = _manifest()
    result = _result(_proof_dict(signer, submission_id, manifest), manifest)

    # First delivery: a derivation failure is a DISTINCT, retryable ingestion error -- NOT a clean
    # finalize. Nothing is scored, the submission stays pending, and no work-unit result is recorded
    # (so a redelivery is genuinely retried rather than idempotent-skipped).
    with pytest.raises(ResultIngestionError) as exc:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref="hk-owner",
            result=result,
        )
    assert exc.value.reason == "finalization_failed"
    assert await repository.submission_status(submission_id) == "pending"
    assert _score(db_path, submission_id) is None
    assert await repository.get_work_unit_result(submission_id) is None

    # Redelivery once the transient condition clears: the SAME forwarded result finalizes cleanly.
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=result,
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert await repository.submission_status(submission_id) == "completed"
    assert _score(db_path, submission_id) is not None


def test_result_route_returns_503_on_finalization_failure(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    signer = worker_signer_from_key(WORKER_KEY)
    headers = {"Authorization": "Bearer secret"}

    def _boom(self, snapshot):  # noqa: ANN001, ANN202
        raise RuntimeError("transient derivation glitch")

    monkeypatch.setattr(PrismWorker, "_component_review", _boom)

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
        body = {
            "api_version": "1.0",
            "work_unit_id": submission_id,
            "assignment_id": submission_id,
            "submission_ref": "hk-owner",
            "challenge_slug": "prism",
            "result": _result(proof, manifest),
            "proof": proof,
        }
        resp = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        # A transient internal derivation failure is retryable -> 503, distinct from the permanent
        # 422 rejections (bad proof / implausible manifest) and the 409 conflict.
        assert resp.status_code == 503, resp.text
        assert resp.json()["detail"]["code"] == "finalization_failed"


# --- Fix 2: per-audit claim/lease so at most one validator replays each pending audit -------------


async def test_claim_audit_unit_is_single_consumer_with_lease(tmp_path) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository

    submission_id, _, outcome = await _finalize_and_sample(app, signer)
    audit_unit_id = outcome.audit_unit_id
    assert audit_unit_id == audit_unit_id_for(submission_id)

    # The first validator wins the claim; a second validator with a live lease is refused.
    assert await repository.claim_audit_unit(
        audit_unit_id, claimant="validator-1", lease_seconds=10_000
    )
    assert not await repository.claim_audit_unit(
        audit_unit_id, claimant="validator-2", lease_seconds=10_000
    )
    # The unit is still pending (claim is orthogonal to lifecycle status).
    unit = await repository.get_audit_unit(audit_unit_id)
    assert unit is not None
    assert unit["status"] == AUDIT_STATUS_PENDING
    assert unit["claimed_by"] == "validator-1"

    # An expired lease (cutoff == now) is reclaimable by another validator.
    assert await repository.claim_audit_unit(audit_unit_id, claimant="validator-3", lease_seconds=0)
    reclaimed = await repository.get_audit_unit(audit_unit_id)
    assert reclaimed is not None
    assert reclaimed["claimed_by"] == "validator-3"


async def test_audit_cycle_skips_units_claimed_by_another_validator(tmp_path) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository

    submission_id, manifest, outcome = await _finalize_and_sample(app, signer)
    audit_unit_id = outcome.audit_unit_id

    # Another validator holds a live claim on the only pending audit.
    assert await repository.claim_audit_unit(
        audit_unit_id, claimant="other-validator", lease_seconds=10_000
    )

    replays: list[str] = []

    async def _replay(sub_id: str) -> str:
        replays.append(sub_id)
        return compute_manifest_sha256(manifest)

    summary = await run_validator_audit_cycle(worker=app.state.worker, audit_replay=_replay)

    # This validator never replays a claimed audit: no wasted GPU/CPU, nothing resolved.
    assert replays == []
    assert summary.pulled == 0
    assert summary.executed == 0
    assert summary.skipped == 1
    assert summary.audits == ()
    still_pending = await repository.get_audit_unit(audit_unit_id)
    assert still_pending is not None
    assert still_pending["status"] == AUDIT_STATUS_PENDING


async def test_single_validator_audit_claims_replays_and_clears_claim(tmp_path) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "robust.sqlite3"

    submission_id, manifest, outcome = await _finalize_and_sample(app, signer)
    score_before = _score(db_path, submission_id)
    assert score_before is not None

    replays: list[str] = []

    async def _replay(sub_id: str) -> str:
        replays.append(sub_id)
        return compute_manifest_sha256(manifest)

    # Single-validator behaviour is unchanged: the sole validator claims, replays, and passes.
    summary = await run_validator_audit_cycle(worker=app.state.worker, audit_replay=_replay)
    assert replays == [submission_id]
    assert summary.pulled == 1
    assert summary.executed == 1
    assert summary.skipped == 0
    assert len(summary.audits) == 1
    assert summary.audits[0].status == AUDIT_STATUS_PASSED
    assert _score(db_path, submission_id) == pytest.approx(score_before)

    # A terminal (passed) audit is no longer pending, so a second cycle bears no work.
    again = await run_validator_audit_cycle(worker=app.state.worker, audit_replay=_replay)
    assert again.pulled == 0
    assert again.executed == 0


async def test_inconclusive_audit_claim_cleared_and_reclaimable(tmp_path) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository

    submission_id, _, outcome = await _finalize_and_sample(app, signer)
    audit_unit_id = outcome.audit_unit_id

    replays: list[str] = []

    async def _inconclusive(sub_id: str) -> str | None:
        replays.append(sub_id)
        return None

    # An inconclusive replay (attempts < max) returns the audit to pending; the claim MUST be
    # cleared so the re-audit is immediately reclaimable (by any validator, with a LIVE lease).
    first = await run_validator_audit_cycle(worker=app.state.worker, audit_replay=_inconclusive)
    assert first.pulled == 1
    reverted = await repository.get_audit_unit(audit_unit_id)
    assert reverted is not None
    assert reverted["status"] == AUDIT_STATUS_PENDING
    assert reverted["claimed_at"] is None

    second = await run_validator_audit_cycle(worker=app.state.worker, audit_replay=_inconclusive)
    assert second.pulled == 1
    assert replays == [submission_id, submission_id]
