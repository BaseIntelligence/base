"""Validator audit-execution wiring + salted audit sampler (architecture.md 3.4/3.5).

Covers the mission finalization assertions this feature delivers:

* VAL-FINAL-005 - with the worker plane ON the validator cycle is AUDIT-ONLY: it pulls a sampled
  ``audit:`` unit, replays the audited submission's evaluation to obtain a fresh manifest hash, and
  resolves it through ``resolve_audit_unit`` (the ``POST /internal/v1/audit_units/{id}/result``
  target). A MATCHING replay leaves the finalized score untouched; a DIVERGENT replay invalidates
  the score AND records a ``worker_fault``. Under the flag the validator cycle NEVER executes a
  primary submission; with the flag OFF the cycle is the legacy primary-execution path (no audits).
* VAL-FINAL-006 - the audit sampler mixes a server-side secret salt into its seed so selection is
  unpredictable from the public ``submission_id`` alone (a different salt selects a different set)
  yet reproducible for a fixed salt, and the per-tier rates are preserved statistically.

Offline, no GPU: finalization + the real replay run through the CPU re-exec seam the other
coordination tests use, and proofs are signed with real sr25519 worker keys.
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
    AUDIT_STATUS_MISMATCH,
    AUDIT_STATUS_PASSED,
    AuditSampler,
    audit_sampler_from_config,
    audit_unit_id_for,
)
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.queue import PrismWorker
from prism_challenge.validator_dispatch import dispatch_assignment
from prism_challenge.validator_executor import (
    run_validator_audit_cycle,
    run_validator_cycle,
)

WORKER_KEY = "//WorkerAuditExec"
EPOCH_SECONDS = 60
BROKER_URL = "http://broker-val:8082"

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
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'audit_exec.sqlite3'}",
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
        epoch_seconds=EPOCH_SECONDS,
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


def _result(signer, unit_id: str, manifest: dict[str, Any], **extra: Any) -> dict[str, Any]:
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(signer=signer, manifest_sha256=digest, unit_id=unit_id)
    return {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
        **extra,
    }


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
    """Finalize a worker result (worker plane) and force it sampled -> a pending audit unit."""
    from prism_challenge.ingestion import ingest_work_unit_result

    submission_id = await _seed(app, hotkey)
    manifest = _manifest()
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref=hotkey,
        result=_result(signer, submission_id, manifest),
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


# --- VAL-FINAL-006: salted audit sampler ---------------------------------------------------------


def test_audit_salt_shifts_selection_but_is_reproducible_for_fixed_salt() -> None:
    ids = [f"submission-{i}" for i in range(400)]

    def _select(salt: str) -> list[bool]:
        sampler = AuditSampler(audit_rate_tier0=0.10, seed=7, salt=salt)
        return [sampler.should_sample(work_unit_id=i, effective_tier=0) for i in ids]

    no_salt = _select("")
    salt_a1 = _select("SERVER-SECRET-A")
    salt_a2 = _select("SERVER-SECRET-A")
    salt_b = _select("SERVER-SECRET-B")

    # A fixed salt reproduces the same selection ...
    assert salt_a1 == salt_a2
    # ... but a different salt selects a different set (unpredictable from submission_id alone) ...
    assert salt_a1 != salt_b
    # ... and the salted selection is not the same as the public (submission_id-only) one.
    assert salt_a1 != no_salt


def test_audit_salt_preserves_per_tier_rates_statistically() -> None:
    n = 6000

    def _bound(p: float) -> float:
        return 4.0 * (p * (1.0 - p) / n) ** 0.5

    for salt in ("", "salt-one", "another-secret-salt"):
        sampler = AuditSampler(
            audit_rate_tier0=0.10, audit_rate_tier1=0.05, audit_rate_tier2=0.02, seed=99, salt=salt
        )
        for tier, rate in ((0, 0.10), (1, 0.05), (2, 0.02)):
            hits = sum(
                sampler.should_sample(work_unit_id=f"{salt}-t{tier}-{i}", effective_tier=tier)
                for i in range(n)
            )
            assert abs(hits / n - rate) < _bound(rate), (salt, tier, hits / n, rate)


def test_audit_sampler_from_config_mixes_secret_salt_repr_hidden() -> None:
    ids = [f"unit-{i}" for i in range(300)]
    cfg_a = WorkerPlaneConfig(enabled=True, audit_salt="TOP-SECRET-SALT")
    cfg_b = WorkerPlaneConfig(enabled=True, audit_salt="OTHER-SECRET-SALT")
    cfg_none = WorkerPlaneConfig(enabled=True)

    sel_a = [
        audit_sampler_from_config(cfg_a).should_sample(work_unit_id=i, effective_tier=0)
        for i in ids
    ]
    sel_a_again = [
        audit_sampler_from_config(cfg_a).should_sample(work_unit_id=i, effective_tier=0)
        for i in ids
    ]
    sel_b = [
        audit_sampler_from_config(cfg_b).should_sample(work_unit_id=i, effective_tier=0)
        for i in ids
    ]
    sel_none = [
        audit_sampler_from_config(cfg_none).should_sample(work_unit_id=i, effective_tier=0)
        for i in ids
    ]

    assert sel_a == sel_a_again  # reproducible for a fixed salt
    assert sel_a != sel_b  # a different salt shifts the selected set
    assert sel_a != sel_none  # unpredictable from submission_id (the public value) alone
    # The salt is a server-side SECRET: it never appears in the config repr.
    assert "TOP-SECRET-SALT" not in repr(cfg_a)
    assert "audit_salt" not in repr(cfg_a)


# --- VAL-FINAL-005: validator audit cycle (injectable replay for deterministic hashes) -----------


async def test_matching_replay_passes_and_leaves_score_untouched(tmp_path) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository
    db_path = tmp_path / "audit_exec.sqlite3"

    submission_id, manifest, outcome = await _finalize_and_sample(app, signer)
    assert outcome.audit_unit_id == audit_unit_id_for(submission_id)
    score_before = _score(db_path, submission_id)
    assert score_before is not None

    # A validator replay reproducing the audited manifest hash: the audit passes, score untouched.
    async def _replay(sub_id: str) -> str:
        assert sub_id == submission_id
        return compute_manifest_sha256(manifest)

    summary = await run_validator_audit_cycle(
        worker=app.state.worker,
        work_unit_ids=[outcome.audit_unit_id],
        audit_replay=_replay,
    )
    assert summary.pulled == 1
    assert summary.executed == 1
    assert len(summary.audits) == 1
    assert summary.audits[0].status == AUDIT_STATUS_PASSED
    assert summary.audits[0].invalidated is False
    assert _score(db_path, submission_id) == pytest.approx(score_before)
    assert await repository.submission_status(submission_id) == "completed"
    assert await repository.list_worker_faults(submission_id=submission_id) == []


async def test_divergent_replay_invalidates_score_and_records_worker_fault(tmp_path) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository
    db_path = tmp_path / "audit_exec.sqlite3"

    submission_id, _, outcome = await _finalize_and_sample(app, signer)
    assert _score(db_path, submission_id) is not None

    async def _replay(_sub_id: str) -> str:
        return "f" * 64  # authoritative replay diverges from the worker's manifest hash

    summary = await run_validator_audit_cycle(
        worker=app.state.worker,
        work_unit_ids=[outcome.audit_unit_id],
        audit_replay=_replay,
    )
    assert summary.audits[0].status == AUDIT_STATUS_MISMATCH
    assert summary.audits[0].invalidated is True
    # Score invalidated + submission dropped from the leaderboard.
    assert _score(db_path, submission_id) is None
    assert await repository.submission_status(submission_id) == "failed"

    # A worker_fault is recorded against the divergent (lying) worker for the audited unit.
    faults = await repository.list_worker_faults(submission_id=submission_id)
    assert len(faults) == 1
    fault = faults[0]
    assert fault["submission_id"] == submission_id
    assert fault["worker_pubkey"] == signer.worker_pubkey
    assert fault["audit_unit_id"] == outcome.audit_unit_id
    assert fault["replay_manifest_sha256"] == "f" * 64


async def test_validator_cycle_is_audit_only_when_flag_on(tmp_path, monkeypatch) -> None:
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    repository = app.state.repository

    # A pending PRIMARY submission that the validator cycle must NOT execute while the flag is on.
    primary_id = await _seed(app, "hk-primary")

    # An independent finalized+sampled submission (its audit unit is the only work the cycle bears).
    audited_id, manifest, outcome = await _finalize_and_sample(app, signer, hotkey="hk-audited")

    # If the cycle ever tried to execute a primary submission, these would fire.
    def _boom_container(*_a: Any, **_k: Any):
        raise AssertionError("validator cycle executed a primary submission while the flag was ON")

    monkeypatch.setattr(PrismWorker, "_evaluate_within_wall_time", _boom_container)

    async def _replay(sub_id: str) -> str:
        assert sub_id == audited_id
        return compute_manifest_sha256(manifest)

    summary = await run_validator_cycle(worker=app.state.worker, audit_replay=_replay)

    # The primary was never pulled/executed; it stays pending, unscored.
    assert summary.executed == 0 or all(res.audit_unit_id != primary_id for res in summary.audits)
    assert await repository.submission_status(primary_id) == "pending"
    # The sampled audit was executed + resolved.
    assert len(summary.audits) == 1
    assert summary.audits[0].status == AUDIT_STATUS_PASSED
    assert await repository.submission_status(audited_id) == "completed"


async def test_validator_cycle_flag_off_executes_primary_and_runs_no_audit(
    tmp_path, monkeypatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path, worker_plane=WorkerPlaneConfig(enabled=False))
    app = await _make_app(settings)
    repository = app.state.repository
    db_path = tmp_path / "audit_exec.sqlite3"

    primary_id = await _seed(app, "hk-legacy")

    # Even if a stray audit row exists, the flag-off cycle never touches audits.
    await repository.create_audit_unit(
        submission_id=primary_id,
        origin_work_unit_id=primary_id,
        audited_manifest_sha256="a" * 64,
        effective_tier=0,
    )

    def _boom_replay(*_a: Any, **_k: Any):
        raise AssertionError("flag-off validator cycle attempted an audit replay")

    monkeypatch.setattr(PrismWorker, "replay_audit_manifest_sha256", _boom_replay)

    summary = await run_validator_cycle(worker=app.state.worker)
    # Legacy primary execution: the submission is re-executed and finalized, no audit resolutions.
    assert summary.executed == 1
    assert primary_id in summary.completed_submissions
    assert summary.audits == ()
    assert _score(db_path, primary_id) is not None


async def test_dispatch_audit_unit_real_replay_diverges_invalidates_and_faults(
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
    repository = app.state.repository
    db_path = tmp_path / "audit_exec.sqlite3"

    # Finalize from a hand-crafted worker manifest; the honest replay produces the REAL runner
    # manifest which differs, so the audit must catch the divergence.
    submission_id, _, outcome = await _finalize_and_sample(app, signer)
    await app.state.database.close()
    assert _score(db_path, submission_id) is not None

    # The base validator agent dispatches the pulled audit unit here. The audit payload carries NO
    # gateway token (audits skip the LLM review), so the audit path must not require one.
    result = await dispatch_assignment(
        work_unit_id=outcome.audit_unit_id,
        payload={"audit": True, "audited_submission_id": submission_id},
        broker_url=BROKER_URL,
        settings=settings,
    )
    assert result["audits_resolved"] == 1
    assert result["audits_invalidated"] == 1

    app2 = await _make_app(settings)
    repository = app2.state.repository
    assert _score(db_path, submission_id) is None
    assert await repository.submission_status(submission_id) == "failed"
    faults = await repository.list_worker_faults(submission_id=submission_id)
    assert len(faults) == 1
    assert faults[0]["worker_pubkey"] == signer.worker_pubkey


def test_audit_result_route_records_worker_fault_on_mismatch(tmp_path, monkeypatch) -> None:
    import anyio
    from fastapi.testclient import TestClient

    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    signer = worker_signer_from_key(WORKER_KEY)
    headers = {"Authorization": "Bearer secret"}
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    with TestClient(create_app(settings)) as client:
        app = client.app

        async def _drive() -> tuple[str, str]:
            sid, _, out = await _finalize_and_sample(app, signer)
            return sid, out.audit_unit_id

        submission_id, audit_unit_id = anyio.run(_drive)

        # A validator posts its divergent replay hash to the internal audit route -> resolve_audit.
        resolved = client.post(
            f"/internal/v1/audit_units/{audit_unit_id}/result",
            json={"manifest_sha256": "e" * 64},
            headers=headers,
        )
        assert resolved.status_code == 200, resolved.text
        assert resolved.json()["status"] == AUDIT_STATUS_MISMATCH
        assert resolved.json()["invalidated"] is True

        async def _faults() -> list[dict[str, Any]]:
            return await app.state.repository.list_worker_faults(submission_id=submission_id)

        faults = anyio.run(_faults)
        assert len(faults) == 1
        assert faults[0]["worker_pubkey"] == signer.worker_pubkey
