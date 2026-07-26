"""Wave 4 todos 20–23: attestation_mode, effective_tier, fail-closed, break-glass."""

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
from prism_challenge.breakglass import (
    BreakGlassAuditLog,
    BreakGlassRequest,
    evaluate_break_glass,
)
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.constation import (
    CheckOutcome,
    ConstationBundle,
    ConstationFailReason,
    constation_ok,
)
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    ATTESTATION_MODE_V1,
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    ExecutionProof,
    ProviderInfo,
    WorkerSignature,
    attach_attestation_mode,
    attestation_mode_of,
    build_execution_proof,
    compute_manifest_sha256,
    elevation_image_digest,
    image_digest_from_env,
    normalize_attestation_mode,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerConstationGate"
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


def _code_bundle() -> str:
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


def _constation_bundle(*, digest: str = DIGEST) -> ConstationBundle:
    man = {"src/prism_recipe/harness.py": "a" * 64}
    return ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest=digest,
        work_unit_id="wu",
        miner_hotkey="hk",
        pod_id="pod-1",
        nonce="nonce-1",
        signed_attestation={"sig": "ok"},
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


def _fail_manifest_checkers():
    def allow(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def nonce(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def sig(_s: object) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    return allow, nonce, sig


def _tier1_proof(signer, unit_id: str, manifest: dict[str, Any], *, image_digest: str):
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=unit_id,
        image_digest=image_digest,
        constation_digest=image_digest,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
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


async def _make_app(settings: PrismSettings):
    app = create_app(settings)
    await app.state.database.init()
    return app


async def _seed(app, hotkey: str = "hk-owner") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_code_bundle(), filename="project.zip")
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


# --- todo 20 ------------------------------------------------------------------------------------


def test_attestation_mode_is_miner_rent_image_pin_evidence_v1() -> None:
    att = attach_attestation_mode(None)
    assert att["attestation_mode"] == ATTESTATION_MODE_V1
    assert normalize_attestation_mode(ATTESTATION_MODE_V1) == ATTESTATION_MODE_V1
    with pytest.raises(ValueError, match="forbidden"):
        normalize_attestation_mode("lium_attested")
    with pytest.raises(ValueError, match="forbidden"):
        normalize_attestation_mode("tee_attested")


def test_env_digest_is_telemetry_not_elevation() -> None:
    assert elevation_image_digest(constation_digest=DIGEST, env_digest=OTHER) == DIGEST
    assert elevation_image_digest(constation_digest=None, env_digest=OTHER) is None
    # image_digest_from_env still reads env but docstring marks telemetry-only
    assert image_digest_from_env({"PRISM_IMAGE_DIGEST": OTHER}) == OTHER


def test_build_proof_always_sets_attestation_mode() -> None:
    signer = worker_signer_from_key(WORKER_KEY)
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256="c" * 64,
        unit_id="u1",
        image_digest=PINNED,
        provider=ProviderInfo(name="lium", pod_id="p"),
        tier=1,  # type: ignore[arg-type]
    )
    assert attestation_mode_of(proof) == ATTESTATION_MODE_V1


def test_selfreport_digest_match_without_constation_is_tier0() -> None:
    """Correct PRISM_IMAGE_DIGEST alone cannot reach tier 1 (todo 20 failure path)."""
    proof = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256="c" * 64,
        image_digest=PINNED,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
        attestation=attach_attestation_mode(None),
    )
    assert effective_tier(proof, pinned_image_digest=PINNED) == 0
    assert effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=False) == 0
    assert effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=True) == 1


# --- todo 21 ------------------------------------------------------------------------------------


def test_tier1_only_when_constation_ok_true() -> None:
    proof = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256="c" * 64,
        image_digest=PINNED,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
    )
    assert effective_tier(proof, constation_ok_result=True) == 1
    assert effective_tier(proof, constation_ok_result=False) == 0
    assert effective_tier(proof, constation_ok_result=None) == 0


def test_claimed_tier2_always_zero_even_with_constation() -> None:
    proof = ExecutionProof(
        version=1,
        tier=2,
        manifest_sha256="c" * 64,
        image_digest=PINNED,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
        attestation={"tdx_quote_b64": "x", "gpu_eat_jwt": "y"},
    )
    assert effective_tier(proof, constation_ok_result=True) == 0


def test_constation_result_object_drives_tier() -> None:
    proof = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256="c" * 64,
        image_digest=DIGEST,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
    )
    allow, nonce, sig = _ok_checkers()
    ok = constation_ok(
        _constation_bundle(),
        check_allowlist=allow,
        check_nonce=nonce,
        verify_signature=sig,
    )
    assert ok.ok is True
    assert effective_tier(proof, constation_ok_result=ok) == 1

    bad_bundle = _constation_bundle()
    # force sealed mismatch
    from dataclasses import replace

    bad = replace(
        bad_bundle,
        reported_sealed_manifest_hashes={"src/prism_recipe/harness.py": "f" * 64},
    )
    fail = constation_ok(bad, check_allowlist=allow, check_nonce=nonce, verify_signature=sig)
    assert fail.ok is False
    assert fail.reason == ConstationFailReason.SEALED_MANIFEST_MISMATCH
    assert effective_tier(proof, constation_ok_result=fail) == 0


# --- todo 22 ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_bundle_writes_score_with_attestation_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app)
    manifest = _manifest("ok")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=DIGEST)
    allow, nonce, sig = _ok_checkers()

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
        pinned_image_digest=DIGEST,
        constation_bundle=_constation_bundle(digest=DIGEST),
        check_allowlist=allow,
        check_nonce=nonce,
        verify_constation_signature=sig,
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert outcome.score_written is True
    assert outcome.effective_tier == 1
    assert outcome.attestation_mode == ATTESTATION_MODE_V1
    score = _final_score(db_path, submission_id)
    assert score is not None and score > 0.0


@pytest.mark.asyncio
async def test_no_bundle_writes_no_score_miner_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app)
    manifest = _manifest("nobundle")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=PINNED)

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


@pytest.mark.asyncio
async def test_manifest_mismatch_miner_fault_no_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app, hotkey="hk-mm")
    manifest = _manifest("mm")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=DIGEST)
    allow, nonce, sig = _ok_checkers()
    from dataclasses import replace

    bad = replace(
        _constation_bundle(digest=DIGEST),
        reported_sealed_manifest_hashes={"src/prism_recipe/harness.py": "f" * 64},
    )
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-mm",
        result=_result(proof, manifest),
        constation_bundle=bad,
        check_allowlist=allow,
        check_nonce=nonce,
        verify_constation_signature=sig,
    )
    assert outcome.status == "rejected"
    assert outcome.score_written is False
    assert outcome.reason == "miner_fault:manifest_mismatch"
    assert _final_score(db_path, submission_id) is None


@pytest.mark.asyncio
async def test_infra_fault_constation_unavailable_no_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app, hotkey="hk-infra")
    manifest = _manifest("infra")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=DIGEST)

    # Exhaust retries → no score
    from prism_challenge.ingestion import ResultIngestionError

    with pytest.raises(ResultIngestionError) as ei:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref="hk-infra",
            result=_result(proof, manifest),
            constation_infra_fault="constation_unavailable",
            constation_attempt=1,
            max_constation_attempts=3,
        )
    assert "infra_fault" in ei.value.reason

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-infra",
        result=_result(proof, manifest),
        constation_infra_fault="constation_unavailable",
        constation_attempt=3,
        max_constation_attempts=3,
    )
    assert outcome.status == "rejected"
    assert outcome.score_written is False
    assert outcome.reason is not None and outcome.reason.startswith("infra_fault:")
    assert _final_score(db_path, submission_id) is None


@pytest.mark.asyncio
async def test_revoked_digest_no_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app, hotkey="hk-rev")
    manifest = _manifest("rev")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=DIGEST)

    def allow(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=False, reason="revoked")

    def nonce(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def sig(_s: object) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-rev",
        result=_result(proof, manifest),
        constation_bundle=_constation_bundle(digest=DIGEST),
        check_allowlist=allow,
        check_nonce=nonce,
        verify_constation_signature=sig,
    )
    assert outcome.status == "rejected"
    assert outcome.reason == "miner_fault:revoked_digest"
    assert _final_score(db_path, submission_id) is None


# --- todo 23 ------------------------------------------------------------------------------------


def test_breakglass_admits_infra_fault_only() -> None:
    log = BreakGlassAuditLog()
    req = BreakGlassRequest(
        operator_id="ops-alice",
        reason="constation outage window",
        work_unit_id="wu-1",
        fault_code="infra_fault:constation_unavailable",
    )
    ok = evaluate_break_glass(req, fault_reason="infra_fault:constation_unavailable", audit_log=log)
    assert ok.admitted is True
    assert log.entries and log.entries[0]["admitted"] is True
    assert log.entries[0]["operator_id"] == "ops-alice"

    log2 = BreakGlassAuditLog()
    bad = evaluate_break_glass(req, fault_reason="miner_fault:replayed_nonce", audit_log=log2)
    assert bad.admitted is False
    assert bad.reason == "breakglass_refused_miner_fault"
    assert log2.entries and log2.entries[0]["admitted"] is False


@pytest.mark.asyncio
async def test_breakglass_admits_infra_run_and_writes_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app, hotkey="hk-bg")
    manifest = _manifest("bg")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=DIGEST)
    log = BreakGlassAuditLog()
    bg = BreakGlassRequest(
        operator_id="ops-bob",
        reason="confirmed BASE outage",
        work_unit_id=submission_id,
        fault_code="infra_fault:constation_unavailable",
    )
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-bg",
        result=_result(proof, manifest),
        constation_infra_fault="constation_unavailable",
        constation_attempt=3,
        max_constation_attempts=3,
        break_glass=bg,
        break_glass_audit_log=log,
    )
    assert outcome.status == "accepted"
    assert outcome.break_glass_admitted is True
    assert outcome.effective_tier == 0  # no elevation without real constation
    assert outcome.score_written is True
    assert _final_score(db_path, submission_id) is not None
    assert any(e.get("admitted") for e in log.entries)


@pytest.mark.asyncio
async def test_breakglass_refuses_miner_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    app = await _make_app(_settings(tmp_path))
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    submission_id = await _seed(app, hotkey="hk-bgm")
    manifest = _manifest("bgm")
    proof = _tier1_proof(signer, submission_id, manifest, image_digest=DIGEST)
    log = BreakGlassAuditLog()
    bg = BreakGlassRequest(
        operator_id="ops-eve",
        reason="please admit anyway",
        work_unit_id=submission_id,
        fault_code="miner_fault:replayed_nonce",
    )
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-bgm",
        result=_result(proof, manifest),
        # missing bundle = miner_fault
        constation_bundle=None,
        break_glass=bg,
        break_glass_audit_log=log,
    )
    assert outcome.status == "rejected"
    assert outcome.break_glass_admitted is False
    assert outcome.score_written is False
    assert _final_score(db_path, submission_id) is None
    assert any(e.get("detail") == "miner_fault_override_refused" for e in log.entries)
