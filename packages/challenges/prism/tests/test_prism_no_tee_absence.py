"""Prism NO TEE residual: package absence + score finalize without tee (VAL-NOTEE-001..008).

Extended by todo 24: image-attestation path, attestation_mode never TEE, tier cannot exceed 1.
Never delete, skip, or xfail this file.
"""

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
from prism_challenge.constation import CheckOutcome, ConstationBundle
from prism_challenge.ingestion import ResultIngestionError, ingest_work_unit_result
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
    normalize_attestation_mode,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerNoTee"
PINNED = "sha256:" + ("ab" * 32)
OTHER = "sha256:" + ("cd" * 32)
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
    # Claimed tier 2 never elevates, even with constation_ok.
    assert effective_tier(proof_t2, pinned_image_digest=PINNED, constation_ok_result=True) == 0
    # Tier 1 requires constation_ok (not pin match alone).
    assert effective_tier(proof_t1, pinned_image_digest=PINNED) == 0
    assert effective_tier(proof_t1, pinned_image_digest=PINNED, constation_ok_result=True) == 1
    assert effective_tier(proof_t1, pinned_image_digest=OTHER, constation_ok_result=False) == 0


def test_attestation_mode_never_implies_tee() -> None:
    """Todo 24: attestation_mode is miner_rent_image_pin_evidence_v1; TEE labels forbidden."""
    assert ATTESTATION_MODE_V1 == "miner_rent_image_pin_evidence_v1"
    assert "tee" not in ATTESTATION_MODE_V1.lower()
    assert "tdx" not in ATTESTATION_MODE_V1.lower()
    assert normalize_attestation_mode(ATTESTATION_MODE_V1) == ATTESTATION_MODE_V1
    for bad in ("lium_attested", "tee", "tee_attested", "tdx", "sev", "cvm"):
        with pytest.raises(ValueError):
            normalize_attestation_mode(bad)
    att = attach_attestation_mode({"tdx_quote_b64": "x", "gpu_eat_jwt": "y"})
    assert att["attestation_mode"] == ATTESTATION_MODE_V1


def test_image_attestation_path_exists_and_functions() -> None:
    """Todo 24: image-attestation path (constation_ok + attestation_mode) is present."""
    from prism_challenge.constation import constation_ok
    from prism_challenge.queue import LIUM_EXECUTION_BACKEND, is_execution_backend_supported

    man = {"h.py": "a" * 64}
    bundle = ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest=DIGEST,
        work_unit_id="wu",
        miner_hotkey="hk",
        pod_id="pod-1",
        nonce="n",
        signed_attestation={"s": "1"},
        expected_sealed_manifest_hashes=dict(man),
        reported_sealed_manifest_hashes=dict(man),
        lium_declared_digest=DIGEST,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
    )

    def ok(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    result = constation_ok(
        bundle, check_allowlist=ok, check_nonce=ok, verify_signature=lambda _s: ok()
    )
    assert result.ok is True
    assert is_execution_backend_supported(LIUM_EXECUTION_BACKEND, constation_bundle=bundle)
    signer = worker_signer_from_key(WORKER_KEY)
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256="c" * 64,
        unit_id="u",
        image_digest=DIGEST,
        constation_digest=DIGEST,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        tier=1,  # type: ignore[arg-type]
    )
    assert attestation_mode_of(proof) == ATTESTATION_MODE_V1
    assert effective_tier(proof, constation_ok_result=result) == 1


def test_effective_tier_cannot_exceed_one_by_any_route() -> None:
    """Todo 24: no route yields effective tier > 1."""
    for claimed in (0, 1, 2):
        proof = ExecutionProof(
            version=1,
            tier=claimed,  # type: ignore[arg-type]
            manifest_sha256="c" * 64,
            image_digest=PINNED if claimed >= 1 else None,
            provider=ProviderInfo(name="lium", pod_id="pod-1") if claimed >= 1 else None,
            worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
            attestation={
                "version": 1,
                "provider": "local_fixture",
                "evidence_type": "prism.tee.v1",
                "tdx_quote_b64": "Q",
                "gpu_eat_jwt": "J",
                "attestation_mode": ATTESTATION_MODE_V1,
            }
            if claimed == 2
            else attach_attestation_mode(None),
        )
        for cok in (True, False, None):
            tier = effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=cok)
            assert tier <= 1, f"claimed={claimed} cok={cok} -> {tier}"


def _code_bundle() -> str:
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
        constation_digest=image_digest,
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


def _ok_checkers():
    def allow(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def nonce(**_k: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def sig(_s: object) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    return allow, nonce, sig


def _constation_bundle(digest: str = DIGEST) -> ConstationBundle:
    man = {"h.py": "a" * 64}
    return ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest=digest,
        work_unit_id="wu",
        miner_hotkey="hk",
        pod_id="pod-1",
        nonce="n",
        signed_attestation={"s": "1"},
        expected_sealed_manifest_hashes=dict(man),
        reported_sealed_manifest_hashes=dict(man),
        lium_declared_digest=digest,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
    )


async def _make_app(settings: PrismSettings):
    app = create_app(settings)
    await app.state.database.init()
    return app


async def _seed(app, hotkey: str = "hk-notee") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_code_bundle(), filename="project.zip")
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
    """Worker-plane finalize succeeds with no tee package; requires constation (P1)."""
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    submission_id = await _seed(app)
    manifest = _manifest()
    proof = _proof_payload(signer, submission_id, manifest, tier=1, image_digest=DIGEST)
    allow, nonce, sig = _ok_checkers()
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-notee",
        result=_result(proof, manifest),
        pinned_image_digest=DIGEST,
        constation_bundle=_constation_bundle(DIGEST),
        check_allowlist=allow,
        check_nonce=nonce,
        verify_constation_signature=sig,
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert outcome.effective_tier == 1
    assert outcome.claimed_tier == 1
    assert outcome.tier_downgraded is False
    assert outcome.reason is None
    assert outcome.attestation_mode == ATTESTATION_MODE_V1
    assert _score(tmp_path / "notee.sqlite3", submission_id) is not None


@pytest.mark.asyncio
async def test_pin_mismatch_without_constation_writes_no_score(tmp_path: Path) -> None:
    """NEW contract (todo 22): missing constation ⇒ no score (was: still finalizes)."""
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
        constation_bundle=None,
    )
    assert outcome.status == "rejected"
    assert outcome.finalized is False
    assert outcome.score_written is False
    assert outcome.effective_tier == 0
    assert _score(tmp_path / "notee.sqlite3", submission_id) is None


@pytest.mark.asyncio
async def test_ingestion_never_raises_tee_required(tmp_path: Path) -> None:
    """Attestation-claiming tier-2 proof never raises tee_required (max effective=0)."""
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
        image_digest=DIGEST,
        attestation=attestation,
    )
    allow, nonce, sig = _ok_checkers()
    try:
        outcome = await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref="hk-t2",
            result=_result(proof, manifest),
            pinned_image_digest=DIGEST,
            constation_bundle=_constation_bundle(DIGEST),
            check_allowlist=allow,
            check_nonce=nonce,
            verify_constation_signature=sig,
        )
    except ResultIngestionError as exc:
        assert exc.reason != "tee_required"
        raise
    # With constation, score may write but effective tier stays 0 for claimed tier 2.
    assert outcome.effective_tier == 0
    assert outcome.tier_downgraded is True
    assert outcome.attestation_mode == ATTESTATION_MODE_V1
