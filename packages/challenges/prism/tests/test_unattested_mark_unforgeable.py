"""T21: unattested mark is unforgeable — miner-supplied attested:true is overridden.

When unattested execution is on, accepted prism results must carry:
  attested=false, attestation_status=unattested, execution_mode=no_phala_host
Server-side only — never trust miner/result payload fields for these keys.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import IngestionOutcome
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.unattested_execution import (
    ATTESTATION_STATUS_UNATTESTED,
    CHALLENGE_UNATTESTED_EXECUTION_ENV,
    EXECUTION_MODE_NO_PHALA_HOST,
    NO_PHALA_ENV,
    RESULT_KEY_ATTESTATION_STATUS,
    RESULT_KEY_ATTESTED,
    RESULT_KEY_EXECUTION_MODE,
    is_unattested_execution_enabled,
    mark_result_unattested,
)

WORKER_KEY = "//WorkerUnforgeableMark"
DIGEST = "sha256:" + ("22" * 32)

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

_SHARD = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def _stage_train(root: Path) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD.format(i=i) for i in range(64)), encoding="utf-8"
    )
    return data_dir


def _zip_b64() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return stream.getvalue()


def _settings(tmp_path: Path, **extra: Any) -> PrismSettings:
    kw: dict[str, Any] = dict(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'coord.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=False,
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
        constation_base_url="http://base-constation.test",
        constation_internal_token="constation-tok",
    )
    kw.update(extra)
    return PrismSettings(**kw)


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": "prism_run_manifest.v2",
        "metrics": {"token_accuracy": 0.5, "loss": 1.0, "step": 1},
        "timing": {"wall_seconds": 1.0},
    }


def _clear_unattested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (CHALLENGE_UNATTESTED_EXECUTION_ENV, "CHALLENGE_NO_PHALA", NO_PHALA_ENV):
        monkeypatch.delenv(key, raising=False)
    # PrismSettings (via DockerExecutorSettings) forbids unknown CHALLENGE_* keys.
    # Agent-challenge conftest setdefaults (e.g. REVIEW_EVIDENCE_ENCRYPTION_KEY) and
    # host/prod env must not leak into create_app / PrismSettings construction.
    known = {f"CHALLENGE_{name.upper()}" for name in PrismSettings.model_fields}
    known.add("CHALLENGE_ENV_FILE")
    for key in list(os.environ):
        if key.startswith("CHALLENGE_") and key not in known:
            monkeypatch.delenv(key, raising=False)

# --- Unit: mark_result_unattested (agent-challenge pattern) ---------------------


def test_mark_result_unattested_overrides_miner_attested_true() -> None:
    """S2: miner-supplied attested:true / verified status cannot survive the mark."""
    forged = {
        "score": 0.99,
        "attested": True,
        "attestation_status": "attested",
        "execution_mode": "phala_tee",
        "tdx_quote": "ab" * 40,
        "phala_attestation": {"quote": "x"},
        "attestation_binding": {"agent_hash": "a" * 64},
        # Worker proof must remain for prism verification path.
        "execution_proof": {"version": 1, "manifest_sha256": "ab" * 32},
    }
    out = mark_result_unattested(forged)
    assert out[RESULT_KEY_ATTESTED] is False
    assert out[RESULT_KEY_ATTESTATION_STATUS] == ATTESTATION_STATUS_UNATTESTED
    assert out[RESULT_KEY_EXECUTION_MODE] == EXECUTION_MODE_NO_PHALA_HOST
    assert out["score"] == 0.99
    # TEE-looking claim keys stripped; worker execution_proof kept.
    assert "tdx_quote" not in out
    assert "phala_attestation" not in out
    assert "attestation_binding" not in out
    assert "execution_proof" in out


def test_mark_result_unattested_hardcodes_false_even_on_empty() -> None:
    """S1: empty payload still gets the honest unattested triple."""
    out = mark_result_unattested({})
    assert out[RESULT_KEY_ATTESTED] is False
    assert out[RESULT_KEY_ATTESTATION_STATUS] == ATTESTATION_STATUS_UNATTESTED
    assert out[RESULT_KEY_EXECUTION_MODE] == EXECUTION_MODE_NO_PHALA_HOST


def test_ingestion_outcome_to_response_flag_on_overrides_miner_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2 surface: IngestionOutcome response never echoes miner attested:true when flag on."""
    _clear_unattested_env(monkeypatch)
    monkeypatch.setenv(CHALLENGE_UNATTESTED_EXECUTION_ENV, "true")
    assert is_unattested_execution_enabled() is True

    outcome = IngestionOutcome(
        status="accepted",
        work_unit_id="wu-1",
        submission_id="sub-1",
        claimed_tier=1,
        effective_tier=0,
        tier_downgraded=True,
        idempotent=False,
        finalized=True,
        submission_status="completed",
        score_written=True,
        # Even if a caller tried to smuggle via reason/attestation_mode, honesty fields
        # come only from mark_result_unattested when flag is on.
        attestation_mode="miner_rent_image_pin_evidence_v1",
    )
    payload = outcome.to_response()
    assert payload[RESULT_KEY_ATTESTED] is False
    assert payload[RESULT_KEY_ATTESTATION_STATUS] == ATTESTATION_STATUS_UNATTESTED
    assert payload[RESULT_KEY_EXECUTION_MODE] == EXECUTION_MODE_NO_PHALA_HOST
    # Miner cannot force verified via any residual field.
    assert payload.get("attested") is not True
    assert payload.get("attestation_status") != "attested"


def test_ingestion_outcome_to_response_flag_off_does_not_claim_tee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3 adjacent: flag off — no forged TEE claim; honesty fields absent or unattested."""
    _clear_unattested_env(monkeypatch)
    outcome = IngestionOutcome(
        status="accepted",
        work_unit_id="wu-2",
        submission_id="sub-2",
        claimed_tier=0,
        effective_tier=0,
        tier_downgraded=False,
        idempotent=False,
        finalized=True,
        score_written=True,
    )
    payload = outcome.to_response()
    # Must never claim verified TEE when flag is off either.
    assert payload.get("attested") is not True
    assert payload.get("attestation_status") != "attested"
    assert payload.get("attestation_status") != "verified"


# --- HTTP: miner body attested:true overridden ---------------------------------


def test_http_miner_attested_true_overridden_when_unattested_flag_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S4: POST result with miner attested:true → response attested:false / unattested."""
    _clear_unattested_env(monkeypatch)
    # NO_PHALA alias: ChallengeSettings rejects unknown CHALLENGE_* keys.
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    headers = {"Authorization": "Bearer secret"}
    signer = worker_signer_from_key(WORKER_KEY)

    with TestClient(create_app(settings)) as client:
        seed = client.post(
            "/internal/v1/bridge/submissions",
            content=_zip_b64(),
            headers={
                "Authorization": "Bearer secret",
                "X-Base-Verified-Hotkey": "hk-owner-forge",
                "X-Submission-Filename": "project.zip",
                "Content-Type": "application/octet-stream",
            },
        )
        assert seed.status_code == 200, seed.text
        sid = seed.json()["id"]
        manifest = _manifest()
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(manifest),
            unit_id=sid,
            image_digest=DIGEST,
            constation_digest=DIGEST,
        ).model_dump(mode="json")
        body = {
            "api_version": "1.0",
            "work_unit_id": sid,
            "assignment_id": sid,
            "submission_ref": "hk-owner-forge",
            "challenge_slug": settings.slug,
            "result": {
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
                # Miner forge attempt inside result payload (top-level extras
                # are rejected by ExternalResultEnvelope; result is open dict):
                "attested": True,
                "attestation_status": "attested",
                "execution_mode": "phala_tee",
            },
            "proof": proof,
        }
        resp = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("status") == "accepted", data
        # Server-stamped honesty mark — miner attested:true did not win.
        assert data[RESULT_KEY_ATTESTED] is False
        assert data[RESULT_KEY_ATTESTATION_STATUS] == ATTESTATION_STATUS_UNATTESTED
        assert data[RESULT_KEY_EXECUTION_MODE] == EXECUTION_MODE_NO_PHALA_HOST
