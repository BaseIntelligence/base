"""T20: missing constation_bundle accepted ONLY when unattested flag is on.

Fail-closed (byte-identical miner_fault:missing_constation_bundle) when off.
Narrow gate — does not bypass proof / other validations.
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import _evaluate_constation_gate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.unattested_execution import (
    CHALLENGE_NO_PHALA_ENV,
    CHALLENGE_UNATTESTED_EXECUTION_ENV,
    NO_PHALA_ENV,
    is_unattested_execution_enabled,
    resolve_unattested_execution_from_environ,
)

WORKER_KEY = "//WorkerUnattestedGate"
DIGEST = "sha256:" + ("11" * 32)
MISSING = "miner_fault:missing_constation_bundle"

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
    for key in (
        CHALLENGE_UNATTESTED_EXECUTION_ENV,
        CHALLENGE_NO_PHALA_ENV,
        NO_PHALA_ENV,
    ):
        monkeypatch.delenv(key, raising=False)


# --- Env resolver (T19 parity, thin prism copy) ---------------------------------


def test_resolve_default_off_when_env_empty() -> None:
    assert resolve_unattested_execution_from_environ({}) is False
    assert is_unattested_execution_enabled(environ={}) is False


def test_resolve_canonical_true() -> None:
    env = {CHALLENGE_UNATTESTED_EXECUTION_ENV: "true"}
    assert resolve_unattested_execution_from_environ(env) is True
    assert is_unattested_execution_enabled(environ=env) is True


def test_resolve_canonical_wins_over_no_phala_false() -> None:
    env = {
        CHALLENGE_UNATTESTED_EXECUTION_ENV: "1",
        NO_PHALA_ENV: "false",
    }
    assert resolve_unattested_execution_from_environ(env) is True


def test_resolve_no_phala_alias_true() -> None:
    assert resolve_unattested_execution_from_environ({NO_PHALA_ENV: "yes"}) is True
    assert resolve_unattested_execution_from_environ({CHALLENGE_NO_PHALA_ENV: "on"}) is True


def test_resolve_canonical_false_explicit() -> None:
    assert (
        resolve_unattested_execution_from_environ(
            {CHALLENGE_UNATTESTED_EXECUTION_ENV: "false"}
        )
        is False
    )


# --- Pure gate unit (S1 / S2) ---------------------------------------------------


def test_gate_missing_bundle_flag_off_rejects_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1 / S4: flag off → same reason string as T18 fail-closed."""
    _clear_unattested_env(monkeypatch)
    gate = _evaluate_constation_gate(
        bundle=None,
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault=None,
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.constation_ok is False
    assert gate.reason == MISSING
    assert gate.retryable is False


def test_gate_missing_bundle_flag_on_admits_without_constation_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: flag on → admit scoring path; constation_ok stays False (no elevation claim)."""
    _clear_unattested_env(monkeypatch)
    monkeypatch.setenv(CHALLENGE_UNATTESTED_EXECUTION_ENV, "true")
    gate = _evaluate_constation_gate(
        bundle=None,
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault=None,
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is True
    assert gate.constation_ok is False
    assert gate.reason != MISSING
    assert gate.retryable is False


def test_gate_missing_bundle_flag_explicit_false_still_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_unattested_env(monkeypatch)
    monkeypatch.setenv(CHALLENGE_UNATTESTED_EXECUTION_ENV, "false")
    gate = _evaluate_constation_gate(
        bundle=None,
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault=None,
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.reason == MISSING


def test_gate_infra_fault_not_bypassed_by_unattested_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narrow gate: unattested does not swallow infra_fault path."""
    _clear_unattested_env(monkeypatch)
    monkeypatch.setenv(CHALLENGE_UNATTESTED_EXECUTION_ENV, "true")
    gate = _evaluate_constation_gate(
        bundle=None,
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault="constation_unavailable",
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.retryable is True
    assert gate.reason is not None
    assert gate.reason.startswith("infra_fault:")


# --- HTTP surface (S1 OFF / S2 ON) ----------------------------------------------


def test_http_missing_bundle_flag_off_422_missing_constation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1 surface: POST without bundle → 422 miner_fault:missing_constation_bundle."""
    _clear_unattested_env(monkeypatch)
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
                "X-Base-Verified-Hotkey": "hk-owner",
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
            "submission_ref": "hk-owner",
            "challenge_slug": settings.slug,
            "result": {
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
            },
            "proof": proof,
        }
        resp = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert isinstance(detail, dict)
        assert detail.get("code") == MISSING

        conn = sqlite3.connect(tmp_path / "coord.sqlite3")
        try:
            score = conn.execute(
                "SELECT final_score FROM scores WHERE submission_id=?", (sid,)
            ).fetchone()
        finally:
            conn.close()
        assert score is None


def test_http_missing_bundle_flag_on_does_not_422_missing_constation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 surface: flag ON → not 422 missing_constation_bundle; may accept/score."""
    _clear_unattested_env(monkeypatch)
    # Use NO_PHALA alias: CHALLENGE_* unknown keys are rejected by ChallengeSettings.
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
                "X-Base-Verified-Hotkey": "hk-owner",
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
            "submission_ref": "hk-owner",
            "challenge_slug": settings.slug,
            "result": {
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
            },
            "proof": proof,
        }
        resp = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        # Must NOT be the missing-bundle 422.
        if resp.status_code == 422:
            detail = resp.json().get("detail")
            code = detail.get("code") if isinstance(detail, dict) else None
            assert code != MISSING, resp.text
            assert "missing_constation" not in str(code), resp.text
        else:
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data.get("status") == "accepted", data
            # Unattested path must not claim constation elevation.
            assert data.get("effective_tier", 0) == 0 or data.get("tier_downgraded") is True


def test_http_flag_on_still_rejects_missing_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S3 narrow: unattested does not bypass proof validation."""
    _clear_unattested_env(monkeypatch)
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    headers = {"Authorization": "Bearer secret"}

    with TestClient(create_app(settings)) as client:
        seed = client.post(
            "/internal/v1/bridge/submissions",
            content=_zip_b64(),
            headers={
                "Authorization": "Bearer secret",
                "X-Base-Verified-Hotkey": "hk-owner",
                "X-Submission-Filename": "project.zip",
                "Content-Type": "application/octet-stream",
            },
        )
        assert seed.status_code == 200, seed.text
        sid = seed.json()["id"]
        # Legacy body without proof / envelope — must still fail validation.
        body = {
            "work_unit_id": sid,
            "submission_ref": "hk-owner",
            "result": {"executed": 1, MANIFEST_PAYLOAD_KEY: _manifest()},
        }
        resp = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        assert resp.status_code == 422, resp.text
        detail = resp.json().get("detail")
        code = detail.get("code") if isinstance(detail, dict) else str(detail)
        assert code != MISSING
        assert "missing_constation" not in str(code)
