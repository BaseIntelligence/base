"""Production HTTP constation path (S1/S3) — no legacy auto-inject (module name gate)."""

from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.constation import ConstationBundle, constation_bundle_to_dict
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerProdConstation"
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
        "metrics": {
            "token_accuracy": 0.5,
            "loss": 1.0,
            "step": 1,
        },
        "timing": {"wall_seconds": 1.0},
    }


def _bundle_for(submission_id: str) -> dict[str, Any]:
    man = {"route-test-harness.py": "a" * 64}
    return constation_bundle_to_dict(
        ConstationBundle(
            commit_sha="a" * 40,
            tree_sha="b" * 40,
            variant="cuda",
            digest=DIGEST,
            work_unit_id=submission_id,
            miner_hotkey="hk-owner",
            pod_id="pod-1",
            nonce="nonce-prod-1",
            signed_attestation={"sig": "fixture"},
            expected_sealed_manifest_hashes=dict(man),
            reported_sealed_manifest_hashes=dict(man),
            lium_declared_digest=DIGEST,
            constation_gap_budget_seconds=30.0,
            constation_observed_max_gap_seconds=1.0,
        )
    )


def _ok_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "reason": "ok"})

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _patch_http_checkers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route BaseHttpConstationClient to always-ok MockTransport (S1 checkers)."""
    from prism_challenge import constation_checkers as mod

    orig = mod.BaseHttpConstationClient.__init__

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs = dict(kwargs)
        kwargs.setdefault("transport", _ok_transport())
        orig(self, *args, **kwargs)

    monkeypatch.setattr(mod.BaseHttpConstationClient, "__init__", _init)


def test_s3_missing_bundle_fail_closed_via_http(tmp_path: Path, monkeypatch) -> None:
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
        code = detail.get("code") if isinstance(detail, dict) else None
        assert code in {
            "miner_fault:missing_constation_bundle",
            "constation_rejected",
        } or (isinstance(detail, dict) and "missing_constation" in str(detail.get("code", ""))), (
            detail
        )

        db_path = tmp_path / "coord.sqlite3"
        conn = sqlite3.connect(db_path)
        try:
            score = conn.execute(
                "SELECT final_score FROM scores WHERE submission_id=?", (sid,)
            ).fetchone()
        finally:
            conn.close()
        assert score is None


def test_s1_honest_bundle_scores_via_http(tmp_path: Path, monkeypatch) -> None:
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
            tier=1,  # type: ignore[arg-type]
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
                "constation_bundle": _bundle_for(sid),
            },
            "proof": proof,
        }
        resp = client.post("/internal/v1/work_units/result", json=body, headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "accepted", data
        assert data.get("score_written") is True, data
        assert data.get("effective_tier") == 1, data
        assert data.get("attestation_mode"), data
