"""Worker-plane queue persist for challenge-owned prism_train_series.v1.

``finalize_worker_result`` does not plumb ``artifact_output_path`` /
``run_manifest_path``. Series body must travel on the forwarded
``prism_run_manifest.v2`` metrics so ``submission_curves.train_series`` is
non-null after worker-plane completion and ``GET /curve`` can return it.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.schemas import TRAIN_SERIES_V1_SCHEMA
from prism_challenge.evaluator.train_series import (
    build_train_series_v1,
    series_is_challenge_owned,
    train_series_sha256,
)
from prism_challenge.ingestion import ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.queue import PrismWorker

WORKER_KEY = "//WorkerSeriesPersist"

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
    import base64
    import io
    import zipfile

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


def _challenge_series(*, submission_id: str = "sub-wp", n: int = 4) -> dict[str, Any]:
    points = [
        {
            "i": i,
            "tokens_seen": (i + 1) * 16,
            "covered_bytes": float((i + 1) * 64),
            "train_ce_nats": float(3.0 - 0.2 * i),
            "running_bpb": float(2.5 - 0.15 * i),
            "wall_s": float(0.05 * (i + 1)),
            "grad_norm": float(0.5 + 0.1 * i),
            "clip_event": bool(i % 2),
            "nan_inf": False,
        }
        for i in range(n)
    ]
    return build_train_series_v1(
        submission_id=submission_id,
        run_id=f"prism-reexec-{submission_id}",
        points=points,
        token_budget=10_000,
        nan_inf_batches=0,
    )


def _scoreable_manifest_with_series(
    series: dict[str, Any] | None,
    *,
    marker: str = "series-wp",
) -> dict[str, Any]:
    """Plausible+scoreable v2 manifest optionally carrying embedded train series."""
    covered_bytes = 4096
    sum_nll_nats = 900.0
    baseline = math.log(50257)
    online_loss = [10.0, 6.0, 3.0, 2.0]
    metrics: dict[str, Any] = {
        "online_loss": online_loss,
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "covered_bytes": covered_bytes,
        "predicted_tokens": 96,
        "step0_loss": online_loss[0],
        "consumed_batches": len(online_loss),
        "random_init_baseline_nats": baseline,
        "marker": marker,
    }
    artifacts: dict[str, Any] = {}
    if series is not None:
        digest = train_series_sha256(series)
        metrics["train_series_schema"] = TRAIN_SERIES_V1_SCHEMA
        metrics["train_series_path"] = "prism_train_series.v1.json"
        metrics["train_series_sha256"] = digest
        metrics["train_series_points"] = len(series.get("points") or [])
        metrics["clip_events"] = int(series.get("aggregates", {}).get("clip_events") or 0)
        # Embed full body: this is what worker-plane hosts forward without artifact roots.
        metrics["train_series"] = series
        artifacts["train_series"] = "prism_train_series.v1.json"
        artifacts["train_series_sha256"] = digest
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {
            "covered_bytes": covered_bytes,
            "single_pass": True,
            "covered_bytes_cumulative": [1024, 2048, 3072, 4096],
        },
        "metrics": metrics,
        "artifacts": artifacts,
        "compute": {"devices": [], "realized": {}},
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
    }


def _proof_dict(signer: Any, unit_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(signer=signer, manifest_sha256=digest, unit_id=unit_id)
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


async def _seed(app: Any, hotkey: str = "hk-series") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    return sub.id


def _curve_row(db_path: Path, submission_id: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT online_loss, train_series FROM submission_curves WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "online_loss": json.loads(row[0]) if row[0] else None,
        "train_series": json.loads(row[1]) if row[1] else None,
    }


def test_load_train_series_for_persist_uses_embedded_without_artifact_roots(tmp_path: Path) -> None:
    """Unit: without artifact roots, challenge embed + matching sha256 yields the series."""
    # PrismWorker construction is heavy; exercise the pure loader via a bare instance shell.
    worker = object.__new__(PrismWorker)
    series = _challenge_series(submission_id="unit-embed")
    digest = train_series_sha256(series)
    metrics = {
        "train_series": series,
        "train_series_sha256": digest,
        "train_series_path": "prism_train_series.v1.json",
        "online_loss": [1.0, 0.5],
    }
    manifest = {"metrics": metrics, "artifacts": {"train_series_sha256": digest}}
    loaded = PrismWorker._load_train_series_for_persist(
        worker,
        manifest=manifest,
        metrics=metrics,
        artifact_output_path=None,
        run_manifest_path=None,
    )
    assert loaded is not None
    assert series_is_challenge_owned(loaded)
    assert train_series_sha256(loaded) == digest
    assert loaded["points"] == series["points"]


def test_load_train_series_for_persist_rejects_miner_or_digest_mismatch() -> None:
    worker = object.__new__(PrismWorker)
    series = _challenge_series(submission_id="unit-bad")
    digest = train_series_sha256(series)
    # Digest mismatch without files → None
    metrics_bad_digest = {
        "train_series": series,
        "train_series_sha256": "0" * 64,
    }
    assert (
        PrismWorker._load_train_series_for_persist(
            worker,
            manifest={"metrics": metrics_bad_digest},
            metrics=metrics_bad_digest,
            artifact_output_path=None,
            run_manifest_path=None,
        )
        is None
    )
    # Miner authority embed → None even with matching sha of miner body
    miner = dict(series)
    miner["authority"] = "miner"
    miner["miner_reported_ignored"] = False
    metrics_miner = {
        "train_series": miner,
        "train_series_sha256": train_series_sha256(miner),
    }
    assert (
        PrismWorker._load_train_series_for_persist(
            worker,
            manifest={"metrics": metrics_miner},
            metrics=metrics_miner,
            artifact_output_path=None,
            run_manifest_path=None,
        )
        is None
    )
    # Pointer-only without artifact root (legacy transport) → None
    metrics_pointer_only = {
        "train_series_path": "prism_train_series.v1.json",
        "train_series_sha256": digest,
    }
    assert (
        PrismWorker._load_train_series_for_persist(
            worker,
            manifest={"metrics": metrics_pointer_only},
            metrics=metrics_pointer_only,
            artifact_output_path=None,
            run_manifest_path=None,
        )
        is None
    )


def test_load_train_series_for_persist_prefers_side_car_when_paths_present(
    tmp_path: Path,
) -> None:
    """Local re-exec still loads side-car when artifact roots exist (pointer+sha)."""
    from prism_challenge.evaluator.train_series import write_train_series_artifact

    worker = object.__new__(PrismWorker)
    series = _challenge_series(submission_id="unit-disk")
    path, digest = write_train_series_artifact(tmp_path, series)
    assert path.is_file()
    metrics = {
        "train_series_path": path.name,
        "train_series_sha256": digest,
        # No embed: local path only
    }
    loaded = PrismWorker._load_train_series_for_persist(
        worker,
        manifest={"metrics": metrics, "artifacts": {}},
        metrics=metrics,
        artifact_output_path=str(tmp_path),
        run_manifest_path=None,
    )
    assert loaded is not None
    assert train_series_sha256(loaded) == digest


async def test_worker_plane_finalize_persists_train_series_from_embedded_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: ingest+finalize leaves train_series on submission_curves; /curve returns it."""
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    def _boom_eval(*_a: Any, **_k: Any) -> None:
        raise AssertionError("evaluator must not re-exec on worker-plane finalize")

    monkeypatch.setattr(PrismWorker, "_evaluate_within_wall_time", _boom_eval)

    submission_id = await _seed(app)
    series = _challenge_series(submission_id=submission_id, n=5)
    assert series_is_challenge_owned(series)
    # Manifest digest uses the same object that was embedded for production hashing honesty.
    manifest = _scoreable_manifest_with_series(series)
    assert "train_series" in manifest["metrics"]
    proof = _proof_dict(signer, submission_id, manifest)

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-series",
        result=_result(proof, manifest),
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert await app.state.repository.submission_status(submission_id) == "completed"

    curve = _curve_row(db_path, submission_id)
    assert curve is not None
    assert curve["online_loss"] == [10.0, 6.0, 3.0, 2.0]
    stored = curve["train_series"]
    assert stored is not None, "worker-plane finalize must persist embedded train_series"
    assert series_is_challenge_owned(stored)
    assert train_series_sha256(stored) == train_series_sha256(series)
    assert len(stored["points"]) == 5
    assert all("grad_norm" in p and "clip_event" in p for p in stored["points"])

    # Reservoir / repository agrees
    from_repo = await app.state.repository.get_submission_curve(submission_id)
    assert from_repo is not None
    assert from_repo.get("train_series") is not None
    assert series_is_challenge_owned(from_repo["train_series"])

    # Public curve route returns non-null train_series for operator time-flow.
    client = TestClient(app)
    response = client.get(f"/v1/submissions/{submission_id}/curve")
    assert response.status_code == 200
    body = response.json()
    assert body["loss_curve"]["online_loss"]
    ts = body["train_series"]
    assert ts is not None
    assert ts["schema"] == TRAIN_SERIES_V1_SCHEMA
    assert ts["authority"] == "challenge"
    assert ts["miner_reported_ignored"] is True
    assert len(ts["points"]) == 5
    assert all(p.get("grad_norm") is not None for p in ts["points"])


async def test_worker_plane_finalize_null_train_series_without_challenge_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy/no-series manifests still finalize with online_loss; train_series stays null."""
    settings = _settings(tmp_path)
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"

    monkeypatch.setattr(
        PrismWorker,
        "_evaluate_within_wall_time",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no re-exec")),
    )

    submission_id = await _seed(app)
    manifest = _scoreable_manifest_with_series(None)
    assert "train_series" not in manifest["metrics"]
    proof = _proof_dict(signer, submission_id, manifest)

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref="hk-series",
        result=_result(proof, manifest),
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True

    curve = _curve_row(db_path, submission_id)
    assert curve is not None
    assert curve["online_loss"] == [10.0, 6.0, 3.0, 2.0]
    assert curve["train_series"] is None

    client = TestClient(app)
    body = client.get(f"/v1/submissions/{submission_id}/curve").json()
    assert body["train_series"] is None
    assert body["loss_curve"]["online_loss"]
