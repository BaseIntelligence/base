"""Verify-only plausibility gate for worker-reported results (architecture.md 3.5).

Covers VAL-PRISM-009 (implausible manifests are rejected with a distinguishable, plausibility-
specific reason and are never scored) and VAL-PRISM-010 (a well-formed manifest passes the gate
unchanged and finalizes to the exact same score as the legacy finalization path). Offline, no GPU:
the ingestion integration drives the CPU re-exec seam the other coordination tests use and proofs
are signed with real sr25519 worker keys.
"""

from __future__ import annotations

import base64
import copy
import io
import math
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.anti_cheat import evaluate_anti_cheat
from prism_challenge.evaluator.components import architecture_name
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.plausibility import (
    MANIFEST_SCHEMA_VERSION,
    REASON_LOSS_AT_BASELINE,
    REASON_LOSS_INCONSISTENT,
    REASON_METRICS_MALFORMED,
    REASON_SCHEMA_VERSION,
    REASON_STEP0_ANOMALY,
    REASON_WALLCLOCK_BUDGET,
    PlausibilityError,
    check_manifest_plausibility,
)
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)

WORKER_KEY = "//WorkerPlausible"
VOCAB = 50257
BASELINE_NATS = math.log(VOCAB)
BUDGET_SECONDS = 1800.0


# --- manifest fixtures ---------------------------------------------------------------------------


def _plausible_manifest() -> dict[str, Any]:
    """A well-formed v2 manifest: decreasing loss starting near baseline, wall-clock in budget."""

    online_loss = [10.4, 8.1, 6.0, 4.2, 3.1, 2.6]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run": {"seed": 1234, "forced_init": True, "stopped_reason": "token_budget"},
        "compute": {"schema": "prism_compute.v1", "wall_clock_seconds": 600.0},
        "metrics": {
            "online_loss": online_loss,
            "step0_loss": online_loss[0],
            "random_init_baseline_nats": BASELINE_NATS,
            "prequential_bpb": 1.23,
            "bits_per_byte": 1.23,
            "covered_bytes": 4096,
            "sum_neg_log_likelihood_nats": 3000.0,
        },
        "score": {
            "schema": "prism_score.v2",
            "prequential_bpb": 1.23,
            "bits_per_byte": 1.23,
            "final_score": 1.0 / (1.0 + 1.23),
        },
        "anti_cheat": {"step0_anomaly": False, "no_learning": False, "zero_forward": False},
    }


def test_plausible_manifest_passes() -> None:
    # No raise for a well-formed manifest.
    check_manifest_plausibility(_plausible_manifest(), wall_clock_budget_seconds=BUDGET_SECONDS)


def test_plausible_manifest_is_not_mutated() -> None:
    manifest = _plausible_manifest()
    before = copy.deepcopy(manifest)
    check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert manifest == before


def test_minimal_metrics_only_manifest_passes() -> None:
    # A manifest with the schema + a metrics dict but no trajectory/compute fields cannot be shown
    # implausible, so it must pass (keeps the existing ingestion fixtures accepted).
    minimal = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metrics": {"prequential_bpb": 1.23},
    }
    check_manifest_plausibility(minimal, wall_clock_budget_seconds=BUDGET_SECONDS)


def test_rejects_bad_schema_version() -> None:
    manifest = _plausible_manifest()
    manifest["schema_version"] = "prism_run_manifest.v1"
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_SCHEMA_VERSION


def test_rejects_missing_or_malformed_metrics() -> None:
    for bad in (None, [], "metrics", 3):
        manifest = _plausible_manifest()
        if bad is None:
            manifest.pop("metrics")
        else:
            manifest["metrics"] = bad
        with pytest.raises(PlausibilityError) as exc:
            check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
        assert exc.value.reason == REASON_METRICS_MALFORMED


def test_rejects_final_loss_at_random_baseline() -> None:
    manifest = _plausible_manifest()
    # A trajectory that never leaves the from-scratch baseline (~ln(vocab)): no learning happened.
    manifest["metrics"]["online_loss"] = [BASELINE_NATS, BASELINE_NATS, BASELINE_NATS]
    manifest["metrics"]["step0_loss"] = BASELINE_NATS
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_LOSS_AT_BASELINE


def test_rejects_final_loss_above_random_baseline() -> None:
    manifest = _plausible_manifest()
    manifest["metrics"]["online_loss"] = [10.0, 11.0, 12.0]
    manifest["metrics"]["step0_loss"] = 10.0
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_LOSS_AT_BASELINE


def test_rejects_no_learning_anti_cheat_flag() -> None:
    manifest = _plausible_manifest()
    manifest["anti_cheat"]["no_learning"] = True
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_LOSS_AT_BASELINE


def test_rejects_step0_anomaly_by_value() -> None:
    manifest = _plausible_manifest()
    # An initial loss impossibly far below the from-scratch baseline (smuggled-weights signal).
    manifest["metrics"]["online_loss"] = [0.9, 0.6, 0.4]
    manifest["metrics"]["step0_loss"] = 0.9
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_STEP0_ANOMALY


def test_rejects_step0_anomaly_flag() -> None:
    manifest = _plausible_manifest()
    manifest["anti_cheat"]["step0_anomaly"] = True
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_STEP0_ANOMALY


def test_rejects_wallclock_grossly_over_budget() -> None:
    manifest = _plausible_manifest()
    manifest["compute"]["wall_clock_seconds"] = BUDGET_SECONDS * 50
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_WALLCLOCK_BUDGET


def test_rejects_negative_wallclock() -> None:
    manifest = _plausible_manifest()
    manifest["compute"]["wall_clock_seconds"] = -1.0
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_WALLCLOCK_BUDGET


def test_wallclock_check_skipped_without_budget() -> None:
    manifest = _plausible_manifest()
    manifest["compute"]["wall_clock_seconds"] = BUDGET_SECONDS * 50
    # With no budget known, an over-budget wall-clock cannot be judged implausible.
    check_manifest_plausibility(manifest, wall_clock_budget_seconds=None)


def test_rejects_log_score_inconsistency() -> None:
    manifest = _plausible_manifest()
    # The score block's bpb disagrees with the logged metrics bpb: fabricated/self-inconsistent.
    manifest["score"]["prequential_bpb"] = 0.01
    manifest["score"]["bits_per_byte"] = 0.01
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_LOSS_INCONSISTENT


def test_rejects_step0_vs_online_loss_inconsistency() -> None:
    manifest = _plausible_manifest()
    manifest["metrics"]["step0_loss"] = manifest["metrics"]["online_loss"][0] + 3.0
    with pytest.raises(PlausibilityError) as exc:
        check_manifest_plausibility(manifest, wall_clock_budget_seconds=BUDGET_SECONDS)
    assert exc.value.reason == REASON_LOSS_INCONSISTENT


# --- ingestion integration (VAL-PRISM-009 / VAL-PRISM-010) ---------------------------------------

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


def _proof_dict(signer, unit_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
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


async def test_ingestion_rejects_implausible_manifests_without_scoring(tmp_path, monkeypatch):
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

    def _bad_schema() -> dict[str, Any]:
        m = _plausible_manifest()
        m["schema_version"] = "prism_run_manifest.v1"
        return m

    def _no_metrics() -> dict[str, Any]:
        m = _plausible_manifest()
        m.pop("metrics")
        return m

    def _at_baseline() -> dict[str, Any]:
        m = _plausible_manifest()
        m["metrics"]["online_loss"] = [BASELINE_NATS, BASELINE_NATS, BASELINE_NATS]
        m["metrics"]["step0_loss"] = BASELINE_NATS
        return m

    def _step0() -> dict[str, Any]:
        m = _plausible_manifest()
        m["metrics"]["online_loss"] = [0.9, 0.6, 0.4]
        m["metrics"]["step0_loss"] = 0.9
        return m

    def _wallclock() -> dict[str, Any]:
        m = _plausible_manifest()
        m["compute"]["wall_clock_seconds"] = BUDGET_SECONDS * 50
        return m

    cases = {
        REASON_SCHEMA_VERSION: _bad_schema,
        REASON_METRICS_MALFORMED: _no_metrics,
        REASON_LOSS_AT_BASELINE: _at_baseline,
        REASON_STEP0_ANOMALY: _step0,
        REASON_WALLCLOCK_BUDGET: _wallclock,
    }

    for expected_reason, build in cases.items():
        submission_id = await _seed(app)
        manifest = build()
        proof = _proof_dict(signer, submission_id, manifest)
        with pytest.raises(PlausibilityError) as exc:
            await ingest_work_unit_result(
                worker=app.state.worker,
                work_unit_id=submission_id,
                submission_ref="hk-owner",
                result=_result(proof, manifest),
            )
        assert exc.value.reason == expected_reason
        # Never scored; the unit stays eligible for retry (still pending).
        assert _score(db_path, submission_id) is None
        assert await app.state.repository.submission_status(submission_id) == "pending"


async def test_ingestion_accepts_plausible_manifest_identically_to_legacy(tmp_path):
    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    app = await _make_app(settings)
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "coord.sqlite3"
    worker = app.state.worker

    # Path A: finalize a plausible worker result through the plausibility-gated ingestion path.
    gated_id = await _seed(app)
    manifest = _plausible_manifest()
    before = copy.deepcopy(manifest)
    proof = _proof_dict(signer, gated_id, manifest)
    outcome = await ingest_work_unit_result(
        worker=worker,
        work_unit_id=gated_id,
        submission_ref="hk-owner",
        result=_result(proof, manifest),
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert await app.state.repository.submission_status(gated_id) == "completed"
    gated_score = _score(db_path, gated_id)
    assert gated_score is not None
    # The forwarded manifest / score inputs are passed through UNCHANGED (plausibility only READS).
    assert manifest == before

    # Path B: feed the SAME manifest to the legacy finalization tail directly (VAL-PRISM-010).
    legacy_id = await _seed(app)
    submission = await worker.repository.claim_submission(legacy_id)
    assert submission is not None
    snapshot = worker._snapshot_from_submission(str(submission["code"]), "project.zip", {})
    review = worker._component_review(snapshot)
    code_for_eval = worker._entrypoint_code(snapshot, review.components.entrypoint)
    anti = evaluate_anti_cheat(
        code_for_eval,
        await worker.repository.previous_codes(legacy_id),
        allowed_import_roots=worker._local_import_roots(snapshot),
    )
    await worker._finalize_container_score(
        submission_id=legacy_id,
        arch_hash=review.fingerprints.family_hash,
        anti=anti,
        manifest=copy.deepcopy(manifest),
        hotkey="hk-owner",
        fingerprints=review.fingerprints,
        name=architecture_name(review.components),
    )
    legacy_score = _score(db_path, legacy_id)
    assert legacy_score is not None

    assert gated_score == pytest.approx(legacy_score)
