"""Probabilistic audit scheduler, audit-unit creation, and audit-mismatch invalidation.

Covers the mission audit assertions (architecture.md 3.4/3.5):

* VAL-PRISM-011 - the scheduler samples finalized results at the configured per-tier rates, is
  reproducible under a deterministic seed, and a 0.0 rate samples exactly nothing.
* VAL-PRISM-012 - a sampled accepted result creates a validator audit unit on the existing dispatch
  path with a DISTINCT work_unit_id (authenticated internal listing, pending-only semantics), the
  audited submission is NOT reverted to pending, and a non-sampled result yields none.
* VAL-PRISM-013 - an audit manifest MISMATCH invalidates the submission's score in its epoch
  leaderboard; a matching audit leaves it untouched.
* VAL-PRISM-023 - invalidation propagates to the architecture crown and weights (q_arch_best /
  canonical / training_variants recomputed; a matching control is unchanged).
* VAL-PRISM-024 - an audit failure/timeout never silently accepts: it re-audits within bounds, then
  reaches a terminal observable ``failed`` state with the audited submission left unresolved.
* VAL-PRISM-026 - R=1-degraded results remain audit-eligible at their effective-tier rate and a
  forced-sample R=1 result creates an audit unit exactly like an R=2 one.

Offline, no GPU: finalization runs through the CPU re-exec seam the other prism coordination tests
use, and proofs are signed with real sr25519 worker keys.
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
    AUDIT_STATUS_FAILED,
    AUDIT_STATUS_MISMATCH,
    AUDIT_STATUS_PASSED,
    AUDIT_STATUS_PENDING,
    AuditSampler,
    audit_unit_id_for,
    resolve_audit_unit,
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
from prism_challenge.weights import get_weights

WORKER_KEY = "//WorkerAudit"
EPOCH_SECONDS = 60

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
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'audit.sqlite3'}",
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
            "marker": marker,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
    }


def _proof_dict(signer, unit_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(signer=signer, manifest_sha256=digest, unit_id=unit_id)
    return proof.model_dump(mode="json")


def _result(proof_dict: dict[str, Any], manifest: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof_dict,
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
    return AuditSampler(audit_rate_tier0=1.0, audit_rate_tier1=1.0, audit_rate_tier2=1.0, seed=0)


def _never() -> AuditSampler:
    return AuditSampler(audit_rate_tier0=0.0, audit_rate_tier1=0.0, audit_rate_tier2=0.0, seed=0)


async def _finalize(
    app,
    signer,
    *,
    hotkey: str = "hk-owner",
    sampler: AuditSampler | None = None,
    replication: int = 2,
):
    """Seed a submission and ingest a well-formed worker result (finalized + optionally sampled)."""
    from prism_challenge.ingestion import ingest_work_unit_result

    submission_id = await _seed(app, hotkey)
    manifest = _manifest()
    proof = _proof_dict(signer, submission_id, manifest)
    extra: dict[str, Any] = {"replication": replication}
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref=hotkey,
        result=_result(proof, manifest, **extra),
        audit_sampler=sampler,
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


# --- VAL-PRISM-011: sampler follows the configured per-tier rates, seeded -------------------------


def test_sampling_matches_configured_per_tier_rates_seeded() -> None:
    sampler = AuditSampler(
        audit_rate_tier0=0.10, audit_rate_tier1=0.05, audit_rate_tier2=0.02, seed=2026
    )
    n = 6000

    def _bound(p: float) -> float:
        # 4-sigma binomial interval around the configured rate for this N.
        return 4.0 * (p * (1.0 - p) / n) ** 0.5

    for tier, rate in ((0, 0.10), (1, 0.05), (2, 0.02)):
        hits = sum(
            sampler.should_sample(work_unit_id=f"t{tier}-{i}", effective_tier=tier)
            for i in range(n)
        )
        observed = hits / n
        assert abs(observed - rate) < _bound(rate), (tier, observed, rate)


def test_zero_rate_samples_nothing_and_seed_reproduces() -> None:
    zero = _never()
    assert not any(
        zero.should_sample(work_unit_id=f"u{i}", effective_tier=t)
        for i in range(2000)
        for t in (0, 1, 2)
    )
    a = AuditSampler(audit_rate_tier0=0.10, seed=11)
    b = AuditSampler(audit_rate_tier0=0.10, seed=11)
    c = AuditSampler(audit_rate_tier0=0.10, seed=12)
    ids = [f"unit-{i}" for i in range(400)]
    sa = [a.should_sample(work_unit_id=i, effective_tier=0) for i in ids]
    sb = [b.should_sample(work_unit_id=i, effective_tier=0) for i in ids]
    sc = [c.should_sample(work_unit_id=i, effective_tier=0) for i in ids]
    assert sa == sb  # same seed => identical sample set
    assert sa != sc  # a different seed shifts it


# --- VAL-PRISM-012: sampled result creates a distinct validator audit unit on the dispatch path ---


async def test_sampled_result_creates_distinct_validator_audit_unit(tmp_path, monkeypatch) -> None:
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

    submission_id, _, outcome = await _finalize(app, signer, sampler=_always())
    assert outcome.audit_sampled is True
    assert outcome.audit_unit_id == audit_unit_id_for(submission_id)
    # The audit unit id is DISTINCT from the primary unit id (== submission_id).
    assert outcome.audit_unit_id != submission_id
    # The audited submission is NOT reverted to pending by audit creation.
    assert await repository.submission_status(submission_id) == "completed"

    pending = await repository.list_pending_audit_units()
    assert len(pending) == 1
    row = pending[0]
    assert row["audit_unit_id"] == audit_unit_id_for(submission_id)
    assert row["submission_id"] == submission_id
    assert row["required_capability"] == "gpu"
    assert row["executor_kind"] == "validator"


def test_audit_unit_visible_on_internal_work_units_and_absent_when_not_sampled(
    tmp_path, monkeypatch
) -> None:
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
            sampled_id, _, out1 = await _finalize(app, signer, hotkey="hk-a", sampler=_always())
            control_id, _, out2 = await _finalize(app, signer, hotkey="hk-b", sampler=_never())
            assert out1.audit_sampled is True and out2.audit_sampled is False
            return sampled_id, control_id

        sampled_id, control_id = anyio.run(_drive)

        # Authenticated internal listing shows exactly the sampled submission's audit unit.
        unauthorized = client.get("/internal/v1/work_units")
        assert unauthorized.status_code == 401
        listed = client.get("/internal/v1/work_units", headers=headers)
        assert listed.status_code == 200, listed.text
        audit_units = [unit for unit in listed.json()["work_units"] if unit.get("audit") is True]
        assert len(audit_units) == 1
        unit = audit_units[0]
        assert unit["work_unit_id"] == audit_unit_id_for(sampled_id)
        assert unit["work_unit_id"] != sampled_id
        assert unit["submission_id"] == sampled_id
        assert unit["required_capability"] == "gpu"
        assert unit["executor_kind"] == "validator"
        # The non-sampled control never produced an audit unit.
        assert all(u["submission_id"] != control_id for u in audit_units)


# --- VAL-PRISM-013: audit mismatch invalidates the submission's epoch-leaderboard score -----------


async def test_audit_mismatch_invalidates_score_in_epoch_leaderboard(tmp_path, monkeypatch) -> None:
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
    db_path = tmp_path / "audit.sqlite3"

    submission_id, _, outcome = await _finalize(app, signer, sampler=_always())
    epoch_id = (await repository.get_submission(submission_id)).epoch_id
    # Before audit: the submission holds a score and ranks in its epoch leaderboard.
    assert _score(db_path, submission_id) is not None
    board_before = await repository.leaderboard(epoch_id)
    assert any(row["id"] == submission_id for row in board_before)

    # A validator replay with a DIFFERENT manifest hash resolves to a mismatch and invalidates.
    resolution = await resolve_audit_unit(
        repository,
        audit_unit_id=outcome.audit_unit_id,
        replay_manifest_sha256="f" * 64,
    )
    assert resolution.status == AUDIT_STATUS_MISMATCH
    assert resolution.invalidated is True
    assert _score(db_path, submission_id) is None
    assert await repository.submission_status(submission_id) == "failed"
    board_after = await repository.leaderboard(epoch_id)
    assert all(row["id"] != submission_id for row in board_after)


async def test_matching_audit_leaves_score_untouched(tmp_path, monkeypatch) -> None:
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
    db_path = tmp_path / "audit.sqlite3"

    submission_id, manifest, outcome = await _finalize(app, signer, sampler=_always())
    score_before = _score(db_path, submission_id)
    assert score_before is not None

    # A validator replay with the SAME manifest hash passes: the score is untouched.
    resolution = await resolve_audit_unit(
        repository,
        audit_unit_id=outcome.audit_unit_id,
        replay_manifest_sha256=compute_manifest_sha256(manifest),
    )
    assert resolution.status == AUDIT_STATUS_PASSED
    assert resolution.invalidated is False
    assert _score(db_path, submission_id) == pytest.approx(score_before)
    assert await repository.submission_status(submission_id) == "completed"


# --- VAL-PRISM-023: invalidation propagates to the architecture crown + weights -------------------


async def test_invalidation_propagates_to_crown_and_weights(tmp_path) -> None:
    from prism_challenge.db import Database
    from prism_challenge.repository import PrismRepository

    database = Database(tmp_path / "crown.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=EPOCH_SECONDS)

    # Family A crowned by hk-alice's submission (0.9); family B held by hk-bob (0.4).
    await _seed_family(repository, family="A", owner="hk-alice", submission="sA", score=0.9)
    await _seed_family(repository, family="B", owner="hk-bob", submission="sB", score=0.4)

    weights_before = await get_weights(repository, EPOCH_SECONDS)
    assert weights_before  # hk-alice crowned
    best_before = await repository.best_architecture()
    assert best_before["owner_hotkey"] == "hk-alice"

    # Audit invalidates hk-alice's sole submission: the crown must fall back to hk-bob (family B).
    invalidated = await repository.invalidate_submission_score(
        "sA", reason="audit invalidated: manifest mismatch"
    )
    assert invalidated is True

    best_after = await repository.best_architecture()
    assert best_after["owner_hotkey"] == "hk-bob"
    weights_after = await get_weights(repository, EPOCH_SECONDS)
    assert "hk-alice" not in weights_after
    assert weights_after.get("hk-bob", 0.0) > 0.0
    # The invalidated submission's training variant no longer flags current-best.
    variants = await repository.list_training_variants("af-A")
    assert all(not v["is_current_best"] for v in variants)


async def test_invalidation_advances_owner_hotkey_in_multi_owner_family(tmp_path) -> None:
    """VAL-FINAL-007: multi-owner family sharing one arch_hash.

    The crown holder (family creator hk-alice) is proven faulty and invalidated while a co-owner
    (hk-bob) has a valid, lower-scored submission on the SAME architecture. Ownership of the
    weight-bearing ``architecture_families.owner_hotkey`` (the field ``get_weights`` rewards for the
    0.60 architecture share) must advance to the surviving best submission's owner, so the faulty
    owner loses the architecture emission share and ``get_weights`` pays the survivor.
    """
    from prism_challenge.db import Database
    from prism_challenge.repository import PrismRepository

    database = Database(tmp_path / "multiowner.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=EPOCH_SECONDS)

    # Single architecture family (arch_hash fh-A) crowned by hk-alice (0.9); hk-bob is a co-owner
    # with a valid lower-scored submission on the SAME architecture. Per the persistent-crown
    # semantics the family's owner_hotkey stays the creator hk-alice until an invalidation.
    await _seed_family(repository, family="A", owner="hk-alice", submission="sA", score=0.9)
    await _seed_co_submission(repository, family="A", owner="hk-bob", submission="sB", score=0.4)

    best_before = await repository.best_architecture()
    assert best_before["owner_hotkey"] == "hk-alice"
    weights_before = await get_weights(repository, EPOCH_SECONDS)
    assert weights_before.get("hk-alice", 0.0) > 0.0
    assert "hk-bob" not in weights_before

    # hk-alice's crown submission is proven faulty and invalidated; hk-bob's survives on the same
    # architecture, so the weight-bearing ownership must advance to hk-bob.
    invalidated = await repository.invalidate_submission_score(
        "sA", reason="audit invalidated: manifest mismatch"
    )
    assert invalidated is True

    best_after = await repository.best_architecture()
    assert best_after["owner_hotkey"] == "hk-bob"
    assert float(best_after["q_arch_best"]) > 0.0

    # The family row's weight-bearing owner AND owner_submission_id advance to the survivor.
    async with repository.database.connect() as conn:
        rows = await conn.execute_fetchall(
            "SELECT owner_hotkey, owner_submission_id, canonical_submission_id "
            "FROM architecture_families WHERE family_hash=?",
            ("fh-A",),
        )
    family_row = dict(list(rows)[0])
    assert family_row["owner_hotkey"] == "hk-bob"
    assert family_row["owner_submission_id"] == "sB"
    assert family_row["canonical_submission_id"] == "sB"

    weights_after = await get_weights(repository, EPOCH_SECONDS)
    # The proven-faulty creator loses the architecture emission share; the survivor is paid.
    assert "hk-alice" not in weights_after
    assert weights_after.get("hk-bob", 0.0) > 0.0


async def test_invalidation_burns_when_no_valid_submission_remains(tmp_path) -> None:
    from prism_challenge.db import Database
    from prism_challenge.repository import PrismRepository

    database = Database(tmp_path / "burn.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=EPOCH_SECONDS)

    await _seed_family(repository, family="A", owner="hk-alice", submission="sA", score=0.9)
    assert await get_weights(repository, EPOCH_SECONDS)  # crowned

    await repository.invalidate_submission_score("sA", reason="audit invalidated")
    # No valid submission remains anywhere -> BURN (empty weights).
    assert await get_weights(repository, EPOCH_SECONDS) == {}


async def test_matching_audit_control_leaves_crown_untouched(tmp_path) -> None:
    from prism_challenge.db import Database
    from prism_challenge.repository import PrismRepository

    database = Database(tmp_path / "control.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=EPOCH_SECONDS)
    await _seed_family(repository, family="A", owner="hk-alice", submission="sA", score=0.9)
    await _seed_family(repository, family="B", owner="hk-bob", submission="sB", score=0.4)

    before = await get_weights(repository, EPOCH_SECONDS)
    best_before = await repository.best_architecture()
    # A matching audit does not call invalidate_submission_score, so nothing changes.
    after = await get_weights(repository, EPOCH_SECONDS)
    best_after = await repository.best_architecture()
    assert before == after
    assert best_before["owner_hotkey"] == best_after["owner_hotkey"] == "hk-alice"


# --- VAL-PRISM-024: audit failure / timeout never silently accepts --------------------------------


async def test_audit_failure_reaudits_then_terminal_failed_without_accepting(
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
    db_path = tmp_path / "audit.sqlite3"

    submission_id, _, outcome = await _finalize(app, signer, sampler=_always())
    audit_unit_id = outcome.audit_unit_id
    unit = await repository.get_audit_unit(audit_unit_id)
    max_attempts = int(unit["max_attempts"])
    assert max_attempts >= 2
    score_before = _score(db_path, submission_id)

    # Every failure/timeout short of exhaustion re-audits (back to pending), never accepting.
    for attempt in range(1, max_attempts):
        res = await resolve_audit_unit(repository, audit_unit_id=audit_unit_id, failed=True)
        assert res.invalidated is False
        assert res.terminal is False
        assert res.status == AUDIT_STATUS_PENDING
        assert res.attempts == attempt
        # Still eligible for re-audit: listed as pending, submission unchanged.
        assert any(
            u["audit_unit_id"] == audit_unit_id for u in await repository.list_pending_audit_units()
        )

    # On exhaustion the audit is terminally failed; the audited submission is NEVER accepted-by-
    # -failure: its score/status are unchanged (unresolved), not confirmed and not invalidated.
    final = await resolve_audit_unit(
        repository, audit_unit_id=audit_unit_id, replay_manifest_sha256=None, failed=True
    )
    assert final.status == AUDIT_STATUS_FAILED
    assert final.terminal is True
    assert final.invalidated is False
    assert _score(db_path, submission_id) == pytest.approx(score_before)
    assert await repository.submission_status(submission_id) == "completed"
    # A terminal audit is no longer listed as pending (no unbounded retry loop).
    assert all(
        u["audit_unit_id"] != audit_unit_id for u in await repository.list_pending_audit_units()
    )
    # Resolving a terminal unit again is an idempotent no-op.
    again = await resolve_audit_unit(repository, audit_unit_id=audit_unit_id, failed=True)
    assert again.status == AUDIT_STATUS_FAILED
    assert again.terminal is True


# --- VAL-PRISM-026: R=1-degraded results stay audit-eligible at their effective-tier rate ---------


def test_r1_and_r2_results_sampled_at_same_effective_tier_rate() -> None:
    sampler = AuditSampler(audit_rate_tier0=0.10, seed=4242)
    n = 6000

    def _bound(p: float) -> float:
        return 4.0 * (p * (1.0 - p) / n) ** 0.5

    # A single population keyed by unit id; R is just a label the base plane attaches, so the
    # scheduler samples R=1 and R=2 identically at the effective tier's rate (never exempting R=1).
    r1_hits = sum(sampler.should_sample(work_unit_id=f"r1-{i}", effective_tier=0) for i in range(n))
    r2_hits = sum(sampler.should_sample(work_unit_id=f"r2-{i}", effective_tier=0) for i in range(n))
    assert abs(r1_hits / n - 0.10) < _bound(0.10)
    assert abs(r2_hits / n - 0.10) < _bound(0.10)


async def test_forced_sample_r1_result_creates_audit_unit(tmp_path, monkeypatch) -> None:
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

    # A replication-degraded (R=1) accepted result is sampled and audited like any other.
    submission_id, _, outcome = await _finalize(app, signer, sampler=_always(), replication=1)
    assert outcome.audit_sampled is True
    row = await repository.get_audit_unit(outcome.audit_unit_id)
    assert row is not None
    assert int(row["replication"]) == 1
    assert row["status"] == AUDIT_STATUS_PENDING
    assert row["executor_kind"] == "validator"


# --- Audit resolution HTTP route (curl surface for VAL-PRISM-013/024) -----------------------------


def test_audit_result_route_invalidates_and_drops_from_leaderboard(tmp_path, monkeypatch) -> None:
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

        async def _drive() -> tuple[str, str, int]:
            sid, _, out = await _finalize(app, signer, sampler=_always())
            epoch = (await app.state.repository.get_submission(sid)).epoch_id
            return sid, out.audit_unit_id, epoch

        submission_id, audit_unit_id, epoch_id = anyio.run(_drive)

        board = client.get(f"/v1/leaderboard?epoch_id={epoch_id}").json()
        assert any(e["submission_id"] == submission_id for e in board["entries"])

        # Unauthenticated resolution is rejected.
        assert (
            client.post(f"/internal/v1/audit_units/{audit_unit_id}/result", json={}).status_code
            == 401
        )
        # A mismatching validator replay invalidates via the route.
        resolved = client.post(
            f"/internal/v1/audit_units/{audit_unit_id}/result",
            json={"manifest_sha256": "e" * 64},
            headers=headers,
        )
        assert resolved.status_code == 200, resolved.text
        assert resolved.json()["status"] == AUDIT_STATUS_MISMATCH
        assert resolved.json()["invalidated"] is True

        # The submission is gone from its epoch leaderboard, and reads as failed (not completed).
        board_after = client.get(f"/v1/leaderboard?epoch_id={epoch_id}").json()
        assert all(e["submission_id"] != submission_id for e in board_after["entries"])
        detail = client.get(f"/v1/submissions/{submission_id}").json()
        assert detail["status"] == "failed"
        assert detail["final_score"] is None


def test_audit_result_route_disabled_when_worker_plane_off(tmp_path) -> None:
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path, worker_plane=WorkerPlaneConfig(enabled=False))
    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/internal/v1/audit_units/audit:sub-x/result",
            json={"manifest_sha256": "a" * 64},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 404


def test_work_units_omits_audit_units_when_worker_plane_off(tmp_path) -> None:
    import anyio
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path, worker_plane=WorkerPlaneConfig(enabled=False))
    headers = {"Authorization": "Bearer secret"}
    with TestClient(create_app(settings)) as client:
        app = client.app

        async def _seed() -> str:
            repo = app.state.repository
            sub = await repo.create_submission(
                "hk-a", SubmissionCreate(code=_bundle(), filename="a.py")
            )
            # An audit unit row exists but must stay invisible while the flag is OFF.
            await repo.create_audit_unit(
                submission_id=sub.id,
                origin_work_unit_id=sub.id,
                audited_manifest_sha256="a" * 64,
                effective_tier=0,
            )
            return sub.id

        submission_id = anyio.run(_seed)
        listed = client.get("/internal/v1/work_units", headers=headers).json()
        assert all(not unit.get("audit") for unit in listed["work_units"])
        # The primary pending unit is still exposed exactly as legacy.
        assert any(unit["submission_id"] == submission_id for unit in listed["work_units"])


# --- helpers --------------------------------------------------------------------------------------


async def _seed_family(
    repository: Any,
    *,
    family: str,
    owner: str,
    submission: str,
    score: float,
    epoch_id: int = 1,
) -> None:
    """Directly seed a completed submission + score + architecture family + training variant."""
    created = "2026-06-27T00:00:00+00:00"
    architecture_id = f"af-{family}"
    family_hash = f"fh-{family}"
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
            (epoch_id, created, created, "open"),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO miners(hotkey, first_seen, last_seen) VALUES (?, ?, ?)",
            (owner, created, created),
        )
        await conn.execute(
            "INSERT INTO submissions("
            "id, hotkey, epoch_id, filename, code, code_hash, arch_hash, metadata, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission,
                owner,
                epoch_id,
                "project.zip",
                "x",
                submission,
                family_hash,
                "{}",
                "completed",
                created,
                created,
            ),
        )
        await conn.execute(
            "INSERT INTO scores("
            "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, "
            "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (submission, score, 0.0, 1.0, 0.0, 0.0, score, "{}", created),
        )
        await conn.execute(
            "INSERT INTO architecture_families("
            "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
            "owner_submission_id, canonical_submission_id, q_arch_best, display_name, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                architecture_id,
                family_hash,
                f"fp-{family}",
                f"bp-{family}",
                owner,
                submission,
                submission,
                score,
                f"arch-{family}",
                created,
                created,
            ),
        )
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
            "metric_mean, metric_std, is_current_best, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"tv-{family}",
                architecture_id,
                f"th-{family}",
                owner,
                submission,
                score,
                score,
                0.0,
                1,
                created,
                created,
            ),
        )


async def _seed_co_submission(
    repository: Any,
    *,
    family: str,
    owner: str,
    submission: str,
    score: float,
    epoch_id: int = 1,
) -> None:
    """Add a co-owner's completed submission on an EXISTING family (same arch_hash).

    Models a multi-owner architecture family: the family's owner_hotkey stays the family creator,
    but a distinct owner has a valid, lower-scored submission and a distinct training variant on the
    same architecture, so it survives an invalidation of the crown holder's submission.
    """
    created = "2026-06-27T00:00:01+00:00"
    architecture_id = f"af-{family}"
    family_hash = f"fh-{family}"
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO miners(hotkey, first_seen, last_seen) VALUES (?, ?, ?)",
            (owner, created, created),
        )
        await conn.execute(
            "INSERT INTO submissions("
            "id, hotkey, epoch_id, filename, code, code_hash, arch_hash, metadata, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission,
                owner,
                epoch_id,
                "project.zip",
                "x",
                submission,
                family_hash,
                "{}",
                "completed",
                created,
                created,
            ),
        )
        await conn.execute(
            "INSERT INTO scores("
            "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, "
            "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (submission, score, 0.0, 1.0, 0.0, 0.0, score, "{}", created),
        )
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
            "metric_mean, metric_std, is_current_best, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"tv-{family}-{owner}",
                architecture_id,
                f"th-{family}-{owner}",
                owner,
                submission,
                score,
                score,
                0.0,
                0,
                created,
                created,
            ),
        )
