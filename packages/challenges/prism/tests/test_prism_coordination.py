"""Concurrency-1 own-broker coordination integration (VAL-PRISM-001..004, 037).

A prism submission becomes EXACTLY ONE gpu work unit (``work_unit_id == submission_id``)
assigned to exactly one validator with concurrency 1; the assigned validator runs the
re-execution by dispatching to its OWN broker-backed ``DockerExecutor`` (exercised here via the
CPU re-exec mock that monkeypatches ``DockerExecutor.run``). The master coordinator (work-unit
exposure + pull) never invokes the executor, and re-posting a completed assignment is idempotent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from base.challenge_sdk.executor import DockerRunSpec
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.coordination import (
    PRISM_WORK_UNIT_CAPABILITY,
    PrismWorkUnit,
    capability_can_run,
    list_pending_prism_work_units,
    prism_work_unit_id,
    pull_assigned_work_units,
    work_unit_to_payload,
)
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.models import SubmissionCreate
from prism_challenge.validator_executor import execute_work_unit, run_validator_cycle

# A tiny CPU-torch byte-level next-token model: trains one step at a time over the challenge
# instrument with no GPU, no tokenizer (byte basis), deterministic under the forced seed.
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


def _two_script_bundle(arch: str, train: str) -> str:
    import base64
    import io
    import zipfile

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", arch)
        archive.writestr("training.py", train)
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
    )


async def _create_submission(app, hotkey: str, *, arch: str = TINY_ARCH, train: str = TINY_TRAIN):
    return await app.state.repository.create_submission(
        hotkey,
        SubmissionCreate(code=_two_script_bundle(arch, train), filename="project.zip"),
    )


async def _make_app(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    await app.state.database.init()
    return app


# --- VAL-PRISM-001 / 003: one gpu work unit per submission (work_unit_id == submission_id) -------


async def test_each_submission_becomes_exactly_one_gpu_work_unit(tmp_path):
    app = await _make_app(tmp_path)
    sub_a = await _create_submission(app, "hk-a")
    sub_b = await _create_submission(app, "hk-b")

    units = await list_pending_prism_work_units(app.state.repository)

    assert len(units) == 2
    by_submission = {u.submission_id: u for u in units}
    assert set(by_submission) == {sub_a.id, sub_b.id}
    for submission_id, unit in by_submission.items():
        # Exactly one unit per submission and the work-unit id IS the submission id.
        assert unit.work_unit_id == submission_id
        assert unit.work_unit_id == prism_work_unit_id(submission_id)
        assert unit.required_capability == PRISM_WORK_UNIT_CAPABILITY == "gpu"


async def test_work_units_endpoint_exposes_single_gpu_unit_per_submission(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        # Seed two submissions through the internal bridge so the master plane can enumerate them.
        for hotkey in ("hk-a", "hk-b"):
            body = _two_script_bundle(TINY_ARCH, TINY_TRAIN).encode()
            resp = client.post(
                "/internal/v1/bridge/submissions",
                content=body,
                headers={
                    "Authorization": "Bearer secret",
                    "X-Base-Verified-Hotkey": hotkey,
                    "X-Submission-Filename": "project.zip",
                    "Content-Type": "application/octet-stream",
                },
            )
            assert resp.status_code == 200, resp.text

        unauth = client.get("/internal/v1/work_units")
        assert unauth.status_code == 401

        listing = client.get("/internal/v1/work_units", headers={"Authorization": "Bearer secret"})
        assert listing.status_code == 200, listing.text
        body = listing.json()
        assert body["challenge_slug"] == settings.slug
        units = body["work_units"]
        assert len(units) == 2
        assert all(u["required_capability"] == "gpu" for u in units)
        assert all(u["work_unit_id"] == u["submission_id"] for u in units)


# --- VAL-PRISM-001: only one validator's pull returns a given unit -------------------------------


async def test_only_assigned_validator_pull_returns_the_unit(tmp_path):
    app = await _make_app(tmp_path)
    sub_a = await _create_submission(app, "hk-a")
    sub_b = await _create_submission(app, "hk-b")
    units = await list_pending_prism_work_units(app.state.repository)

    # The master assigns sub_a's unit to validator A and sub_b's to validator B.
    pulled_a = pull_assigned_work_units(units, work_unit_ids=[sub_a.id])
    pulled_b = pull_assigned_work_units(units, work_unit_ids=[sub_b.id])

    assert [u.work_unit_id for u in pulled_a] == [sub_a.id]
    assert [u.work_unit_id for u in pulled_b] == [sub_b.id]
    # Disjoint: no unit is returned to two validators.
    assert {u.work_unit_id for u in pulled_a}.isdisjoint({u.work_unit_id for u in pulled_b})


# --- VAL-PRISM-002: a busy validator is not given a second concurrent prism unit -----------------


async def test_busy_validator_not_given_second_concurrent_unit(tmp_path):
    app = await _make_app(tmp_path)
    sub_a = await _create_submission(app, "hk-a")
    sub_b = await _create_submission(app, "hk-b")
    units = await list_pending_prism_work_units(app.state.repository)
    assigned = [sub_a.id, sub_b.id]

    # Idle validator A pulls at most one unit (concurrency 1) even though two are assigned.
    idle_pull = pull_assigned_work_units(units, work_unit_ids=assigned, in_flight=0)
    assert len(idle_pull) == 1

    # While A is busy (one in-flight prism unit), its next pull yields no second unit.
    busy_pull = pull_assigned_work_units(units, work_unit_ids=assigned, in_flight=1)
    assert busy_pull == []

    # The second unit is instead available to an idle validator B.
    other_pull = pull_assigned_work_units(units, work_unit_ids=[sub_b.id], in_flight=0)
    assert [u.work_unit_id for u in other_pull] == [sub_b.id]


async def test_validator_cycle_runs_one_unit_at_a_time(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    app = await _make_app(tmp_path)
    await _create_submission(app, "hk-a")
    await _create_submission(app, "hk-b")

    summary = await run_validator_cycle(worker=app.state.worker, max_concurrency=1)

    # Concurrency 1: exactly one unit pulled + executed this cycle; the other stays pending.
    assert summary.pulled == 1
    assert summary.executed == 1
    remaining = await list_pending_prism_work_units(app.state.repository)
    assert len(remaining) == 1


# --- VAL-PRISM-003: prism gpu units are capability-gated to gpu validators -----------------------


async def test_prism_unit_capability_gated_to_gpu_validators(tmp_path):
    app = await _make_app(tmp_path)
    sub = await _create_submission(app, "hk-a")
    units = await list_pending_prism_work_units(app.state.repository)

    # A cpu-only validator's pull omits the gpu unit; a gpu validator receives it.
    cpu_only = pull_assigned_work_units(units, work_unit_ids=[sub.id], capabilities=("cpu",))
    gpu_validator = pull_assigned_work_units(
        units, work_unit_ids=[sub.id], capabilities=("cpu", "gpu")
    )

    assert cpu_only == []
    assert [u.work_unit_id for u in gpu_validator] == [sub.id]
    assert capability_can_run("gpu", ("cpu", "gpu")) is True
    assert capability_can_run("gpu", ("cpu",)) is False


# --- VAL-PRISM-037: re-execution dispatches through the validator's OWN broker -------------------


async def test_reexec_dispatches_through_own_broker_not_master(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    app = await _make_app(tmp_path)
    sub = await _create_submission(app, "hk-a")

    units = await list_pending_prism_work_units(app.state.repository)
    pulled = pull_assigned_work_units(units, work_unit_ids=[sub.id])
    # The master coordinator (work-unit exposure + pull) NEVER invokes the executor.
    assert captured == []

    outcome = await execute_work_unit(app.state.worker, pulled[0])

    # The validator's OWN broker-backed executor ran the re-execution (network-isolated).
    assert outcome.executed is True
    assert outcome.status == "completed"
    assert len(captured) == 1
    assert captured[0].limits.network == "none"
    assert "--nproc-per-node=1" in captured[0].command


# --- VAL-PRISM-004: posting a result for a completed prism assignment is idempotent --------------


async def test_repost_completed_prism_assignment_is_idempotent(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    app = await _make_app(tmp_path)
    sub = await _create_submission(app, "hk-a")
    db_path = tmp_path / "coord.sqlite3"

    first = await run_validator_cycle(worker=app.state.worker, work_unit_ids=[sub.id])
    assert first.executed == 1
    assert sub.id in first.completed_submissions
    runs_after_first = len(captured)

    def _score(submission_id: str):
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    score_before = _score(sub.id)
    assert score_before is not None and score_before > 0.0

    # Re-running the now-completed assignment is a no-op: no re-dispatch, score unchanged.
    second = await run_validator_cycle(worker=app.state.worker, work_unit_ids=[sub.id])
    assert second.pulled == 0  # the unit is no longer pending => nothing to pull
    assert second.executed == 0
    assert len(captured) == runs_after_first

    # A direct re-execution of the completed unit is also an idempotent no-op.
    outcome = await execute_work_unit(
        app.state.worker,
        PrismWorkUnit(work_unit_id=sub.id, submission_id=sub.id, hotkey="hk-a"),
    )
    assert outcome.executed is False
    assert outcome.status == "completed"
    assert len(captured) == runs_after_first
    assert _score(sub.id) == pytest.approx(score_before)


# --- Clarity: the work-unit hotkey is surfaced as the master's `submission_ref` contract key ------


async def test_work_unit_payload_carries_hotkey_as_submission_ref(tmp_path):
    app = await _make_app(tmp_path)
    await _create_submission(app, "hk-ref")
    units = await list_pending_prism_work_units(app.state.repository)

    assert len(units) == 1
    unit = units[0]
    # The field is named for what it actually holds (the submitting miner hotkey).
    assert unit.hotkey == "hk-ref"
    # The wire payload preserves the master's `submission_ref` contract key (read by the platform
    # HttpChallengeWorkSource), sourced from that hotkey.
    payload = work_unit_to_payload(unit)
    assert payload["submission_ref"] == "hk-ref"
    assert payload["submission_id"] == unit.submission_id


# --- Defense-in-depth: validator cycle enforces concurrency 1 against the REAL in-flight count ----


async def test_validator_cycle_uses_real_in_flight_count(tmp_path):
    app = await _make_app(tmp_path)
    sub_a = await _create_submission(app, "hk-a")
    sub_b = await _create_submission(app, "hk-b")

    # Simulate a concurrent in-flight run: claim sub_a so it sits in `running` (not terminal).
    claimed = await app.state.repository.claim_submission(sub_a.id)
    assert claimed is not None
    assert await app.state.repository.count_in_flight_submissions() == 1

    # A cycle that does NOT pass in_flight resolves the true count (1) and, with concurrency 1,
    # pulls nothing more -- the cap is enforced, not advisory. No broker is dispatched.
    summary = await run_validator_cycle(worker=app.state.worker, work_unit_ids=[sub_b.id])
    assert summary.pulled == 0
    assert summary.executed == 0

    # Passing in_flight=0 explicitly (ignoring reality) WOULD have pulled the pending unit, proving
    # the default now reflects the validator's real concurrency draw.
    forced = pull_assigned_work_units(
        await list_pending_prism_work_units(app.state.repository),
        work_unit_ids=[sub_b.id],
        in_flight=0,
    )
    assert [u.work_unit_id for u in forced] == [sub_b.id]

    # An explicit in_flight override is honored as-is (the cap is not recomputed): a saturated
    # override pulls nothing and never dispatches the broker.
    override = await run_validator_cycle(
        worker=app.state.worker, work_unit_ids=[sub_b.id], in_flight=5
    )
    assert override.pulled == 0
    assert override.executed == 0
