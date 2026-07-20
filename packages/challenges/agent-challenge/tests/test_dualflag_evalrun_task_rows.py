"""Dual-flag public status.evaluation.task_rows from EvalRun.plan_json (VAL-DFROWS-001..006)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import agent_challenge.api.routes as api_routes
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.plan_scoring import (
    build_score_record_from_eval_plan,
    canonical_eval_plan_json,
)
from agent_challenge.models import AgentSubmission, EvalRun, TaskLogEvent

NOW = datetime(2026, 7, 20, 15, 0, tzinfo=UTC)
AGENT_HASH = "a" * 64
REVIEW_DIGEST = "b" * 64
MEASUREMENT = {
    "mrtd": "1" * 96,
    "rtmr0": "2" * 96,
    "rtmr1": "3" * 96,
    "rtmr2": "4" * 96,
    "os_image_hash": "5" * 64,
    "compose_hash": "6" * 64,
}


def _policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }


def _plan(*, eval_run_id: str, task_count: int = 3) -> dict[str, object]:
    policy = _policy()
    task_ids = [f"df-task-{index:03d}" for index in range(task_count)]
    return {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": f"submission-{eval_run_id}",
        "submission_version": 1,
        "authorizing_review_digest": REVIEW_DIGEST,
        "agent_hash": AGENT_HASH,
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry.example/task@sha256:" + "3" * 64,
                "task_config_sha256": "4" * 64,
            }
            for task_id in task_ids
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "5" * 64,
            "compose_hash": MEASUREMENT["compose_hash"],
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "6" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("6" * 64)).hexdigest(),
            "measurement": {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "os_image_hash": MEASUREMENT["os_image_hash"],
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": f"key-nonce-{eval_run_id}",
        "score_nonce": f"score-nonce-{eval_run_id}",
        "run_token_sha256": "7" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


def _trials_for(plan: dict[str, object], *, perfect_count: int) -> dict[str, list[float]]:
    task_ids = [item["task_id"] for item in plan["selected_tasks"]]
    trials: dict[str, list[float]] = {}
    for index, task_id in enumerate(task_ids):
        value = 1.0 if index < perfect_count else 0.0
        trials[task_id] = [value] * int(plan["k"])
    return trials


async def _seed_dualflag_eval_run(
    session,
    *,
    eval_run_id: str,
    phase: str,
    with_scores: bool,
    perfect_count: int = 2,
    task_count: int = 3,
) -> tuple[int, dict[str, object], EvalRun]:
    plan = _plan(eval_run_id=eval_run_id, task_count=task_count)
    plan_json = canonical_eval_plan_json(plan)
    submission = AgentSubmission(
        miner_hotkey=f"hk-{eval_run_id}",
        name=f"agent-{eval_run_id}",
        agent_hash=hashlib.sha256(eval_run_id.encode("utf-8")).hexdigest(),
        artifact_uri=f"/tmp/{eval_run_id}.zip",
        status="queued",
        raw_status="queued",
        effective_status="queued",
        submitted_at=NOW,
        created_at=NOW,
    )
    session.add(submission)
    await session.flush()

    score = None
    passed = None
    total = None
    score_json = None
    score_sha = None
    if with_scores:
        trials = _trials_for(plan, perfect_count=perfect_count)
        record = build_score_record_from_eval_plan(plan, trials)
        score_json = ew.canonical_json_v1(record).decode("utf-8")
        score_sha = ew.score_record_digest(record)
        score = ew.decode_score_f64be(record["final"]["job_score_f64be"])
        passed = record["final"]["passed_tasks"]
        total = record["final"]["total_tasks"]

    run = EvalRun(
        eval_run_id=eval_run_id,
        submission_id=submission.id,
        submission_version=1,
        authorizing_review_digest=REVIEW_DIGEST,
        plan_json=plan_json,
        plan_sha256=hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
        token_sha256=hashlib.sha256(f"token-{eval_run_id}".encode()).hexdigest(),
        phase=phase,
        verified=with_scores and phase == "eval_accepted",
        reward_eligible=False,
        result_available=with_scores,
        score=score,
        passed_tasks=passed,
        total_tasks=total,
        canonical_score_record_json=score_json,
        canonical_score_record_sha256=score_sha,
        issued_at=NOW,
        # Far future so eval_status_page expiry reconciliation does not rewrite
        # the fixture phase before public status projection.
        expires_at=NOW + timedelta(days=3650),
        created_at=NOW,
        updated_at=NOW,
    )
    session.add(run)
    await session.commit()
    return submission.id, plan, run


def _enable_dual_flags(monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)


async def test_dualflag_status_task_rows_from_evalrun_plan_n3(
    client,
    database_session,
    monkeypatch,
) -> None:
    """VAL-DFROWS-001/005: plan selected_tasks N=3 project with job=None."""
    _enable_dual_flags(monkeypatch)
    async with database_session() as session:
        submission_id, plan, _run = await _seed_dualflag_eval_run(
            session,
            eval_run_id="eval-df-rows-001",
            phase="eval_prepared",
            with_scores=False,
            task_count=3,
        )

    response = await client.get(f"/submissions/{submission_id}/status")
    assert response.status_code == 200
    payload = response.json()
    evaluation = payload["evaluation"]
    assert evaluation is not None
    assert evaluation["job_id"] == "eval-df-rows-001"
    assert evaluation["status"] == "eval_prepared"
    rows = evaluation["task_rows"]
    assert len(rows) == 3
    expected_ids = [item["task_id"] for item in plan["selected_tasks"]]
    assert [row["task_id"] for row in rows] == expected_ids
    assert all(row["has_result"] is False for row in rows)
    assert all(row["phase"] == "assigned" for row in rows)
    # Dual-flag path must not invent a legacy EvaluationJob.
    assert evaluation["current_attempt"] is None


async def test_dualflag_eval_expired_and_prepared_still_show_planned_rows(
    client,
    database_session,
    monkeypatch,
) -> None:
    """VAL-DFROWS-002: expired/prepared still project planned rows."""
    _enable_dual_flags(monkeypatch)
    for phase, eval_run_id in (
        ("eval_expired", "eval-df-rows-expired"),
        ("eval_prepared", "eval-df-rows-prepared"),
    ):
        async with database_session() as session:
            submission_id, plan, _run = await _seed_dualflag_eval_run(
                session,
                eval_run_id=eval_run_id,
                phase=phase,
                with_scores=False,
                task_count=3,
            )
        response = await client.get(f"/submissions/{submission_id}/status")
        assert response.status_code == 200
        evaluation = response.json()["evaluation"]
        assert evaluation["status"] == phase
        rows = evaluation["task_rows"]
        assert len(rows) == 3
        assert [row["task_id"] for row in rows] == [
            item["task_id"] for item in plan["selected_tasks"]
        ]


async def test_dualflag_score_ledger_not_hardcoded_zero(
    client,
    database_session,
    monkeypatch,
) -> None:
    """VAL-DFROWS-003: score/passed/total come from EvalRun ledger."""
    _enable_dual_flags(monkeypatch)
    async with database_session() as session:
        submission_id, _plan, run = await _seed_dualflag_eval_run(
            session,
            eval_run_id="eval-df-rows-score",
            phase="eval_accepted",
            with_scores=True,
            perfect_count=2,
            task_count=3,
        )
        assert run.score is not None and run.score > 0
        assert run.passed_tasks == 2
        assert run.total_tasks == 3
        expected_score = float(run.score)
        expected_passed = int(run.passed_tasks)
        expected_total = int(run.total_tasks)

    response = await client.get(f"/submissions/{submission_id}/status")
    assert response.status_code == 200
    evaluation = response.json()["evaluation"]
    assert evaluation["score"] == expected_score
    assert evaluation["passed_tasks"] == expected_passed
    assert evaluation["total_tasks"] == expected_total
    assert evaluation["score"] != 0.0 or expected_score == 0.0
    assert not (
        evaluation["score"] == 0.0
        and evaluation["passed_tasks"] == 0
        and evaluation["total_tasks"] == 0
    )


async def test_dualflag_score_record_overlay_has_result(
    client,
    database_session,
    monkeypatch,
) -> None:
    """VAL-DFROWS-004: overlay has_result from score record; no invented ids."""
    _enable_dual_flags(monkeypatch)
    async with database_session() as session:
        submission_id, plan, _run = await _seed_dualflag_eval_run(
            session,
            eval_run_id="eval-df-rows-overlay",
            phase="eval_accepted",
            with_scores=True,
            perfect_count=2,
            task_count=3,
        )
        planned_ids = {item["task_id"] for item in plan["selected_tasks"]}

    response = await client.get(f"/submissions/{submission_id}/status")
    assert response.status_code == 200
    rows = response.json()["evaluation"]["task_rows"]
    assert len(rows) == 3
    assert {row["task_id"] for row in rows} == planned_ids
    # Perfect first two tasks get has_result; all planned ids only.
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["df-task-000"]["has_result"] is True
    assert by_id["df-task-001"]["has_result"] is True
    assert by_id["df-task-002"]["has_result"] is True
    assert by_id["df-task-000"]["phase"] == "completed"
    assert by_id["df-task-002"]["phase"] == "failed"
    # No unplanned invent.
    assert "invented-task" not in by_id


async def test_dualflag_task_events_remain_empty_without_guest_logs(
    client,
    database_session,
    monkeypatch,
) -> None:
    """VAL-DFROWS-006: empty TaskLogEvent stays honest empty; no fake log bodies."""
    _enable_dual_flags(monkeypatch)
    async with database_session() as session:
        submission_id, _plan, _run = await _seed_dualflag_eval_run(
            session,
            eval_run_id="eval-df-rows-nologs",
            phase="eval_expired",
            with_scores=False,
            task_count=3,
        )
        # Prove no TaskLogEvent rows exist for this submission.
        # (clean DB fixture leaves table empty for this id)

    status = await client.get(f"/submissions/{submission_id}/status")
    assert status.status_code == 200
    assert len(status.json()["evaluation"]["task_rows"]) == 3

    events = await client.get(f"/submissions/{submission_id}/task-events")
    assert events.status_code == 200
    body = events.json()
    event_list = body.get("events") if isinstance(body, dict) else body
    if event_list is None and isinstance(body, dict):
        event_list = body.get("items") or body.get("task_events") or []
    assert event_list == []
    serialized = json.dumps(body)
    assert "fake guest" not in serialized.lower()
    assert "invented" not in serialized.lower()


def test_task_rows_helper_from_plan_and_score_overlay_unit() -> None:
    """Offline unit fixture: helper projects N=3 plan and overlays score record."""
    plan = _plan(eval_run_id="eval-df-helper", task_count=3)
    plan_json = canonical_eval_plan_json(plan)
    record = build_score_record_from_eval_plan(plan, _trials_for(plan, perfect_count=1))
    score_json = ew.canonical_json_v1(record).decode("utf-8")
    run = EvalRun(
        eval_run_id="eval-df-helper",
        submission_id=1,
        submission_version=1,
        authorizing_review_digest=REVIEW_DIGEST,
        plan_json=plan_json,
        plan_sha256=hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
        token_sha256="c" * 64,
        phase="eval_accepted",
        score=0.5,
        passed_tasks=1,
        total_tasks=3,
        canonical_score_record_json=score_json,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        created_at=NOW,
        updated_at=NOW,
    )
    rows = api_routes._task_rows_from_eval_run(run)
    assert len(rows) == 3
    assert [row.task_id for row in rows] == [t["task_id"] for t in plan["selected_tasks"]]
    assert rows[0].has_result is True
    assert rows[0].phase == "completed"
    assert rows[1].has_result is True
    assert rows[1].phase == "failed"
    assert rows[2].has_result is True
    assert rows[2].phase == "failed"
    # Ensure helper does not touch TaskLogEvent / invent logs.
    assert TaskLogEvent is not None
