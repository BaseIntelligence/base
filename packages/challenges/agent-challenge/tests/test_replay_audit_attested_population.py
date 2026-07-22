"""Production replay-audit population and labelled seam regressions."""

from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import EvalRun, EvaluationJob, TaskAttestation, TaskResult
from agent_challenge.evaluation.plan_scoring import (
    build_score_record_from_eval_plan,
    canonical_eval_plan_json,
    persist_canonical_score_record,
    persist_eval_plan,
)
from agent_challenge.evaluation.replay_audit import (
    REPLAY_AUDIT_LABEL,
    AggregationSpec,
    AuditCandidate,
    InvalidReplayTrialsError,
    ReplayAuditDispute,
    ReplayAuditWireError,
    accepted_verified_replay_population,
    compare_replay_trials,
    persist_replay_dispute,
    replay_request_for_candidate,
    replay_result_from_mapping,
)
from agent_challenge.models import AgentSubmission


def _plan(eval_run_id: str = "eval-replay-population-1") -> dict:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    return ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": eval_run_id,
            "submission_id": "1",
            "submission_version": 1,
            "authorizing_review_digest": "2" * 64,
            "agent_hash": "3" * 64,
            "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "selected_tasks": [
                {
                    "task_id": "task-a",
                    "image_ref": "registry.example/task@sha256:" + "4" * 64,
                    "task_config_sha256": "5" * 64,
                }
            ],
            "k": 2,
            "scoring_policy": policy,
            "scoring_policy_digest": ew.scoring_policy_digest(policy),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "6" * 64,
                "compose_hash": "7" * 64,
                "app_identity": "agent-challenge-eval",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "8" * 64,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("8" * 64)).hexdigest(),
                "measurement": {
                    "mrtd": "a" * 96,
                    "rtmr0": "b" * 96,
                    "rtmr1": "c" * 96,
                    "rtmr2": "d" * 96,
                    "os_image_hash": "e" * 64,
                    "key_provider": "validator-kms",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8700",
            "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
            "key_release_nonce": "key-replay",
            "score_nonce": "score-replay",
            "run_token_sha256": "f" * 64,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


async def _seed_population_row(
    session,
    *,
    plan: dict,
    phase: str = "eval_accepted",
    verified: bool = True,
    result_available: bool = True,
    reward_eligible: bool = True,
    job_plan: dict | None = None,
    task_attested: bool = True,
) -> tuple[AgentSubmission, EvaluationJob, EvalRun]:
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey=f"hotkey-{plan['eval_run_id']}",
        name=f"agent-{plan['eval_run_id']}",
        agent_hash=plan["agent_hash"],
        artifact_uri=f"/tmp/{plan['eval_run_id']}.zip",
        raw_status="tb_completed",
        status="tb_completed",
        effective_status="valid",
        version_number=1,
        submitted_at=now,
        created_at=now,
    )
    session.add(submission)
    await session.flush()
    plan = copy.deepcopy(plan)
    plan["submission_id"] = str(submission.id)
    plan = ew.validate_eval_plan(plan)
    job = EvaluationJob(
        job_id=f"job-{plan['eval_run_id']}",
        submission_id=submission.id,
        status="completed",
        selected_tasks_json="[]",
        score=0.75,
        passed_tasks=0,
        total_tasks=1,
    )
    persist_eval_plan(job, job_plan or plan)
    record = build_score_record_from_eval_plan(plan, {"task-a": [0.5, 1.0]})
    persist_canonical_score_record(job, record)
    session.add(job)
    await session.flush()
    session.add(
        TaskResult(
            job_id=job.id,
            task_id="task-a",
            docker_image=plan["selected_tasks"][0]["image_ref"],
            status="completed",
            score=ew.decode_score_f64be(record["tasks"][0]["aggregate_score_f64be"]),
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )
    )
    session.add(
        TaskAttestation(
            job_id=job.id,
            task_id="task-a",
            verified=task_attested,
            reason=None if task_attested else "missing",
            retryable=False,
        )
    )
    run = EvalRun(
        eval_run_id=plan["eval_run_id"],
        submission_id=submission.id,
        submission_version=1,
        authorizing_review_digest=plan["authorizing_review_digest"],
        plan_json=canonical_eval_plan_json(plan),
        plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode()).hexdigest(),
        token_sha256="9" * 64,
        phase=phase,
        verified=verified,
        result_available=result_available,
        reward_eligible=reward_eligible,
        retryable=False,
        key_granted_at=now,
        issued_at=now,
        expires_at=now + timedelta(hours=1),
        result_job_id=job.id,
    )
    session.add(run)
    await session.flush()
    return submission, job, run


async def _review_allow(*_args, **_kwargs):
    return SimpleNamespace(review_digest="2" * 64)


async def test_population_requires_durable_accepted_full_attested_evidence(
    database_session, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.replay_audit.verified_review_assignment_for_submission",
        _review_allow,
    )
    plan = _plan()
    async with database_session() as session:
        submission, job, run = await _seed_population_row(session, plan=plan)
        await session.commit()

    async with database_session() as session:
        population = await accepted_verified_replay_population(session, enabled=True)
    assert [candidate.eval_run_id for candidate in population] == [run.eval_run_id]
    assert population[0].attested_score == pytest.approx(0.75)
    assert population[0].n_attempts == 2
    assert population[0].eval_plan["scoring_policy"] == plan["scoring_policy"]

    # Caller-controlled candidate flags cannot substitute for durable eligibility.
    assert AuditCandidate("spoofed", attested=True, verified=True).population_eligible is None
    assert submission.id == job.submission_id


@pytest.mark.parametrize(
    ("phase", "verified", "result_available", "reward_eligible", "task_attested"),
    [
        ("eval_rejected", False, False, False, True),
        ("eval_error", False, False, False, True),
        ("eval_accepted", False, True, False, True),
        ("eval_accepted", True, False, False, True),
        ("eval_accepted", True, True, True, False),
    ],
)
async def test_failed_unverified_liveness_and_incomplete_runs_are_inert(
    database_session,
    monkeypatch,
    phase,
    verified,
    result_available,
    reward_eligible,
    task_attested,
) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.replay_audit.verified_review_assignment_for_submission",
        _review_allow,
    )
    plan = _plan(f"eval-ineligible-{phase}-{verified}-{result_available}-{task_attested}")
    async with database_session() as session:
        await _seed_population_row(
            session,
            plan=plan,
            phase=phase,
            verified=verified,
            result_available=result_available,
            reward_eligible=reward_eligible,
            task_attested=task_attested,
        )
        await session.commit()

    async with database_session() as session:
        assert await accepted_verified_replay_population(session, enabled=True) == []
        assert await accepted_verified_replay_population(session, enabled=False) == []


def test_labelled_request_and_result_require_exact_plan_identity() -> None:
    plan = _plan()
    candidate = AuditCandidate(
        "1",
        attested_score=0.75,
        n_attempts=2,
        eval_plan=plan,
        eval_run_id=plan["eval_run_id"],
        plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode()).hexdigest(),
        population_eligible=True,
    )
    request = replay_request_for_candidate(candidate)
    body = request.to_dict()
    assert body["audit_label"] == REPLAY_AUDIT_LABEL
    assert body["kind"] == "replay_audit_request"
    assert body["k"] == plan["k"]
    assert body["scoring_policy"] == plan["scoring_policy"]

    result = {
        "schema_version": 1,
        "audit_label": REPLAY_AUDIT_LABEL,
        "kind": "replay_audit_result",
        "audit_id": request.audit_id,
        "submission_id": "1",
        "eval_run_id": plan["eval_run_id"],
        "replay_attempt": 1,
        "plan_sha256": request.plan_sha256,
        "trial_scores_by_task": {"task-a": [0.5, 1.0]},
    }
    assert replay_result_from_mapping(result).trial_scores_by_task["task-a"] == [0.5, 1.0]
    malformed = copy.deepcopy(result)
    malformed["audit_label"] = "unlabelled"
    with pytest.raises(ReplayAuditWireError):
        replay_result_from_mapping(malformed)

    tampered = copy.deepcopy(plan)
    tampered["k"] = 1
    with pytest.raises(ReplayAuditWireError):
        replay_request_for_candidate(
            AuditCandidate(
                "1",
                attested_score=0.75,
                eval_plan=tampered,
                eval_run_id=plan["eval_run_id"],
                plan_sha256=candidate.plan_sha256,
                population_eligible=True,
            )
        )


@pytest.mark.parametrize(
    "trial_scores_by_task",
    [
        {"task-a": [0.5, 1.0]},
        {"task-b": [0.5, 1.0], "task-a": [0.5, 1.0]},
        {"task-a": [0.5, 1.0], "task-b": [0.5]},
        {"task-a": [0.5, 1.0], "task-b": [0.5, 1.0], "task-extra": [1.0, 1.0]},
    ],
)
def test_replay_result_validation_requires_complete_ordered_selected_set(
    trial_scores_by_task,
) -> None:
    plan = _plan("eval-replay-result-shape-1")
    plan["selected_tasks"].append(
        {
            "task_id": "task-b",
            "image_ref": "registry.example/task-b@sha256:" + "9" * 64,
            "task_config_sha256": "a" * 64,
        }
    )
    plan = ew.validate_eval_plan(plan)
    candidate = AuditCandidate(
        "1",
        attested_score=0.75,
        eval_plan=plan,
        eval_run_id=plan["eval_run_id"],
        plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode()).hexdigest(),
        population_eligible=True,
    )
    request = replay_request_for_candidate(candidate)
    result = replay_result_from_mapping(
        {
            "schema_version": 1,
            "audit_label": REPLAY_AUDIT_LABEL,
            "kind": "replay_audit_result",
            "audit_id": request.audit_id,
            "submission_id": request.submission_id,
            "eval_run_id": request.eval_run_id,
            "replay_attempt": request.replay_attempt,
            "plan_sha256": request.plan_sha256,
            "trial_scores_by_task": trial_scores_by_task,
        }
    )
    with pytest.raises(ReplayAuditWireError):
        result.validate_against(request)


def test_plan_comparison_rejects_digest_mutation_and_uses_immutable_k() -> None:
    plan = _plan()
    candidate = AuditCandidate(
        "1",
        attested_score=0.75,
        eval_plan=plan,
        eval_run_id=plan["eval_run_id"],
        plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode()).hexdigest(),
        population_eligible=True,
    )
    matching = compare_replay_trials(
        candidate,
        {"task-a": [0.5, 1.0]},
        spec=AggregationSpec(),
        tolerance=0.2,
    )
    assert matching.delta == 0.0
    with pytest.raises(InvalidReplayTrialsError):
        compare_replay_trials(
            candidate,
            {"task-a": [0.5]},
            spec=AggregationSpec(),
            tolerance=0.2,
        )


async def test_replay_request_endpoint_exposes_only_sampled_plan(client, monkeypatch) -> None:
    plan = _plan("eval-replay-endpoint-1")
    candidate = AuditCandidate(
        "1",
        attested_score=0.75,
        eval_plan=plan,
        eval_run_id=plan["eval_run_id"],
        plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode()).hexdigest(),
        population_eligible=True,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.phala_attestation_enabled",
        True,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.attested_review_enabled",
        True,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.accepted_verified_replay_population",
        lambda *_args, **_kwargs: _async_value([candidate]),
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.replay_audit_sampler_from_settings",
        lambda _settings: SimpleNamespace(sample=lambda _candidates: ["1"]),
    )

    response = await client.get(
        "/internal/v1/replay-audits/requests",
        headers={
            "Authorization": "Bearer test-token",
            "X-Base-Challenge-Slug": "agent-challenge",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["requests"]) == 1
    assert body["requests"][0]["eval_plan"] == plan
    assert body["requests"][0]["k"] == 2


async def _async_value(value):
    return value


async def test_mismatch_dispute_is_idempotent_and_does_not_change_job(
    database_session, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.replay_audit.verified_review_assignment_for_submission",
        _review_allow,
    )
    plan = _plan("eval-dispute-1")
    async with database_session() as session:
        _submission, job, _run = await _seed_population_row(session, plan=plan)
        await session.commit()

    candidate = AuditCandidate(
        "1",
        attested_score=0.75,
        eval_plan=plan,
        eval_run_id=plan["eval_run_id"],
        plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode()).hexdigest(),
        population_eligible=True,
    )
    comparison = compare_replay_trials(
        candidate,
        {"task-a": [0.0, 0.0]},
        spec=AggregationSpec(),
        tolerance=0.2,
    )
    assert comparison.flagged
    async with database_session() as session:
        first = await persist_replay_dispute(
            session,
            candidate=candidate,
            comparison=comparison,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
        second = await persist_replay_dispute(session, candidate=candidate, comparison=comparison)
        await session.commit()
        disputes = (await session.scalars(select(ReplayAuditDispute))).all()
        persisted_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.id == job.id)
        )
    assert first is not None
    assert second is not None
    assert first.id == second.id
    assert len(disputes) == 1
    assert persisted_job.score == pytest.approx(0.75)
