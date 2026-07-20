"""Immutable Eval-plan scoring policy regressions.

These tests deliberately exercise the plan-derived scoring seam rather than
the mutable runtime settings used by fully legacy jobs.  They cover the policy
and count vectors required by VAL-SCORE-001..015 and prove that a persisted
attested plan is authoritative across job finalization and replay.
"""

from __future__ import annotations

import copy
import hashlib
import uuid

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.plan_scoring import (
    CanonicalPlanScoringError,
    build_score_record_from_eval_plan,
    persist_canonical_score_record,
    persist_direct_eval_result,
    persist_eval_plan,
    scoring_policy_from_settings,
    validate_eval_result_from_plan,
    validate_score_record_from_eval_plan,
)
from agent_challenge.evaluation.replay_audit import (
    AggregationSpec,
    AuditCandidate,
    audit_submission,
)
from agent_challenge.evaluation.validator_executor import finalize_job_if_complete
from agent_challenge.evaluation.weights import is_reward_eligible_job
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.config import ChallengeSettings

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b" * 96,
    "rtmr1": "c" * 96,
    "rtmr2": "d" * 96,
    "compose_hash": "e" * 64,
    "os_image_hash": "f" * 64,
}
AGENT_HASH = "1" * 64


def _policy(
    *,
    per_task_aggregation: str = "mean",
    keep_policy: str = "off",
    drop_lowest_n: int = 0,
    threshold_f64be: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "per_task_aggregation": per_task_aggregation,
        "keep_policy": keep_policy,
        "drop_lowest_n": drop_lowest_n,
        "threshold_f64be": threshold_f64be,
    }


def _plan(
    *,
    policy: dict[str, object],
    k: int,
    task_ids: tuple[str, ...] = ("task-a", "task-b", "task-c"),
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "eval_run_id": "eval-plan-score-001",
        "submission_id": "submission-score-001",
        "submission_version": 1,
        "authorizing_review_digest": "2" * 64,
        "agent_hash": AGENT_HASH,
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry.example/task@sha256:" + "3" * 64,
                "task_config_sha256": "4" * 64,
            }
            for task_id in task_ids
        ],
        "k": k,
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
        "result_endpoint": "/evaluation/v1/runs/eval-plan-score-001/result",
        "key_release_nonce": "key-nonce-score-001",
        "score_nonce": "score-nonce-score-001",
        "run_token_sha256": "7" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


@pytest.mark.parametrize(
    ("policy", "trials", "expected_aggregates", "expected_score"),
    [
        (
            _policy(),
            {"task-a": [1.0, 0.0, 1.0], "task-b": [0.0, 0.0, 0.0], "task-c": [1.0, 1.0, 1.0]},
            [2.0 / 3.0, 0.0, 1.0],
            (2.0 / 3.0 + 0.0 + 1.0) / 3.0,
        ),
        (
            _policy(per_task_aggregation="best_of_k"),
            {"task-a": [0.0, 1.0, 0.0], "task-b": [0.4, 0.6, 0.5], "task-c": [0.0, 0.0, 0.0]},
            [1.0, 0.6, 0.0],
            1.6 / 3.0,
        ),
        (
            _policy(keep_policy="drop_lowest_n", drop_lowest_n=1),
            {"task-a": [1.0], "task-b": [1.0], "task-c": [0.0]},
            [1.0, 1.0, 0.0],
            1.0,
        ),
        (
            _policy(keep_policy="threshold_band", threshold_f64be="3fe0000000000000"),
            {"task-a": [1.0], "task-b": [0.4], "task-c": [0.5]},
            [1.0, 0.4, 0.5],
            0.75,
        ),
    ],
)
def test_plan_reconstructs_ordered_trials_aggregates_and_final_score(
    policy, trials, expected_aggregates, expected_score
) -> None:
    plan = _plan(policy=policy, k=len(next(iter(trials.values()))))

    record = build_score_record_from_eval_plan(plan, trials)

    assert [ew.decode_score_f64be(task["aggregate_score_f64be"]) for task in record["tasks"]] == (
        expected_aggregates
    )
    assert ew.decode_score_f64be(record["final"]["job_score_f64be"]) == expected_score
    assert record["final"]["total_tasks"] == 3
    assert record["final"]["passed_tasks"] == sum(score == 1.0 for score in expected_aggregates)
    assert [task["passed_trials"] for task in record["tasks"]] == [
        sum(score == 1.0 for score in scores) for scores in trials.values()
    ]


def test_plan_clamps_drop_lowest_and_defines_empty_threshold_score() -> None:
    drop_plan = _plan(policy=_policy(keep_policy="drop_lowest_n", drop_lowest_n=3), k=1)
    dropped = build_score_record_from_eval_plan(
        drop_plan, {"task-a": [0.4], "task-b": [0.9], "task-c": [0.5]}
    )
    assert ew.decode_score_f64be(dropped["final"]["job_score_f64be"]) == 0.9

    threshold_plan = _plan(
        policy=_policy(keep_policy="threshold_band", threshold_f64be="3fe0000000000000"), k=1
    )
    empty = build_score_record_from_eval_plan(
        threshold_plan, {"task-a": [0.4], "task-b": [0.0], "task-c": [0.49]}
    )
    assert ew.decode_score_f64be(empty["final"]["job_score_f64be"]) == 0.0
    assert empty["final"]["passed_tasks"] == 0
    assert empty["final"]["total_tasks"] == 3


def test_plan_policy_and_score_record_mutations_fail_closed() -> None:
    plan = _plan(policy=_policy(), k=3)
    record = build_score_record_from_eval_plan(
        plan, {"task-a": [1.0, 0.0, 1.0], "task-b": [0.0, 0.0, 0.0], "task-c": [1.0, 1.0, 1.0]}
    )

    changed_policy = copy.deepcopy(plan)
    changed_policy["scoring_policy"]["per_task_aggregation"] = "best_of_k"
    with pytest.raises(CanonicalPlanScoringError):
        build_score_record_from_eval_plan(
            changed_policy,
            {"task-a": [1.0], "task-b": [0.0], "task-c": [1.0]},
        )

    wrong_trial_count = copy.deepcopy(record)
    wrong_trial_count["tasks"][0]["trial_scores_f64be"].pop()
    with pytest.raises(CanonicalPlanScoringError):
        validate_score_record_from_eval_plan(plan, wrong_trial_count)

    reduced_count = copy.deepcopy(record)
    reduced_count["final"]["total_tasks"] = 1
    with pytest.raises(CanonicalPlanScoringError):
        validate_score_record_from_eval_plan(plan, reduced_count)


async def _seed_plan_job(session, *, plan, scores, tmp_path) -> str:
    agent_dir = tmp_path / f"agent-{uuid.uuid4().hex}"
    agent_dir.mkdir()
    tasks = [
        BenchmarkTask(
            task_id=task["task_id"],
            docker_image=task["image_ref"],
            benchmark="terminal_bench",
            metadata={"task_id": task["task_id"]},
        )
        for task in plan["selected_tasks"]
    ]
    submission = AgentSubmission(
        miner_hotkey="plan-score-hotkey",
        name=f"plan-score-{uuid.uuid4().hex}",
        agent_hash=AGENT_HASH,
        artifact_uri=str(agent_dir),
        raw_status="tb_running",
        effective_status="evaluating",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=uuid.uuid4().hex,
        submission_id=submission.id,
        status="running",
        selected_tasks_json=benchmark_tasks_to_json(tasks),
    )
    persist_eval_plan(job, plan)
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    for task, score in zip(tasks, scores, strict=True):
        session.add(
            TaskResult(
                job_id=job.id,
                task_id=task.task_id,
                docker_image=task.docker_image,
                status="completed",
                score=score,
                returncode=0,
                stdout="",
                stderr="",
                duration_seconds=0.0,
            )
        )
    await session.flush()
    return job.job_id


async def test_mainline_finalizer_uses_persisted_plan_not_mutable_settings(
    database_session, monkeypatch, tmp_path
) -> None:
    plan = _plan(policy=_policy(keep_policy="drop_lowest_n", drop_lowest_n=1), k=1)
    async with database_session() as session:
        job_id = await _seed_plan_job(session, plan=plan, scores=[1.0, 1.0, 0.0], tmp_path=tmp_path)
        await session.commit()

    # A later config change must not change a plan already issued to the CVM.
    monkeypatch.setattr(
        "agent_challenge.evaluation.validator_executor.settings",
        ChallengeSettings(keep_good_tasks_policy="threshold-band", keep_good_tasks_threshold=1.0),
    )
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))

    assert summary is not None
    assert ew.encode_score_f64be(job.score) == ew.encode_score_f64be(1.0)
    assert (job.passed_tasks, job.total_tasks) == (2, 3)
    assert is_reward_eligible_job(job, 3) is True

    # The mutable count cannot shrink after finalization without invalidating the
    # exact plan-bound score record and reward eligibility.
    job.total_tasks = 1
    assert is_reward_eligible_job(job, 1) is False


async def test_mainline_finalizer_uses_persisted_k_trial_score_record(
    database_session, monkeypatch, tmp_path
) -> None:
    plan = _plan(policy=_policy(per_task_aggregation="best_of_k"), k=3)
    record = build_score_record_from_eval_plan(
        plan, {"task-a": [0.0, 1.0, 0.0], "task-b": [0.4, 0.6, 0.5], "task-c": [0.0, 0.0, 0.0]}
    )
    async with database_session() as session:
        job_id = await _seed_plan_job(session, plan=plan, scores=[1.0, 0.6, 0.0], tmp_path=tmp_path)
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        persist_canonical_score_record(job, record)
        await session.commit()

    monkeypatch.setattr(
        "agent_challenge.evaluation.validator_executor.settings",
        ChallengeSettings(per_task_aggregation="mean", keep_good_tasks_policy="off"),
    )
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()

    assert summary is not None
    assert ew.encode_score_f64be(summary.score) == ew.encode_score_f64be(1.6 / 3.0)
    assert (summary.passed_tasks, summary.total_tasks) == (1, 3)


def test_direct_result_validation_reconstructs_every_binding_from_plan() -> None:
    plan = _plan(policy=_policy(per_task_aggregation="best_of_k"), k=3)
    record = build_score_record_from_eval_plan(
        plan, {"task-a": [0.0, 1.0, 0.0], "task-b": [0.4, 0.6, 0.5], "task-c": [0.0, 0.0, 0.0]}
    )
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=[task["task_id"] for task in plan["selected_tasks"]],
    )
    request = {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": AGENT_HASH,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "8" * 64,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": "ab",
                "event_log": [],
                "report_data": ew.score_report_data_hex(binding),
                "measurement": {**MEASUREMENT, "rtmr3": "9" * 96},
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            },
        },
    }

    assert validate_eval_result_from_plan(plan, request)["score_record"] == record

    mismatched_measurement = copy.deepcopy(request)
    mismatched_measurement["execution_proof"]["attestation"]["measurement"]["compose_hash"] = (
        "a" * 64
    )
    with pytest.raises(CanonicalPlanScoringError):
        validate_eval_result_from_plan(plan, mismatched_measurement)

    mismatched_final = copy.deepcopy(request)
    mismatched_final["score_record"]["final"]["total_tasks"] = 1
    with pytest.raises(CanonicalPlanScoringError):
        validate_eval_result_from_plan(plan, mismatched_final)


async def test_direct_result_persistence_uses_the_persisted_plan(
    database_session, tmp_path
) -> None:
    plan = _plan(policy=_policy(), k=1)
    record = build_score_record_from_eval_plan(
        plan,
        {"task-a": [1.0], "task-b": [0.0], "task-c": [1.0]},
    )
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=[task["task_id"] for task in plan["selected_tasks"]],
    )
    request = {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": AGENT_HASH,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "8" * 64,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": "ab",
                "event_log": [],
                "report_data": ew.score_report_data_hex(binding),
                "measurement": {**MEASUREMENT, "rtmr3": "9" * 96},
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            },
        },
    }
    async with database_session() as session:
        job_id = await _seed_plan_job(session, plan=plan, scores=[1.0, 0.0, 1.0], tmp_path=tmp_path)
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        final = persist_direct_eval_result(job, request)
        await session.commit()

    assert (final.score, final.passed_tasks, final.total_tasks) == (2.0 / 3.0, 2, 3)
    assert (job.score, job.passed_tasks, job.total_tasks) == (2.0 / 3.0, 2, 3)


def test_issuance_materializes_hyphenated_legacy_settings_once() -> None:
    policy = scoring_policy_from_settings(
        ChallengeSettings(
            per_task_aggregation="best-of-k",
            keep_good_tasks_policy="threshold-band",
            keep_good_tasks_threshold=0.5,
        )
    )
    assert policy == _policy(
        per_task_aggregation="best_of_k",
        keep_policy="threshold_band",
        threshold_f64be="3fe0000000000000",
    )
    with pytest.raises(CanonicalPlanScoringError):
        scoring_policy_from_settings(ChallengeSettings(keep_good_tasks_policy="best-of-k"))


def test_replay_plan_derives_k_and_policy_from_the_same_immutable_bytes() -> None:
    plan = _plan(policy=_policy(per_task_aggregation="best_of_k"), k=3)
    record = build_score_record_from_eval_plan(
        plan, {"task-a": [0.0, 1.0, 0.0], "task-b": [0.4, 0.6, 0.5], "task-c": [0.0, 0.0, 0.0]}
    )
    score = ew.decode_score_f64be(record["final"]["job_score_f64be"])
    seen: list[int] = []

    def broker(_submission_id: str, *, k: int):
        seen.append(k)
        return {"task-a": [0.0, 1.0, 0.0], "task-b": [0.4, 0.6, 0.5], "task-c": [0.0, 0.0, 0.0]}

    candidate = AuditCandidate(
        "submission-score-001",
        attested_score=score,
        eval_plan=plan,
    )
    result = audit_submission(
        candidate,
        broker,
        spec=AggregationSpec(per_task_aggregation="mean", keep_policy="off"),
        tolerance=0.0,
    )

    assert seen == [3]
    assert result.replay_score == score
    assert result.flagged is False
