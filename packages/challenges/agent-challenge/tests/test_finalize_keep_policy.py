"""Keep-good-scoring-tasks policy applied through the real job-finalize path.

End-to-end behavioral contract for the ``keep-good-tasks-policy`` feature over
:func:`finalize_job_if_complete` and :func:`is_reward_eligible_job`, anchored to
the mission validation assertions:

* VAL-SCORE-006 -- drop-lowest-N job score = mean over the surviving tasks.
* VAL-SCORE-008 -- drop-lowest-N with N=0 = legacy mean over all tasks.
* VAL-SCORE-009 -- threshold-band keeps only at/above-threshold tasks.
* VAL-SCORE-010 -- threshold-band all-below finalizes with job score 0.0 and the
  submission is NOT reward-eligible.
* VAL-SCORE-012 -- the keep policy NEVER shrinks the reward-eligibility
  task-count gate: ``total_tasks`` stays the full selected count regardless of
  how many tasks the policy retains for the score (anti-gaming).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.own_runner.reward import floats_bit_identical
from agent_challenge.evaluation.validator_executor import finalize_job_if_complete
from agent_challenge.evaluation.weights import is_reward_eligible_job
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.config import ChallengeSettings


def _tasks(count: int) -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id=f"terminal-bench/task-{index}",
            docker_image=f"ghcr.io/baseintelligence/terminal-bench-runner:{index}",
            prompt=f"task {index}",
            benchmark="terminal_bench",
            metadata={"task_id": f"terminal-bench/task-{index}"},
        )
        for index in range(count)
    ]


async def _seed_job_with_scores(session, *, scores: list[float], tmp_path):
    """Create a running job whose selected tasks already have terminal results."""

    agent_hash = uuid.uuid4().hex[:12]
    agent_dir = tmp_path / agent_hash
    agent_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(len(scores))
    submission = AgentSubmission(
        miner_hotkey=f"hotkey-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=str(agent_dir),
        status="evaluating",
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
        total_tasks=len(tasks),
        verdict="valid",
    )
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


def _use_policy(monkeypatch, **overrides) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.validator_executor.settings",
        ChallengeSettings(**overrides),
    )


async def _finalize(database_session, *, scores, tmp_path, **overrides):
    async with database_session() as session:
        job_id = await _seed_job_with_scores(session, scores=scores, tmp_path=tmp_path)
        await session.commit()
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    return summary, job


# --------------------------------------------------------------------------- #
# VAL-SCORE-006 -- drop-lowest-N through finalize
# --------------------------------------------------------------------------- #
async def test_finalize_drop_lowest_n1(database_session, monkeypatch, tmp_path):
    _use_policy(monkeypatch, keep_good_tasks_policy="drop-lowest-n", keep_good_tasks_drop_lowest=1)
    summary, job = await _finalize(database_session, scores=[1.0, 1.0, 0.5, 0.0], tmp_path=tmp_path)
    assert summary is not None
    assert summary.status == "completed"
    assert floats_bit_identical(job.score, 0.8333333333333334)
    # Anti-gaming: the eligibility denominator is untouched by the score policy.
    assert job.total_tasks == 4
    assert job.passed_tasks == 2


# --------------------------------------------------------------------------- #
# VAL-SCORE-008 -- N=0 equals the legacy mean over all tasks
# --------------------------------------------------------------------------- #
async def test_finalize_drop_lowest_n0_is_legacy_mean(database_session, monkeypatch, tmp_path):
    scores = [1.0, 0.0, 0.6666666666666666, 0.5]
    _use_policy(monkeypatch, keep_good_tasks_policy="drop-lowest-n", keep_good_tasks_drop_lowest=0)
    _summary, job = await _finalize(database_session, scores=scores, tmp_path=tmp_path)
    assert floats_bit_identical(job.score, sum(scores) / len(scores))
    assert job.total_tasks == 4


async def test_finalize_off_matches_drop_lowest_n0(database_session, monkeypatch, tmp_path):
    scores = [0.3, 0.9, 0.1, 0.8]
    _use_policy(monkeypatch, keep_good_tasks_policy="off")
    _s1, off_job = await _finalize(database_session, scores=scores, tmp_path=tmp_path)
    _use_policy(monkeypatch, keep_good_tasks_policy="drop-lowest-n", keep_good_tasks_drop_lowest=0)
    _s2, n0_job = await _finalize(database_session, scores=scores, tmp_path=tmp_path)
    assert floats_bit_identical(off_job.score, n0_job.score)


# --------------------------------------------------------------------------- #
# VAL-SCORE-009 -- threshold-band through finalize
# --------------------------------------------------------------------------- #
async def test_finalize_threshold_band(database_session, monkeypatch, tmp_path):
    _use_policy(monkeypatch, keep_good_tasks_policy="threshold-band", keep_good_tasks_threshold=0.5)
    _summary, job = await _finalize(
        database_session, scores=[1.0, 0.4, 0.5, 0.0], tmp_path=tmp_path
    )
    assert floats_bit_identical(job.score, 0.75)
    assert job.total_tasks == 4
    # Only task-0 scored >= 1.0.
    assert job.passed_tasks == 1


# --------------------------------------------------------------------------- #
# VAL-SCORE-010 -- all-below threshold -> defined 0.0, finalizes, NOT eligible
# --------------------------------------------------------------------------- #
async def test_finalize_threshold_band_all_below(database_session, monkeypatch, tmp_path):
    _use_policy(monkeypatch, keep_good_tasks_policy="threshold-band", keep_good_tasks_threshold=0.5)
    summary, job = await _finalize(database_session, scores=[0.4, 0.0, 0.49], tmp_path=tmp_path)
    assert summary is not None
    assert summary.status == "completed"
    assert job.score == 0.0
    assert job.passed_tasks == 0
    # Ineligible: no task passed on the full selected set.
    assert is_reward_eligible_job(job, job.total_tasks) is False


# --------------------------------------------------------------------------- #
# VAL-SCORE-012 -- keep policy cannot shrink the eligibility task-count gate
# --------------------------------------------------------------------------- #
async def test_keep_policy_does_not_shrink_eligibility_gate(
    database_session, monkeypatch, tmp_path
):
    # A policy that excludes tasks from the SCORE must not reduce the eligibility
    # denominator: a job evaluated on 3 tasks cannot qualify against a gate of 4.
    _use_policy(monkeypatch, keep_good_tasks_policy="drop-lowest-n", keep_good_tasks_drop_lowest=2)
    _summary, job = await _finalize(
        database_session, scores=[1.0, 1.0, 0.0, 0.0], tmp_path=tmp_path
    )
    # Score policy dropped the 2 lowest -> job score is mean(1.0, 1.0) = 1.0 ...
    assert floats_bit_identical(job.score, 1.0)
    # ... but the eligibility count still reflects ALL 4 selected tasks.
    assert job.total_tasks == 4
    assert job.passed_tasks == 2
    # Eligible against its true task count, but NOT against a larger required gate
    # (the policy did not let it masquerade as a full run of a bigger set).
    assert is_reward_eligible_job(job, 4) is True
    assert is_reward_eligible_job(job, 5) is False


async def test_eligibility_gate_uses_full_count_not_kept_count(
    database_session, monkeypatch, tmp_path
):
    # threshold-band keeps only 1 of 4 tasks for the score; eligibility must still
    # key on 4, so an under-count run can never qualify by excluding tasks.
    _use_policy(monkeypatch, keep_good_tasks_policy="threshold-band", keep_good_tasks_threshold=0.9)
    _summary, job = await _finalize(
        database_session, scores=[1.0, 0.5, 0.4, 0.3], tmp_path=tmp_path
    )
    assert floats_bit_identical(job.score, 1.0)  # only the 1.0 task kept
    assert job.total_tasks == 4  # NOT shrunk to the single kept task
    assert is_reward_eligible_job(job, 4) is True


# --------------------------------------------------------------------------- #
# Backward-compat: default (off) finalize is the legacy mean over all tasks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scores",
    [[1.0, 0.0], [1.0, 0.0, 0.6666666666666666, 0.5, 1.0]],
)
async def test_finalize_default_off_is_legacy_mean(database_session, monkeypatch, tmp_path, scores):
    _use_policy(monkeypatch)  # defaults: policy off
    _summary, job = await _finalize(database_session, scores=scores, tmp_path=tmp_path)
    assert floats_bit_identical(job.score, sum(scores) / len(scores))
    assert job.total_tasks == len(scores)
