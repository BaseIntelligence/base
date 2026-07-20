"""Hard legacy-parity contract for the M5 variance / keep-policy scoring layer.

Pins the mission's backward-compatibility invariants for the
``scoring-legacy-parity`` feature:

* VAL-SCORE-013 -- ``k == 1`` + keep policy ``off`` (the defaults) yields a
  finalized JOB score byte-identical to today's legacy scoring:
  ``score = sum(per-task 0/1 scores)/total_tasks`` and
  ``passed = count(task score >= 1.0)`` (epsilon=0 harbor math).
* VAL-SCORE-014 -- keep policy ``off`` with ``k > 1`` reproduces the legacy
  ``n_attempts = k`` behaviour exactly (per-task = harbor mean over the k trials,
  job = mean over tasks); the keep-policy layer is inert when off.
* VAL-SCORE-015 -- the aggregation preserves epsilon=0 harbor parity: an
  independent reimplementation of the aggregation using stock harbor 0.13.1
  reward math (mean via CPython list ``sum``/``len``, preserved trial order, no
  ``fsum``/``Decimal``, nan-aware) reproduces the finalized task scores and job
  score bit-for-bit.

The M5 behaviour is already landed, so these are regression/contract tests. Each
assertion is paired with a DISCRIMINATOR proving non-vacuity: a plausible wrong
implementation (``math.fsum`` last-ULP drift, a shrunk denominator, or best-of-k
substituted for the mean) is shown to DIVERGE from the produced output, so the
byte-equality assertions cannot pass trivially.

These assert over the real scoring aggregation surfaces only:
* per-task trial aggregation -- ``own_runner.variance.aggregate_task_scores``,
* the keep-good-tasks JOB layer -- ``own_runner.keep_policy.keep_good_job_score``,
* the DB job-finalize path -- ``validator_executor.finalize_job_if_complete``.
"""

from __future__ import annotations

import math
import statistics
import uuid
from collections import OrderedDict

import pytest
from sqlalchemy import select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.own_runner.keep_policy import keep_good_job_score
from agent_challenge.evaluation.own_runner.reward import (
    Mean,
    floats_bit_identical,
    reward_values_equal,
)
from agent_challenge.evaluation.own_runner.variance import aggregate_task_scores
from agent_challenge.evaluation.validator_executor import finalize_job_if_complete
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.config import ChallengeSettings


# --------------------------------------------------------------------------- #
# Independent legacy harbor 0.13.1 reward math (deliberately NOT delegating to
# the production module: this is the byte-for-byte reference the produced scores
# must match). Stock harbor ``Mean`` is a CPython left-to-right list ``sum`` then
# a single division -- NEVER ``math.fsum`` / ``statistics.mean`` / ``Decimal``,
# any of which drift in the last ULP and would break epsilon=0 parity.
# --------------------------------------------------------------------------- #
def _harbor_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _legacy_task_score(trial_scores: list[float]) -> float:
    """Legacy own_runner per-task score = harbor mean over the k trial scores."""

    return _harbor_mean([float(score) for score in trial_scores])


def _legacy_job_score(task_scores: list[float]) -> float:
    """Legacy job score = harbor mean over the per-task scores (no tasks dropped)."""

    return _harbor_mean([float(score) for score in task_scores])


def _legacy_passed(task_scores: list[float]) -> int:
    """Legacy passed count = number of tasks whose score is >= 1.0."""

    return sum(1 for score in task_scores if score >= 1.0)


# --------------------------------------------------------------------------- #
# DB finalize seeding (compact, self-contained). Creates a running job whose
# selected tasks already carry terminal per-task results, then finalizes it.
# --------------------------------------------------------------------------- #
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


async def _seed_job_with_scores(session, *, scores: list[float], tmp_path) -> str:
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


def _use_defaults(monkeypatch) -> None:
    """Pin the finalize path to the default settings (keep policy OFF)."""

    monkeypatch.setattr(
        "agent_challenge.evaluation.validator_executor.settings",
        ChallengeSettings(),
    )


async def _finalize(database_session, *, scores, tmp_path):
    async with database_session() as session:
        job_id = await _seed_job_with_scores(session, scores=scores, tmp_path=tmp_path)
        await session.commit()
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    return summary, job


# =========================================================================== #
# VAL-SCORE-013: k=1 + policy=off is a byte-identical legacy job score
# =========================================================================== #
@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 0.0, 1.0, 1.0],  # 3 of 4 pass
        [0.0, 0.0, 1.0, 0.0, 1.0],  # 2 of 5 pass
        [1.0],  # single-task all-pass
        [0.0, 0.0, 0.0],  # all-fail
    ],
)
async def test_finalize_k1_policy_off_is_byte_identical_legacy(
    database_session, monkeypatch, tmp_path, scores
):
    # k=1 => each task's single 0/1 trial IS its per-task score (seeded directly).
    _use_defaults(monkeypatch)  # keep policy OFF (the default)
    summary, job = await _finalize(database_session, scores=scores, tmp_path=tmp_path)

    assert summary is not None
    assert summary.status == "completed"
    # Job score is byte-identical to the independent legacy recompute.
    assert floats_bit_identical(job.score, _legacy_job_score(scores))
    assert floats_bit_identical(summary.score, _legacy_job_score(scores))
    # Passed-count is count(task score >= 1.0); total is the FULL task count.
    assert job.passed_tasks == _legacy_passed(scores)
    assert summary.passed_tasks == _legacy_passed(scores)
    assert job.total_tasks == len(scores)
    assert summary.total_tasks == len(scores)


def test_k1_per_task_score_equals_the_single_trial_score() -> None:
    # k=1: the default per-task mean over one trial is exactly that trial score,
    # so the whole k=1 pipeline reduces to the legacy single-trial scoring.
    trial_by_task = OrderedDict(t0=[1.0], t1=[0.0], t2=[1.0])
    per_task = aggregate_task_scores(trial_by_task, mode="mean")
    for task, trials in trial_by_task.items():
        assert floats_bit_identical(per_task[task], float(trials[0]))
    produced = keep_good_job_score(list(per_task.values()), policy="off")
    assert floats_bit_identical(produced, _legacy_job_score([1.0, 0.0, 1.0]))


async def test_discriminator_off_keeps_every_task_in_the_denominator(
    database_session, monkeypatch, tmp_path
):
    # Non-vacuity: a wrong "off" that shrank the denominator (e.g. dropped the
    # lowest task) would change the score. Prove off keeps ALL tasks.
    scores = [1.0, 0.0]
    _use_defaults(monkeypatch)
    _summary, job = await _finalize(database_session, scores=scores, tmp_path=tmp_path)
    assert floats_bit_identical(job.score, 0.5)  # mean over BOTH tasks
    # A drop-lowest-1 denominator (the plausible wrong impl) would score 1.0.
    dropped = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=1)
    assert floats_bit_identical(dropped, 1.0)
    assert not floats_bit_identical(job.score, dropped)


# =========================================================================== #
# VAL-SCORE-014: policy=off with k>1 equals the legacy n_attempts mean
# =========================================================================== #
@pytest.mark.parametrize(
    "trials_by_task",
    [
        # Binary trials (k=3): per-task mean then job mean == legacy n_attempts=3.
        OrderedDict(a=[1.0, 1.0, 0.0], b=[0.0, 0.0, 0.0], c=[1.0, 1.0, 1.0]),
        # Fractional trials (k=2 and k=4 mixed widths within a job).
        OrderedDict(x=[0.4, 0.6, 0.5], y=[1.0, 0.0], z=[0.25, 0.75, 0.5, 1.0]),
    ],
)
def test_policy_off_kgt1_equals_legacy_n_attempts_mean(trials_by_task) -> None:
    # Production pipeline: per-task harbor mean over the k trials, then the
    # keep-policy=OFF job layer (mean over tasks).
    per_task = aggregate_task_scores(trials_by_task, mode="mean")  # default mode
    produced_job = keep_good_job_score(list(per_task.values()), policy="off")

    # Independent legacy n_attempts=k recompute.
    legacy_per_task = {task: _legacy_task_score(trials) for task, trials in trials_by_task.items()}
    legacy_job = _legacy_job_score(list(legacy_per_task.values()))

    for task in trials_by_task:
        assert floats_bit_identical(per_task[task], legacy_per_task[task])
    assert floats_bit_identical(produced_job, legacy_job)


async def test_finalize_policy_off_kgt1_matches_legacy_end_to_end(
    database_session, monkeypatch, tmp_path
):
    # End-to-end: the k>1 per-task means feed the DB finalize path (policy off),
    # whose job score must equal the legacy mean over the per-task means.
    trials_by_task = OrderedDict(
        a=[1.0, 0.0, 1.0],
        b=[0.5, 0.6, 0.4],
        c=[0.0, 0.0, 0.0],
        d=[1.0, 1.0, 1.0],
    )
    per_task = aggregate_task_scores(trials_by_task, mode="mean")
    task_scores = list(per_task.values())

    _use_defaults(monkeypatch)
    _summary, job = await _finalize(database_session, scores=task_scores, tmp_path=tmp_path)

    legacy_job = _legacy_job_score(
        [_legacy_task_score(trials) for trials in trials_by_task.values()]
    )
    assert floats_bit_identical(job.score, legacy_job)
    # passed = tasks whose aggregated mean is a full 1.0 (only the unanimous one).
    assert job.passed_tasks == _legacy_passed(task_scores) == 1
    assert job.total_tasks == 4


def test_discriminator_off_is_mean_not_best_of_k() -> None:
    # Non-vacuity: for a flaky task the mean (used when off) differs from the max
    # (best-of-k). Prove the off path uses the mean, not the best trial.
    trials_by_task = OrderedDict(flaky=[0.0, 1.0, 0.0])
    mean_per_task = aggregate_task_scores(trials_by_task, mode="mean")["flaky"]
    max_per_task = aggregate_task_scores(trials_by_task, mode="best-of-k")["flaky"]
    assert floats_bit_identical(mean_per_task, _legacy_task_score([0.0, 1.0, 0.0]))
    assert floats_bit_identical(max_per_task, 1.0)
    assert not floats_bit_identical(mean_per_task, max_per_task)


# =========================================================================== #
# VAL-SCORE-015: aggregation preserves epsilon=0 harbor parity (bit-for-bit)
# =========================================================================== #
@pytest.mark.parametrize(
    "trials_by_task",
    [
        # Default-mean binary fixture.
        OrderedDict(a=[1.0, 0.0, 1.0], b=[1.0, 1.0, 1.0], c=[0.0, 0.0, 0.0]),
        # Non-trivial fractional-trial fixture (last-ULP sensitive).
        OrderedDict(
            p=[0.1, 0.2, 0.3],
            q=[0.7, 0.1, 0.9, 0.3],
            r=[1.0 / 3.0, 2.0 / 3.0],
        ),
    ],
)
def test_aggregation_bit_for_bit_matches_independent_harbor(trials_by_task) -> None:
    produced_task = aggregate_task_scores(trials_by_task, mode="mean")
    produced_job = keep_good_job_score(list(produced_task.values()), policy="off")

    # Independent harbor recompute (list sum/len, preserved trial order).
    expected_task = {task: _legacy_task_score(t) for task, t in trials_by_task.items()}
    expected_job = _legacy_job_score(list(expected_task.values()))

    for task in trials_by_task:
        assert floats_bit_identical(produced_task[task], expected_task[task])
    assert floats_bit_identical(produced_job, expected_job)
    # The produced per-task score is single-sourced from harbor 0.13.1 ``Mean``.
    for task, trials in trials_by_task.items():
        assert floats_bit_identical(produced_task[task], float(Mean.aggregate(list(trials))))


def test_harbor_parity_is_nan_aware() -> None:
    # An FP-nondeterministic (nan) trial propagates to a nan per-task score, and
    # the parity comparator treats nan == nan (IEEE-754 would say False).
    trials_by_task = OrderedDict(good=[1.0, 1.0], bad=[float("nan"), 0.0])
    produced_task = aggregate_task_scores(trials_by_task, mode="mean")
    expected_task = {task: _legacy_task_score(t) for task, t in trials_by_task.items()}
    for task in trials_by_task:
        assert reward_values_equal(produced_task[task], expected_task[task])
    assert math.isnan(produced_task["bad"])
    assert reward_values_equal(produced_task["good"], 1.0)


def test_discriminator_epsilon_zero_rejects_statistics_mean_drift() -> None:
    # Non-vacuity: the bit-for-bit assertions above are only meaningful because
    # harbor's plain list ``sum``/``len`` differs from the most plausible "wrong"
    # implementation, ``statistics.mean`` (exact-rational, rounds once at the end),
    # in the last ULP. Prove the produced mean matches plain ``sum``/``len`` and
    # NOT the statistics variant, so an epsilon-tolerant swap would be caught.
    fixture = [0.1, 0.2, 0.3]
    produced = aggregate_task_scores(OrderedDict(t=fixture), mode="mean")["t"]
    plain = sum(fixture) / len(fixture)  # 0.19999999999999998
    exact = statistics.mean(fixture)  # 0.2
    # The fixture is genuinely last-ULP divergent (guards against a no-op test).
    assert not floats_bit_identical(plain, exact)
    assert floats_bit_identical(produced, plain)
    assert not floats_bit_identical(produced, exact)
