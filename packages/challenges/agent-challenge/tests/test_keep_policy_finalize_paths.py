"""Keep-good-scoring-tasks JOB policy consistency across ALL job-finalize paths.

Architecture C5 names ``validator_executor.finalize_job_if_complete`` the
decentralized scoring mainline, but the same job score is also produced by two
other sites:

* ``runner.run_evaluation_job`` -- the legacy COMBINED-runner path (analyzer +
  all tasks + finalize in one process, still the ``python -m ...evaluation.worker``
  default), and
* ``reconciler._mark_job_completed_from_attempt`` -- SINGLE-ATTEMPT recovery of a
  terminal-bench task that completed but was never finalized.

Both historically computed the plain mean over the task scores. With the default
policy ``off`` that is byte-identical to the mainline, but once a non-off keep
policy is configured the three paths must NOT diverge (miners finalized via
different paths would otherwise be scored on different bases). These tests pin
that every path applies the SAME settings-driven keep policy while keeping
``total_tasks``/``passed_tasks`` over the FULL selected set (the anti-gaming
eligibility gate is never shrunk), and that ``off`` stays byte-identical.

The two discriminator tests (``drop-lowest-n`` through the combined runner,
``threshold-band`` through the reconciler) FAIL against the legacy plain-mean
implementation and PASS once the policy is applied.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

# Reuse the reconciler / mainline seeding helpers (repo convention: sibling test
# modules import each other by bare name -- see test_combined_worker, etc.).
from test_finalize_keep_policy import _seed_job_with_scores
from test_reconciler import (
    _mark_submission_tb_running,
    _submission_and_job,
    _terminal_bench_task,
    _write_trial,
)

from agent_challenge.evaluation import create_evaluation_job, run_evaluation_job
from agent_challenge.evaluation.own_runner.keep_policy import keep_good_job_score
from agent_challenge.evaluation.own_runner.reward import floats_bit_identical
from agent_challenge.evaluation.reconciler import run_reconciler_once
from agent_challenge.evaluation.terminal_bench import create_terminal_bench_attempt
from agent_challenge.evaluation.validator_executor import finalize_job_if_complete
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.swe_forge import SweForgeTask


class _ScoredSweForgeExecutor:
    """SWE-forge executor mapping each task to a returncode (0 => score 1.0)."""

    def __init__(self, returncodes_by_task: dict[str, int]) -> None:
        self._returncodes = returncodes_by_task
        self.specs: list = []

    def run(self, spec, timeout_seconds: int) -> DockerRunResult:
        self.specs.append(spec)
        task = spec.labels["base.task"]
        if task == "analyzer":
            return DockerRunResult(container_name="analyzer", stdout="ok", stderr="", returncode=0)
        return DockerRunResult(
            container_name="fake",
            stdout=f"ran {task}",
            stderr="",
            returncode=self._returncodes.get(task, 0),
        )


class _ValidReport:
    rules_version = "rules-test"
    overall_verdict = "valid"
    reason_codes = ["rules_passed"]

    def to_json_compatible(self) -> dict[str, object]:
        return {
            "rules_version": self.rules_version,
            "overall_verdict": self.overall_verdict,
            "reason_codes": self.reason_codes,
        }


async def _run_combined_swe_forge_job(
    database_session,
    monkeypatch,
    tmp_path,
    *,
    returncodes_by_task: dict[str, int],
    **policy_overrides: object,
):
    """Finalize a job through the legacy COMBINED runner and return its summary."""

    agent_dir = tmp_path / f"agent-{uuid.uuid4().hex[:8]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    task_ids = list(returncodes_by_task)
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [
            SweForgeTask(task_id=tid, docker_image=f"baseintelligence/swe-forge:{tid}")
            for tid in task_ids
        ],
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.evaluation_task_count", len(task_ids)
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_concurrency", 1)
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend", "swe_forge"
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.run_rules_analyzer",
        lambda _workspace, *, reviewer=None: _ValidReport(),
    )
    for key, value in policy_overrides.items():
        monkeypatch.setattr(f"agent_challenge.evaluation.runner.settings.{key}", value)

    executor = _ScoredSweForgeExecutor(returncodes_by_task)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"hotkey-{uuid.uuid4().hex[:8]}",
            name="agent-kp",
            agent_hash=f"kp-{uuid.uuid4().hex[:8]}",
            artifact_uri=str(agent_dir),
        )
        session.add(submission)
        await session.flush()
        job = await create_evaluation_job(session, submission)
        summary = await run_evaluation_job(session, job.job_id, executor=executor)
    return summary


async def _finalize_mainline(database_session, monkeypatch, tmp_path, *, scores, **overrides):
    """Finalize a multi-task job through the decentralized mainline."""

    monkeypatch.setattr(
        "agent_challenge.evaluation.validator_executor.settings",
        ChallengeSettings(**overrides),
    )
    async with database_session() as session:
        job_id = await _seed_job_with_scores(session, scores=scores, tmp_path=tmp_path)
        await session.commit()
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    return summary


# --------------------------------------------------------------------------- #
# Combined runner (runner.run_evaluation_job)
# --------------------------------------------------------------------------- #
async def test_combined_runner_applies_keep_policy(database_session, monkeypatch, tmp_path):
    # Three tasks score [1.0, 1.0, 0.0]; drop-lowest-1 keeps the top two -> 1.0,
    # NOT the legacy plain mean 2/3. (Red against the plain-mean implementation.)
    returncodes = {"kp-task-0": 0, "kp-task-1": 0, "kp-task-2": 1}
    summary = await _run_combined_swe_forge_job(
        database_session,
        monkeypatch,
        tmp_path,
        returncodes_by_task=returncodes,
        keep_good_tasks_policy="drop-lowest-n",
        keep_good_tasks_drop_lowest=1,
    )
    assert summary is not None
    assert summary.status == "completed"
    assert floats_bit_identical(summary.score, 1.0)
    # Discriminator: the legacy plain mean over all three tasks is 2/3.
    assert not floats_bit_identical(summary.score, 2.0 / 3.0)
    # Anti-gaming: the eligibility gate stays over the FULL selected set.
    assert summary.total_tasks == 3
    assert summary.passed_tasks == 2


async def test_combined_runner_policy_off_is_byte_identical_plain_mean(
    database_session, monkeypatch, tmp_path
):
    returncodes = {"kp-task-0": 0, "kp-task-1": 0, "kp-task-2": 1}
    summary = await _run_combined_swe_forge_job(
        database_session,
        monkeypatch,
        tmp_path,
        returncodes_by_task=returncodes,
        keep_good_tasks_policy="off",
    )
    # off (default) == the legacy plain mean over all tasks.
    assert floats_bit_identical(summary.score, 2.0 / 3.0)
    assert summary.total_tasks == 3
    assert summary.passed_tasks == 2


# --------------------------------------------------------------------------- #
# Reconciler single-attempt recovery (reconciler._mark_job_completed_from_attempt)
# --------------------------------------------------------------------------- #
async def test_reconciler_recovery_applies_keep_policy(database_session, monkeypatch, tmp_path):
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root", str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.reconciler.settings.keep_good_tasks_policy", "threshold-band"
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.reconciler.settings.keep_good_tasks_threshold", 0.5
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, task=task)
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 0.4)
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert summary.terminal_bench_finalized == 1
    async with database_session() as session:
        job = (await session.execute(select(EvaluationJob))).scalar_one()
    assert job.status == "completed"
    # threshold-band(0.5) over the single recovered task score 0.4 -> below the
    # band -> defined 0.0, NOT the legacy plain single-task score 0.4. (Red.)
    assert floats_bit_identical(job.score, 0.0)
    assert not floats_bit_identical(job.score, 0.4)
    # Anti-gaming: eligibility gate stays over the full (single-task) set.
    assert job.total_tasks == 1
    assert job.passed_tasks == 0


async def test_reconciler_recovery_policy_off_is_byte_identical(
    database_session, monkeypatch, tmp_path
):
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root", str(tmp_path)
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, task=task)
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 0.4)
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert summary.terminal_bench_finalized == 1
    async with database_session() as session:
        job = (await session.execute(select(EvaluationJob))).scalar_one()
    # off (default): job score is the plain single-task score, byte-identical.
    assert floats_bit_identical(job.score, 0.4)
    assert job.total_tasks == 1
    assert job.passed_tasks == 0


# --------------------------------------------------------------------------- #
# Cross-path consistency: combined runner and mainline agree under the SAME policy
# --------------------------------------------------------------------------- #
async def test_combined_runner_and_mainline_agree_under_nonoff_policy(
    database_session, monkeypatch, tmp_path
):
    scores = [1.0, 1.0, 0.0]
    expected = keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=1)

    runner_summary = await _run_combined_swe_forge_job(
        database_session,
        monkeypatch,
        tmp_path,
        returncodes_by_task={"kp-task-0": 0, "kp-task-1": 0, "kp-task-2": 1},
        keep_good_tasks_policy="drop-lowest-n",
        keep_good_tasks_drop_lowest=1,
    )
    mainline_summary = await _finalize_mainline(
        database_session,
        monkeypatch,
        tmp_path,
        scores=scores,
        keep_good_tasks_policy="drop-lowest-n",
        keep_good_tasks_drop_lowest=1,
    )

    assert floats_bit_identical(runner_summary.score, expected)
    assert mainline_summary is not None
    assert floats_bit_identical(mainline_summary.score, expected)
    # The two independent finalize paths produce the SAME policy-applied score ...
    assert floats_bit_identical(runner_summary.score, mainline_summary.score)
    # ... while both keep the eligibility gate over the full selected set.
    assert runner_summary.total_tasks == mainline_summary.total_tasks == 3
    assert runner_summary.passed_tasks == mainline_summary.passed_tasks == 2
