"""Tests for the own-runner trial/job orchestrator (Task 15).

The orchestrator owns the Task -> Trial -> Job multiplicity that stock harbor
implements in ``harbor/job.py`` + ``harbor/trial/queue.py``:

* run ``k`` trials per task (pass@k; harbor ``--n-attempts``/``-k``, default 1);
* bound concurrency (harbor ``--n-concurrent``/``-n``, default 4) with an
  ``asyncio.Semaphore`` -- never more than ``n_concurrent`` trials in flight;
* aggregate the per-trial results into a single job result, reusing the Task-9
  reward path (pass@k + metric aggregation) and the Task-8 ``result_schema``
  benchmark-result derivation;
* resume/lock so completed trials are never re-run (no double-count) and a
  resume with a different config is rejected.

Layers (mirrors the sibling own-runner test modules):

* **Pure orchestration** (no docker): an injected stub ``trial_runner`` records
  concurrency / which trials actually executed, so we can pin the semaphore
  bound, the pass@k aggregation (cross-checked against the REAL harbor 0.13.1
  wheel where available), and resume/lock behaviour deterministically.
* **Composition** (no docker): ``driver_verifier_trial_runner`` wired to a REAL
  Task-13 :class:`AgentDriver` (injected fake agent class) + an injected
  verifier seam, proving the drive -> verify -> map ordering and the
  ``VerifierOutcome`` -> ``TrialOutcome`` mapping.
* **Docker integration** (``@docker_required``): the full orchestrator with the
  real ``run_verifier`` against throwaway ``python:3.12-slim`` containers,
  k=2 trials, oracle pass.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner.driver import AgentDriver
from agent_challenge.evaluation.own_runner.orchestrator import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_N_ATTEMPTS,
    DEFAULT_N_CONCURRENT,
    TRIAL_TIMEOUT_REASON_CODE,
    JobConfig,
    JobResult,
    OrchestratorLockError,
    TaskSpec,
    TrialId,
    TrialJobOrchestrator,
    TrialOutcome,
    driver_verifier_trial_runner,
    plan_trials,
)
from agent_challenge.evaluation.own_runner.verifier_runner import VerifierOutcome

# ===========================================================================
# Defaults parity with harbor JobConfig
# ===========================================================================


def test_defaults_match_harbor() -> None:
    # harbor JobConfig: n_attempts=1, n_concurrent_trials=4, retry.max_retries=0.
    assert DEFAULT_N_ATTEMPTS == 1
    assert DEFAULT_N_CONCURRENT == 4
    assert DEFAULT_MAX_RETRIES == 0
    config = JobConfig()
    assert config.n_attempts == 1
    assert config.n_concurrent == 4
    assert config.max_retries == 0


# ===========================================================================
# Trial planning (k trials per task; harbor nesting: attempt outer, task inner)
# ===========================================================================


def test_plan_trials_k_per_task_attempt_outer() -> None:
    tasks = [TaskSpec("t1"), TaskSpec("t2")]
    plan = plan_trials(tasks, n_attempts=3)
    # k trials per task -> 3 * 2 == 6 trials.
    assert len(plan) == 6
    # attempt is the OUTER loop, task the inner (harbor job.py _init_trial_configs).
    assert [(t.task_name, t.attempt) for t in plan] == [
        ("t1", 0),
        ("t2", 0),
        ("t1", 1),
        ("t2", 1),
        ("t1", 2),
        ("t2", 2),
    ]


def test_plan_trials_default_k_is_one() -> None:
    plan = plan_trials([TaskSpec("only")])
    assert len(plan) == 1
    assert plan[0].task_name == "only"
    assert plan[0].attempt == 0


def test_trial_name_is_stable_and_fs_safe() -> None:
    # Trial names must be deterministic (resume matching) and filesystem-safe.
    a = TrialId("caffe/cifar 10", 2).trial_name
    b = TrialId("caffe/cifar 10", 2).trial_name
    assert a == b
    assert "/" not in a
    assert " " not in a
    assert "2" in a


# ===========================================================================
# Helpers
# ===========================================================================


def _outcome(
    task_name: str,
    attempt: int,
    reward: float | int | None,
    *,
    errored: bool = False,
    reason_code: str | None = None,
    source: str | None = None,
) -> TrialOutcome:
    rewards = None if reward is None else {"reward": reward}
    return TrialOutcome(
        task_name=task_name,
        trial_name=TrialId(task_name, attempt).trial_name,
        status="failed" if errored else "completed",
        rewards=rewards,
        reason_code=reason_code,
        errored=errored,
        source=source,
    )


def _make_runner(reward_map: dict[tuple[str, int], float | int | None]):
    """A deterministic stub trial runner driven by a (task, attempt)->reward map."""

    async def _run(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        reward = reward_map[(trial_id.task_name, trial_id.attempt)]
        return _outcome(trial_id.task_name, trial_id.attempt, reward, source=task.source)

    return _run


# ===========================================================================
# S1: pass@k aggregation == harbor rule
# ===========================================================================


async def test_passk_aggregation_three_trials_one_pass(tmp_path: Path) -> None:
    # k=3 trials of one task, rewards [1, 0, 0], none errored.
    runner = _make_runner({("task", 0): 1, ("task", 1): 0, ("task", 2): 0})
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=3, n_concurrent=4),
        job_dir=tmp_path / "job",
        trial_runner=runner,
    )
    result = await orch.run([TaskSpec("task")])

    assert isinstance(result, JobResult)
    # none errored -> job status completed.
    assert result.status == "completed"
    assert result.reason_code is None
    assert result.n_total_trials == 3
    assert result.n_errored_trials == 0
    # flat-mean score = mean([1,0,0]) = 1/3 ; resolved = round(1/3 * 3) = round(1.0) = 1.
    assert result.score == pytest.approx(1 / 3)
    assert result.resolved == 1
    assert result.total == 3
    # pass@k: min_trials=3 -> k_values=[2]; pass@2(n=3,c=1) = 1 - C(2,2)/C(3,2).
    assert result.pass_at_k == {"agent__adhoc": {2: 0.6666666666666667}}
    # benchmark result is the Task-8 schema shape, sorted keys, five fields.
    assert set(result.benchmark_result) == {
        "status",
        "score",
        "resolved",
        "total",
        "reason_code",
    }
    assert result.benchmark_result["resolved"] == 1


async def test_passk_default_k1_is_empty(tmp_path: Path) -> None:
    # Default -k 1 -> min_trials=1 -> k_values=[] -> pass_at_k {} (harbor rule).
    runner = _make_runner({("task", 0): 1})
    orch = TrialJobOrchestrator(
        config=JobConfig(),  # n_attempts defaults to 1
        job_dir=tmp_path / "job",
        trial_runner=runner,
    )
    result = await orch.run([TaskSpec("task")])
    assert result.pass_at_k == {}
    assert result.status == "completed"
    assert result.score == 1.0
    assert result.resolved == 1
    assert result.total == 1


async def test_passk_matches_real_harbor_wheel(tmp_path: Path) -> None:
    # Cross-check our aggregation against the REAL harbor 0.13.1 wheel if present.
    harbor_passk = pytest.importorskip("harbor.utils.pass_at_k")
    from harbor.models.trial.result import (  # type: ignore[import-not-found]
        AgentInfo,
        TrialResult,
    )

    # 4 trials, 2 pass -> min_trials=4 -> k=[2,4].
    rewards = [1, 0, 1, 0]
    runner = _make_runner({("task", i): r for i, r in enumerate(rewards)})
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=4, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=runner,
    )
    result = await orch.run([TaskSpec("task")])

    # Build the equivalent harbor TrialResults and compute pass@k with the wheel.
    def _harbor_trial(reward: int) -> Any:
        return TrialResult.model_construct(
            task_name="task",
            source=None,
            agent_info=AgentInfo(name="agent", version="0", model_info=None),
            verifier_result=type("VR", (), {"rewards": {"reward": reward}})(),
        )

    harbor_groups = harbor_passk.compute_pass_at_k_by_evals([_harbor_trial(r) for r in rewards])
    # Our pass@k must equal harbor's, ε=0.
    assert result.pass_at_k == harbor_groups


# ===========================================================================
# S2: bounded concurrency (never more than n_concurrent in flight)
# ===========================================================================


async def test_concurrency_never_exceeds_bound(tmp_path: Path) -> None:
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def _run(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            # Force overlap so the bound is actually exercised.
            await asyncio.sleep(0.02)
        finally:
            async with lock:
                in_flight -= 1
        return _outcome(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=4, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=_run,
    )
    # 3 tasks * 4 attempts = 12 trials; with n_concurrent=2 the peak must be 2.
    result = await orch.run([TaskSpec("a"), TaskSpec("b"), TaskSpec("c")])

    assert peak == 2  # actually reaches the bound...
    assert peak <= 2  # ...and never exceeds it.
    assert orch.peak_in_flight == peak
    assert result.n_total_trials == 12
    assert len(result.trial_outcomes) == 12


async def test_concurrency_single_when_bound_is_one(tmp_path: Path) -> None:
    in_flight = 0
    peak = 0

    async def _run(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _outcome(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=3, n_concurrent=1),
        job_dir=tmp_path / "job",
        trial_runner=_run,
    )
    await orch.run([TaskSpec("a")])
    assert peak == 1


# ===========================================================================
# S3: resume/lock -- completed trials skipped, no double-count
# ===========================================================================


async def test_resume_skips_finished_trials_no_double_count(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    config = JobConfig(n_attempts=3, n_concurrent=1)
    tasks = [TaskSpec("task")]
    reward_map = {("task", 0): 1, ("task", 1): 0, ("task", 2): 1}

    # First run: complete 2 trials then a hard process kill on the 3rd. A real
    # kill (SIGKILL) is NOT a catchable ``Exception`` -- an ordinary trial
    # exception now folds to a fail-closed errored outcome (VAL-ORCH-030/036), so
    # resume must be exercised with a genuine ``BaseException`` that models the
    # orchestrator process going away mid-trial.
    first_executed: list[str] = []

    class _SimulatedProcessKill(BaseException):
        pass

    async def _crashing(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        if len(first_executed) >= 2:
            raise _SimulatedProcessKill("simulated kill mid-run")
        first_executed.append(trial_id.trial_name)
        return _outcome(
            trial_id.task_name, trial_id.attempt, reward_map[(trial_id.task_name, trial_id.attempt)]
        )

    orch1 = TrialJobOrchestrator(config=config, job_dir=job_dir, trial_runner=_crashing)
    with pytest.raises(_SimulatedProcessKill, match="simulated kill"):
        await orch1.run(tasks)
    assert len(first_executed) == 2  # exactly 2 persisted before the crash

    # Resume: a fresh orchestrator on the same job_dir only runs the unfinished trial.
    second_executed: list[str] = []

    async def _resume_runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        second_executed.append(trial_id.trial_name)
        return _outcome(
            trial_id.task_name, trial_id.attempt, reward_map[(trial_id.task_name, trial_id.attempt)]
        )

    orch2 = TrialJobOrchestrator(config=config, job_dir=job_dir, trial_runner=_resume_runner)
    result = await orch2.run(tasks)

    # Only the trials NOT finished in run 1 were executed on resume.
    assert len(second_executed) == 1
    assert set(second_executed).isdisjoint(set(first_executed))
    # No double-count: exactly 3 trial outcomes total, each trial once.
    assert result.n_total_trials == 3
    assert len(result.trial_outcomes) == 3
    names = [o.trial_name for o in result.trial_outcomes]
    assert len(names) == len(set(names))
    # Aggregation correct over the full set: mean([1,0,1]) = 2/3.
    assert result.score == pytest.approx(2 / 3)


async def test_resume_with_different_config_is_rejected(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    runner = _make_runner({("task", 0): 1})
    orch1 = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1), job_dir=job_dir, trial_runner=runner
    )
    await orch1.run([TaskSpec("task")])

    # Resuming with an incompatible config (different k) must be rejected (harbor
    # raises FileExistsError on a config/lock mismatch).
    orch2 = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2), job_dir=job_dir, trial_runner=runner
    )
    with pytest.raises(OrchestratorLockError):
        await orch2.run([TaskSpec("task")])


async def test_resume_identical_config_after_full_completion_is_idempotent(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    config = JobConfig(n_attempts=2, n_concurrent=2)
    reward_map = {("task", 0): 1, ("task", 1): 0}

    executed: list[str] = []

    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        executed.append(trial_id.trial_name)
        return _outcome(
            trial_id.task_name, trial_id.attempt, reward_map[(trial_id.task_name, trial_id.attempt)]
        )

    orch1 = TrialJobOrchestrator(config=config, job_dir=job_dir, trial_runner=_runner)
    first = await orch1.run([TaskSpec("task")])
    assert len(executed) == 2

    # Re-running the already-complete job re-runs NOTHING and yields the same result.
    orch2 = TrialJobOrchestrator(config=config, job_dir=job_dir, trial_runner=_runner)
    second = await orch2.run([TaskSpec("task")])
    assert len(executed) == 2  # no new executions
    assert second.score == first.score
    assert second.n_total_trials == 2
    assert len(second.trial_outcomes) == 2


# ===========================================================================
# Aggregation edges: errored trials, multi-task pass@k
# ===========================================================================


async def test_errored_trial_makes_job_failed_and_counts_zero(tmp_path: Path) -> None:
    # 2 trials: one clean reward 1, one errored (reward None). Errored -> job failed.
    async def _run(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        if trial_id.attempt == 1:
            return _outcome(
                trial_id.task_name,
                trial_id.attempt,
                None,
                errored=True,
                reason_code="harbor_reward_missing",
            )
        return _outcome(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=_run,
    )
    result = await orch.run([TaskSpec("task")])

    assert result.n_errored_trials == 1
    assert result.status == "failed"  # any errored trial -> job failed
    # errored trial contributes 0 to the mean: mean([1, 0]) = 0.5.
    assert result.score == 0.5
    # pass@k still computed; None reward counts as a failure (success 0), k=[2].
    # successes for the task: [1, 0] -> c=1, n=2 -> pass@2(2,1) = 1 - C(1,2)/C(2,2) = 1.0.
    assert result.pass_at_k == {"agent__adhoc": {2: 1.0}}


async def test_multitask_passk_mean_over_tasks(tmp_path: Path) -> None:
    # 2 tasks, k=2. task A: [1,1] (c=2), task B: [0,0] (c=0).
    reward_map = {
        ("A", 0): 1,
        ("A", 1): 1,
        ("B", 0): 0,
        ("B", 1): 0,
    }
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2, n_concurrent=4),
        job_dir=tmp_path / "job",
        trial_runner=_make_runner(reward_map),
    )
    result = await orch.run([TaskSpec("A"), TaskSpec("B")])
    # pass@2: A(n=2,c=2)=1.0 ; B(n=2,c=0)=0.0 ; mean over 2 tasks = 0.5.
    assert result.pass_at_k == {"agent__adhoc": {2: 0.5}}


async def test_fractional_rewards_disable_passk(tmp_path: Path) -> None:
    # A fractional reward disables pass@k for the whole group (harbor rule).
    reward_map = {("task", 0): 0.5, ("task", 1): 1}
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=_make_runner(reward_map),
    )
    result = await orch.run([TaskSpec("task")])
    assert result.pass_at_k == {}
    # score still aggregates the metric mean: mean([0.5, 1]) = 0.75.
    assert result.score == 0.75


async def test_empty_task_list_yields_empty_job(tmp_path: Path) -> None:
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=3),
        job_dir=tmp_path / "job",
        trial_runner=_make_runner({}),
    )
    result = await orch.run([])
    assert result.n_total_trials == 0
    assert result.trial_outcomes == []
    assert result.pass_at_k == {}
    assert result.status == "completed"
    assert result.score == 0.0


# ===========================================================================
# Per-trial backstop timeout (asyncio.wait_for) -- guarantees gather completes
# ===========================================================================


def test_default_trial_timeout_sec_formula() -> None:
    from agent_challenge.evaluation.own_runner.orchestrator import (
        DEFAULT_AGENT_TIMEOUT_SEC,
        DEFAULT_TRIAL_BUILD_SLACK_SEC,
        default_trial_timeout_sec,
    )
    from agent_challenge.evaluation.own_runner.verifier_runner import (
        DEFAULT_VERIFIER_TIMEOUT_SEC,
    )

    # Explicit per-task budgets win: agent + verifier + build slack.
    assert default_trial_timeout_sec(agent_sec=100, verifier_sec=20, build_slack_sec=5) == 125.0
    # Missing budgets fall back to the conservative module defaults.
    assert default_trial_timeout_sec() == float(
        DEFAULT_AGENT_TIMEOUT_SEC + DEFAULT_VERIFIER_TIMEOUT_SEC + DEFAULT_TRIAL_BUILD_SLACK_SEC
    )


async def test_trial_backstop_timeout_errors_trial_and_job_finalizes(tmp_path: Path) -> None:
    # A trial runner that never returns must not wedge the job: the per-trial
    # asyncio.wait_for backstop aborts it into an errored outcome so gather
    # completes and the job still aggregates to a valid result.
    async def _hang(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        await asyncio.sleep(30)  # far beyond the tiny backstop below
        return _outcome(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1, n_concurrent=1),
        job_dir=tmp_path / "job",
        trial_runner=_hang,
        trial_timeout_sec=0.05,
    )
    result = await orch.run([TaskSpec("task")])

    assert result.n_total_trials == 1
    assert len(result.trial_outcomes) == 1
    outcome = result.trial_outcomes[0]
    assert outcome.errored is True
    assert outcome.status == "failed"
    assert outcome.reason_code == TRIAL_TIMEOUT_REASON_CODE
    assert outcome.rewards is None
    # Any errored trial -> job failed, but it DID finalize (aggregated, not hung).
    assert result.status == "failed"
    assert result.score == 0.0
    assert result.total == 1
    # The timed-out trial is persisted durably (resume-safe).
    trial_dir = tmp_path / "job" / "trials" / TrialId("task", 0).trial_name
    assert (trial_dir / "result.json").is_file()


async def test_trial_backstop_lets_healthy_trials_finish(tmp_path: Path) -> None:
    # A single hung trial must not take down its siblings: gather still completes
    # with the healthy trial completed and only the hung one errored.
    async def _mixed(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        if trial_id.attempt == 0:
            await asyncio.sleep(30)  # hangs -> backstop aborts it
        return _outcome(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=_mixed,
        trial_timeout_sec=0.1,
    )
    result = await orch.run([TaskSpec("task")])

    assert result.n_total_trials == 2
    assert result.n_errored_trials == 1
    assert sorted(o.status for o in result.trial_outcomes) == ["completed", "failed"]
    # The job still finalizes (any errored trial -> failed) with a valid aggregate.
    assert result.status == "failed"


# ===========================================================================
# Composition: driver_verifier_trial_runner (real Task-13 driver + verifier seam)
# ===========================================================================


class _FakeEnvironment:
    """Minimal recording exec-bridge stand-in (duck-typed TerminalEnvironment)."""

    def __init__(self) -> None:
        self.removed = False
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs: Any) -> Any:
        self.commands.append(command)
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": None})()

    def remove(self) -> None:
        self.removed = True


class _OracleAgent:
    """A trivial agent whose setup/run succeed (drive -> completed)."""

    def __init__(self, *, logs_dir: Any = None, model_name: Any = None, **kwargs: Any) -> None:
        self.logs_dir = logs_dir
        self.model_name = model_name

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        return "DONE"


class _CrashAgent:
    def __init__(self, *, logs_dir: Any = None, model_name: Any = None, **kwargs: Any) -> None:
        pass

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        raise RuntimeError("boom")


async def test_driver_verifier_runner_maps_pass(tmp_path: Path) -> None:
    driver = AgentDriver(agent_class=_OracleAgent)
    env = _FakeEnvironment()

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> Any:
        from agent_challenge.evaluation.own_runner.orchestrator import PreparedTrial

        return PreparedTrial(
            environment=env,
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        # Verifier ran on the SAME environment the agent used (still alive).
        assert environment is env
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
        )

    runner = driver_verifier_trial_runner(
        driver=driver, preparer=_preparer, verifier=_fake_verifier
    )
    outcome = await runner(TrialId("task", 0), TaskSpec("task"))

    assert outcome.status == "completed"
    assert outcome.errored is False
    assert outcome.rewards == {"reward": 1.0}
    assert outcome.reason_code is None
    # The environment is torn down by the runner AFTER verification.
    assert env.removed is True


async def test_driver_verifier_runner_agent_crash_skips_verifier(tmp_path: Path) -> None:
    driver = AgentDriver(agent_class=_CrashAgent)
    env = _FakeEnvironment()
    verifier_called = False

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> Any:
        from agent_challenge.evaluation.own_runner.orchestrator import PreparedTrial

        return PreparedTrial(
            environment=env,
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        nonlocal verifier_called
        verifier_called = True
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1},
        )

    runner = driver_verifier_trial_runner(
        driver=driver, preparer=_preparer, verifier=_fake_verifier
    )
    outcome = await runner(TrialId("task", 0), TaskSpec("task"))

    # Agent crashed -> trial errored, verifier NOT run, env still torn down.
    assert outcome.status == "failed"
    assert outcome.errored is True
    assert outcome.reason_code == "harbor_trial_failed"
    assert outcome.rewards is None
    assert verifier_called is False
    assert env.removed is True


async def test_orchestrator_with_driver_verifier_runner_end_to_end(tmp_path: Path) -> None:
    # Full orchestration over the REAL driver + an injected verifier seam, k=2.
    driver = AgentDriver(agent_class=_OracleAgent)

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> Any:
        from agent_challenge.evaluation.own_runner.orchestrator import PreparedTrial

        return PreparedTrial(
            environment=_FakeEnvironment(),
            instruction="go",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1},
        )

    runner = driver_verifier_trial_runner(
        driver=driver, preparer=_preparer, verifier=_fake_verifier
    )
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=runner,
    )
    result = await orch.run([TaskSpec("task")])
    assert result.status == "completed"
    assert result.score == 1.0
    assert result.resolved == 2
    assert result.total == 2
    # k=2 binary all-pass -> pass@2(n=2,c=2) = 1.0.
    assert result.pass_at_k == {"agent__adhoc": {2: 1.0}}


# ===========================================================================
# Task 17: capture the agent's OWN output as a distinct stream=agent artifact
# ===========================================================================


class _MarkerAgent:
    """An agent whose ``run`` returns a distinctive marker summary string.

    The driver surfaces this as ``AgentRunResult.output``; Task 17 must capture
    it as the agent's OWN output (the source of the ``stream=agent`` channel),
    kept entirely separate from harness / install / verifier (test) output.
    """

    MARKER = "AGENT-OWN-OUTPUT::caffe-marker-7f3a"

    def __init__(self, *, logs_dir: Any = None, model_name: Any = None, **kwargs: Any) -> None:
        self.logs_dir = logs_dir
        self.model_name = model_name

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        return self.MARKER


async def test_driver_verifier_runner_captures_agent_output(tmp_path: Path) -> None:
    # The driver's AgentRunResult.output (the agent's OWN returned summary) must
    # be carried on the TrialOutcome so the orchestrator can persist it as the
    # agent stream -- NOT discarded as it was before Task 17.
    driver = AgentDriver(agent_class=_MarkerAgent)
    env = _FakeEnvironment()

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> Any:
        from agent_challenge.evaluation.own_runner.orchestrator import PreparedTrial

        return PreparedTrial(
            environment=env,
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
        )

    runner = driver_verifier_trial_runner(
        driver=driver, preparer=_preparer, verifier=_fake_verifier
    )
    outcome = await runner(TrialId("task", 0), TaskSpec("task"))

    assert outcome.status == "completed"
    assert outcome.agent_output == _MarkerAgent.MARKER


async def test_driver_verifier_runner_agent_crash_has_no_agent_output(tmp_path: Path) -> None:
    # On an agent crash the driver yields output=None -> no agent_output captured
    # (an errored trial produces no agent stream, only a harness-level reason).
    driver = AgentDriver(agent_class=_CrashAgent)

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> Any:
        from agent_challenge.evaluation.own_runner.orchestrator import PreparedTrial

        return PreparedTrial(
            environment=_FakeEnvironment(),
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        raise AssertionError("verifier must not run after an agent crash")

    runner = driver_verifier_trial_runner(
        driver=driver, preparer=_preparer, verifier=_fake_verifier
    )
    outcome = await runner(TrialId("task", 0), TaskSpec("task"))

    assert outcome.status == "failed"
    assert outcome.errored is True
    assert outcome.agent_output is None


async def test_persist_trial_writes_agent_log_under_agent_dir(tmp_path: Path) -> None:
    # _persist_trial must write the captured agent output to <trial_dir>/agent/
    # so the unchanged host-side seam (_separated_log_refs -> agent_log_files ->
    # record_separated_trial_logs stream="agent") picks it up for own-runner.
    async def _noop_runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        raise AssertionError("runner is never invoked in this test")

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1),
        job_dir=tmp_path / "job",
        trial_runner=_noop_runner,
    )
    trial_id = TrialId("caffe/cifar 10", 0)
    outcome = TrialOutcome(
        task_name="caffe/cifar 10",
        trial_name=trial_id.trial_name,
        status="completed",
        rewards={"reward": 1.0},
        agent_output=_MarkerAgent.MARKER,
    )

    orch._persist_trial(trial_id, outcome)

    trial_dir = tmp_path / "job" / "trials" / trial_id.trial_name
    agent_log = trial_dir / "agent" / "agent.log"
    assert agent_log.is_file()
    assert agent_log.read_text() == _MarkerAgent.MARKER
    # result.json still written alongside (lean, agent_output not embedded in it).
    assert (trial_dir / "result.json").is_file()


async def test_persist_trial_writes_no_agent_dir_without_output(tmp_path: Path) -> None:
    # No agent output (errored / empty) -> no agent/ dir is created (the host seam
    # only emits a stream=agent event when agent files exist).
    async def _noop_runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        raise AssertionError("runner is never invoked in this test")

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1),
        job_dir=tmp_path / "job",
        trial_runner=_noop_runner,
    )
    trial_id = TrialId("task", 0)
    outcome = TrialOutcome(
        task_name="task",
        trial_name=trial_id.trial_name,
        status="failed",
        rewards=None,
        errored=True,
        agent_output=None,
    )

    orch._persist_trial(trial_id, outcome)

    trial_dir = tmp_path / "job" / "trials" / trial_id.trial_name
    assert (trial_dir / "result.json").is_file()
    assert not (trial_dir / "agent").exists()


def test_to_dict_excludes_agent_output_keeping_result_json_lean() -> None:
    # The agent output lives in agent/agent.log on disk (harbor-faithful: logs in
    # files, not result.json). It must NOT bloat the per-trial result.json, and a
    # round-trip through to_dict/from_dict must not depend on it.
    outcome = TrialOutcome(
        task_name="task",
        trial_name="task__attempt-0",
        status="completed",
        rewards={"reward": 1.0},
        agent_output=_MarkerAgent.MARKER,
    )
    serialized = outcome.to_dict()
    assert "agent_output" not in serialized
    assert _MarkerAgent.MARKER not in repr(serialized)

    restored = TrialOutcome.from_dict(serialized)
    # The persisted record carries no agent_output (it is reconstructed from the
    # on-disk agent.log, which survives a resume independently).
    assert restored.agent_output is None
    assert restored.task_name == "task"
    assert restored.rewards == {"reward": 1.0}


# ===========================================================================
# Docker integration: full orchestrator with the real run_verifier
# ===========================================================================

_IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", _IMAGE], capture_output=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


docker_required = pytest.mark.skipif(
    not _docker_ready(), reason=f"docker + {_IMAGE} image required for orchestrator container tests"
)

_BINARY_TEST_SH = """#!/bin/bash
if [ -f /app/solved ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def _make_tests_dir(tmp_path: Path) -> Path:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_BINARY_TEST_SH)
    test_sh.chmod(0o755)
    return tests_dir


@docker_required
async def test_orchestrator_docker_oracle_passk(tmp_path: Path) -> None:
    from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
    from agent_challenge.evaluation.own_runner.orchestrator import PreparedTrial
    from agent_challenge.evaluation.own_runner.verifier_runner import run_verifier

    tests_dir = _make_tests_dir(tmp_path)

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> PreparedTrial:
        env = DockerExecEnvironment.launch(_IMAGE, network="host")
        # Oracle: solve the task before the agent runs so the verifier writes 1.
        await env.exec("touch /app/solved", user="root")
        return PreparedTrial(
            environment=env,
            instruction="solve it",
            tests_source_dir=tests_dir,
            start_session=False,
        )

    driver = AgentDriver(agent_class=_OracleAgent)
    runner = driver_verifier_trial_runner(driver=driver, preparer=_preparer, verifier=run_verifier)
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=2, n_concurrent=2),
        job_dir=tmp_path / "job",
        trial_runner=runner,
    )
    result = await orch.run([TaskSpec("task")])

    assert result.status == "completed"
    assert result.score == 1.0
    assert result.resolved == 2
    assert result.total == 2
    assert result.pass_at_k == {"agent__adhoc": {2: 1.0}}
