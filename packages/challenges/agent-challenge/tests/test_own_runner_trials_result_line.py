"""Behavioral tests for k-trials + the single fail-closed result line.

These pin the ``orchestrator-trials-and-result-line`` feature's validation
assertions (see ``validation-contract.md``), exercised through the observable
surfaces only (trial launch counts, the aggregated ``JobResult``, the persisted
per-trial ``result.json``, and the emitted ``BASE_BENCHMARK_RESULT=`` line):

* VAL-ORCH-025 -- exactly ``k = n_attempts`` trials run per task, aggregated.
* VAL-ORCH-026 -- ``k=1`` reproduces byte-identical legacy single-trial scoring.
* VAL-ORCH-027 -- a persisted trial is resume-safe (never re-launched).
* VAL-ORCH-028 -- exactly one parseable ``BASE_BENCHMARK_RESULT=`` line.
* VAL-ORCH-029 -- a backend crash still emits one fail-closed (score 0) line.
* VAL-ORCH-030 -- a single task-container crash yields a parseable per-task
  failed result and the job still finalizes with siblings intact.
* VAL-ORCH-036 -- a single crashed trial among ``k`` folds to a fail-closed 0
  (denominator preserved) without hanging or corrupting sibling trials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner.orchestrator import (
    TRIAL_CRASH_REASON_CODE,
    JobConfig,
    TaskSpec,
    TrialId,
    TrialJobOrchestrator,
    TrialOutcome,
)
from agent_challenge.evaluation.own_runner.reason_codes import is_known_reason_code
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
    derive_benchmark_result_from_stats,
)


def _completed(task_name: str, attempt: int, reward: float | int) -> TrialOutcome:
    return TrialOutcome(
        task_name=task_name,
        trial_name=TrialId(task_name, attempt).trial_name,
        status="completed",
        rewards={"reward": reward},
    )


# ===========================================================================
# VAL-ORCH-025: exactly k = n_attempts trials per task, aggregated
# ===========================================================================
async def test_runs_exactly_k_trials_per_task_and_aggregates(tmp_path: Path) -> None:
    k = 3
    launches: list[tuple[str, int]] = []
    reward_map = {
        ("A", 0): 1,
        ("A", 1): 1,
        ("A", 2): 0,
        ("B", 0): 0,
        ("B", 1): 0,
        ("B", 2): 0,
    }

    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        launches.append((trial_id.task_name, trial_id.attempt))
        reward = reward_map[(trial_id.task_name, trial_id.attempt)]
        return _completed(trial_id.task_name, trial_id.attempt, reward)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=k, n_concurrent=4),
        job_dir=tmp_path / "job",
        trial_runner=_runner,
    )
    result = await orch.run([TaskSpec("A"), TaskSpec("B")])

    # Each task launched EXACTLY k trials -- no more, no fewer.
    per_task = {"A": 0, "B": 0}
    for name, _attempt in launches:
        per_task[name] += 1
    assert per_task == {"A": k, "B": k}

    # The k trials of each task are aggregated (not left as k separate results):
    # every attempt of each task appears exactly once in the outcome set.
    outcomes_by_task: dict[str, list[TrialOutcome]] = {"A": [], "B": []}
    for outcome in result.trial_outcomes:
        outcomes_by_task[outcome.task_name].append(outcome)
    assert len(outcomes_by_task["A"]) == k
    assert len(outcomes_by_task["B"]) == k
    assert result.n_total_trials == 2 * k
    # Aggregated per-task score folds into the single job score: mean over all
    # trials = mean([1,1,0, 0,0,0]) = 2/6.
    assert result.score == pytest.approx(2 / 6)


# ===========================================================================
# VAL-ORCH-026: k=1 reproduces byte-identical legacy single-trial scoring
# ===========================================================================
@pytest.mark.parametrize("reward", [1, 0])
async def test_k1_is_byte_identical_to_legacy_single_trial(tmp_path: Path, reward: int) -> None:
    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        return _completed(trial_id.task_name, trial_id.attempt, reward)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1, n_concurrent=1),
        job_dir=tmp_path / "job",
        trial_runner=_runner,
    )
    result = await orch.run([TaskSpec("task")])

    # The legacy own_runner single-trial derivation for the SAME reward: one
    # completed trial, mean metric == reward, no errored trials.
    legacy = derive_benchmark_result_from_stats(
        {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "evals": {"agent__adhoc": {"metrics": [{"mean": float(reward)}]}},
            },
        }
    )
    # Byte-identical: same dict AND same serialized wire form.
    assert result.benchmark_result == legacy
    assert json.dumps(result.benchmark_result, sort_keys=True) == json.dumps(legacy, sort_keys=True)
    # k=1 => no pass@k (harbor rule).
    assert result.pass_at_k == {}


# ===========================================================================
# VAL-ORCH-027: completed trials are resume-safe (never re-launched)
# ===========================================================================
async def test_persisted_trial_is_not_relaunched(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    config = JobConfig(n_attempts=2, n_concurrent=1)

    # Pre-seed a persisted result.json for attempt-0 exactly as _persist_trial
    # writes it (result.json presence == "this trial is done").
    seeded = TrialId("task", 0)
    seed_dir = job_dir / "trials" / seeded.trial_name
    seed_dir.mkdir(parents=True)
    seeded_outcome = TrialOutcome(
        task_name="task",
        trial_name=seeded.trial_name,
        status="completed",
        rewards={"reward": 1},
    )
    (seed_dir / "result.json").write_text(json.dumps(seeded_outcome.to_dict(), sort_keys=True))
    # Write the lock so the resume passes the config-fingerprint guard.
    (job_dir / "lock.json").write_text(json.dumps(config.fingerprint(), sort_keys=True))

    launched: list[str] = []

    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        launched.append(trial_id.trial_name)
        return _completed(trial_id.task_name, trial_id.attempt, 0)

    orch = TrialJobOrchestrator(config=config, job_dir=job_dir, trial_runner=_runner)
    result = await orch.run([TaskSpec("task")])

    # Only the MISSING trial (attempt-1) was launched; the persisted attempt-0
    # was never re-run.
    assert launched == [TrialId("task", 1).trial_name]
    assert result.n_total_trials == 2
    # The seeded trial's stored reward (1) is reused in the aggregate: mean([1,0]).
    assert result.score == pytest.approx(0.5)


# ===========================================================================
# VAL-ORCH-036: a single crashed trial among k folds to a fail-closed 0
# ===========================================================================
async def test_one_crashed_trial_folds_to_zero_denominator_preserved(tmp_path: Path) -> None:
    k = 3

    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        if trial_id.attempt == 1:
            raise RuntimeError("container exited non-zero mid-trial")
        return _completed(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=k, n_concurrent=3),
        job_dir=tmp_path / "job",
        trial_runner=_runner,
    )
    # The job FINALIZES (does not raise / hang) despite the crashed trial.
    result = await orch.run([TaskSpec("task")])

    # Denominator preserved: the crashed trial did not silently vanish.
    assert result.n_total_trials == k
    assert len(result.trial_outcomes) == k
    assert result.n_errored_trials == 1

    by_attempt = {o.trial_name: o for o in result.trial_outcomes}
    crashed = by_attempt[TrialId("task", 1).trial_name]
    # Crashed trial folds to a fail-closed 0 (never a fabricated / passing score).
    assert crashed.errored is True
    assert crashed.status == "failed"
    assert crashed.rewards is None
    assert is_known_reason_code(crashed.reason_code)
    # Sibling trials' recorded scores are intact.
    for attempt in (0, 2):
        sibling = by_attempt[TrialId("task", attempt).trial_name]
        assert sibling.rewards == {"reward": 1}
        assert sibling.errored is False
    # Aggregate mean folds the crashed trial as 0: mean([1, 0, 1]) = 2/3.
    assert result.score == pytest.approx(2 / 3)
    # Any errored trial -> job failed, but it DID finalize.
    assert result.status == "failed"
    # The crashed trial is persisted durably (resume-safe, not re-run later).
    crashed_dir = tmp_path / "job" / "trials" / TrialId("task", 1).trial_name
    assert (crashed_dir / "result.json").is_file()


async def test_crashed_trial_uses_generic_crash_reason_when_unclassified(tmp_path: Path) -> None:
    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        raise ValueError("unclassified boom")

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1, n_concurrent=1),
        job_dir=tmp_path / "job",
        trial_runner=_runner,
    )
    result = await orch.run([TaskSpec("task")])

    assert result.n_total_trials == 1
    outcome = result.trial_outcomes[0]
    assert outcome.errored is True
    assert outcome.reason_code == TRIAL_CRASH_REASON_CODE


async def test_crashed_trial_prefers_carried_reason_code(tmp_path: Path) -> None:
    class _CodedError(RuntimeError):
        reason_code = "harbor_nonzero_exit"

    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        raise _CodedError("nonzero")

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1, n_concurrent=1),
        job_dir=tmp_path / "job",
        trial_runner=_runner,
    )
    result = await orch.run([TaskSpec("task")])
    assert result.trial_outcomes[0].reason_code == "harbor_nonzero_exit"


# ===========================================================================
# VAL-ORCH-030: a single task-container crash still finalizes the job with
# the OTHER tasks' results intact
# ===========================================================================
async def test_one_task_crash_does_not_abort_other_tasks(tmp_path: Path) -> None:
    async def _runner(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        if trial_id.task_name == "crasher":
            raise RuntimeError("task container crashed")
        return _completed(trial_id.task_name, trial_id.attempt, 1)

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1, n_concurrent=4),
        job_dir=tmp_path / "job",
        trial_runner=_runner,
    )
    # The job finalizes (does not raise) even though one task crashed.
    result = await orch.run([TaskSpec("healthy1"), TaskSpec("crasher"), TaskSpec("healthy2")])

    assert result.n_total_trials == 3
    by_task = {o.task_name: o for o in result.trial_outcomes}
    # Crashed task -> parseable failed per-task result (score 0), not a hang.
    assert by_task["crasher"].errored is True
    assert by_task["crasher"].status == "failed"
    assert by_task["crasher"].rewards is None
    # Sibling tasks' results are intact.
    assert by_task["healthy1"].rewards == {"reward": 1}
    assert by_task["healthy2"].rewards == {"reward": 1}
    # Job finalized as failed (one errored trial) with a valid aggregate.
    assert result.status == "failed"
    # Each task's per-trial result.json is persisted (parseable).
    for name in ("healthy1", "crasher", "healthy2"):
        result_json = tmp_path / "job" / "trials" / TrialId(name, 0).trial_name / "result.json"
        assert result_json.is_file()
        parsed = json.loads(result_json.read_text())
        assert parsed["task_name"] == name


# ===========================================================================
# VAL-ORCH-028 / VAL-ORCH-029: exactly one parseable BASE_BENCHMARK_RESULT= line
# (fail-closed on crash)
# ===========================================================================
def _result_lines(stdout: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len(RESULT_LINE_PREFIX) :])
        for line in stdout.splitlines()
        if line.startswith(RESULT_LINE_PREFIX)
    ]


def test_exactly_one_well_formed_result_line(monkeypatch, tmp_path, capsys) -> None:
    from agent_challenge.evaluation import own_runner_backend
    from agent_challenge.evaluation.own_runner.orchestrator import JobResult

    canned = JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )

    async def _fake_run(**kwargs: Any) -> JobResult:
        return canned

    monkeypatch.setattr(own_runner_backend, "run_own_runner_job", _fake_run)

    rc = own_runner_backend.main(
        ["run", "--task", "t", "--job-dir", str(tmp_path / "job"), "--n-attempts", "1"]
    )

    assert rc == 0
    parsed = _result_lines(capsys.readouterr().out)
    # Exactly one result line, carrying at least the five core fields.
    assert len(parsed) == 1
    assert {"status", "score", "resolved", "total", "reason_code"} <= set(parsed[0])
    assert parsed[0]["status"] == "completed"


def test_backend_crash_emits_one_failclosed_line(monkeypatch, tmp_path, capsys) -> None:
    from agent_challenge.evaluation import own_runner_backend

    async def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("injected mid-run crash")

    monkeypatch.setattr(own_runner_backend, "run_own_runner_job", _boom)

    rc = own_runner_backend.main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])

    # Fail-closed: nonzero exit AND exactly one parseable failed line (score 0,
    # a reason code) -- never a missing line and never a passing score.
    assert rc != 0
    parsed = _result_lines(capsys.readouterr().out)
    assert len(parsed) == 1
    assert parsed[0]["status"] == "failed"
    assert parsed[0]["score"] == 0.0
    assert parsed[0]["resolved"] == 0
    assert is_known_reason_code(parsed[0]["reason_code"])
