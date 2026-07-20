"""Agent-driving parity on a subset — Task 23.

Fast, docker-free unit tests pin the parity harness logic (record building from
both the stock-harbor ``result.json`` shape and the own-runner trial outcomes,
the argv builder, and the epsilon=0 diff) plus the recorded agent's transcript
driving. A final opt-in integration test (``RUN_AGENT_PARITY_DOCKER=1``) runs the
REAL both-path parity over a 3-task subset.

The committed suite stays harbor-free: it imports only the harbor-free
:mod:`recorded_parity_agent` and the harness (whose imports are harbor-free). The
harbor ``BaseAgent`` adapter (:mod:`recorded_harbor_agent`) is exercised only by
the harbor subprocess in the integration test, so harbor's later removal (Task
24) does not break collection.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import recorded_parity_agent as rpa  # noqa: E402

from agent_challenge.evaluation.own_runner.orchestrator import TrialOutcome  # noqa: E402
from tools.agent_parity_harness import (  # noqa: E402
    build_harbor_argv,
    diff_records,
    digest_key,
    harbor_records_from_run_dir,
    harbor_records_from_run_dirs,
    own_records_from_job_dir,
    own_records_from_outcomes,
    record_from_rewards,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _FakeExecResult:
    def __init__(self, return_code: int, stdout: str | None, stderr: str | None) -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _RecordingEnv:
    """Records every ``exec`` call so we can assert the driven command sequence."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def exec(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        return _FakeExecResult(return_code=0, stdout=f"out:{command}", stderr=None)


def _harbor_result(task_name: str, reward: float) -> dict:
    """A minimal stock-harbor ``result.json`` mapping (only fields the harness reads)."""
    return {
        "task_name": task_name,
        "exception_info": None,
        "verifier_result": {"rewards": {"reward": reward}},
    }


def _write_harbor_run_dir(root: Path, results: dict[str, float]) -> Path:
    run_dir = root / "2026-06-17__00-00-00"
    for i, (task_name, reward) in enumerate(results.items()):
        trial_dir = run_dir / f"{task_name.split('/')[-1]}__abc{i}"
        trial_dir.mkdir(parents=True)
        (trial_dir / "result.json").write_text(json.dumps(_harbor_result(task_name, reward)))
    return run_dir


def _write_own_job_dir(root: Path, results: dict[str, float]) -> Path:
    job_dir = root / "own-job"
    trials_dir = job_dir / "trials"
    for task_name, reward in results.items():
        # Mirror the orchestrator's filesystem-safe trial_name (no path separators).
        safe = task_name.replace("/", "_")
        outcome = TrialOutcome(
            task_name=task_name,
            trial_name=f"{safe}__attempt-0",
            status="completed",
            rewards={"reward": reward},
            reason_code=None,
            errored=False,
        )
        trial_dir = trials_dir / outcome.trial_name
        trial_dir.mkdir(parents=True)
        (trial_dir / "result.json").write_text(json.dumps(outcome.to_dict()))
    return job_dir


# ---------------------------------------------------------------------------
# Recorded agent (harbor-free) transcript driving
# ---------------------------------------------------------------------------
async def test_recorded_agent_drives_probe_transcript_verbatim():
    env = _RecordingEnv()
    agent = rpa.RecordedAgent(logs_dir=None, model_name=None)
    await agent.setup(env)
    summary = await agent.run("instruction", env, context=None)

    driven = [(cmd, kwargs) for cmd, kwargs in env.calls]
    assert driven == rpa.TRANSCRIPTS["probe"]
    assert "mode=probe" in summary


def test_recorded_agent_construction_matches_driver_contract():
    # The own-runner driver constructs agent_cls(logs_dir=, model_name=, extra_env=).
    agent = rpa.RecordedAgent(logs_dir=Path("/tmp/x"), model_name="m", extra_env={"K": "V"})
    assert agent.name() == "recorded-parity"
    assert agent.version() == "1.0.0"


def test_select_mode_precedence(monkeypatch):
    monkeypatch.delenv("RECORDED_MODE", raising=False)
    assert rpa.select_mode(None) == rpa.DEFAULT_MODE
    assert rpa.select_mode({"RECORDED_MODE": "probe"}) == "probe"
    monkeypatch.setenv("RECORDED_MODE", "probe")
    assert rpa.select_mode(None) == "probe"
    # extra_env wins over os.environ.
    assert rpa.select_mode({"RECORDED_MODE": "probe"}) == "probe"


# ---------------------------------------------------------------------------
# Record building: both paths normalize through map_rewards_to_outcome
# ---------------------------------------------------------------------------
def test_digest_key_strips_prefix():
    assert digest_key("terminal-bench/foo") == "foo"
    assert digest_key("foo") == "foo"


def test_record_from_rewards_resolved_mapping():
    solved = record_from_rewards({"reward": 1.0})
    assert solved == {"reward": 1.0, "status": "completed", "reason_code": None, "resolved": 1}
    unsolved = record_from_rewards({"reward": 0.0})
    assert unsolved == {"reward": 0.0, "status": "completed", "reason_code": None, "resolved": 0}


def test_harbor_records_from_run_dir(tmp_path):
    run_dir = _write_harbor_run_dir(
        tmp_path,
        {"terminal-bench/task-a": 0.0, "terminal-bench/task-b": 1.0},
    )
    records = harbor_records_from_run_dir(run_dir)
    assert set(records) == {"task-a", "task-b"}
    assert records["task-a"]["resolved"] == 0
    assert records["task-b"]["resolved"] == 1


def test_harbor_records_rejects_exception(tmp_path):
    run_dir = tmp_path / "run"
    trial = run_dir / "task-x__abc"
    trial.mkdir(parents=True)
    bad = _harbor_result("terminal-bench/task-x", 0.0)
    bad["exception_info"] = {"type": "RuntimeError"}
    (trial / "result.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exception"):
        harbor_records_from_run_dir(run_dir)


def test_harbor_records_rejects_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no harbor result.json"):
        harbor_records_from_run_dir(tmp_path / "empty")


def test_harbor_records_from_run_dirs_merges_per_task(tmp_path):
    dir_a = _write_harbor_run_dir(tmp_path / "a", {"terminal-bench/task-a": 0.0})
    dir_b = _write_harbor_run_dir(tmp_path / "b", {"terminal-bench/task-b": 1.0})
    merged = harbor_records_from_run_dirs([dir_a, dir_b])
    assert set(merged) == {"task-a", "task-b"}
    assert merged["task-b"]["resolved"] == 1


def test_harbor_records_from_run_dirs_rejects_duplicate(tmp_path):
    dir_a = _write_harbor_run_dir(tmp_path / "a", {"terminal-bench/task-a": 0.0})
    dir_b = _write_harbor_run_dir(tmp_path / "b", {"terminal-bench/task-a": 1.0})
    with pytest.raises(ValueError, match="more than one harbor run dir"):
        harbor_records_from_run_dirs([dir_a, dir_b])


def test_own_records_from_outcomes():
    outcomes = [
        TrialOutcome(
            task_name="terminal-bench/task-a",
            trial_name="task-a__attempt-0",
            status="completed",
            rewards={"reward": 0.0},
        ),
        TrialOutcome(
            task_name="task-b",
            trial_name="task-b__attempt-0",
            status="completed",
            rewards={"reward": 1.0},
        ),
    ]
    records = own_records_from_outcomes(outcomes)
    assert set(records) == {"task-a", "task-b"}
    assert records["task-a"]["resolved"] == 0
    assert records["task-b"]["resolved"] == 1


def test_own_records_rejects_errored():
    outcomes = [
        TrialOutcome(
            task_name="task-a",
            trial_name="task-a__attempt-0",
            status="failed",
            rewards=None,
            errored=True,
            reason_code="harbor_trial_failed",
        )
    ]
    with pytest.raises(ValueError, match="errored"):
        own_records_from_outcomes(outcomes)


def test_own_records_from_job_dir(tmp_path):
    job_dir = _write_own_job_dir(tmp_path, {"terminal-bench/task-a": 0.0})
    records = own_records_from_job_dir(job_dir)
    assert records["task-a"]["resolved"] == 0


# ---------------------------------------------------------------------------
# Diff: the parity assertion itself
# ---------------------------------------------------------------------------
def test_diff_identical_records_zero_deltas():
    harbor = {"task-a": record_from_rewards({"reward": 0.0})}
    own = {"task-a": record_from_rewards({"reward": 0.0})}
    assert diff_records(harbor, own) == []


def test_diff_detects_resolved_mismatch():
    harbor = {"task-a": record_from_rewards({"reward": 1.0})}
    own = {"task-a": record_from_rewards({"reward": 0.0})}
    deltas = diff_records(harbor, own)
    assert any(d["kind"] == "field_mismatch" and d["field"] == "resolved" for d in deltas)


def test_parity_by_construction_same_rewards_zero_deltas(tmp_path):
    # Same observed rewards on BOTH paths -> both normalize through the SAME
    # map_rewards_to_outcome -> identical records -> zero deltas at eps=0.
    rewards = {"terminal-bench/task-a": 0.0, "terminal-bench/task-b": 0.0}
    harbor = harbor_records_from_run_dir(_write_harbor_run_dir(tmp_path / "h", rewards))
    own = own_records_from_job_dir(_write_own_job_dir(tmp_path / "o", rewards))
    assert diff_records(harbor, own, reward_eps=0.0) == []


# ---------------------------------------------------------------------------
# Argv builder
# ---------------------------------------------------------------------------
def test_build_harbor_argv_dataset_subset():
    argv = build_harbor_argv(
        harbor_bin="/venv/bin/harbor",
        jobs_dir="/jobs",
        dataset="terminal-bench==2.1",
        include_task_names=["task-a", "task-b"],
        agent_env={"RECORDED_MODE": "probe"},
        job_name="parity",
    )
    assert argv[:2] == ["/venv/bin/harbor", "run"]
    assert "--agent-import-path" in argv
    assert argv[argv.index("--agent-import-path") + 1] == "recorded_harbor_agent:Agent"
    assert argv[argv.index("-d") + 1] == "terminal-bench==2.1"
    assert argv.count("-i") == 2
    assert "--ae" in argv and "RECORDED_MODE=probe" in argv
    assert argv[argv.index("-o") + 1] == "/jobs"
    assert argv[argv.index("-k") + 1] == "1"
    assert argv[-1] == "--yes"


def test_build_harbor_argv_path_subset():
    argv = build_harbor_argv(
        harbor_bin="harbor",
        jobs_dir="/jobs",
        task_paths=["/cache/task-a", "/cache/task-b"],
    )
    assert argv.count("-p") == 2
    assert "/cache/task-a" in argv and "/cache/task-b" in argv


# ---------------------------------------------------------------------------
# Opt-in real-docker both-path integration (heavy; gated by env var)
# ---------------------------------------------------------------------------
_DOCKER_PARITY = os.environ.get("RUN_AGENT_PARITY_DOCKER") == "1"


@pytest.mark.skipif(
    not _DOCKER_PARITY,
    reason="set RUN_AGENT_PARITY_DOCKER=1 (needs docker + harbor venv) for real both-path parity",
)
def test_real_both_path_parity_subset(tmp_path):
    """Run the recorded agent under stock harbor AND own_runner over the subset.

    Requires: docker; the harbor venv binary at ``HARBOR_BIN``; the subset task
    images pulled; and ``AGENT_PARITY_TASK_PATHS`` -- a comma-separated list of
    cached task-root dirs, parallel to ``AGENT_PARITY_SUBSET`` (offline harbor
    runs one ``-p`` task per job, so PATH A is one harbor job per task, merged).
    Asserts zero deltas at eps=0.
    """
    import asyncio

    from agent_challenge.evaluation.own_runner_backend import run_own_runner_job
    from tools.agent_parity_harness import run_harbor_subprocess

    harbor_bin = os.environ.get("HARBOR_BIN")
    if not harbor_bin or (not shutil.which(harbor_bin) and not Path(harbor_bin).exists()):
        pytest.skip("HARBOR_BIN not set / not found")

    subset = os.environ.get(
        "AGENT_PARITY_SUBSET",
        "adaptive-rejection-sampler,configure-git-webserver,count-dataset-tokens",
    ).split(",")
    task_paths_env = os.environ.get("AGENT_PARITY_TASK_PATHS")
    if not task_paths_env:
        pytest.skip("AGENT_PARITY_TASK_PATHS (cached task roots) not set")
    task_paths = task_paths_env.split(",")
    assert len(task_paths) == len(subset), "AGENT_PARITY_TASK_PATHS must parallel the subset"
    agent_env = {"RECORDED_MODE": "probe"}

    # PATH A — stock harbor, one job per task (offline -p), then merge.
    run_dirs = [
        run_harbor_subprocess(
            harbor_bin=harbor_bin,
            jobs_dir=tmp_path / "harbor-jobs",
            task_paths=[path],
            agent_env=agent_env,
            job_name=f"task23-{task_id}",
        )
        for task_id, path in zip(subset, task_paths, strict=True)
    ]
    harbor_records = harbor_records_from_run_dirs(run_dirs)

    # PATH B — own_runner.
    job_result = asyncio.run(
        run_own_runner_job(
            task_ids=subset,
            job_dir=tmp_path / "own-job",
            agent_class=rpa.RecordedAgent,
            agent_env=agent_env,
            n_attempts=1,
            n_concurrent=1,
            stage_solution=False,
        )
    )
    own_records = own_records_from_outcomes(job_result.trial_outcomes)

    deltas = diff_records(harbor_records, own_records, reward_eps=0.0)
    assert deltas == [], f"parity deltas: {deltas}"
    assert set(harbor_records) == set(own_records) == set(subset)
