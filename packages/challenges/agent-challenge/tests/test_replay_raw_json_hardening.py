"""VAL-SCORE-028 hardening: reject raw replay JSON before last-key-wins.

Scrutiny found that the production path
``DockerRunResult.stdout`` → ``_run_terminal_bench_task`` → ``_replay_trial_scores``
used ordinary ``json.loads``, which collapses duplicate task-result keys before
``replay_runner._extract_task_replay_scores`` can apply its duplicate-aware
hook. These tests bind the raw boundary so a forged outer payload cannot be
sanitized into an accepted one-task result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation import replay_runner, runner
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.evaluation.replay_audit import ReplayAuditRequest
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.executors import DockerRunResult


def _plan() -> dict:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    plan = {
        "schema_version": 1,
        "eval_run_id": "replay-raw-json-1",
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": "2" * 64,
        "agent_hash": "3" * 64,
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task-a@sha256:" + "4" * 64,
                "task_config_sha256": "5" * 64,
            },
            {
                "task_id": "task-b",
                "image_ref": "registry.example/task-b@sha256:" + "6" * 64,
                "task_config_sha256": "7" * 64,
            },
        ],
        "k": 2,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "8" * 64,
            "compose_hash": "9" * 64,
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "a" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("a" * 64)).hexdigest(),
            "measurement": {
                "mrtd": "b" * 96,
                "rtmr0": "c" * 96,
                "rtmr1": "d" * 96,
                "rtmr2": "e" * 96,
                "os_image_hash": "f" * 64,
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "validator.example:8700",
        "result_endpoint": "/evaluation/v1/runs/replay-raw-json-1/result",
        "key_release_nonce": "key-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": "0" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    return ew.validate_eval_plan(plan)


class _Session:
    def __init__(self, submission: AgentSubmission) -> None:
        self._submission = submission

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def scalar(self, _statement):
        return SimpleNamespace(submission_id=self._submission.id, submission=self._submission)


class _Database:
    def __init__(self, submission: AgentSubmission) -> None:
        self._submission = submission

    def session(self):
        return _Session(self._submission)


class _RecordingReplayExecutor:
    """Fake broker that returns the raw stdout bytes for each task unit."""

    def __init__(self, stdout_by_task: dict[str, str]) -> None:
        self.stdout_by_task = stdout_by_task
        self.calls: list[str] = []

    def run(self, spec, timeout_seconds: int) -> DockerRunResult:
        task_id = spec.labels["base.task"]
        self.calls.append(task_id)
        return DockerRunResult(
            container_name="replay-broker",
            stdout=self.stdout_by_task[task_id],
            stderr="",
            returncode=0,
        )


def _submission_with_workspace(tmp_path: Path) -> AgentSubmission:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text("# replay\n", encoding="utf-8")
    return AgentSubmission(
        id=1,
        miner_hotkey="hotkey-replay-raw",
        name="agent-replay-raw",
        agent_hash="replay-raw-hash",
        artifact_uri=str(agent_dir),
        raw_status="tb_running",
        effective_status="evaluating",
    )


def test_replay_trial_scores_rejects_duplicate_task_keys_before_last_key_wins() -> None:
    # Ordinary json.loads would accept this as {"task-a": [0.5, 0.5]}.
    raw = (
        "prefix noise\n"
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-a":[0.25,0.75],"task-a":[0.5,0.5]}}\n'
    )
    with pytest.raises(ValueError, match="duplicate|invalid|replay"):
        runner._replay_trial_scores(raw)


def test_replay_trial_scores_rejects_duplicate_outer_field_keys() -> None:
    raw = (
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-a":[0.25,0.75]},'
        '"replay_trial_scores_by_task":{"task-a":[0.5,0.5]}}'
    )
    with pytest.raises(ValueError, match="duplicate|invalid|replay"):
        runner._replay_trial_scores(raw)


def test_replay_trial_scores_accepts_exact_one_task_mapping() -> None:
    raw = (
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-a":[0.25,0.75]}}'
    )
    assert runner._replay_trial_scores(raw) == {"task-a": [0.25, 0.75]}


def test_run_terminal_bench_task_rejects_duplicate_keys_on_docker_run_result_stdout(
    monkeypatch, tmp_path
) -> None:
    """Production path integrates parse before re-serializing TaskResult.stdout."""

    monkeypatch.setattr(runner.settings, "docker_backend", "cli")
    monkeypatch.setattr(runner.settings, "evaluation_timeout_seconds", 5)
    monkeypatch.setattr(runner.settings, "harbor_output_dir", str(tmp_path / "harbor"))
    (tmp_path / "harbor").mkdir()

    stdout = (
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-a":[0.25,0.75],"task-a":[0.5,0.5]}}'
    )
    executor = _RecordingReplayExecutor({"task-a": stdout})
    submission = _submission_with_workspace(tmp_path)
    job = EvaluationJob(
        id=0,
        job_id="job-replay-raw",
        submission_id=submission.id,
        selected_tasks_json="[]",
        total_tasks=1,
    )
    task = BenchmarkTask(
        task_id="task-a",
        docker_image="registry.example/task-a@sha256:" + "4" * 64,
        benchmark="terminal_bench",
        prompt="replay task-a",
        metadata={"task_id": "task-a"},
    )

    with pytest.raises(ValueError, match="duplicate|invalid|replay"):
        runner._run_terminal_bench_task(
            executor,
            submission,
            job,
            task,
            own_runner_attempts=2,
            replay_audit=True,
            replay_eval_plan=_plan(),
            replay_task_ids=["task-a"],
        )


async def test_replay_runner_production_path_rejects_duplicate_raw_stdout(
    monkeypatch, tmp_path
) -> None:
    """End-to-end: realRunner path over DockerRunResult.stdout, not mocked TaskResult."""

    plan = _plan()
    request = ReplayAuditRequest(
        audit_id="replay:replay-raw-json-1:1",
        submission_id="1",
        eval_run_id=plan["eval_run_id"],
        replay_attempt=1,
        plan_sha256=hashlib.sha256(ew.canonical_json_v1(plan)).hexdigest(),
        eval_plan=plan,
        attested_score=0.5,
    )
    submission = _submission_with_workspace(tmp_path)
    monkeypatch.setattr(runner.settings, "docker_backend", "cli")
    monkeypatch.setattr(runner.settings, "evaluation_timeout_seconds", 5)
    monkeypatch.setattr(runner.settings, "harbor_output_dir", str(tmp_path / "harbor"))
    (tmp_path / "harbor").mkdir()

    # First-key / last-key conflict on the exact single selected task. Under the
    # pre-hardening path, ordinary last-key-wins would yield a valid set of size 1.
    duplicate_stdout = (
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-a":[0.25,0.75],"task-a":[0.5,0.5]}}'
    )
    good_stdout = (
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-b":[0.5,0.5]}}'
    )
    executor = _RecordingReplayExecutor(
        {
            "task-a": duplicate_stdout,
            "task-b": good_stdout,
        }
    )

    monkeypatch.setattr(replay_runner, "DockerExecutor", lambda **_: executor)
    monkeypatch.setattr(replay_runner, "database", _Database(submission))
    monkeypatch.setattr(replay_runner, "_gateway_from_assignment", lambda _: None)
    # Intentionally leave replay_runner._run_terminal_bench_task as the real
    # production function so DockerRunResult.stdout is parsed first.

    with pytest.raises(ValueError, match="duplicate|invalid|replay"):
        await replay_runner.run_replay_request(
            request,
            assignment_payload={},
            broker_url="http://validator-broker:8082",
            broker_token="broker-token",
            broker_token_file="/run/broker-token",
            broker_allowed_images=("registry.example/",),
            work_unit_id=request.audit_id,
        )

    # Fail closed on the first task unit; second selected task is never needed
    # for a malformed first unit, but at most both units may be launched.
    assert executor.calls[0] == "task-a"
    assert executor.calls in (["task-a"], ["task-a", "task-b"])


@pytest.mark.parametrize(
    "stdout",
    [
        # Extra task smuggled into a single-task unit.
        (
            'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
            '"replay_trial_scores_by_task":'
            '{"task-a":[0.25,0.75],"task-extra":[1.0,1.0]}}'
        ),
        # Missing scores map.
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed"}',
        # Wrong k.
        (
            'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
            '"replay_trial_scores_by_task":{"task-a":[0.25]}}'
        ),
        # Partial / empty mapping.
        (
            'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
            '"replay_trial_scores_by_task":{}}'
        ),
    ],
)
async def test_replay_runner_production_path_rejects_malformed_selected_set(
    monkeypatch, tmp_path, stdout
) -> None:
    plan = _plan()
    request = ReplayAuditRequest(
        audit_id="replay:replay-raw-json-1:1",
        submission_id="1",
        eval_run_id=plan["eval_run_id"],
        replay_attempt=1,
        plan_sha256=hashlib.sha256(ew.canonical_json_v1(plan)).hexdigest(),
        eval_plan=plan,
        attested_score=0.5,
    )
    submission = _submission_with_workspace(tmp_path)
    monkeypatch.setattr(runner.settings, "docker_backend", "cli")
    monkeypatch.setattr(runner.settings, "evaluation_timeout_seconds", 5)
    monkeypatch.setattr(runner.settings, "harbor_output_dir", str(tmp_path / "harbor"))
    (tmp_path / "harbor").mkdir()

    good_b = (
        'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
        '"replay_trial_scores_by_task":{"task-b":[0.5,0.5]}}'
    )
    executor = _RecordingReplayExecutor({"task-a": stdout, "task-b": good_b})
    monkeypatch.setattr(replay_runner, "DockerExecutor", lambda **_: executor)
    monkeypatch.setattr(replay_runner, "database", _Database(submission))
    monkeypatch.setattr(replay_runner, "_gateway_from_assignment", lambda _: None)

    with pytest.raises(ValueError, match="replay|invalid|omitted|wrong|trial"):
        await replay_runner.run_replay_request(
            request,
            assignment_payload={},
            broker_url="http://validator-broker:8082",
            broker_token="broker-token",
            broker_token_file="/run/broker-token",
            broker_allowed_images=("registry.example/",),
            work_unit_id=request.audit_id,
        )


async def test_replay_runner_production_path_accepts_clean_selected_set(
    monkeypatch, tmp_path
) -> None:
    plan = _plan()
    request = ReplayAuditRequest(
        audit_id="replay:replay-raw-json-1:1",
        submission_id="1",
        eval_run_id=plan["eval_run_id"],
        replay_attempt=1,
        plan_sha256=hashlib.sha256(ew.canonical_json_v1(plan)).hexdigest(),
        eval_plan=plan,
        attested_score=0.5,
    )
    submission = _submission_with_workspace(tmp_path)
    monkeypatch.setattr(runner.settings, "docker_backend", "cli")
    monkeypatch.setattr(runner.settings, "evaluation_timeout_seconds", 5)
    monkeypatch.setattr(runner.settings, "harbor_output_dir", str(tmp_path / "harbor"))
    (tmp_path / "harbor").mkdir()

    executor = _RecordingReplayExecutor(
        {
            "task-a": (
                'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
                '"replay_trial_scores_by_task":{"task-a":[0.25,0.75]}}'
            ),
            "task-b": (
                'BASE_BENCHMARK_RESULT={"score":0.5,"status":"completed",'
                '"replay_trial_scores_by_task":{"task-b":[0.5,0.5]}}'
            ),
        }
    )
    monkeypatch.setattr(replay_runner, "DockerExecutor", lambda **_: executor)
    monkeypatch.setattr(replay_runner, "database", _Database(submission))
    monkeypatch.setattr(replay_runner, "_gateway_from_assignment", lambda _: None)

    result = await replay_runner.run_replay_request(
        request,
        assignment_payload={},
        broker_url="http://validator-broker:8082",
        broker_token="broker-token",
        broker_token_file="/run/broker-token",
        broker_allowed_images=("registry.example/",),
        work_unit_id=request.audit_id,
    )

    assert executor.calls == ["task-a", "task-b"]
    assert result["trial_scores_by_task"] == {
        "task-a": [0.25, 0.75],
        "task-b": [0.5, 0.5],
    }
    # Positive control: a clean mapping survives the full parse/re-serialize
    # round-trip without relying on last-key-wins collapse.
    assert json.loads(json.dumps({"replay_trial_scores_by_task": result["trial_scores_by_task"]}))[
        "replay_trial_scores_by_task"
    ] == {
        "task-a": [0.25, 0.75],
        "task-b": [0.5, 0.5],
    }
