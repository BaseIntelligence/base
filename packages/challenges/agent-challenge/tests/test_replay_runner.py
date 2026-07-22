from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation import replay_runner
from agent_challenge.evaluation.replay_audit import ReplayAuditRequest


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def scalar(self, _statement):
        submission = SimpleNamespace(id=1)
        return SimpleNamespace(submission_id=1, submission=submission)


class _Database:
    def session(self):
        return _Session()


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
        "eval_run_id": "replay-runner-1",
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": "2" * 64,
        "agent_hash": "3" * 64,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
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
        "result_endpoint": "/evaluation/v1/runs/replay-runner-1/result",
        "key_release_nonce": "key-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": "0" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    return ew.validate_eval_plan(plan)


async def test_replay_runner_executes_every_selected_task_once_in_plan_order(monkeypatch) -> None:
    plan = _plan()
    request = ReplayAuditRequest(
        audit_id="replay:replay-runner-1:1",
        submission_id="1",
        eval_run_id=plan["eval_run_id"],
        replay_attempt=1,
        plan_sha256=hashlib.sha256(ew.canonical_json_v1(plan)).hexdigest(),
        eval_plan=plan,
        attested_score=0.5,
    )
    captured: dict[str, object] = {}
    calls: list[tuple[str, int, object, str, object, object]] = []

    def broker_factory(**kwargs):
        captured.update(kwargs)
        return object()

    def run_task(
        _broker,
        _submission,
        _job,
        task,
        *,
        gateway,
        own_runner_attempts,
        replay_audit,
        replay_eval_plan,
        replay_task_ids,
        **_,
    ):
        calls.append(
            (
                task.task_id,
                own_runner_attempts,
                gateway,
                str(replay_audit),
                replay_eval_plan,
                replay_task_ids,
            )
        )
        return SimpleNamespace(
            score=0.5,
            stdout=f'{{"replay_trial_scores_by_task":{{"{task.task_id}":[0.25,0.75]}}}}',
        )

    monkeypatch.setattr(replay_runner, "DockerExecutor", broker_factory)
    monkeypatch.setattr(replay_runner, "_run_terminal_bench_task", run_task)
    monkeypatch.setattr(replay_runner, "database", _Database())
    monkeypatch.setattr(replay_runner, "_gateway_from_assignment", lambda _: "scoped-gateway")

    result = await replay_runner.run_replay_request(
        request,
        assignment_payload={"gateway_token": "token", "gateway_url": "http://gateway"},
        broker_url="http://validator-broker:8082",
        broker_token="broker-token",
        broker_token_file="/run/broker-token",
        broker_allowed_images=("registry.example/",),
        work_unit_id=request.audit_id,
    )

    assert captured == {
        "challenge": "agent-challenge",
        "backend": "broker",
        "broker_url": "http://validator-broker:8082",
        "broker_token": "broker-token",
        "broker_token_file": "/run/broker-token",
        "allowed_images": ("registry.example/",),
    }
    assert [(task_id, attempts) for task_id, attempts, *_ in calls] == [
        ("task-a", 2),
        ("task-b", 2),
    ]
    assert all(
        gateway == "scoped-gateway"
        and replay == "True"
        and plan_value == plan
        and replay_ids == [task_id]
        for task_id, _, gateway, replay, plan_value, replay_ids in calls
    )
    assert result["trial_scores_by_task"] == {
        "task-a": [0.25, 0.75],
        "task-b": [0.25, 0.75],
    }


@pytest.mark.parametrize(
    ("stdout_by_task", "message"),
    [
        (
            {
                "task-a": '{"replay_trial_scores_by_task":{"task-a":[0.25,0.75]}}',
                "task-b": '{"replay_trial_scores_by_task":{}}',
            },
            "missing",
        ),
        (
            {
                "task-a": '{"replay_trial_scores_by_task":'
                '{"task-a":[0.25,0.75],"task-a":[0.5,0.5]}}',
                "task-b": '{"replay_trial_scores_by_task":{"task-b":[0.5,0.5]}}',
            },
            "duplicate",
        ),
        (
            {
                "task-a": '{"replay_trial_scores_by_task":'
                '{"task-b":[0.5,0.5],"task-a":[0.25,0.75]}}',
                "task-b": '{"replay_trial_scores_by_task":{"task-b":[0.5,0.5]}}',
            },
            "reordered",
        ),
        (
            {
                "task-a": '{"replay_trial_scores_by_task":{"task-a":[0.25,0.75]}}',
                "task-b": '{"replay_trial_scores_by_task":'
                '{"task-b":[0.5,0.5],"task-extra":[1.0,1.0]}}',
            },
            "extra",
        ),
    ],
)
async def test_replay_runner_rejects_incomplete_or_mutated_task_results(
    monkeypatch, stdout_by_task, message
) -> None:
    plan = _plan()
    request = ReplayAuditRequest(
        audit_id="replay:replay-runner-1:1",
        submission_id="1",
        eval_run_id=plan["eval_run_id"],
        replay_attempt=1,
        plan_sha256=hashlib.sha256(ew.canonical_json_v1(plan)).hexdigest(),
        eval_plan=plan,
        attested_score=0.5,
    )
    calls: list[str] = []

    def run_task(_broker, _submission, _job, task, **_):
        calls.append(task.task_id)
        return SimpleNamespace(score=0.5, stdout=stdout_by_task[task.task_id])

    monkeypatch.setattr(replay_runner, "DockerExecutor", lambda **_: object())
    monkeypatch.setattr(replay_runner, "_run_terminal_bench_task", run_task)
    monkeypatch.setattr(replay_runner, "database", _Database())
    monkeypatch.setattr(replay_runner, "_gateway_from_assignment", lambda _: "scoped-gateway")

    with pytest.raises(ValueError, match="replay"):
        await replay_runner.run_replay_request(
            request,
            assignment_payload={"gateway_token": "token", "gateway_url": "http://gateway"},
            broker_url="http://validator-broker:8082",
            broker_token="broker-token",
            broker_token_file="/run/broker-token",
            broker_allowed_images=("registry.example/",),
            work_unit_id=request.audit_id,
        )

    # A malformed task result is rejected before the replay can be reported as
    # complete. The runner still executes each task at most once.
    assert calls in (["task-a"], ["task-a", "task-b"])
