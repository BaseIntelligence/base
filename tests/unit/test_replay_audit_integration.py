from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from base.master.replay_audit import (
    REPLAY_AUDIT_LABEL,
    REPLAY_AUDIT_REQUEST_KIND,
    REPLAY_AUDIT_RESULT_KIND,
    ReplayAuditRequest,
    ReplayAuditResult,
    replay_assignment_payload,
)


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eval_run_id": "run-1",
        "submission_id": "sub-1",
        "submission_version": 2,
        "selected_tasks": [{"task_id": "task-b"}, {"task_id": "task-a"}],
        "k": 3,
        "scoring_policy": {
            "schema_version": 1,
            "mode": "best_of_k",
            "keep": {"mode": "threshold_band", "threshold": 0.5},
        },
        "scoring_policy_digest": "a" * 64,
    }


def _request() -> dict[str, Any]:
    plan = _plan()
    return {
        "schema_version": 1,
        "audit_label": REPLAY_AUDIT_LABEL,
        "kind": REPLAY_AUDIT_REQUEST_KIND,
        "audit_id": "replay:run-1:1",
        "submission_id": "sub-1",
        "eval_run_id": "run-1",
        "replay_attempt": 1,
        "plan_sha256": "b" * 64,
        "eval_plan": plan,
        "k": 3,
        "selected_tasks": plan["selected_tasks"],
        "scoring_policy": plan["scoring_policy"],
        "scoring_policy_digest": plan["scoring_policy_digest"],
        "attested_score": 0.75,
    }


def _result() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_label": REPLAY_AUDIT_LABEL,
        "kind": REPLAY_AUDIT_RESULT_KIND,
        "audit_id": "replay:run-1:1",
        "submission_id": "sub-1",
        "eval_run_id": "run-1",
        "replay_attempt": 1,
        "plan_sha256": "b" * 64,
        "trial_scores_by_task": {
            "task-b": [0.5, 0.75, 1.0],
            "task-a": [0.25, 0.5, 0.75],
        },
    }


def test_replay_wire_preserves_complete_plan_and_policy() -> None:
    request = ReplayAuditRequest.from_mapping(_request())
    payload = replay_assignment_payload(request)

    assert payload["replay_audit_label"] == REPLAY_AUDIT_LABEL
    assert payload["replay_audit_request"] == _request()
    assert (
        payload["replay_audit_request"]["eval_plan"]["scoring_policy"]
        == _plan()["scoring_policy"]
    )


def test_replay_wire_rejects_label_or_plan_mutations() -> None:
    raw = _request()
    raw["kind"] = "ordinary_assignment"
    with pytest.raises(ValueError):
        ReplayAuditRequest.from_mapping(raw)

    raw = _request()
    raw["k"] = 1
    with pytest.raises(ValueError):
        ReplayAuditRequest.from_mapping(raw)


def test_replay_result_requires_ordered_trials_not_an_aggregate() -> None:
    result = ReplayAuditResult.from_mapping(_result())
    assert list(result.trial_scores_by_task["task-b"]) == [0.5, 0.75, 1.0]

    raw = _result()
    raw.pop("trial_scores_by_task")
    raw["score"] = 0.75
    with pytest.raises(ValueError):
        ReplayAuditResult.from_mapping(raw)


@dataclass
class _Record:
    slug: str
    internal_base_url: str


class _Registry:
    def __init__(self) -> None:
        self.record = _Record("agent-challenge", "http://challenge:8000")

    async def list(self, *, active_only: bool = False) -> list[_Record]:
        return [self.record]

    async def get(self, slug: str) -> _Record:
        return self.record

    async def get_token(self, slug: str) -> str:
        return "challenge-token"


async def test_replay_client_consumes_only_the_labelled_request() -> None:
    from base.master.challenge_work_source import HttpChallengeReplayClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer challenge-token"
        return httpx.Response(200, json=_request())

    client = HttpChallengeReplayClient(
        _Registry(),
        transport=httpx.MockTransport(handler),
    )
    request = await client.fetch_request(
        challenge_slug="agent-challenge", eval_run_id="run-1"
    )
    assert request.eval_run_id == "run-1"

    def unlabeled(_: httpx.Request) -> httpx.Response:
        body = _request()
        body["audit_label"] = "not-replay"
        return httpx.Response(200, json=body)

    client = HttpChallengeReplayClient(
        _Registry(),
        transport=httpx.MockTransport(unlabeled),
    )
    with pytest.raises(ValueError):
        await client.fetch_request(
            challenge_slug="agent-challenge", eval_run_id="run-1"
        )


async def test_replay_adapter_uses_replay_entrypoint_only() -> None:
    from base.schemas.assignment import AssignmentView
    from base.validator.agent import AssignmentContext, BrokerConfig
    from base.validator.agent.adapters import AgentChallengeCycleExecutor

    calls: list[dict[str, Any]] = []

    async def dispatch_replay(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _result()

    assignment = AssignmentView(
        id="11111111-1111-1111-1111-111111111111",
        challenge_slug="agent-challenge",
        work_unit_id="replay:run-1:1",
        submission_ref="sub-1",
        payload=replay_assignment_payload(ReplayAuditRequest.from_mapping(_request())),
        required_capability="cpu",
        status="running",
        attempt_count=1,
        max_attempts=3,
    )
    context = AssignmentContext(
        assignment=assignment,
        gateway_env={},
        broker=BrokerConfig(broker_url="http://validator-broker:8082"),
    )
    executor = AgentChallengeCycleExecutor(dispatch_replay=dispatch_replay)
    result = await executor.execute(context, progress=lambda **_: None)  # type: ignore[arg-type]

    assert result.success is True
    assert result.payload["replay_audit_result"]["kind"] == REPLAY_AUDIT_RESULT_KIND
    assert calls[0]["request"] == _request()
    assert calls[0]["broker_url"] == "http://validator-broker:8082"


async def test_normal_adapter_does_not_invoke_replay_entrypoint() -> None:
    from base.schemas.assignment import AssignmentView
    from base.validator.agent import AssignmentContext, BrokerConfig
    from base.validator.agent.adapters import AgentChallengeCycleExecutor

    normal_calls: list[dict[str, Any]] = []
    replay_calls: list[dict[str, Any]] = []

    async def dispatch(**kwargs: Any) -> dict[str, Any]:
        normal_calls.append(kwargs)
        return _result()

    async def dispatch_replay(**kwargs: Any) -> dict[str, Any]:
        replay_calls.append(kwargs)
        return _result()

    assignment = AssignmentView(
        id="22222222-2222-2222-2222-222222222222",
        challenge_slug="agent-challenge",
        work_unit_id="ordinary:run-1",
        submission_ref="sub-1",
        payload={"gateway_token": "token", "gateway_url": "http://gateway"},
        required_capability="cpu",
        status="running",
        attempt_count=1,
        max_attempts=3,
    )
    context = AssignmentContext(
        assignment=assignment,
        gateway_env={},
        broker=BrokerConfig(broker_url="http://validator-broker:8082"),
    )
    executor = AgentChallengeCycleExecutor(
        dispatch=dispatch,
        dispatch_replay=dispatch_replay,
    )
    await executor.execute(context, progress=lambda **_: None)  # type: ignore[arg-type]

    assert len(normal_calls) == 1
    assert replay_calls == []


def test_result_wire_serializes_without_losing_trial_order() -> None:
    result = ReplayAuditResult.from_mapping(_result())
    encoded = json.dumps(result.to_dict(), separators=(",", ":"))
    assert encoded.index("task-b") < encoded.index("task-a")
