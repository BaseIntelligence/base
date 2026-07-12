from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from base.master.replay_audit import (
    REPLAY_AUDIT_LABEL,
    REPLAY_AUDIT_REQUEST_KIND,
    ReplayAuditRequest,
    ReplayAuditWireError,
    plan_sha256,
    scoring_policy_digest,
)


def _policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }


def _plan() -> dict[str, Any]:
    policy = _policy()
    public_key = "ab" * 32
    return {
        "schema_version": 1,
        "eval_run_id": "run-1",
        "submission_id": "sub-1",
        "submission_version": 2,
        "authorizing_review_digest": "01" * 32,
        "agent_hash": "02" * 32,
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task@sha256:" + "03" * 32,
                "task_config_sha256": "04" * 32,
            },
            {
                "task_id": "task-b",
                "image_ref": "registry.example/task@sha256:" + "05" * 32,
                "task_config_sha256": "06" * 32,
            },
        ],
        "k": 2,
        "scoring_policy": policy,
        "scoring_policy_digest": scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "07" * 32,
            "compose_hash": "08" * 32,
            "app_identity": "eval-app-v1",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": public_key,
            "kms_public_key_sha256": hashlib.sha256(
                bytes.fromhex(public_key)
            ).hexdigest(),
            "measurement": {
                "mrtd": "09" * 48,
                "rtmr0": "0a" * 48,
                "rtmr1": "0b" * 48,
                "rtmr2": "0c" * 48,
                "os_image_hash": "0d" * 32,
                "key_provider": "validator-kms",
                "vm_shape": "tdx.small",
            },
        },
        "key_release_endpoint": "tcp://release.example:8701",
        "result_endpoint": "/evaluation/v1/runs/run-1/result",
        "key_release_nonce": "key-nonce-1",
        "score_nonce": "score-nonce-1",
        "run_token_sha256": "0e" * 32,
        "issued_at_ms": 1_000,
        "expires_at_ms": 2_000,
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
        "plan_sha256": plan_sha256(plan),
        "eval_plan": plan,
        "k": plan["k"],
        "selected_tasks": plan["selected_tasks"],
        "scoring_policy": plan["scoring_policy"],
        "scoring_policy_digest": plan["scoring_policy_digest"],
        "attested_score": 0.75,
    }


def test_replay_request_recomputes_complete_plan_and_policy_digests() -> None:
    request = ReplayAuditRequest.from_mapping(_request())

    assert request.plan_sha256 == plan_sha256(request.eval_plan)
    assert request.scoring_policy_digest == scoring_policy_digest(
        request.scoring_policy
    )
    assert [task["task_id"] for task in request.selected_tasks] == ["task-a", "task-b"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["eval_plan"].update({"k": 3}),
        lambda body: body["eval_plan"]["selected_tasks"].reverse(),
        lambda body: body["eval_plan"]["scoring_policy"].update(
            {"keep_policy": "drop_lowest_n"}
        ),
        lambda body: body["eval_plan"].pop("eval_app"),
        lambda body: body["eval_plan"]["scoring_policy"].update({"drop_lowest_n": 1}),
    ],
)
def test_replay_request_rejects_single_field_plan_mutations(mutate) -> None:
    body = _request()
    mutate(body)
    with pytest.raises(ReplayAuditWireError):
        ReplayAuditRequest.from_mapping(body)


def test_replay_request_rejects_forged_digests_and_aliases() -> None:
    body = _request()
    body["plan_sha256"] = "ff" * 32
    with pytest.raises(ReplayAuditWireError):
        ReplayAuditRequest.from_mapping(body)

    body = _request()
    body["scoring_policy_digest"] = "ff" * 32
    with pytest.raises(ReplayAuditWireError):
        ReplayAuditRequest.from_mapping(body)

    body = _request()
    body["eval_plan"]["scoring_policy"]["threshold"] = None
    with pytest.raises(ReplayAuditWireError):
        ReplayAuditRequest.from_mapping(body)


def test_replay_request_rejects_duplicate_nested_json_keys() -> None:
    body = _request()
    raw = json.dumps(body, separators=(",", ":"))
    raw = raw.replace('"k":2', '"k":2,"k":2', 1)
    with pytest.raises(ReplayAuditWireError):
        ReplayAuditRequest.from_mapping(json.loads(raw), raw_body=raw.encode())


def test_replay_request_rejects_partial_plan_before_assignment_payload() -> None:
    body = _request()
    body["eval_plan"] = {
        "schema_version": 1,
        "eval_run_id": "run-1",
        "selected_tasks": body["selected_tasks"],
        "k": 2,
        "scoring_policy": body["scoring_policy"],
    }
    with pytest.raises(ReplayAuditWireError):
        ReplayAuditRequest.from_mapping(body)
