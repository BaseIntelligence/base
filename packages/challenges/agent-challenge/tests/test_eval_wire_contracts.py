"""Strict Eval v1 wire-contract regressions shared with the BASE oracle."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.canonical import report_data as rd

VECTOR_PATH = Path(__file__).with_name("eval_execution_proof_v2_vectors.json")
VECTORS: dict[str, Any] = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
POSITIVE: dict[str, Any] = VECTORS["positive"]


def _set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    *parents, leaf = path.split(".")
    cursor: Any = value
    for part in parents:
        cursor = cursor[int(part)] if part.isdigit() else cursor[part]
    cursor[int(leaf) if leaf.isdigit() else leaf] = replacement


def _rename_path(value: dict[str, Any], source: str, target: str) -> None:
    *source_parents, source_leaf = source.split(".")
    cursor: Any = value
    for part in source_parents:
        cursor = cursor[int(part)] if part.isdigit() else cursor[part]
    moved = cursor.pop(source_leaf)
    _set_path(value, target, moved)


def _malformed_proof(vector: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    if vector["kind"] == "set":
        _set_path(payload, vector["path"], vector["value"])
    else:
        _rename_path(payload, vector["from"], vector["to"])
    return payload


def _score_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "policy_digest": "2" * 64,
        "k": 2,
        "tasks": [
            {
                "task_id": "task-a",
                "trial_scores_f64be": ["0000000000000000", "3ff0000000000000"],
                "aggregate_score_f64be": "3fe0000000000000",
                "passed_trials": 1,
            },
            {
                "task_id": "task-b",
                "trial_scores_f64be": ["3ff0000000000000", "3ff0000000000000"],
                "aggregate_score_f64be": "3ff0000000000000",
                "passed_trials": 2,
            },
        ],
        "final": {
            "job_score_f64be": "3fe8000000000000",
            "passed_tasks": 1,
            "total_tasks": 2,
        },
    }


def test_shared_v2_vector_matches_exact_canonical_bytes_and_report_data() -> None:
    binding = POSITIVE["binding"]
    built = ew.build_score_binding(
        canonical_measurement=binding["canonical_measurement"],
        agent_hash=binding["agent_hash"],
        eval_run_id=binding["eval_run_id"],
        score_nonce=binding["score_nonce"],
        scores_digest=binding["scores_digest"],
        task_ids=binding["task_ids"],
    )

    assert ew.canonical_json_v1(built).hex() == POSITIVE["canonical_json_utf8_hex"]
    assert ew.score_report_data_hex(built) == POSITIVE["report_data_hex"]
    assert (
        rd.report_data_hex(
            canonical_measurement=binding["canonical_measurement"],
            agent_hash=binding["agent_hash"],
            task_ids=binding["task_ids"],
            scores_digest=binding["scores_digest"],
            eval_run_id=binding["eval_run_id"],
            score_nonce=binding["score_nonce"],
        )
        == POSITIVE["report_data_hex"]
    )


@pytest.mark.parametrize("task_ids", [["task-b", "task-a"], ["task-a", "task-a"]])
def test_score_binding_rejects_noncanonical_task_arrays(task_ids: list[str]) -> None:
    binding = POSITIVE["binding"]
    with pytest.raises(ew.EvalWireError):
        ew.build_score_binding(
            canonical_measurement=binding["canonical_measurement"],
            agent_hash=binding["agent_hash"],
            eval_run_id=binding["eval_run_id"],
            score_nonce=binding["score_nonce"],
            scores_digest=binding["scores_digest"],
            task_ids=task_ids,
        )


def test_producer_task_id_normalization_is_order_independent_but_never_deduplicates() -> None:
    assert ew.canonical_task_ids(["task-b", "task-a"]) == ["task-a", "task-b"]
    with pytest.raises(ew.EvalWireError):
        ew.canonical_task_ids(["task-a", "task-a"])


@pytest.mark.parametrize(
    "vector",
    VECTORS["malformed"],
    ids=lambda vector: vector["name"],
)
def test_strict_execution_proof_rejects_shared_malformed_vectors(vector: dict[str, Any]) -> None:
    if vector["kind"] == "raw_json":
        with pytest.raises(ew.EvalWireError):
            ew.parse_eval_execution_proof_json(vector["value"])
    else:
        with pytest.raises(ew.EvalWireError):
            ew.validate_eval_execution_proof(_malformed_proof(vector))


def test_only_empty_placeholder_signature_is_accepted_on_eval_wire() -> None:
    ew.validate_eval_execution_proof(POSITIVE["execution_proof"])
    forged = copy.deepcopy(POSITIVE["execution_proof"])
    forged["worker_signature"]["sig"] = "not-empty"
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_execution_proof(forged)


@pytest.mark.parametrize(
    "value",
    [
        "8000000000000000",
        "7ff0000000000000",
        "7ff8000000000000",
        "3FF0000000000000",
        "3ff000000000000",
        "3ff00000000000000",
    ],
)
def test_binary64_score_encoding_rejects_noncanonical_or_nonfinite_values(value: str) -> None:
    with pytest.raises(ew.EvalWireError):
        ew.decode_score_f64be(value)


def test_score_record_reconstructs_aggregates_and_full_set_counts() -> None:
    record = _score_record()
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    record["policy_digest"] = ew.scoring_policy_digest(policy)
    validated = ew.validate_canonical_score_record(
        record,
        scoring_policy=policy,
        expected_eval_run_id="eval-run-001",
        expected_task_ids=["task-a", "task-b"],
        expected_k=2,
    )
    assert validated == record
    assert ew.score_record_digest(record) == ew.score_record_digest(validated)

    tampered = copy.deepcopy(record)
    tampered["tasks"][0]["aggregate_score_f64be"] = "3ff0000000000000"
    with pytest.raises(ew.EvalWireError):
        ew.validate_canonical_score_record(
            tampered,
            scoring_policy=policy,
            expected_eval_run_id="eval-run-001",
            expected_task_ids=["task-a", "task-b"],
            expected_k=2,
        )


def test_result_request_and_receipt_are_closed_and_bind_the_proof() -> None:
    record = _score_record()
    request = {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "submission_id": "submission-001",
        "agent_hash": POSITIVE["binding"]["agent_hash"],
        "score_record": record,
        "scores_digest": ew.score_record_digest(record),
        "execution_proof": POSITIVE["execution_proof"],
    }
    assert ew.validate_eval_result_request(request) == request
    request["caller_measurement"] = {}
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_result_request(request)

    receipt = {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "receipt_id": "receipt-001",
        "body_sha256": "3" * 64,
        "received_at_ms": 1,
        "phase": "received",
        "terminal": False,
        "verified": False,
        "retryable": True,
        "reason_code": None,
        "result_available": False,
        "finalized_at_ms": None,
    }
    assert ew.validate_eval_receipt(receipt) == receipt


def test_key_release_v2_binding_is_purpose_separated_and_nonce_sensitive() -> None:
    value = ew.key_release_report_data_hex(
        eval_run_id="eval-run-001",
        key_release_nonce="key-nonce-001",
        ra_tls_spki_digest="4" * 64,
    )
    assert len(value) == 128
    assert value[64:] == "0" * 64
    assert value != ew.key_release_report_data_hex(
        eval_run_id="eval-run-001",
        key_release_nonce="score-nonce-001",
        ra_tls_spki_digest="4" * 64,
    )


def _eval_plan() -> dict[str, Any]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    measurement = POSITIVE["binding"]["canonical_measurement"]
    return {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "submission_id": "submission-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": POSITIVE["binding"]["agent_hash"],
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
            }
            for task_id in POSITIVE["binding"]["task_ids"]
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "d" * 64,
            "compose_hash": measurement["compose_hash"],
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "3" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("3" * 64)).hexdigest(),
            "measurement": {
                "mrtd": measurement["mrtd"],
                "rtmr0": measurement["rtmr0"],
                "rtmr1": measurement["rtmr1"],
                "rtmr2": measurement["rtmr2"],
                "os_image_hash": measurement["os_image_hash"],
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-run-001/result",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


def test_eval_plan_is_closed_and_requires_distinct_purpose_nonces() -> None:
    plan = _eval_plan()
    assert ew.validate_eval_plan(plan) == plan

    crossed = copy.deepcopy(plan)
    crossed["score_nonce"] = crossed["key_release_nonce"]
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_plan(crossed)

    extra = copy.deepcopy(plan)
    extra["caller_measurement"] = {}
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_plan(extra)


def test_eval_plan_rejects_unbound_kms_public_key_digest() -> None:
    plan = _eval_plan()
    plan["eval_app"]["kms_public_key_sha256"] = "4" * 64
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_plan(plan)


class _QuoteResponse:
    quote = "ab" * 8
    event_log = [
        {
            "imr": 3,
            "event_type": 134217729,
            "digest": "c" * 96,
            "event": "compose-hash",
            "event_payload": "d" * 64,
        }
    ]
    vm_config = {
        "vcpu": 1,
        "memory_mb": 2048,
        "os_image_hash": POSITIVE["binding"]["canonical_measurement"]["os_image_hash"],
    }


class _QuoteProvider:
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def get_quote(self, report_data: bytes) -> _QuoteResponse:
        self.calls.append(report_data)
        return _QuoteResponse()


def test_attested_image_emits_exact_eval_result_request_with_v2_binding() -> None:
    binding = POSITIVE["binding"]
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    record = ew.build_canonical_score_record(
        eval_run_id=binding["eval_run_id"],
        policy=policy,
        trial_scores_by_task={
            "task-a": [1.0],
            "task-b": [0.0],
            "task-c": [0.5],
        },
    )
    provider = _QuoteProvider()
    line = ar.emit_attested_benchmark_result(
        benchmark_result={
            "status": "completed",
            "score": 0.5,
            "resolved": 1,
            "total": 3,
            "reason_code": None,
        },
        canonical_measurement=binding["canonical_measurement"],
        rtmr3="e" * 96,
        agent_hash=binding["agent_hash"],
        task_ids=binding["task_ids"],
        scores={},
        quote_provider=provider,
        manifest_sha256="1" * 64,
        eval_run_id=binding["eval_run_id"],
        submission_id="submission-001",
        score_nonce=binding["score_nonce"],
        score_record=record,
        image_digest="registry.example/eval@sha256:" + "d" * 64,
    )
    payload = json.loads(line.split("=", 1)[1])

    assert set(payload) == {
        "schema_version",
        "eval_run_id",
        "submission_id",
        "agent_hash",
        "score_record",
        "scores_digest",
        "execution_proof",
    }
    assert ew.validate_eval_result_request(payload) == payload
    assert payload["execution_proof"]["worker_signature"] == {"worker_pubkey": "", "sig": ""}
    report_data = payload["execution_proof"]["attestation"]["report_data"]
    assert provider.calls == [bytes.fromhex(report_data)]


def test_plan_driven_emitter_rejects_crossed_run_before_quote_emission() -> None:
    plan = _eval_plan()
    record = ew.build_canonical_score_record(
        eval_run_id="other-run",
        policy=plan["scoring_policy"],
        trial_scores_by_task={task["task_id"]: [1.0] for task in plan["selected_tasks"]},
    )
    provider = _QuoteProvider()
    with pytest.raises(ar.AttestationEmissionError):
        ar.emit_attested_eval_result_from_plan(
            eval_plan=plan,
            score_record=record,
            rtmr3="e" * 96,
            quote_provider=provider,
            manifest_sha256="1" * 64,
        )
    assert provider.calls == []
