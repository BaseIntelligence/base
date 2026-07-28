"""Host-side enforcement of guest_artifact_proof at Eval result ingestion/scoring.

Scenarios (contract):
  S1 — valid proof matching plan agent_hash ⇒ accepted for scoring
  S2 — proof absent on success path ⇒ rejected (not scored)
  S3 — match=False ⇒ rejected
  S4 — executed_hash != download_hash ⇒ rejected
  S5 — proof internally consistent but hash ≠ plan agent_hash ⇒ rejected (attack)
  S6 — wire-level older envelope without field still validates (no retroactive break)
  S7 — reject-as-invalid is distinguishable from a legitimate score-0 burn
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.plan_scoring import (
    GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH,
    GUEST_ARTIFACT_PROOF_HASH_MISMATCH,
    GUEST_ARTIFACT_PROOF_MISSING,
    CanonicalPlanScoringError,
    build_score_record_from_eval_plan,
    validate_eval_result_from_plan,
)

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b" * 96,
    "rtmr1": "c" * 96,
    "rtmr2": "d" * 96,
    "compose_hash": "e" * 64,
    "os_image_hash": "f" * 64,
}
AGENT_HASH = "1" * 64
OTHER_HASH = "2" * 64


def _policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }


def _plan(*, agent_hash: str = AGENT_HASH) -> dict[str, Any]:
    policy = _policy()
    return {
        "schema_version": 1,
        "eval_run_id": "eval-host-gap-001",
        "submission_id": "submission-host-gap-001",
        "submission_version": 1,
        "authorizing_review_digest": "3" * 64,
        "agent_hash": agent_hash,
        "package_tree_sha": "b" * 64,
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task@sha256:" + "4" * 64,
                "task_config_sha256": "5" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "6" * 64,
            "compose_hash": MEASUREMENT["compose_hash"],
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "7" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("7" * 64)).hexdigest(),
            "measurement": {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "os_image_hash": MEASUREMENT["os_image_hash"],
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-host-gap-001/result",
        "key_release_nonce": "key-nonce-host-gap-001",
        "score_nonce": "score-nonce-host-gap-001",
        "run_token_sha256": "8" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


def _matching_proof(*, agent_hash: str = AGENT_HASH, byte_size: int = 32) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "expected_hash": agent_hash,
        "download_hash": agent_hash,
        "executed_hash": agent_hash,
        "byte_size": byte_size,
        "match": True,
    }


def _result_request(
    plan: dict[str, Any],
    *,
    proof: dict[str, Any] | None = ...,  # type: ignore[assignment]
    agent_hash: str | None = None,
) -> dict[str, Any]:
    """Build a plan-bound result request. ``proof=...`` means attach matching proof."""
    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    scores_digest = ew.score_record_digest(record)
    hash_value = agent_hash if agent_hash is not None else plan["agent_hash"]
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=hash_value,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=[task["task_id"] for task in plan["selected_tasks"]],
    )
    request: dict[str, Any] = {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": hash_value,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "9" * 64,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": "ab" * 8,
                "event_log": [],
                "report_data": ew.score_report_data_hex(binding),
                "measurement": {**MEASUREMENT, "rtmr3": "c" * 96},
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            },
        },
    }
    if proof is ...:
        request["guest_artifact_proof"] = _matching_proof(agent_hash=hash_value)
    elif proof is not None:
        request["guest_artifact_proof"] = proof
    return request


# --------------------------------------------------------------------------- #
# S6 — wire remains optional (legacy / non-success stored bodies)
# --------------------------------------------------------------------------- #
def test_wire_still_accepts_result_request_without_guest_artifact_proof() -> None:
    """Given: older envelope without proof. When: wire validate. Then: accepted."""
    plan = _plan()
    request = _result_request(plan, proof=None)
    assert "guest_artifact_proof" not in request
    validated = ew.validate_eval_result_request(request)
    assert "guest_artifact_proof" not in validated


# --------------------------------------------------------------------------- #
# S1 — valid matching proof accepted + scored
# --------------------------------------------------------------------------- #
def test_host_accepts_valid_guest_artifact_proof_matching_plan_agent_hash() -> None:
    """Given: matching proof. When: plan-backed validate. Then: accepted with proof."""
    plan = _plan()
    request = _result_request(plan)
    validated = validate_eval_result_from_plan(plan, request)
    assert validated["guest_artifact_proof"] == _matching_proof()
    assert validated["score_record"]["final"]["total_tasks"] == 1
    # Score is reconstructible (not rejected as invalid).
    score = ew.decode_score_f64be(validated["score_record"]["final"]["job_score_f64be"])
    assert score == 1.0


# --------------------------------------------------------------------------- #
# S2 — missing proof on success path rejected
# --------------------------------------------------------------------------- #
def test_host_rejects_success_result_missing_guest_artifact_proof() -> None:
    """Given: success-shaped body without proof. When: host validate. Then: missing code."""
    plan = _plan()
    request = _result_request(plan, proof=None)
    with pytest.raises(CanonicalPlanScoringError) as exc_info:
        validate_eval_result_from_plan(plan, request)
    assert exc_info.value.reason_code == GUEST_ARTIFACT_PROOF_MISSING
    assert "guest_artifact_proof" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# S3 — match=False rejected
# --------------------------------------------------------------------------- #
def test_host_rejects_guest_artifact_proof_with_match_false() -> None:
    """Given: match=False. When: host validate. Then: rejected (wire or host)."""
    plan = _plan()
    proof = _matching_proof()
    proof["match"] = False
    request = _result_request(plan, proof=proof)
    with pytest.raises((CanonicalPlanScoringError, ew.EvalWireError)):
        validate_eval_result_from_plan(plan, request)


# --------------------------------------------------------------------------- #
# S4 — executed_hash != download_hash rejected
# --------------------------------------------------------------------------- #
def test_host_rejects_executed_hash_not_equal_download_hash() -> None:
    """Given: match=True but executed≠download. When: validate. Then: hash mismatch."""
    plan = _plan()
    proof = _matching_proof()
    proof["executed_hash"] = OTHER_HASH
    # Keep match True to force host/wire internal equality check (not match flag).
    proof["match"] = True
    request = _result_request(plan, proof=proof)
    with pytest.raises((CanonicalPlanScoringError, ew.EvalWireError)) as exc_info:
        validate_eval_result_from_plan(plan, request)
    if isinstance(exc_info.value, CanonicalPlanScoringError):
        assert exc_info.value.reason_code in {
            GUEST_ARTIFACT_PROOF_HASH_MISMATCH,
            GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH,
        }


# --------------------------------------------------------------------------- #
# S5 — attack: consistent proof for a different artifact than submitted
# --------------------------------------------------------------------------- #
def test_host_rejects_proof_for_different_artifact_than_plan_agent_hash() -> None:
    """Given: proof hashes equal each other but ≠ plan agent_hash. Then: reject."""
    plan = _plan(agent_hash=AGENT_HASH)
    # Top-level agent_hash still matches plan (so only the proof is the attack).
    proof = _matching_proof(agent_hash=OTHER_HASH)
    request = _result_request(plan, proof=proof)
    with pytest.raises(CanonicalPlanScoringError) as exc_info:
        validate_eval_result_from_plan(plan, request)
    assert exc_info.value.reason_code == GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH
    assert (
        "agent_hash" in str(exc_info.value).lower()
        or "guest_artifact" in str(exc_info.value).lower()
    )


# --------------------------------------------------------------------------- #
# S7 — reject-as-invalid distinguishable from score-0 burn
# --------------------------------------------------------------------------- #
def test_host_reject_reason_codes_are_distinct_from_score_zero() -> None:
    """Missing/mismatch codes must not look like a scored zero."""
    assert GUEST_ARTIFACT_PROOF_MISSING != "score_zero"
    assert GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH != "score_zero"
    assert GUEST_ARTIFACT_PROOF_MISSING.startswith("guest_artifact_proof_")
    assert GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH.startswith("guest_artifact_proof_")

    plan = _plan()
    # A zero-score body with a valid matching proof is still a valid admission
    # shape (score may be 0.0); missing proof is never that path.
    zero_record = build_score_record_from_eval_plan(plan, {"task-a": [0.0]})
    scores_digest = ew.score_record_digest(zero_record)
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=["task-a"],
    )
    zero_request = {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": AGENT_HASH,
        "score_record": zero_record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "9" * 64,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": "ab" * 8,
                "event_log": [],
                "report_data": ew.score_report_data_hex(binding),
                "measurement": {**MEASUREMENT, "rtmr3": "c" * 96},
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            },
        },
        "guest_artifact_proof": _matching_proof(),
    }
    validated = validate_eval_result_from_plan(plan, zero_request)
    assert ew.decode_score_f64be(validated["score_record"]["final"]["job_score_f64be"]) == 0.0

    missing = copy.deepcopy(zero_request)
    del missing["guest_artifact_proof"]
    with pytest.raises(CanonicalPlanScoringError) as exc_info:
        validate_eval_result_from_plan(plan, missing)
    assert exc_info.value.reason_code == GUEST_ARTIFACT_PROOF_MISSING
