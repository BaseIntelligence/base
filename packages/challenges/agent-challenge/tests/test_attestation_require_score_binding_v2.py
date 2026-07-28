"""Require schema-v2 SCORE BINDING inside report_data on the production path.

Naming subtlety (do not conflate):
- Outer wire ``score_record.schema_version`` / ``eval_result_request.schema_version``
  may legitimately remain **1**.
- The SCORE BINDING hashed into TDX ``report_data`` must be **schema_version: 2**
  (``build_score_binding`` / ``score_report_data_hex``).

When ``phala_attestation_enabled`` is true, a ``report_data`` that only validates
under the legacy ``validator_nonce`` construction is fail-closed rejected with
reason code ``legacy_report_data_rejected`` — never a warning or downgrade.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.canonical import report_data as rd
from agent_challenge.evaluation.plan_scoring import (
    LEGACY_REPORT_DATA_REJECTED,
    CanonicalPlanScoringError,
    build_score_record_from_eval_plan,
    require_schema_v2_score_report_data,
    validate_eval_result_from_plan,
)
from agent_challenge.evaluation.score_chain_gate import verify_score_domain_binding
from agent_challenge.keyrelease.quote import os_image_hash_from_registers

REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH = "ab" * 32
OS_IMAGE_HASH = os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"])
AGENT_HASH = "55" * 32


def _guest_proof(*, agent_hash: str = AGENT_HASH) -> dict:
    return {
        "schema_version": 1,
        "expected_hash": agent_hash,
        "download_hash": agent_hash,
        "executed_hash": agent_hash,
        "byte_size": 32,
        "match": True,
    }


MEASUREMENT = {
    **REGS,
    "compose_hash": COMPOSE_HASH,
    "os_image_hash": OS_IMAGE_HASH,
}


def _plan() -> dict[str, Any]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    return ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": "eval-binding-v2-1",
            "submission_id": "submission-binding-v2-1",
            "submission_version": 1,
            "authorizing_review_digest": "66" * 32,
            "agent_hash": AGENT_HASH,
            "package_tree_sha": "bb" * 32,
            "selected_tasks": [
                {
                    "task_id": "task-a",
                    "image_ref": "registry.example/task@sha256:" + "77" * 32,
                    "task_config_sha256": "88" * 32,
                }
            ],
            "k": 1,
            "scoring_policy": policy,
            "scoring_policy_digest": ew.scoring_policy_digest(policy),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "99" * 32,
                "compose_hash": COMPOSE_HASH,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": {
                    **REGS,
                    "os_image_hash": OS_IMAGE_HASH,
                    "key_provider": "validator-kms",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-binding-v2-1/result",
            "key_release_nonce": "key-release-binding-v2-1",
            "score_nonce": "score-binding-v2-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _score_materials(plan: dict[str, Any]) -> tuple[dict[str, Any], str, list[str]]:
    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    scores_digest = ew.score_record_digest(record)
    task_ids = [task["task_id"] for task in plan["selected_tasks"]]
    return record, scores_digest, task_ids


def _v2_report_data(plan: dict[str, Any], scores_digest: str, task_ids: list[str]) -> str:
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    return ew.score_report_data_hex(binding)


def _legacy_report_data(
    plan: dict[str, Any],
    scores_digest: str,
    task_ids: list[str],
    *,
    validator_nonce: str | None = None,
) -> str:
    """Archival/unit-only legacy construction (validator_nonce preimage)."""
    return rd.report_data_hex(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        task_ids=task_ids,
        scores_digest=scores_digest,
        # Production bug: selfdeploy bound key_release_nonce as validator_nonce.
        validator_nonce=validator_nonce or plan["key_release_nonce"],
    )


def _result_request(
    plan: dict[str, Any],
    *,
    report_data: str,
    record: dict[str, Any],
    scores_digest: str,
) -> dict[str, Any]:
    return {
        # Outer wire schema_version stays 1 — not the score-binding schema.
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": AGENT_HASH,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "cc" * 32,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": "ab",
                "event_log": [],
                "report_data": report_data,
                "measurement": {**MEASUREMENT, "rtmr3": "9" * 96},
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": OS_IMAGE_HASH,
                },
            },
        },
        "guest_artifact_proof": _guest_proof(),
    }


def test_legacy_validator_nonce_report_data_rejected_when_attestation_enabled() -> None:
    """S1: legacy validator_nonce report_data is hard-rejected under attestation."""
    plan = _plan()
    record, scores_digest, task_ids = _score_materials(plan)
    legacy_hex = _legacy_report_data(plan, scores_digest, task_ids)
    v2_hex = _v2_report_data(plan, scores_digest, task_ids)
    assert legacy_hex != v2_hex  # constructions must differ

    with pytest.raises(CanonicalPlanScoringError) as excinfo:
        require_schema_v2_score_report_data(
            reported_report_data=legacy_hex,
            canonical_measurement=MEASUREMENT,
            agent_hash=AGENT_HASH,
            task_ids=task_ids,
            scores_digest=scores_digest,
            eval_run_id=plan["eval_run_id"],
            score_nonce=plan["score_nonce"],
            key_release_nonce=plan["key_release_nonce"],
            phala_attestation_enabled=True,
        )
    assert excinfo.value.reason_code == LEGACY_REPORT_DATA_REJECTED
    assert LEGACY_REPORT_DATA_REJECTED == "legacy_report_data_rejected"

    request = _result_request(
        plan, report_data=legacy_hex, record=record, scores_digest=scores_digest
    )
    with pytest.raises(CanonicalPlanScoringError) as plan_exc:
        validate_eval_result_from_plan(plan, request)
    assert plan_exc.value.reason_code == "legacy_report_data_rejected"


def test_v2_score_binding_report_data_accepted() -> None:
    """S2: schema-v2 score binding report_data passes the binding check (quote mocked)."""
    plan = _plan()
    record, scores_digest, task_ids = _score_materials(plan)
    v2_hex = _v2_report_data(plan, scores_digest, task_ids)

    # Binding check itself — no DCAP/quote required.
    expected = require_schema_v2_score_report_data(
        reported_report_data=v2_hex,
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        task_ids=task_ids,
        scores_digest=scores_digest,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        key_release_nonce=plan["key_release_nonce"],
        phala_attestation_enabled=True,
    )
    assert expected == v2_hex

    request = _result_request(plan, report_data=v2_hex, record=record, scores_digest=scores_digest)
    validated = validate_eval_result_from_plan(plan, request)
    assert validated["scores_digest"] == scores_digest

    # Mock DCAP/quote verification surface: binding check is independent of quote crypto.
    quote_verifier = MagicMock()
    quote_verifier.verify.return_value = MagicMock(tcb_status="UpToDate")
    assert quote_verifier.verify  # mock present; binding already accepted above

    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    err, expected_hex = verify_score_domain_binding(
        score_binding=binding,
        reported_report_data_hex=v2_hex,
        eval_plan=plan,
        scores_digest=scores_digest,
    )
    assert err is None
    assert expected_hex == v2_hex


def test_outer_score_record_schema_v1_still_allowed_when_binding_is_v2() -> None:
    """S3: outer score_record.schema_version=1 is OK when report_data binding is v2.

    Guards against over-rejecting: the outer wire schema is not the score binding.
    """
    plan = _plan()
    record, scores_digest, task_ids = _score_materials(plan)
    assert record["schema_version"] == 1  # outer wire remains v1 by contract
    v2_hex = _v2_report_data(plan, scores_digest, task_ids)

    request = _result_request(plan, report_data=v2_hex, record=record, scores_digest=scores_digest)
    assert request["schema_version"] == 1
    assert request["score_record"]["schema_version"] == 1
    # Binding inside report_data is schema_version 2 (not present on outer score_record).
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    assert binding["schema_version"] == 2

    validated = validate_eval_result_from_plan(plan, request)
    assert validated["score_record"]["schema_version"] == 1
    assert validated["execution_proof"]["attestation"]["report_data"] == v2_hex


def test_v2_binding_with_tampered_scores_digest_rejected() -> None:
    """S4: v2 binding whose scores_digest does not match the request is rejected."""
    plan = _plan()
    record, scores_digest, task_ids = _score_materials(plan)
    tampered_digest = "ff" * 32
    assert tampered_digest != scores_digest

    # report_data bound to a different scores_digest than the request carries.
    tampered_hex = _v2_report_data(plan, tampered_digest, task_ids)
    good_hex = _v2_report_data(plan, scores_digest, task_ids)
    assert tampered_hex != good_hex

    with pytest.raises(CanonicalPlanScoringError) as excinfo:
        require_schema_v2_score_report_data(
            reported_report_data=tampered_hex,
            canonical_measurement=MEASUREMENT,
            agent_hash=AGENT_HASH,
            task_ids=task_ids,
            scores_digest=scores_digest,
            eval_run_id=plan["eval_run_id"],
            score_nonce=plan["score_nonce"],
            key_release_nonce=plan["key_release_nonce"],
            phala_attestation_enabled=True,
        )
    # Tamper is not the legacy path — must not mis-label as legacy.
    assert excinfo.value.reason_code != LEGACY_REPORT_DATA_REJECTED

    request = _result_request(
        plan, report_data=tampered_hex, record=record, scores_digest=scores_digest
    )
    with pytest.raises(CanonicalPlanScoringError) as plan_exc:
        validate_eval_result_from_plan(plan, request)
    assert plan_exc.value.reason_code != "legacy_report_data_rejected"

    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    err, _ = verify_score_domain_binding(
        score_binding=binding,
        reported_report_data_hex=tampered_hex,
        eval_plan=plan,
        scores_digest=scores_digest,
    )
    assert err is not None
    assert err != "legacy_report_data_rejected"


def test_legacy_report_data_also_rejected_on_score_chain_gate() -> None:
    """Dual-flag score-domain re-check surfaces the same stable reason code."""
    plan = _plan()
    _, scores_digest, task_ids = _score_materials(plan)
    legacy_hex = _legacy_report_data(plan, scores_digest, task_ids)
    # A well-formed v2 binding object with a legacy quote report_data field.
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    err, _ = verify_score_domain_binding(
        score_binding=binding,
        reported_report_data_hex=legacy_hex,
        eval_plan=plan,
        scores_digest=scores_digest,
    )
    assert err == "legacy_report_data_rejected"


@pytest.mark.parametrize("bad_schema_version", ["abc", {}, [], b"2"])
def test_malformed_binding_schema_version_refuses_cleanly(
    bad_schema_version: Any,
) -> None:
    """A non-coercible binding ``schema_version`` must refuse, never raise.

    ``verify_score_domain_binding`` is contracted to return
    ``tuple[str | None, str | None]``.  An attacker-influenceable field must not
    escape as an uncaught ValueError/TypeError: that would surface as a 500 and
    erase the stable refuse code from the attestation evidence trail.
    """
    plan = _plan()
    _, scores_digest, task_ids = _score_materials(plan)
    binding = ew.build_score_binding(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    mutated = {**binding, "schema_version": bad_schema_version}
    err, expected_hex = verify_score_domain_binding(
        score_binding=mutated,
        reported_report_data_hex=None,
        eval_plan=plan,
        scores_digest=scores_digest,
    )
    assert err is not None
    assert expected_hex is None
