"""AGATE residual producer wire: harness_entry produces + binds package_residual.

Product residual (agate-wire-residual-producer-harness):
1. After measured LLM rules residual completes successfully, harness path produces
   residual materials and binds them into durable review materials (outcome).
2. On residual reject/fail, bind reject so prepare stays refuse.
3. Honest prepare can authorize when materials present.
4. Keep fail-closed without residual.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agent_challenge.evaluation.fresh_review_gate import (
    admit_eval_cvm_fresh_review,
    admit_eval_cvm_launch_from_assignment,
)
from agent_challenge.evaluation.llm_rules_residual import (
    MEASURED_RESIDUAL_KIND,
    REFUSE_RESIDUAL_FAIL,
    REFUSE_RESIDUAL_MISSING,
    admit_package_residual_for_eval,
    extract_package_residual,
)
from agent_challenge.review.harness_entry import (
    PRODUCT_HARNESS_KIND,
    admit_product_review_entry,
    bind_measured_residual_into_review_materials,
    map_decision_verdict_to_residual_verdict,
    produce_package_residual_from_identity,
)
from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    build_decision,
    build_observed_openrouter_transport,
    build_openrouter_observation,
    build_planned_openrouter_request,
    build_policy_observation,
    build_review_core_minimal,
    planned_request_sha256,
    review_digest,
    review_report_data_hex,
    sha256_hex,
)
from agent_challenge.review.report import merge_package_residual_into_outcome_dict

SAMPLE_ZIP = b"PK\x03\x04agate-residual-producer-zip-v1"
ENTRY_ID = "python -m agent_challenge.selfdeploy"
ENTRY_BYTES = b'#!/usr/bin/env python3\n"""selfdeploy residual producer"""\n'
SAMPLE_RULES: dict[str, bytes] = {
    ".rules/acceptance.md": b"# acceptance\nMeasured residual required.\n",
    ".rules/security.md": b"# security\nFail closed without residual.\n",
}
TREE = "ab" * 32
T0 = 1_700_000_000_000
ROUTING = sha256_hex(b'{"order":["residual-producer"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"gen-producer","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-residual-producer")


def _admit_identity():
    return admit_product_review_entry(
        agent_zip_bytes=SAMPLE_ZIP,
        entry_script_identity=ENTRY_ID,
        entry_script_bytes=ENTRY_BYTES,
        rules_files=SAMPLE_RULES,
    )


def _times() -> dict[str, int]:
    return {
        "issued_at_ms": T0,
        "started_at_ms": T0,
        "model_call_marked_at_ms": T0 + 1,
        "request_started_at_ms": T0 + 2,
        "request_finished_at_ms": T0 + 3,
        "verifier_finished_at_ms": T0 + 4,
        "report_finished_at_ms": T0 + 5,
        "expires_at_ms": T0 + 3_600_000,
        "submission_received_at_ms": T0 + 60_000,
    }


def _core(*, verdict: str = "allow") -> dict[str, Any]:
    planned = build_planned_openrouter_request(
        body_sha256=BODY_SHA,
        body_length=len(BODY),
        routing_sha256=ROUTING,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_observed_openrouter_transport(
        planned_request_sha256_=p_digest,
        response_body_sha256=RESP_SHA,
        response_body_length=len(RESP),
        metadata_sha256=META,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=BODY_SHA,
        request_body_length=len(BODY),
        response_id="gen-producer",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-producer",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-producer",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-producer",
        routing_sha256=ROUTING,
    )
    return build_review_core_minimal(
        session_id="rs-producer",
        assignment_id="ra-producer",
        submission_id="sub-producer",
        review_nonce="nonce-producer",
        assignment_digest="aa" * 32,
        rules_observation={
            "snapshot_sha256": "55" * 32,
            "revision_id": "rules-rev-producer",
        },
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict=verdict),
        times=_times(),
    )


def _envelope_without_residual(*, verdict: str = "allow") -> dict[str, Any]:
    core = _core(verdict=verdict)
    return {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": review_digest(core),
        "report_data_hex": review_report_data_hex(core),
        "review_core": core,
        "attestation": {
            "tdx_quote_hex": "00" * 16,
            "event_log": [],
            "measurement": {},
        },
    }


# ---------------------------------------------------------------------------
# Producer from harness identity
# --------------------------------------------------------------------------- #


def test_harness_path_produces_residual_materials_on_allow() -> None:
    identity = _admit_identity()
    materials = produce_package_residual_from_identity(
        identity,
        residual_verdict="allow",
        package_tree_sha=TREE,
    )
    assert materials.residual_kind == MEASURED_RESIDUAL_KIND
    assert materials.residual_verdict == "allow"
    assert materials.rules_bundle_sha256 == identity.rules_bundle_sha256
    assert materials.rules_version == identity.rules_version
    assert materials.rules_file_digests == dict(identity.rules_file_digests)
    assert materials.package_tree_sha == TREE
    assert materials.harness_kind == PRODUCT_HARNESS_KIND
    assert len(materials.residual_digest) == 64

    decision = admit_package_residual_for_eval(
        residual=materials.as_dict(),
        dual_flags_on=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.admitted is True


def test_harness_path_produces_reject_residual() -> None:
    identity = _admit_identity()
    materials = produce_package_residual_from_identity(
        identity,
        residual_verdict="reject",
        package_tree_sha=TREE,
    )
    assert materials.residual_verdict == "reject"
    decision = admit_package_residual_for_eval(
        residual=materials.as_dict(),
        dual_flags_on=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_FAIL


def test_bind_measured_residual_into_outcome_and_envelope() -> None:
    identity = _admit_identity()
    env0 = {"schema_version": 1, "domain": REVIEW_REPORT_DOMAIN}
    out0 = {
        "status": "verified_allow",
        "terminal": True,
        "retryable": False,
        "reason_code": "review_verified",
        "nonce_consumed": True,
        "measurement_allowlisted": True,
        "report_data_matched": True,
        "verified_at_ms": T0,
    }
    bound = bind_measured_residual_into_review_materials(
        identity=identity,
        residual_verdict="allow",
        package_tree_sha=TREE,
        envelope=env0,
        outcome=out0,
    )
    assert "package_residual" in bound["envelope"]
    assert "package_residual" in bound["outcome"]
    bag = bound["outcome"]["package_residual"]
    assert bag["residual_verdict"] == "allow"
    assert bag["rules_bundle_sha256"] == identity.rules_bundle_sha256
    assert bag["package_tree_sha"] == TREE
    extracted = extract_package_residual(outcome=bound["outcome"])
    assert extracted is not None
    assert extracted.residual_digest == bag["residual_digest"]


def test_map_decision_verdict_to_residual_verdict() -> None:
    assert map_decision_verdict_to_residual_verdict("allow") == "allow"
    assert map_decision_verdict_to_residual_verdict("ALLOW") == "allow"
    assert map_decision_verdict_to_residual_verdict("reject") == "reject"
    assert map_decision_verdict_to_residual_verdict("escalate") == "fail"
    assert map_decision_verdict_to_residual_verdict("unknown") == "fail"
    assert map_decision_verdict_to_residual_verdict(None) == "fail"


def test_merge_package_residual_into_outcome_dict_from_identity_json() -> None:
    identity = _admit_identity()
    identity_json = json.dumps(identity.as_dict(), sort_keys=True, separators=(",", ":"))
    base = {
        "status": "verified_allow",
        "terminal": True,
        "retryable": False,
        "reason_code": "review_verified",
        "nonce_consumed": True,
        "measurement_allowlisted": True,
        "report_data_matched": True,
        "verified_at_ms": T0,
    }
    merged = merge_package_residual_into_outcome_dict(
        outcome=base,
        harness_identity_json=identity_json,
        package_tree_sha=TREE,
        decision_verdict="allow",
    )
    assert merged is not None
    assert merged["status"] == "verified_allow"
    assert merged["package_residual"]["residual_verdict"] == "allow"
    assert merged["package_residual"]["package_tree_sha"] == TREE


def test_merge_skips_when_identity_missing() -> None:
    base = {"status": "verified_allow", "terminal": True}
    merged = merge_package_residual_into_outcome_dict(
        outcome=base,
        harness_identity_json=None,
        package_tree_sha=TREE,
        decision_verdict="allow",
    )
    assert merged is None


# ---------------------------------------------------------------------------
# Prepare / authorize path with producer-bound materials
# --------------------------------------------------------------------------- #


def test_honest_prepare_admits_when_producer_residual_on_outcome() -> None:
    identity = _admit_identity()
    env = _envelope_without_residual(verdict="allow")
    out0 = {
        "status": "verified_allow",
        "terminal": True,
        "retryable": False,
        "reason_code": "review_verified",
        "nonce_consumed": True,
        "measurement_allowlisted": True,
        "report_data_matched": True,
        "verified_at_ms": T0,
    }
    bound = bind_measured_residual_into_review_materials(
        identity=identity,
        residual_verdict="allow",
        package_tree_sha=TREE,
        outcome=out0,
    )
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
        outcome=bound["outcome"],
    )
    assert decision.may_launch is True
    assert decision.verdict == "allow"

    assignment = SimpleNamespace(
        phase="review_allowed",
        review_verification_outcome_json=json.dumps(bound["outcome"]),
        review_report_envelope_json=json.dumps(env),
        review_report_data_hex=env["report_data_hex"],
        review_digest=env["review_digest"],
    )
    launch = admit_eval_cvm_launch_from_assignment(
        assignment,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
    )
    assert launch.may_launch is True


def test_reject_residual_from_producer_stops_prepare() -> None:
    identity = _admit_identity()
    env = _envelope_without_residual(verdict="reject")
    out0 = {
        "status": "verified_reject",
        "terminal": True,
        "retryable": False,
        "reason_code": "review_verified",
        "nonce_consumed": True,
        "measurement_allowlisted": True,
        "report_data_matched": True,
        "verified_at_ms": T0,
    }
    bound = bind_measured_residual_into_review_materials(
        identity=identity,
        residual_verdict="reject",
        package_tree_sha=TREE,
        outcome=out0,
    )
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
        outcome=bound["outcome"],
        # plain allow would still be refused by residual reject
    )
    # Fresh gate may fail earlier on reject verdict; residual also refuse.
    assert decision.may_launch is False
    residual = admit_package_residual_for_eval(
        outcome=bound["outcome"],
        dual_flags_on=True,
        expected_package_tree_sha=TREE,
    )
    assert residual.admitted is False
    assert residual.reason_code == REFUSE_RESIDUAL_FAIL


def test_fail_closed_without_residual_materials() -> None:
    env = _envelope_without_residual(verdict="allow")
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
        outcome={
            "status": "verified_allow",
            "terminal": True,
            "retryable": False,
            "reason_code": "review_verified",
            "nonce_consumed": True,
            "measurement_allowlisted": True,
            "report_data_matched": True,
            "verified_at_ms": T0,
        },
    )
    assert decision.may_launch is False
    assert decision.reason_code in {REFUSE_RESIDUAL_MISSING, "package_residual_missing"}


# ---------------------------------------------------------------------------
# Guest POST /report payload contract (extra=forbid)
# --------------------------------------------------------------------------- #


def _guest_shaped_report_payload(
    *,
    envelope: dict[str, Any] | None = None,
    with_package_residual: bool = False,
) -> dict[str, Any]:
    """Mirror review_runtime submission builder shape for POST /report."""
    env = envelope if envelope is not None else _envelope_without_residual(verdict="allow")
    evidence = {
        "planned_request_b64": "cGxhbm5lZA==",
        "transport_observation_b64": "b2JzZXJ2ZWQ=",
        "request_body_b64": "cmVxdWVzdA==",
        "response_body_b64": "cmVzcG9uc2U=",
    }
    payload: dict[str, Any] = {"envelope": env, "evidence": evidence}
    if with_package_residual:
        # Deliberately forbidden top-level key (guest must never send this).
        identity = _admit_identity()
        residual = produce_package_residual_from_identity(
            identity,
            residual_verdict="allow",
            package_tree_sha=TREE,
        ).as_dict()
        payload["package_residual"] = residual
    return payload


def test_guest_shaped_report_payload_is_envelope_and_evidence_only() -> None:
    """Guest POST body must be envelope+evidence; residual is host-bound only."""
    from pydantic import ValidationError

    from agent_challenge.api.routes import ReviewReportSubmission

    guest_payload = _guest_shaped_report_payload(with_package_residual=False)
    assert set(guest_payload.keys()) == {"envelope", "evidence"}
    assert "package_residual" not in guest_payload

    accepted = ReviewReportSubmission.model_validate(guest_payload)
    assert accepted.envelope == guest_payload["envelope"]
    assert accepted.evidence == guest_payload["evidence"]

    # Top-level package_residual is extra=forbid → ValidationError (live 422).
    forbidden = _guest_shaped_report_payload(with_package_residual=True)
    assert "package_residual" in forbidden
    try:
        ReviewReportSubmission.model_validate(forbidden)
        raise AssertionError("expected ValidationError for top-level package_residual")
    except ValidationError as exc:
        err_text = str(exc)
        assert "package_residual" in err_text
        assert "extra" in err_text.lower() or "forbid" in err_text.lower()


def test_host_bound_residual_still_admits_honest_allow_after_guest_schema_fix() -> None:
    """After guest schema fix, host merge path still produces residual for allow."""
    identity = _admit_identity()
    identity_json = json.dumps(identity.as_dict(), sort_keys=True, separators=(",", ":"))
    guest_payload = _guest_shaped_report_payload(with_package_residual=False)
    from agent_challenge.api.routes import ReviewReportSubmission

    # Guest payload validates (no residual on wire).
    ReviewReportSubmission.model_validate(guest_payload)

    base_outcome = {
        "status": "verified_allow",
        "terminal": True,
        "retryable": False,
        "reason_code": "review_verified",
        "nonce_consumed": True,
        "measurement_allowlisted": True,
        "report_data_matched": True,
        "verified_at_ms": T0,
    }
    merged = merge_package_residual_into_outcome_dict(
        outcome=base_outcome,
        harness_identity_json=identity_json,
        package_tree_sha=TREE,
        decision_verdict="allow",
    )
    assert merged is not None
    assert "package_residual" in merged
    assert merged["package_residual"]["residual_verdict"] == "allow"

    decision = admit_eval_cvm_fresh_review(
        envelope=guest_payload["envelope"],
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
        outcome=merged,
    )
    assert decision.may_launch is True
    assert decision.verdict == "allow"
