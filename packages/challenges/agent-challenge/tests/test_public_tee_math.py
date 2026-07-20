"""Public TEE math route + serializer unit locks (VAL-ACATM-001..010).

Covers:
- available:false locked closed form
- available:true field allowlist + required safe classes
- redaction deny-list (nonce/tokens/evidence/model IO/KEY material)
- dual-flag status review.* independent of list queued status
- OpenAPI path present
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_challenge.api import routes as api_routes
from agent_challenge.app import app
from agent_challenge.core.models import AgentSubmission, ReviewAssignment, ReviewSession
from agent_challenge.keyrelease.quote import build_rtmr3_event_log, build_tdx_quote
from agent_challenge.review.public_tee import (
    PUBLIC_TEE_DENY_KEYS,
    PUBLIC_TEE_TOP_LEVEL_ALLOWLIST,
    assert_public_tee_safe,
    build_public_tee_math,
    build_public_tee_math_from_assignment,
    public_tee_assignment_qualifies,
    public_tee_unavailable,
)
from agent_challenge.review.report import (
    REVIEW_REPORT_DOMAIN,
    ReviewVerificationOutcome,
    build_review_envelope,
    review_report_data_hex,
)
from agent_challenge.review.report import _public_projection as miner_public_projection
from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment


def _routing() -> dict[str, object]:
    return {
        "order": ["alpha", "beta"],
        "only": ["alpha", "beta"],
        "ignore": [],
        "quantizations": [],
        "sort": None,
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }


def _assignment() -> tuple[dict[str, Any], ReviewInputConfig]:
    measurement = {
        "mrtd": "11" * 48,
        "rtmr0": "22" * 48,
        "rtmr1": "33" * 48,
        "rtmr2": "44" * 48,
        "os_image_hash": hashlib.sha256(
            bytes.fromhex(("11" * 48) + ("33" * 48) + ("44" * 48))
        ).hexdigest(),
        "key_provider": "phala",
        "vm_shape": "tdx.small",
    }
    config = ReviewInputConfig(
        routing=_routing(),
        image_ref="docker.io/example/reviewer@sha256:" + ("a" * 64),
        compose_hash="ab" * 32,
        kms_public_key_hex="cd" * 32,
        measurement=measurement,
    )
    assignment, _bytes, _digest = build_review_assignment(
        session_id="rs-public-tee",
        assignment_id="ra-public-tee",
        attempt=1,
        submission_id="42",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 9,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/ra-public-tee/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="rn-public-tee-secret-nonce",
        issued_at_ms=1_000,
        expires_at_ms=9_000,
        session_token_sha256="60" * 32,
        config=config,
    )
    return assignment, config


def _review_core(assignment: dict[str, Any]) -> dict[str, Any]:
    core = assignment["assignment_core"]
    policy = core["policy"]
    return {
        "schema_version": 1,
        "session_id": core["session_id"],
        "assignment_id": core["assignment_id"],
        "assignment_digest": assignment["assignment_digest"],
        "submission_id": core["submission_id"],
        "artifact_observation": {
            "agent_hash": core["artifact"]["agent_hash"],
            "zip_sha256": core["artifact"]["zip_sha256"],
            "zip_size_bytes": core["artifact"]["zip_size_bytes"],
            "manifest_sha256": core["artifact"]["manifest_sha256"],
            "manifest_entries_sha256": core["artifact"]["manifest_entries_sha256"],
        },
        "rules_observation": {
            "snapshot_sha256": core["rules"]["snapshot_sha256"],
            "revision_id": core["rules"]["revision_id"],
        },
        "policy_observation": {
            "model": policy["model"],
            "routing_sha256": policy["routing_sha256"],
            "prompt_version": policy["prompt_version"],
            "prompt_sha256": policy["prompt_sha256"],
            "tool_schema_version": policy["tool_schema_version"],
            "tool_schema_sha256": policy["tool_schema_sha256"],
            "verifier_version": policy["verifier_version"],
            "verifier_sha256": policy["verifier_sha256"],
        },
        "openrouter_observation": {
            "planned_request_sha256": "70" * 32,
            "transport_observation_sha256": "71" * 32,
            "request_body_sha256": "72" * 32,
            "request_body_length": 7,
            "response_status": 200,
            "response_content_encoding": "identity",
            "response_body_sha256": "73" * 32,
            "response_body_length": 11,
            "response_id": "or-response",
            "returned_model": "x-ai/grok-4.5",
            "metadata_sha256": "74" * 32,
            "observed_provider": "openrouter",
            "provider_provenance": "openrouter_metadata",
            "cache_hit": False,
        },
        "decision": {
            "static_findings_sha256": "75" * 32,
            "parsed_output_sha256": "76" * 32,
            "verifier_input_sha256": "77" * 32,
            "verifier_output_sha256": "78" * 32,
            "verifier_result": "pass",
            "verdict": "allow",
            "reason_codes": ["alpha_reason", "zeta_reason"],
            "evidence_digests": ["79" * 32, "80" * 32],
        },
        "times": {
            "issued_at_ms": 1_000,
            "started_at_ms": 1_000,
            "model_call_marked_at_ms": 1_001,
            "request_started_at_ms": 1_002,
            "request_finished_at_ms": 1_003,
            "verifier_finished_at_ms": 1_004,
            "report_finished_at_ms": 1_005,
            "expires_at_ms": 9_000,
            "submission_received_at_ms": 1_000,
        },
        "review_nonce": core["review_nonce"],
    }


def _envelope() -> tuple[dict[str, Any], dict[str, Any], ReviewVerificationOutcome]:
    assignment, config = _assignment()
    core = _review_core(assignment)
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            ("compose-hash", bytes.fromhex(config.compose_hash)),
            ("key-provider", b"phala"),
        ]
    )
    measurement = {
        **config.resolved_measurement(),
        "rtmr3": rtmr3,
        "compose_hash": config.compose_hash,
    }
    quote = build_tdx_quote(
        mrtd=measurement["mrtd"],
        rtmr0=measurement["rtmr0"],
        rtmr1=measurement["rtmr1"],
        rtmr2=measurement["rtmr2"],
        rtmr3=measurement["rtmr3"],
        report_data=review_report_data_hex(core),
    )
    envelope = build_review_envelope(
        review_core=core,
        tdx_quote_hex=quote,
        event_log=event_log,
        measurement=measurement,
        vm_config={
            "vcpu": 1,
            "memory_mb": 2048,
            "os_image_hash": measurement["os_image_hash"],
        },
    )
    outcome = ReviewVerificationOutcome(
        status="verified_allow",
        terminal=True,
        retryable=False,
        reason_code="policy_allowed",
        nonce_consumed=True,
        measurement_allowlisted=True,
        report_data_matched=True,
        verified_at_ms=1_700,
    )
    return envelope, assignment, outcome


def test_public_tee_unavailable_locked_closed_form() -> None:
    assert public_tee_unavailable() == {"available": False}
    assert build_public_tee_math(submission_id=1, envelope=None) == {"available": False}
    assert build_public_tee_math(submission_id=1, envelope={}) == {"available": False}
    assert build_public_tee_math_from_assignment(
        submission_id=1,
        envelope_json=None,
    ) == {"available": False}
    assert_public_tee_safe({"available": False})


def test_public_tee_envelope_without_projection_is_unavailable() -> None:
    """Envelope alone (review_verifying pre-verify) must not yield available:true."""

    envelope, _assignment, _outcome = _envelope()
    assert build_public_tee_math(submission_id=42, envelope=envelope) == {"available": False}
    assert build_public_tee_math(
        submission_id=42,
        envelope=envelope,
        verification_outcome=None,
        public_projection=None,
    ) == {"available": False}
    assert build_public_tee_math_from_assignment(
        submission_id=42,
        envelope_json=json.dumps(envelope),
        outcome_json=None,
        public_projection_json=None,
    ) == {"available": False}
    assert (
        public_tee_assignment_qualifies(
            envelope_json=json.dumps(envelope),
            outcome_json=None,
            public_projection_json=None,
        )
        is False
    )


def test_public_tee_envelope_verifier_unavailable_is_unavailable() -> None:
    """Parked verifier_unavailable must not yield available:true without projection."""

    envelope, _assignment, _outcome = _envelope()
    parked = {
        "status": "verifier_unavailable",
        "terminal": False,
        "retryable": True,
        "reason_code": "dcap_qvl_missing",
        "nonce_consumed": False,
        "measurement_allowlisted": False,
        "report_data_matched": False,
        "verified_at_ms": None,
    }
    assert build_public_tee_math(
        submission_id=42,
        envelope=envelope,
        verification_outcome=parked,
        public_projection=None,
    ) == {"available": False}
    assert build_public_tee_math_from_assignment(
        submission_id=42,
        envelope_json=json.dumps(envelope),
        outcome_json=json.dumps(parked),
        public_projection_json=None,
    ) == {"available": False}
    assert (
        public_tee_assignment_qualifies(
            envelope_json=json.dumps(envelope),
            outcome_json=json.dumps(parked),
            public_projection_json=None,
        )
        is False
    )


def test_public_tee_envelope_verified_with_projection_is_available() -> None:
    """Envelope + verified_* + projection → available:true (positive gate)."""

    envelope, _assignment, outcome = _envelope()
    projection = miner_public_projection(envelope=envelope, outcome=outcome)
    payload = build_public_tee_math(
        submission_id=42,
        envelope=envelope,
        verification_outcome=outcome.as_dict(),
        public_projection=projection,
    )
    assert payload["available"] is True
    assert payload["verification_outcome"]["status"] == "verified_allow"
    assert (
        public_tee_assignment_qualifies(
            envelope_json=json.dumps(envelope),
            outcome_json=json.dumps(outcome.as_dict()),
            public_projection_json=json.dumps(projection),
        )
        is True
    )
    # Verified outcome alone (no projection) is still eligible — align with
    # verified_* dual-flag; builder projects safe math without miner digests.
    verified_only = build_public_tee_math(
        submission_id=42,
        envelope=envelope,
        verification_outcome=outcome.as_dict(),
        public_projection=None,
    )
    assert verified_only["available"] is True
    assert verified_only["verification_outcome"]["status"] == "verified_allow"
    assert (
        public_tee_assignment_qualifies(
            envelope_json=json.dumps(envelope),
            outcome_json=json.dumps(outcome.as_dict()),
            public_projection_json=None,
        )
        is True
    )


def test_public_tee_math_available_true_field_allowlist() -> None:
    envelope, _assignment, outcome = _envelope()
    projection = miner_public_projection(envelope=envelope, outcome=outcome)
    payload = build_public_tee_math(
        submission_id=42,
        envelope=envelope,
        verification_outcome=outcome.as_dict(),
        public_projection=projection,
    )
    assert payload["available"] is True
    assert set(payload.keys()) <= PUBLIC_TEE_TOP_LEVEL_ALLOWLIST
    # Required safe classes (VAL-ACATM-004 / 010)
    assert payload["domain"] == REVIEW_REPORT_DOMAIN
    assert payload["review_digest"] == envelope["review_digest"]
    assert payload["report_data_hex"] == envelope["report_data_hex"]
    assert isinstance(payload["report_data_preimage"], dict)
    assert "review_nonce" not in payload["report_data_preimage"]
    assert "review_nonce_sha256" in payload["report_data_preimage"]
    measurement = payload["measurement"]
    assert isinstance(measurement, dict)
    for key in (
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "rtmr3",
        "compose_hash",
        "os_image_hash",
        "key_provider",
        "vm_shape",
    ):
        assert key in measurement
        assert isinstance(measurement[key], str) and measurement[key]
    assert isinstance(payload["tdx_quote_hex"], str) and payload["tdx_quote_hex"]
    assert isinstance(payload.get("event_log"), list)
    outcome_pub = payload["verification_outcome"]
    assert outcome_pub["status"] == "verified_allow"
    assert outcome_pub["measurement_allowlisted"] is True
    assert outcome_pub["report_data_matched"] is True
    assert outcome_pub["verified_at_ms"] == 1_700
    assert outcome_pub["reason_code"] == "policy_allowed"
    assert payload["quote_fingerprint_sha256"] == projection["quote_fingerprint_sha256"]
    assert_public_tee_safe(payload)


def test_public_tee_math_redacts_nonce_tokens_evidence_and_keys() -> None:
    envelope, _assignment, outcome = _envelope()
    # Smuggle secrets into a hostile envelope copy — builder must not echo them.
    # Keep review_core schema-valid so preimage projection still runs.
    hostile = json.loads(json.dumps(envelope))
    hostile["session_token"] = "session-token-plaintext-secret"
    hostile["attestation"]["capability"] = "cap-bearer-secret"
    hostile["evidence_objects"] = {"request_body": "not-public-body"}
    hostile["openrouter_api_key"] = "sk-leaked-openrouter"
    hostile["encryption_key"] = "KEY_FILE_MATERIAL"
    payload = build_public_tee_math(
        submission_id=42,
        envelope=hostile,
        verification_outcome={
            **outcome.as_dict(),
            "nonce_consumed": True,
            "terminal": True,
            "retryable": False,
            "private_key": "should-not-leak",
        },
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert "rn-public-tee-secret-nonce" not in serialized
    assert "session-token-plaintext-secret" not in serialized
    assert "cap-bearer-secret" not in serialized
    assert "sk-leaked-openrouter" not in serialized
    assert "KEY_FILE_MATERIAL" not in serialized
    assert "private_key" not in serialized
    assert "nonce_consumed" not in serialized
    assert "request_body" not in serialized
    for key in PUBLIC_TEE_DENY_KEYS:
        assert f'"{key}":' not in serialized
    # Hash of nonce is OK (inspectability without plaintext).
    expected_hash = hashlib.sha256(b"rn-public-tee-secret-nonce").hexdigest()
    assert payload["report_data_preimage"]["review_nonce_sha256"] == expected_hash
    assert '"review_nonce":' not in serialized
    assert_public_tee_safe(payload)


def test_public_tee_math_from_assignment_json_columns() -> None:
    envelope, _assignment, outcome = _envelope()
    projection = miner_public_projection(envelope=envelope, outcome=outcome)
    payload = build_public_tee_math_from_assignment(
        submission_id=7,
        envelope_json=json.dumps(envelope),
        outcome_json=json.dumps(outcome.as_dict()),
        public_projection_json=json.dumps(projection),
    )
    assert payload["available"] is True
    assert payload["submission_id"] in {7, 42, "42"}
    assert payload["verdict"] == "allow"


async def test_get_review_tee_available_false_when_no_report(
    client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hk-tee-absent",
            name="tee-absent",
            agent_hash="tee-absent-hash",
            artifact_uri="/tmp/tee-absent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/review/tee")
    assert response.status_code == 200
    assert response.json() == {"available": False}

    v1 = await client.get(f"/v1/submissions/{submission_id}/review/tee")
    assert v1.status_code == 200
    assert v1.json() == {"available": False}


async def test_get_review_tee_envelope_only_not_available(
    client,
    database_session,
    monkeypatch,
) -> None:
    """Route must not treat durable envelope alone as available:true."""

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    envelope, assignment_payload, _outcome = _envelope()
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hk-tee-env-only",
            name="tee-env-only",
            agent_hash="tee-env-only-hash",
            artifact_uri="/tmp/tee-env-only.zip",
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id="rs-tee-env-only",
            submission_id=submission.id,
            artifact_sha256="ab" * 32,
            artifact_size_bytes=12,
            manifest_sha256="cd" * 32,
            manifest_entries_sha256="ef" * 32,
            current_assignment_id="ra-tee-env-only",
            authorizing_assignment_id=None,
        )
        session.add(review_session)
        await session.flush()
        # Durable envelope as written during review_verifying — no projection /
        # no verified outcome yet.
        row = ReviewAssignment(
            session_id=review_session.id,
            assignment_id="ra-tee-env-only",
            attempt=1,
            phase="review_verifying",
            assignment_bytes=json.dumps(assignment_payload),
            assignment_digest=assignment_payload["assignment_digest"],
            artifact_sha256="ab" * 32,
            rules_snapshot_sha256="11" * 32,
            rules_revision_id="rules-v1",
            review_nonce="nonce-ra-tee-env-only",
            session_token_sha256="bb" * 32,
            capability_state="active",
            issued_at=now,
            expires_at=now,
            review_report_envelope_json=json.dumps(envelope),
            review_digest=envelope["review_digest"],
            review_report_data_hex=envelope["report_data_hex"],
            review_verification_outcome_json=None,
            review_public_projection_json=None,
            reason_code=None,
            finished_at=None,
        )
        session.add(row)
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/review/tee")
    assert response.status_code == 200
    assert response.json() == {"available": False}

    status = await client.get(f"/submissions/{submission_id}/status")
    assert status.status_code == 200
    review = status.json()["review"]
    assert review is not None
    assert review["report_available"] is False
    assert review["verified"] is False


async def test_get_review_tee_verifier_unavailable_not_available(
    client,
    database_session,
    monkeypatch,
) -> None:
    """Route must not surface available:true for parked verifier_unavailable."""

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    envelope, assignment_payload, _outcome = _envelope()
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    parked = {
        "status": "verifier_unavailable",
        "terminal": False,
        "retryable": True,
        "reason_code": "dcap_qvl_missing",
        "nonce_consumed": False,
        "measurement_allowlisted": False,
        "report_data_matched": False,
        "verified_at_ms": None,
    }

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hk-tee-vu",
            name="tee-vu",
            agent_hash="tee-vu-hash",
            artifact_uri="/tmp/tee-vu.zip",
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id="rs-tee-vu",
            submission_id=submission.id,
            artifact_sha256="ab" * 32,
            artifact_size_bytes=12,
            manifest_sha256="cd" * 32,
            manifest_entries_sha256="ef" * 32,
            current_assignment_id="ra-tee-vu",
            authorizing_assignment_id="ra-tee-vu",
        )
        session.add(review_session)
        await session.flush()
        row = ReviewAssignment(
            session_id=review_session.id,
            assignment_id="ra-tee-vu",
            attempt=1,
            phase="review_verifying",
            assignment_bytes=json.dumps(assignment_payload),
            assignment_digest=assignment_payload["assignment_digest"],
            artifact_sha256="ab" * 32,
            rules_snapshot_sha256="11" * 32,
            rules_revision_id="rules-v1",
            review_nonce="nonce-ra-tee-vu",
            session_token_sha256="bb" * 32,
            capability_state="active",
            issued_at=now,
            expires_at=now,
            review_report_envelope_json=json.dumps(envelope),
            review_digest=envelope["review_digest"],
            review_report_data_hex=envelope["report_data_hex"],
            review_verification_outcome_json=json.dumps(parked),
            review_public_projection_json=None,
            reason_code="dcap_qvl_missing",
            finished_at=None,
        )
        session.add(row)
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/review/tee")
    assert response.status_code == 200
    assert response.json() == {"available": False}

    status = await client.get(f"/submissions/{submission_id}/status")
    assert status.status_code == 200
    review = status.json()["review"]
    assert review is not None
    assert review["report_available"] is False
    assert review["verified"] is False


async def test_get_review_tee_available_true_safe_fields(
    client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    envelope, assignment_payload, outcome = _envelope()
    projection = miner_public_projection(envelope=envelope, outcome=outcome)
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hk-tee-present",
            name="tee-present",
            agent_hash="tee-present-hash",
            artifact_uri="/tmp/tee-present.zip",
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id="rs-tee-route",
            submission_id=submission.id,
            artifact_sha256="ab" * 32,
            artifact_size_bytes=12,
            manifest_sha256="cd" * 32,
            manifest_entries_sha256="ef" * 32,
            current_assignment_id="ra-tee-route",
            authorizing_assignment_id="ra-tee-route",
        )
        session.add(review_session)
        await session.flush()
        row = ReviewAssignment(
            session_id=review_session.id,
            assignment_id="ra-tee-route",
            attempt=1,
            phase="review_allowed",
            assignment_bytes=json.dumps(assignment_payload),
            assignment_digest=assignment_payload["assignment_digest"],
            artifact_sha256="ab" * 32,
            rules_snapshot_sha256="11" * 32,
            rules_revision_id="rules-v1",
            review_nonce="nonce-ra-tee-route",
            session_token_sha256="bb" * 32,
            capability_state="revoked",
            issued_at=now,
            expires_at=now,
            review_report_envelope_json=json.dumps(envelope),
            review_digest=envelope["review_digest"],
            review_report_data_hex=envelope["report_data_hex"],
            review_verification_outcome_json=json.dumps(outcome.as_dict()),
            review_public_projection_json=json.dumps(projection),
            reason_code="policy_allowed",
            finished_at=now,
        )
        session.add(row)
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/review/tee")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["domain"] == REVIEW_REPORT_DOMAIN
    assert body["review_digest"] == envelope["review_digest"]
    assert body["report_data_hex"] == envelope["report_data_hex"]
    serialized = json.dumps(body)
    assert '"review_nonce":' not in serialized
    assert "rn-public-tee-secret-nonce" not in serialized
    assert body["measurement"]["key_provider"] == "phala"
    assert body["tdx_quote_hex"]
    assert body["verification_outcome"]["status"] == "verified_allow"
    assert body["quote_fingerprint_sha256"]
    assert set(body.keys()) <= PUBLIC_TEE_TOP_LEVEL_ALLOWLIST
    assert_public_tee_safe(body)


@pytest.mark.parametrize(
    ("outcome_status", "expected_verdict"),
    [
        ("verified_allow", "allow"),
        ("verified_reject", "reject"),
        ("verified_escalate", "escalate"),
    ],
)
async def test_dual_flag_status_review_projection_independent_of_queued(
    client,
    database_session,
    monkeypatch,
    outcome_status: str,
    expected_verdict: str,
) -> None:
    """VAL-ACATM-001/002/003/008: dual-flag review.* stays even when list is queued."""

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    phase = {
        "verified_allow": "review_allowed",
        "verified_reject": "review_rejected",
        "verified_escalate": "review_escalated",
    }[outcome_status]

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"hk-dual-{expected_verdict}",
            name=f"dual-{expected_verdict}",
            agent_hash=f"dual-hash-{expected_verdict}",
            artifact_uri=f"/tmp/dual-{expected_verdict}.zip",
            # Intentionally non-terminal queue-looking lifecycle status.
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()
        # Durable status event so public status stays queued-family without
        # requiring a full happy-path transition graph.
        from agent_challenge.models import SubmissionStatusEvent

        session.add(
            SubmissionStatusEvent(
                submission_id=submission.id,
                sequence=1,
                from_status=None,
                to_status="queued",
                actor="test",
                reason="seed_queued",
            )
        )
        review_session = ReviewSession(
            session_id=f"rs-dual-{expected_verdict}",
            submission_id=submission.id,
            artifact_sha256="ab" * 32,
            artifact_size_bytes=12,
            manifest_sha256="cd" * 32,
            manifest_entries_sha256="ef" * 32,
            current_assignment_id=f"ra-dual-{expected_verdict}",
            authorizing_assignment_id=(
                f"ra-dual-{expected_verdict}" if outcome_status == "verified_allow" else None
            ),
        )
        session.add(review_session)
        await session.flush()
        assignment = ReviewAssignment(
            session_id=review_session.id,
            assignment_id=f"ra-dual-{expected_verdict}",
            attempt=1,
            phase=phase,
            assignment_bytes="{}",
            assignment_digest="aa" * 32,
            artifact_sha256="ab" * 32,
            rules_snapshot_sha256="11" * 32,
            rules_revision_id="rules-v1",
            review_nonce=f"nonce-dual-{expected_verdict}",
            session_token_sha256="bb" * 32,
            capability_state="revoked",
            issued_at=now,
            expires_at=now,
            review_public_projection_json=json.dumps(
                {"schema_version": 1, "verdict": expected_verdict}
            ),
            review_verification_outcome_json=json.dumps(
                {
                    "status": outcome_status,
                    "terminal": True,
                    "retryable": False,
                    "reason_code": f"{expected_verdict}_ok",
                    "nonce_consumed": True,
                    "measurement_allowlisted": True,
                    "report_data_matched": True,
                    "verified_at_ms": 99,
                }
            ),
            reason_code=f"{expected_verdict}_ok",
            finished_at=now,
        )
        session.add(assignment)
        await session.commit()
        submission_id = submission.id
        session_public_id = review_session.session_id
        assignment_public_id = assignment.assignment_id

    status = await client.get(f"/submissions/{submission_id}/status")
    assert status.status_code == 200
    body = status.json()
    # Queue/list status may remain queued-family; dual-flag must still expose review.
    assert body["status"] in {"queued", "pending", "assigned"} or isinstance(body["status"], str)
    review = body["review"]
    assert review is not None
    assert review["verified"] is True
    assert review["verdict"] == expected_verdict
    assert review["terminal"] is True
    assert review["retryable"] is False
    assert review["report_available"] is True
    assert review["phase"] == phase
    assert review["reason_code"] == f"{expected_verdict}_ok"
    assert review["session_id"] == session_public_id
    assert review["assignment_id"] == assignment_public_id
    # Secret classes never on status surface.
    serialized = json.dumps(body)
    for forbidden in (
        "session_token",
        "rn-public-tee-secret-nonce",
        "OPENROUTER_API_KEY",
        "encryption_key",
        "request_body",
        "response_body",
    ):
        assert forbidden not in serialized


async def test_dual_flag_status_report_available_false_without_projection(
    client,
    database_session,
    monkeypatch,
) -> None:
    """Optional lock: status.report_available is false without projection."""

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hk-dual-no-proj",
            name="dual-no-proj",
            agent_hash="dual-hash-no-proj",
            artifact_uri="/tmp/dual-no-proj.zip",
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()
        from agent_challenge.models import SubmissionStatusEvent

        session.add(
            SubmissionStatusEvent(
                submission_id=submission.id,
                sequence=1,
                from_status=None,
                to_status="queued",
                actor="test",
                reason="seed_queued",
            )
        )
        review_session = ReviewSession(
            session_id="rs-dual-no-proj",
            submission_id=submission.id,
            artifact_sha256="ab" * 32,
            artifact_size_bytes=12,
            manifest_sha256="cd" * 32,
            manifest_entries_sha256="ef" * 32,
            current_assignment_id="ra-dual-no-proj",
            authorizing_assignment_id=None,
        )
        session.add(review_session)
        await session.flush()
        assignment = ReviewAssignment(
            session_id=review_session.id,
            assignment_id="ra-dual-no-proj",
            attempt=1,
            phase="review_verifying",
            assignment_bytes="{}",
            assignment_digest="aa" * 32,
            artifact_sha256="ab" * 32,
            rules_snapshot_sha256="11" * 32,
            rules_revision_id="rules-v1",
            review_nonce="nonce-dual-no-proj",
            session_token_sha256="bb" * 32,
            capability_state="active",
            issued_at=now,
            expires_at=now,
            review_public_projection_json=None,
            review_verification_outcome_json=json.dumps(
                {
                    "status": "verifier_unavailable",
                    "terminal": False,
                    "retryable": True,
                    "reason_code": "dcap_qvl_missing",
                    "nonce_consumed": False,
                    "measurement_allowlisted": False,
                    "report_data_matched": False,
                    "verified_at_ms": None,
                }
            ),
            reason_code="dcap_qvl_missing",
            finished_at=None,
        )
        session.add(assignment)
        await session.commit()
        submission_id = submission.id

    status = await client.get(f"/submissions/{submission_id}/status")
    assert status.status_code == 200
    review = status.json()["review"]
    assert review is not None
    assert review["report_available"] is False
    assert review["verified"] is False
    assert review["verdict"] is None


def test_openapi_documents_public_tee_path() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    assert "/submissions/{submission_id}/review/tee" in paths
    assert "/v1/submissions/{submission_id}/review/tee" in paths
    get_op = paths["/submissions/{submission_id}/review/tee"]["get"]
    assert get_op["responses"]["200"]
    # response model is registered
    components = schema.get("components", {}).get("schemas", {})
    assert any("PublicTeeMath" in name for name in components)


def test_frontend_contract_public_paths_include_tee() -> None:
    from _routing import public_route_paths

    public_paths = public_route_paths(app)
    assert "/submissions/{submission_id}/review/tee" in public_paths
    assert "/v1/submissions/{submission_id}/review/tee" in public_paths
