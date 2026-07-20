"""Offline contract tests for review-domain report binding and verification.

These tests use a synthetic quote layout and a static quote verifier.  They
prove byte-level schema and verifier discrimination only, and do not claim that
any quote came from a live TDX CVM.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from agent_challenge.api import routes as api_routes
from agent_challenge.app import app
from agent_challenge.auth.security import SignedRequestAuth
from agent_challenge.core.models import AgentSubmission, ReviewNonce, ReviewSession
from agent_challenge.keyrelease.quote import (
    QuoteStructureError,
    QuoteVerificationError,
    QuoteVerifierUnavailable,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    parse_td_report,
    runtime_event_digest,
)
from agent_challenge.review.openrouter import (
    build_model_call_started,
    build_openrouter_request_body,
    build_planned_openrouter_request,
)
from agent_challenge.review.report import (
    REVIEW_REPORT_DOMAIN,
    DcapReviewQuoteVerifier,
    ReviewMeasurementAllowlist,
    ReviewVerifierUnavailable,
    build_review_envelope,
    review_digest,
    review_report_data_hex,
    submit_review_report,
    validate_review_core,
    validate_review_envelope,
    verify_review_envelope,
)
from agent_challenge.review.schemas import (
    ReviewInputConfig,
    build_review_assignment,
    validate_observed_openrouter_transport,
)
from agent_challenge.review.sessions import (
    create_review_session,
    mark_model_call_started,
    recover_pending_review_reports,
    review_audit_page,
)
from agent_challenge.sdk.config import ChallengeSettings

# Regenerated after REVIEW_MODEL pin flip to x-ai/grok-4.5 (and report_data preimage v2).
REVIEW_CORE_GOLDEN_DIGEST = "78cf224da0836df1dbc6a8fdad5910a6c58fadc69a9fdac4d15198876400993f"
REVIEW_REPORT_DATA_GOLDEN_HEX = (
    "97f1284cdfa6ebcfee9ff887fea55d1d5f77840cf45f34f5d5cc5234a691e7bc" + ("00" * 32)
)


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
        session_id="rs-report",
        assignment_id="ra-report",
        attempt=1,
        submission_id="17",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 9,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/ra-report/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="rn-report",
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
            # Challenge-domain submission/send receive (attested into report_data v2).
            "submission_received_at_ms": 1_000,
        },
        "review_nonce": core["review_nonce"],
    }


def _envelope() -> tuple[dict[str, Any], dict[str, Any], ReviewMeasurementAllowlist]:
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
    allowlist = ReviewMeasurementAllowlist.from_measurements(
        [
            {
                "mrtd": measurement["mrtd"],
                "rtmr0": measurement["rtmr0"],
                "rtmr1": measurement["rtmr1"],
                "rtmr2": measurement["rtmr2"],
                "compose_hash": measurement["compose_hash"],
                "os_image_hash": measurement["os_image_hash"],
            }
        ]
    )
    return envelope, assignment, allowlist


def test_review_decision_rejects_reason_or_evidence_lists_over_256() -> None:
    """Architecture §resource limits: reason/evidence entries ≤ 256 before serialization."""

    assignment, _config = _assignment()
    core = _review_core(assignment)
    over_reasons = {
        **core,
        "decision": {
            **core["decision"],
            "reason_codes": [f"r{i:03d}" for i in range(257)],
        },
    }
    over_evidence = {
        **core,
        "decision": {
            **core["decision"],
            "evidence_digests": [f"{i:064x}" for i in range(257)],
        },
    }
    with pytest.raises(ValueError, match="256|bound|reason|evidence"):
        validate_review_core(over_reasons)
    with pytest.raises(ValueError, match="256|bound|reason|evidence"):
        validate_review_core(over_evidence)

    at_cap = {
        **core,
        "decision": {
            **core["decision"],
            "verifier_result": "reject",
            "verdict": "reject",
            "reason_codes": [f"r{i:03d}" for i in range(256)],
            "evidence_digests": [f"{i:064x}" for i in range(256)],
        },
    }
    assert validate_review_core(at_cap)


def test_review_core_is_schema_closed_and_uses_declared_set_ordering() -> None:
    assignment, _config = _assignment()
    core = _review_core(assignment)

    assert validate_review_core(core)
    baseline = review_digest(core)
    assert review_digest({**core, "session_id": "rs-other"}) != baseline
    assert (
        review_digest(
            {
                **core,
                "policy_observation": {
                    **core["policy_observation"],
                    "routing_sha256": "ff" * 32,
                },
            }
        )
        != baseline
    )

    for invalid in (
        {**core, "quote": "forbidden"},
        {**core, "review_digest": "00" * 32},
        {
            **core,
            "decision": {
                **core["decision"],
                "reason_codes": list(reversed(core["decision"]["reason_codes"])),
            },
        },
        {
            **core,
            "decision": {
                **core["decision"],
                "evidence_digests": [*core["decision"]["evidence_digests"], "79" * 32],
            },
        },
    ):
        with pytest.raises(ValueError):
            validate_review_core(invalid)


def test_review_digest_and_report_data_match_independent_two_layer_derivation() -> None:
    assignment, _config = _assignment()
    core = _review_core(assignment)
    canonical_core = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_review_digest = hashlib.sha256(canonical_core).hexdigest()
    expected_report_preimage = {
        "domain": REVIEW_REPORT_DOMAIN,
        "schema_version": 2,
        "review_digest": expected_review_digest,
        "session_id": core["session_id"],
        "review_nonce": core["review_nonce"],
        "issued_at_ms": core["times"]["issued_at_ms"],
        "received_at_ms": core["times"]["submission_received_at_ms"],
    }
    expected_report_digest = hashlib.sha256(
        json.dumps(
            expected_report_preimage,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert review_digest(core) == expected_review_digest
    assert review_report_data_hex(core) == expected_report_digest + ("00" * 32)
    assert review_digest(core) == REVIEW_CORE_GOLDEN_DIGEST
    assert review_report_data_hex(core) == REVIEW_REPORT_DATA_GOLDEN_HEX


def test_review_runtime_emits_quote_only_for_derived_review_domain_report_data() -> None:
    runtime_path = Path(__file__).parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    class QuoteClient:
        def __init__(self) -> None:
            self.report_data: list[bytes] = []

        def get_quote(self, report_data: bytes) -> object:
            self.report_data.append(report_data)
            return type("Quote", (), {"quote": "beef", "event_log": [], "vm_config": {}})()

    assignment, _config = _assignment()
    client = QuoteClient()
    emitted = runtime._quote_review_core(_review_core(assignment), client=client)

    assert emitted["report_data_hex"] == REVIEW_REPORT_DATA_GOLDEN_HEX
    assert client.report_data == [bytes.fromhex(REVIEW_REPORT_DATA_GOLDEN_HEX)]
    assert "--report-data-hex" not in runtime_path.read_text(encoding="utf-8")


def test_timestamps_are_bound_ordered_and_assignment_bounded() -> None:
    assignment, _config = _assignment()
    core = _review_core(assignment)
    baseline = review_digest(core)
    changed = copy.deepcopy(core)
    changed["times"]["request_finished_at_ms"] += 1
    assert review_digest(changed) != baseline

    for field, value in (
        ("started_at_ms", 999),
        ("request_started_at_ms", 1_000),
        ("report_finished_at_ms", 9_000),
    ):
        invalid = copy.deepcopy(core)
        invalid["times"][field] = value
        with pytest.raises(ValueError):
            validate_review_core(invalid)


def test_inverted_model_call_vs_request_started_times_reject_fixed_order_validates() -> None:
    """Previous live bug: request_started stamped before model_call announce.

    Offline unit proves the inverted pair is rejected by _validate_times and the
    fixed ordering (model_call_marked_at_ms <= request_started_at_ms) validates.
    """

    assignment, _config = _assignment()
    core = _review_core(assignment)
    # Happy path already has marked@1001 <= started@1002.
    assert validate_review_core(core)

    inverted = copy.deepcopy(core)
    inverted["times"]["model_call_marked_at_ms"] = 1_005
    inverted["times"]["request_started_at_ms"] = 1_002
    with pytest.raises(ValueError, match="timestamp|order|strict"):
        validate_review_core(inverted)

    fixed = copy.deepcopy(core)
    fixed["times"] = {
        "issued_at_ms": 1_000,
        "started_at_ms": 1_000,
        "model_call_marked_at_ms": 1_010,
        "request_started_at_ms": 1_010,  # equal after announce is allowed (sorted equal ok).
        "request_finished_at_ms": 1_020,
        "verifier_finished_at_ms": 1_030,
        "report_finished_at_ms": 1_040,
        "expires_at_ms": 9_000,
        "submission_received_at_ms": 1_000,
    }
    assert validate_review_core(fixed)
    strict = copy.deepcopy(fixed)
    strict["times"]["request_started_at_ms"] = 1_011
    assert validate_review_core(strict)


def test_outer_envelope_requires_quote_report_data_and_strict_event_measurement_shapes() -> None:
    envelope, _assignment, _allowlist = _envelope()
    assert validate_review_envelope(envelope)

    for mutate in (
        lambda item: item.update({"domain": "base-agent-challenge-v1"}),
        lambda item: item.update({"report_data_hex": "00" * 64}),
        lambda item: item["attestation"]["event_log"][0].update({"extra": True}),
        lambda item: item["attestation"]["measurement"].update({"unknown": "x"}),
        lambda item: item["attestation"]["vm_config"].update({"vcpu": 0}),
        lambda item: item["attestation"]["measurement"].update({"mrtd": "11" * 47}),
    ):
        invalid = copy.deepcopy(envelope)
        mutate(invalid)
        with pytest.raises(ValueError):
            validate_review_envelope(invalid)


def test_verifier_rejects_each_review_domain_binding_tamper() -> None:
    envelope, assignment, allowlist = _envelope()
    assignment["model_call_started_json"] = json.dumps(
        build_model_call_started(
            assignment_id="ra-report",
            planned_request_sha256="70" * 32,
            request_body_sha256="72" * 32,
            request_body_length=7,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    controls = (
        lambda item: item["review_core"].update({"review_nonce": "rn-other"}),
        lambda item: item.update({"report_data_hex": "00" * 64}),
        lambda item: item["attestation"]["event_log"][0].update({"event_payload": "00" * 32}),
        lambda item: item["attestation"]["measurement"].update({"compose_hash": "00" * 32}),
        lambda item: item["attestation"]["vm_config"].update({"os_image_hash": "00" * 32}),
    )
    for mutate in controls:
        invalid = copy.deepcopy(envelope)
        mutate(invalid)
        outcome = verify_review_envelope(
            envelope=invalid,
            assignment=assignment,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=allowlist,
            received_at_ms=1_006,
        )
        assert outcome.status == "trust_failed"
        assert outcome.terminal
        assert outcome.nonce_consumed


class _UnavailableVerifier:
    def verify(self, quote_hex: str) -> object:
        raise ReviewVerifierUnavailable("offline verifier unavailable")


@pytest.mark.asyncio
async def test_review_verification_is_conjunctive_and_preserves_nonce_on_transient_outage(
    database_session,
) -> None:
    envelope, assignment_object, allowlist = _envelope()
    now = datetime.now(UTC).replace(microsecond=0)
    artifact_bytes = b"report-zip"
    submission = AgentSubmission(
        miner_hotkey="review-miner",
        name="report-agent",
        agent_hash=assignment_object["assignment_core"]["artifact"]["agent_hash"],
        artifact_uri="/tmp/report.zip",
        artifact_path="/tmp/report.zip",
        zip_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        zip_size_bytes=len(artifact_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )
    config = ReviewInputConfig(
        routing=_routing(),
        image_ref=assignment_object["assignment_core"]["review_app"]["image_ref"],
        compose_hash=assignment_object["assignment_core"]["review_app"]["compose_hash"],
        kms_public_key_hex=assignment_object["assignment_core"]["review_app"]["kms_public_key_hex"],
        measurement=assignment_object["assignment_core"]["review_app"]["measurement"],
    )

    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_bytes,
            rules_files={".rules/policy.md": b"review"},
            rules_revision_id="rules-v1",
            settings=_settings_with_evidence_key(shared_token="report-token"),
            input_config=config,
            now=now,
        )
        # This report-focused fixture begins after the separately tested signed
        # deployment acknowledgement has moved the review CVM to running.
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        settings = _settings_with_evidence_key(shared_token="report-token")
        durable_assignment = json.loads(created.assignment.assignment_bytes)
        report_core = _review_core(durable_assignment)
        evidence = _minimal_bound_evidence(
            assignment_id=created.assignment.assignment_id,
            openrouter_observation=report_core["openrouter_observation"],
        )
        await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=build_model_call_started(
                assignment_id=created.assignment.assignment_id,
                planned_request_sha256=report_core["openrouter_observation"][
                    "planned_request_sha256"
                ],
                request_body_sha256=report_core["openrouter_observation"]["request_body_sha256"],
                request_body_length=report_core["openrouter_observation"]["request_body_length"],
            ),
            now=now + timedelta(milliseconds=1),
        )
        report_core["times"] = {
            "issued_at_ms": int(now.timestamp() * 1000),
            "started_at_ms": int(now.timestamp() * 1000),
            "model_call_marked_at_ms": int(now.timestamp() * 1000) + 1,
            "request_started_at_ms": int(now.timestamp() * 1000) + 2,
            "request_finished_at_ms": int(now.timestamp() * 1000) + 3,
            "verifier_finished_at_ms": int(now.timestamp() * 1000) + 4,
            "report_finished_at_ms": int(now.timestamp() * 1000) + 5,
            "expires_at_ms": int((now + timedelta(minutes=30)).timestamp() * 1000),
            "submission_received_at_ms": int(now.timestamp() * 1000),
        }
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
        report = build_review_envelope(
            review_core=report_core,
            tdx_quote_hex=build_tdx_quote(
                mrtd=measurement["mrtd"],
                rtmr0=measurement["rtmr0"],
                rtmr1=measurement["rtmr1"],
                rtmr2=measurement["rtmr2"],
                rtmr3=measurement["rtmr3"],
                report_data=review_report_data_hex(report_core),
            ),
            event_log=event_log,
            measurement=measurement,
            vm_config={
                "vcpu": 1,
                "memory_mb": 2048,
                "os_image_hash": measurement["os_image_hash"],
            },
        )
        runtime_allowlist = ReviewMeasurementAllowlist.from_measurements(
            [
                {
                    "mrtd": measurement["mrtd"],
                    "rtmr0": measurement["rtmr0"],
                    "rtmr1": measurement["rtmr1"],
                    "rtmr2": measurement["rtmr2"],
                    "compose_hash": measurement["compose_hash"],
                    "os_image_hash": measurement["os_image_hash"],
                }
            ]
        )

        transient = await submit_review_report(
            session,
            assignment=created.assignment,
            envelope=report,
            evidence_objects=evidence,
            evidence_settings=settings,
            quote_verifier=_UnavailableVerifier(),
            allowlist=runtime_allowlist,
            now=now + timedelta(seconds=10),
        )
        assert transient.status == "verifier_unavailable", transient.reason_code
        nonce = await session.scalar(
            select(ReviewNonce).where(ReviewNonce.assignment_id == created.assignment.id)
        )
        assert nonce is not None and nonce.state == "active"
        assert created.assignment.phase == "review_verifying"

        verified = await submit_review_report(
            session,
            assignment=created.assignment,
            envelope=report,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=runtime_allowlist,
            now=now + timedelta(hours=1),
        )
        assert verified.status == "verified_allow", verified.reason_code
        assert created.assignment.phase == "review_allowed"
        assert nonce.state == "consumed"


def test_review_allowlist_is_rotatable_and_fail_closed() -> None:
    envelope, _assignment, allowlist = _envelope()
    measurement = envelope["attestation"]["measurement"]
    rotated = ReviewMeasurementAllowlist.from_measurements(
        [
            {
                field: measurement[field]
                for field in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "compose_hash", "os_image_hash")
            },
            {
                **{
                    field: measurement[field]
                    for field in (
                        "mrtd",
                        "rtmr0",
                        "rtmr1",
                        "rtmr2",
                        "compose_hash",
                        "os_image_hash",
                    )
                },
                "mrtd": "ff" * 48,
            },
        ]
    )
    assert allowlist.contains(measurement)
    assert rotated.contains(measurement)
    assert not ReviewMeasurementAllowlist().contains(measurement)
    assert not rotated.contains({**measurement, "compose_hash": "00" * 32})


def test_quote_verifier_failure_is_not_treated_as_transient() -> None:
    class InvalidVerifier:
        def verify(self, quote_hex: str) -> object:
            raise QuoteVerificationError("bad quote")

    with pytest.raises(QuoteVerificationError):
        InvalidVerifier().verify("00")


def _marked_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    marked = copy.deepcopy(assignment)
    marked["model_call_started_json"] = json.dumps(
        build_model_call_started(
            assignment_id=str(assignment["assignment_core"]["assignment_id"]),
            planned_request_sha256="70" * 32,
            request_body_sha256="72" * 32,
            request_body_length=7,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return marked


def test_review_acceptance_uses_strict_tdx_v4_and_event_log_validators() -> None:
    """VAL-REVIEW-034/035: unsupported quote family/layout and event aliases reject."""

    envelope, assignment, allowlist = _envelope()
    assignment = _marked_assignment(assignment)

    def _unsupported_version(item: dict[str, Any]) -> None:
        raw = bytes.fromhex(item["attestation"]["tdx_quote_hex"])
        mutated = bytearray(raw)
        mutated[0:2] = (5).to_bytes(2, "little")
        item["attestation"]["tdx_quote_hex"] = bytes(mutated).hex()

    def _truncated_below_declared_length(item: dict[str, Any]) -> None:
        # Live Intel quotes may append certification-data after the signed region.
        # What must still fail is a declared signature length that runs past EOF.
        raw = bytes.fromhex(item["attestation"]["tdx_quote_hex"])
        if len(raw) <= 16:
            item["attestation"]["tdx_quote_hex"] = raw.hex()
            return
        item["attestation"]["tdx_quote_hex"] = raw[:-16].hex()

    def _malformed_declared_length(item: dict[str, Any]) -> None:
        raw = bytes.fromhex(item["attestation"]["tdx_quote_hex"])
        # Prefix is 48-byte header + 584-byte body = 632, then LE u32 signature length.
        prefix = raw[:632]
        real_sig = raw[636:]
        # Claim more signature bytes than remain in the buffer.
        declared = (len(real_sig) + 64).to_bytes(4, "little")
        item["attestation"]["tdx_quote_hex"] = (prefix + declared + real_sig).hex()

    def _event_alias(item: dict[str, Any]) -> None:
        entry = item["attestation"]["event_log"][0]
        entry["event"] = "compose_hash"
        entry["digest"] = runtime_event_digest(
            "compose_hash", bytes.fromhex(entry["event_payload"])
        ).hex()

    def _duplicate_identity(item: dict[str, Any]) -> None:
        item["attestation"]["event_log"].append(copy.deepcopy(item["attestation"]["event_log"][0]))

    def _shadowing_alias(item: dict[str, Any]) -> None:
        payload = "00" * 32
        item["attestation"]["event_log"].insert(
            0,
            {
                "imr": 3,
                "event_type": 0x08000001,
                "digest": runtime_event_digest("composehash", bytes.fromhex(payload)).hex(),
                "event": "composehash",
                "event_payload": payload,
            },
        )

    def _wrong_ordering(item: dict[str, Any]) -> None:
        item["attestation"]["event_log"][0], item["attestation"]["event_log"][1] = (
            item["attestation"]["event_log"][1],
            item["attestation"]["event_log"][0],
        )

    def _malformed_width(item: dict[str, Any]) -> None:
        item["attestation"]["event_log"][0]["digest"] = "00" * 47

    controls = (
        _unsupported_version,
        _truncated_below_declared_length,
        _malformed_declared_length,
        _event_alias,
        _duplicate_identity,
        _shadowing_alias,
        _wrong_ordering,
        _malformed_width,
    )
    for mutate in controls:
        invalid = copy.deepcopy(envelope)
        mutate(invalid)
        with pytest.raises((ValueError, QuoteStructureError, QuoteVerificationError)):
            validate_review_envelope(invalid)
        outcome = verify_review_envelope(
            envelope=invalid,
            assignment=assignment,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=allowlist,
            received_at_ms=1_006,
        )
        assert outcome.status == "trust_failed"
        assert outcome.terminal
        assert outcome.nonce_consumed


def test_dcap_review_quote_verifier_maps_quote_verifier_unavailable() -> None:
    """VAL-REVIEW-035: canonical Unavailable maps to retryable receipt disposition."""

    class UnavailableBackend:
        def verify(self, quote_hex: str) -> object:
            raise QuoteVerifierUnavailable("dcap-qvl not available: missing binary")

    class TimeoutBackend:
        def verify(self, quote_hex: str) -> object:
            raise QuoteVerifierUnavailable("dcap-qvl timed out: 60s")

    class MalformedOutputBackend:
        def verify(self, quote_hex: str) -> object:
            raise QuoteVerifierUnavailable("dcap-qvl output is not JSON: x")

    for backend in (UnavailableBackend(), TimeoutBackend(), MalformedOutputBackend()):
        adapter = DcapReviewQuoteVerifier(verifier=backend)
        with pytest.raises(ReviewVerifierUnavailable):
            adapter.verify("00")
        envelope, assignment, allowlist = _envelope()
        assignment = _marked_assignment(assignment)
        outcome = verify_review_envelope(
            envelope=envelope,
            assignment=assignment,
            quote_verifier=adapter,
            allowlist=allowlist,
            received_at_ms=1_006,
        )
        assert outcome.status == "verifier_unavailable"
        assert not outcome.terminal
        assert not outcome.nonce_consumed
        assert outcome.retryable


def test_future_or_post_receipt_report_timestamps_reject() -> None:
    """VAL-REVIEW-033: report timestamps may not run past durable receipt time."""

    envelope, assignment, allowlist = _envelope()
    assignment = _marked_assignment(assignment)
    # Future relative to receipt — still ordered and before expires.
    future = copy.deepcopy(envelope)
    future["review_core"]["times"]["report_finished_at_ms"] = 5_000
    future["review_digest"] = review_digest(future["review_core"])
    future["report_data_hex"] = review_report_data_hex(future["review_core"])
    report = parse_td_report(future["attestation"]["tdx_quote_hex"])
    future["attestation"]["tdx_quote_hex"] = build_tdx_quote(
        mrtd=report.mrtd,
        rtmr0=report.rtmr0,
        rtmr1=report.rtmr1,
        rtmr2=report.rtmr2,
        rtmr3=report.rtmr3,
        report_data=future["report_data_hex"],
    )
    outcome = verify_review_envelope(
        envelope=future,
        assignment=assignment,
        quote_verifier=StaticQuoteVerifier(),
        allowlist=allowlist,
        received_at_ms=1_006,
    )
    assert outcome.status == "trust_failed"
    assert outcome.terminal

    # report_finished after receipt rejects even when smaller than expires.
    post_receipt = copy.deepcopy(envelope)
    post_receipt["review_core"]["times"]["report_finished_at_ms"] = 2_000
    post_receipt["review_digest"] = review_digest(post_receipt["review_core"])
    post_receipt["report_data_hex"] = review_report_data_hex(post_receipt["review_core"])
    post_receipt["attestation"]["tdx_quote_hex"] = build_tdx_quote(
        mrtd=report.mrtd,
        rtmr0=report.rtmr0,
        rtmr1=report.rtmr1,
        rtmr2=report.rtmr2,
        rtmr3=report.rtmr3,
        report_data=post_receipt["report_data_hex"],
    )
    outcome = verify_review_envelope(
        envelope=post_receipt,
        assignment=assignment,
        quote_verifier=StaticQuoteVerifier(),
        allowlist=allowlist,
        received_at_ms=1_500,
    )
    assert outcome.status == "trust_failed"


@pytest.mark.asyncio
async def test_recovery_verifies_original_report_against_original_receipt_boundary(
    database_session,
) -> None:
    """VAL-REVIEW-035: recovery re-checks exact receipt against original received_at."""

    _envelope_unused, assignment_object, _baseline_allowlist = _envelope()
    now = datetime.now(UTC).replace(microsecond=0)
    artifact_bytes = b"receipt-boundary-zip"
    submission = AgentSubmission(
        miner_hotkey="review-boundary-miner",
        name="boundary-agent",
        agent_hash=assignment_object["assignment_core"]["artifact"]["agent_hash"],
        artifact_uri="/tmp/boundary.zip",
        artifact_path="/tmp/boundary.zip",
        zip_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        zip_size_bytes=len(artifact_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )
    config = ReviewInputConfig(
        routing=_routing(),
        image_ref=assignment_object["assignment_core"]["review_app"]["image_ref"],
        compose_hash=assignment_object["assignment_core"]["review_app"]["compose_hash"],
        kms_public_key_hex=assignment_object["assignment_core"]["review_app"]["kms_public_key_hex"],
        measurement=assignment_object["assignment_core"]["review_app"]["measurement"],
    )
    settings = _settings_with_evidence_key(shared_token="boundary-token")

    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_bytes,
            rules_files={".rules/policy.md": b"review"},
            rules_revision_id="rules-v1",
            settings=settings,
            input_config=config,
            now=now,
        )
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        durable_assignment = json.loads(created.assignment.assignment_bytes)
        report_core = _review_core(durable_assignment)
        evidence = _minimal_bound_evidence(
            assignment_id=created.assignment.assignment_id,
            openrouter_observation=report_core["openrouter_observation"],
        )
        await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=build_model_call_started(
                assignment_id=created.assignment.assignment_id,
                planned_request_sha256=report_core["openrouter_observation"][
                    "planned_request_sha256"
                ],
                request_body_sha256=report_core["openrouter_observation"]["request_body_sha256"],
                request_body_length=report_core["openrouter_observation"]["request_body_length"],
            ),
            now=now + timedelta(milliseconds=1),
        )
        start_ms = int(now.timestamp() * 1000)
        receipt_time = now + timedelta(seconds=2)
        receipt_ms = int(receipt_time.timestamp() * 1000)
        # Timely first receipt: finished <= original received_at, well under expiry.
        report_core["times"] = {
            "issued_at_ms": start_ms,
            "started_at_ms": start_ms,
            "model_call_marked_at_ms": start_ms + 1,
            "request_started_at_ms": start_ms + 2,
            "request_finished_at_ms": start_ms + 3,
            "verifier_finished_at_ms": start_ms + 4,
            "report_finished_at_ms": receipt_ms,
            "expires_at_ms": int((now + timedelta(minutes=30)).timestamp() * 1000),
            "submission_received_at_ms": start_ms,
        }
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
        report = build_review_envelope(
            review_core=report_core,
            tdx_quote_hex=build_tdx_quote(
                mrtd=measurement["mrtd"],
                rtmr0=measurement["rtmr0"],
                rtmr1=measurement["rtmr1"],
                rtmr2=measurement["rtmr2"],
                rtmr3=measurement["rtmr3"],
                report_data=review_report_data_hex(report_core),
            ),
            event_log=event_log,
            measurement=measurement,
            vm_config={
                "vcpu": 1,
                "memory_mb": 2048,
                "os_image_hash": measurement["os_image_hash"],
            },
        )
        runtime_allowlist = ReviewMeasurementAllowlist.from_measurements(
            [
                {
                    "mrtd": measurement["mrtd"],
                    "rtmr0": measurement["rtmr0"],
                    "rtmr1": measurement["rtmr1"],
                    "rtmr2": measurement["rtmr2"],
                    "compose_hash": measurement["compose_hash"],
                    "os_image_hash": measurement["os_image_hash"],
                }
            ]
        )

        first = await submit_review_report(
            session,
            assignment=created.assignment,
            envelope=report,
            evidence_objects=evidence,
            evidence_settings=settings,
            quote_verifier=_UnavailableVerifier(),
            allowlist=runtime_allowlist,
            now=receipt_time,
        )
        assert first.status == "verifier_unavailable"
        assert not first.nonce_consumed
        assert created.assignment.phase == "review_verifying"
        assert created.assignment.review_report_received_at == receipt_time
        assert created.assignment.review_report_envelope_json is not None
        original_sha = created.assignment.review_report_sha256
        await session.commit()

    async with database_session() as session:
        # Post-expiry recovery must re-verify the exact original report against the
        # original durable received_at (not recovery wall clock) and allow.
        recovered = await recover_pending_review_reports(
            session,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=runtime_allowlist,
            now=now + timedelta(hours=2),
            evidence_settings=settings,
        )
        durable = await session.get(type(created.assignment), created.assignment.id)
        assert recovered == 1
        assert durable is not None
        assert durable.phase == "review_allowed"
        assert durable.review_report_received_at.replace(tzinfo=UTC) == receipt_time
        assert durable.review_report_sha256 == original_sha
        nonce = await session.scalar(
            select(ReviewNonce).where(ReviewNonce.assignment_id == durable.id)
        )
        assert nonce is not None and nonce.state == "consumed"


@pytest.mark.asyncio
async def test_post_receipt_timeline_rejects_even_when_first_receipted_via_unavailable(
    database_session,
) -> None:
    """VAL-REVIEW-033/035: finished_at after original receipt terminalizes, not parks."""

    _envelope_unused, assignment_object, _allowlist = _envelope()
    now = datetime.now(UTC).replace(microsecond=0)
    artifact_bytes = b"post-receipt-zip"
    submission = AgentSubmission(
        miner_hotkey="review-post-receipt-miner",
        name="post-receipt-agent",
        agent_hash=assignment_object["assignment_core"]["artifact"]["agent_hash"],
        artifact_uri="/tmp/post-receipt.zip",
        artifact_path="/tmp/post-receipt.zip",
        zip_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        zip_size_bytes=len(artifact_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )
    config = ReviewInputConfig(
        routing=_routing(),
        image_ref=assignment_object["assignment_core"]["review_app"]["image_ref"],
        compose_hash=assignment_object["assignment_core"]["review_app"]["compose_hash"],
        kms_public_key_hex=assignment_object["assignment_core"]["review_app"]["kms_public_key_hex"],
        measurement=assignment_object["assignment_core"]["review_app"]["measurement"],
    )
    settings = _settings_with_evidence_key(shared_token="post-receipt-token")

    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_bytes,
            rules_files={".rules/policy.md": b"review"},
            rules_revision_id="rules-v1",
            settings=settings,
            input_config=config,
            now=now,
        )
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        durable_assignment = json.loads(created.assignment.assignment_bytes)
        report_core = _review_core(durable_assignment)
        evidence = _minimal_bound_evidence(
            assignment_id=created.assignment.assignment_id,
            openrouter_observation=report_core["openrouter_observation"],
        )
        await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=build_model_call_started(
                assignment_id=created.assignment.assignment_id,
                planned_request_sha256=report_core["openrouter_observation"][
                    "planned_request_sha256"
                ],
                request_body_sha256=report_core["openrouter_observation"]["request_body_sha256"],
                request_body_length=report_core["openrouter_observation"]["request_body_length"],
            ),
            now=now + timedelta(milliseconds=1),
        )
        start_ms = int(now.timestamp() * 1000)
        receipt_time = now + timedelta(seconds=2)
        receipt_ms = int(receipt_time.timestamp() * 1000)
        report_core["times"] = {
            "issued_at_ms": start_ms,
            "started_at_ms": start_ms,
            "model_call_marked_at_ms": start_ms + 1,
            "request_started_at_ms": start_ms + 2,
            "request_finished_at_ms": start_ms + 3,
            "verifier_finished_at_ms": start_ms + 4,
            "report_finished_at_ms": receipt_ms + 5_000,
            "expires_at_ms": int((now + timedelta(minutes=30)).timestamp() * 1000),
            "submission_received_at_ms": start_ms,
        }
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
        report = build_review_envelope(
            review_core=report_core,
            tdx_quote_hex=build_tdx_quote(
                mrtd=measurement["mrtd"],
                rtmr0=measurement["rtmr0"],
                rtmr1=measurement["rtmr1"],
                rtmr2=measurement["rtmr2"],
                rtmr3=measurement["rtmr3"],
                report_data=review_report_data_hex(report_core),
            ),
            event_log=event_log,
            measurement=measurement,
            vm_config={
                "vcpu": 1,
                "memory_mb": 2048,
                "os_image_hash": measurement["os_image_hash"],
            },
        )
        runtime_allowlist = ReviewMeasurementAllowlist.from_measurements(
            [
                {
                    "mrtd": measurement["mrtd"],
                    "rtmr0": measurement["rtmr0"],
                    "rtmr1": measurement["rtmr1"],
                    "rtmr2": measurement["rtmr2"],
                    "compose_hash": measurement["compose_hash"],
                    "os_image_hash": measurement["os_image_hash"],
                }
            ]
        )
        outcome = await submit_review_report(
            session,
            assignment=created.assignment,
            envelope=report,
            evidence_objects=evidence,
            evidence_settings=settings,
            quote_verifier=_UnavailableVerifier(),
            allowlist=runtime_allowlist,
            now=receipt_time,
        )
        assert outcome.status == "trust_failed"
        assert outcome.terminal
        assert outcome.nonce_consumed
        assert created.assignment.phase == "review_error"
        assert created.assignment.review_report_received_at == receipt_time


async def _durable_report_fixture(
    database_session,
    *,
    label: str = "primary",
) -> tuple[
    object,
    object,
    ChallengeSettings,
    dict[str, Any],
    ReviewMeasurementAllowlist,
    dict[str, bytes],
    datetime,
]:
    """Create one marked assignment and its exact bounded transport evidence."""

    now = datetime.now(UTC).replace(microsecond=0)
    settings = _settings_with_evidence_key()
    assignment_fixture, config = _assignment()
    artifact_bytes = b"report-evidence-zip"
    submission = AgentSubmission(
        miner_hotkey="review-evidence-miner",
        name="review-evidence-agent",
        agent_hash=hashlib.sha256(f"review-evidence-{label}".encode()).hexdigest(),
        artifact_uri="/tmp/review-evidence.zip",
        artifact_path="/tmp/review-evidence.zip",
        zip_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        zip_size_bytes=len(artifact_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )
    request_body = build_openrouter_request_body(
        messages=[{"role": "user", "content": "Review supplied bytes only."}],
        routing=_routing(),
    )
    planned, planned_bytes, planned_digest = build_planned_openrouter_request(
        body=request_body,
        routing_sha256=hashlib.sha256(
            json.dumps(_routing(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    response_body = b'{"id":"or-evidence-response","model":"x-ai/grok-4.5","choices":[]}'
    metadata = b'{"provider":"offline"}'
    observed = {
        "schema_version": 1,
        "planned_request_sha256": planned_digest,
        "final_origin": "https://openrouter.ai:443",
        "final_path": "/api/v1/chat/completions",
        "tls_hostname": "openrouter.ai",
        "tls_hostname_verified": True,
        "redirected": False,
        "proxied": False,
        "response_status": 200,
        "response_content_encoding": "identity",
        "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
        "response_body_length": len(response_body),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
    }
    observed_bytes = validate_observed_openrouter_transport(observed)

    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_bytes,
            rules_files={".rules/policy.md": b"review"},
            rules_revision_id="rules-v1",
            settings=settings,
            input_config=config,
            now=now,
        )
        # This transport fixture begins after a valid deployment acknowledgement.
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=build_model_call_started(
                assignment_id=created.assignment.assignment_id,
                planned_request_sha256=planned_digest,
                request_body_sha256=planned["body_sha256"],
                request_body_length=planned["body_length"],
            ),
            now=now + timedelta(milliseconds=1),
        )
        assignment = json.loads(created.assignment.assignment_bytes)
        core = _review_core(assignment)
        core["openrouter_observation"] = {
            "planned_request_sha256": planned_digest,
            "transport_observation_sha256": hashlib.sha256(observed_bytes).hexdigest(),
            "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
            "request_body_length": len(request_body),
            "response_status": 200,
            "response_content_encoding": "identity",
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "response_body_length": len(response_body),
            "response_id": "or-evidence-response",
            "returned_model": "x-ai/grok-4.5",
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "observed_provider": "openrouter",
            "provider_provenance": "openrouter_metadata",
            "cache_hit": False,
        }
        start_ms = int(now.timestamp() * 1000)
        core["times"] = {
            "issued_at_ms": start_ms,
            "started_at_ms": start_ms,
            "model_call_marked_at_ms": start_ms + 1,
            "request_started_at_ms": start_ms + 2,
            "request_finished_at_ms": start_ms + 3,
            "verifier_finished_at_ms": start_ms + 4,
            "report_finished_at_ms": start_ms + 5,
            "expires_at_ms": int((now + timedelta(minutes=30)).timestamp() * 1000),
            "submission_received_at_ms": start_ms,
        }
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
        envelope = build_review_envelope(
            review_core=core,
            tdx_quote_hex=build_tdx_quote(
                mrtd=measurement["mrtd"],
                rtmr0=measurement["rtmr0"],
                rtmr1=measurement["rtmr1"],
                rtmr2=measurement["rtmr2"],
                rtmr3=measurement["rtmr3"],
                report_data=review_report_data_hex(core),
            ),
            event_log=event_log,
            measurement=measurement,
            vm_config={
                "vcpu": 1,
                "memory_mb": 2048,
                "os_image_hash": measurement["os_image_hash"],
            },
        )
        allowlist = ReviewMeasurementAllowlist.from_measurements(
            [
                {
                    field: measurement[field]
                    for field in (
                        "mrtd",
                        "rtmr0",
                        "rtmr1",
                        "rtmr2",
                        "compose_hash",
                        "os_image_hash",
                    )
                }
            ]
        )
        await session.commit()

    return (
        created.session,
        created.assignment,
        settings,
        envelope,
        allowlist,
        {
            "planned_request": planned_bytes,
            "transport_observation": observed_bytes,
            "request_body": request_body,
            "response_body": response_body,
            "metadata": metadata,
        },
        now,
    )


@pytest.mark.asyncio
async def test_receipted_evidence_is_recomputable_projected_and_recovered_after_expiry(
    database_session,
) -> None:
    (
        review_session,
        assignment,
        settings,
        envelope,
        allowlist,
        evidence,
        now,
    ) = await _durable_report_fixture(database_session, label="second")

    async with database_session() as session:
        transient = await submit_review_report(
            session,
            assignment=assignment,
            envelope=envelope,
            evidence_objects=evidence,
            evidence_settings=settings,
            quote_verifier=_UnavailableVerifier(),
            allowlist=allowlist,
            now=now + timedelta(seconds=1),
        )
        assert transient.status == "verifier_unavailable"
        receipted = await session.get(type(assignment), assignment.id)
        assert receipted is not None
        assert receipted.phase == "review_verifying"
        assert receipted.review_evidence_descriptor_json is not None
        assert receipted.review_public_projection_json is None
        await session.commit()

    async with database_session() as session:
        recovered = await recover_pending_review_reports(
            session,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=allowlist,
            now=now + timedelta(hours=1),
        )
        await session.commit()
        durable_assignment = await session.get(type(assignment), assignment.id)
        durable_session = await session.get(ReviewSession, review_session.id)
        assert recovered == 1
        assert durable_assignment is not None
        assert durable_assignment.phase == "review_allowed"
        assert durable_assignment.review_public_projection_json is not None
        assert durable_session is not None
        assert durable_session.authorizing_assignment_id == durable_assignment.assignment_id

        public = await review_audit_page(
            session,
            session_row=durable_session,
            cursor=None,
            limit=10,
        )
        internal = await review_audit_page(
            session,
            session_row=durable_session,
            cursor=None,
            limit=10,
            internal=True,
        )

    projection = public["items"][0]["report_projection"]
    assert projection is not None
    assert projection["request_body_sha256"] == hashlib.sha256(evidence["request_body"]).hexdigest()
    assert (
        projection["response_body_sha256"] == hashlib.sha256(evidence["response_body"]).hexdigest()
    )
    assert "Review supplied bytes only." not in json.dumps(public)
    descriptor = internal["items"][0]["evidence_descriptor"]
    assert descriptor["request_object_ref"].startswith("re_")
    assert descriptor["response_object_ref"].startswith("re_")
    assert descriptor["request_body_sha256"] == projection["request_body_sha256"]
    assert internal["items"][0]["report_envelope"]["review_digest"] == projection["review_digest"]


@pytest.mark.asyncio
async def test_report_route_receipts_are_202_or_retryable_503_and_reposts_are_immutable(
    client,
    database_session,
    monkeypatch,
) -> None:
    (
        _review_session,
        assignment,
        _settings,
        envelope,
        allowlist,
        evidence,
        _now,
    ) = await _durable_report_fixture(database_session, label="second")

    monkeypatch.setattr(api_routes, "_review_quote_verifier", lambda: _UnavailableVerifier())
    payload = {
        "envelope": envelope,
        "evidence": {
            f"{kind}_b64": base64.b64encode(value).decode("ascii")
            for kind, value in evidence.items()
        },
    }
    headers = {
        "Authorization": f"Bearer {_derive_test_token(assignment.assignment_id)}",
        "Content-Type": "application/json",
    }

    missing_evidence = await client.post(
        f"/review/v1/assignments/{assignment.assignment_id}/report",
        json={"envelope": envelope},
        headers=headers,
    )
    assert missing_evidence.status_code == 422

    first = await client.post(
        f"/review/v1/assignments/{assignment.assignment_id}/report",
        json=payload,
        headers=headers,
    )
    assert first.status_code == 503
    assert first.json()["status"] == "verifier_unavailable"
    assert first.json()["nonce_consumed"] is False

    changed = copy.deepcopy(payload)
    changed["evidence"]["response_body_b64"] = base64.b64encode(b'{"altered":true}').decode("ascii")
    conflicting = await client.post(
        f"/review/v1/assignments/{assignment.assignment_id}/report",
        json=changed,
        headers=headers,
    )
    assert conflicting.status_code == 409

    monkeypatch.setattr(api_routes, "_review_quote_verifier", lambda: StaticQuoteVerifier())
    monkeypatch.setattr(
        api_routes.settings,
        "review_app_measurement_allowlist",
        allowlist.entries,
    )
    terminal = await client.post(
        f"/review/v1/assignments/{assignment.assignment_id}/report",
        json=payload,
        headers=headers,
    )
    assert terminal.status_code == 200
    assert terminal.json()["status"] == "verified_allow"
    assert terminal.json()["nonce_consumed"] is True

    (
        _second_session,
        second_assignment,
        _second_settings,
        second_envelope,
        _second_allowlist,
        second_evidence,
        _second_now,
    ) = await _durable_report_fixture(database_session)
    initial = await client.post(
        f"/review/v1/assignments/{second_assignment.assignment_id}/report",
        json={
            "envelope": second_envelope,
            "evidence": {
                f"{kind}_b64": base64.b64encode(value).decode("ascii")
                for kind, value in second_evidence.items()
            },
        },
        headers={
            "Authorization": f"Bearer {_derive_test_token(second_assignment.assignment_id)}",
            "Content-Type": "application/json",
        },
    )
    assert initial.status_code == 202
    assert initial.json()["status"] == "verified_allow"


@pytest.mark.asyncio
async def test_report_and_evidence_routes_separate_redacted_public_from_internal_bytes(
    client,
    database_session,
    internal_headers,
) -> None:
    (
        review_session,
        assignment,
        settings,
        envelope,
        allowlist,
        evidence,
        now,
    ) = await _durable_report_fixture(database_session)

    async def signed_auth() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="review-evidence-miner",
            signature="offline",
            nonce="offline",
            timestamp="2026-07-11T00:00:00+00:00",
            body_sha256="0" * 64,
            canonical_request="GET\n/review\n0\noffline\nhash",
        )

    app.dependency_overrides[api_routes.signed_submission_auth] = signed_auth
    try:
        unavailable = await client.get(f"/submissions/{review_session.submission_id}/review/report")
        assert unavailable.status_code == 404

        async with database_session() as session:
            outcome = await submit_review_report(
                session,
                assignment=assignment,
                envelope=envelope,
                evidence_objects=evidence,
                evidence_settings=settings,
                quote_verifier=StaticQuoteVerifier(),
                allowlist=allowlist,
                now=now + timedelta(seconds=1),
            )
            assert outcome.status == "verified_allow"
            await session.commit()

        public = await client.get(f"/submissions/{review_session.submission_id}/review/report")
        internal = await client.get(
            f"/internal/v1/reviews/{review_session.session_id}/report",
            headers=internal_headers,
        )
        assert public.status_code == 200
        assert internal.status_code == 200
        public_text = public.text
        assert "Review supplied bytes only." not in public_text
        assert "or-evidence-response" not in public_text
        public_item = public.json()["items"][0]
        internal_item = internal.json()["items"][0]
        assert public_item["report_projection"]["review_digest"] == envelope["review_digest"]
        assert internal_item["report_envelope"] == envelope
        response_ref = internal_item["evidence_descriptor"]["response_object_ref"]
        response = await client.get(
            f"/internal/v1/reviews/{review_session.session_id}/evidence/{response_ref}",
            headers={**internal_headers, "Range": "bytes=1-10"},
        )
        assert response.status_code == 206
        assert response.content == evidence["response_body"][1:11]
        denied = await client.get(
            f"/internal/v1/reviews/{review_session.session_id}/evidence/{response_ref}"
        )
        assert denied.status_code in {401, 403}
    finally:
        app.dependency_overrides.pop(api_routes.signed_submission_auth, None)


def _derive_test_token(assignment_id: str) -> str:
    import hmac

    mac = hmac.new(
        b"test-token",
        b"agent-challenge:review-session:v1:" + assignment_id.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{assignment_id}.{mac}"


def _settings_with_evidence_key(
    *,
    shared_token: str = "test-token",
    evidence_key: str = "test-evidence-key",
) -> ChallengeSettings:
    """Build settings with bearer auth and evidence encryption split.

    Default evidence key matches CHALLENGE_REVIEW_EVIDENCE_ENCRYPTION_KEY from
    tests/conftest so internal evidence routes can decrypt fixture ciphertext.
    """

    return ChallengeSettings(
        shared_token=shared_token,
        review_evidence_encryption_key=evidence_key,
    )


def _minimal_bound_evidence(
    *,
    assignment_id: str,
    openrouter_observation: dict[str, Any],
) -> dict[str, bytes]:
    """Build exact evidence objects that bind to a synthetic openrouter observation."""

    from agent_challenge.review.schemas import (
        validate_observed_openrouter_transport,
        validate_planned_openrouter_request,
    )

    request_body = b"request"
    response_body = b"response-xx"
    metadata = b"metadata"
    planned = {
        "schema_version": 1,
        "method": "POST",
        "origin": "https://openrouter.ai:443",
        "path": "/api/v1/chat/completions",
        "headers": {
            "accept": "application/json",
            "accept-encoding": "identity",
            "content-type": "application/json",
            "x-openrouter-metadata": "enabled",
        },
        "body_sha256": hashlib.sha256(request_body).hexdigest(),
        "body_length": len(request_body),
        "model": "x-ai/grok-4.5",
        "routing_sha256": "11" * 32,
    }
    planned_bytes = validate_planned_openrouter_request(planned)
    observed = {
        "schema_version": 1,
        "planned_request_sha256": hashlib.sha256(planned_bytes).hexdigest(),
        "final_origin": "https://openrouter.ai:443",
        "final_path": "/api/v1/chat/completions",
        "tls_hostname": "openrouter.ai",
        "tls_hostname_verified": True,
        "redirected": False,
        "proxied": False,
        "response_status": 200,
        "response_content_encoding": "identity",
        "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
        "response_body_length": len(response_body),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
    }
    observed_bytes = validate_observed_openrouter_transport(observed)
    # Replace the caller's observation digests so binding checks accept the blob.
    openrouter_observation.update(
        {
            "planned_request_sha256": observed["planned_request_sha256"],
            "transport_observation_sha256": hashlib.sha256(observed_bytes).hexdigest(),
            "request_body_sha256": planned["body_sha256"],
            "request_body_length": planned["body_length"],
            "response_body_sha256": observed["response_body_sha256"],
            "response_body_length": observed["response_body_length"],
            "metadata_sha256": observed["metadata_sha256"],
        }
    )
    return {
        "planned_request": planned_bytes,
        "transport_observation": observed_bytes,
        "request_body": request_body,
        "response_body": response_body,
        "metadata": metadata,
    }


@pytest.mark.asyncio
async def test_evidence_encryption_is_independent_of_internal_bearer(
    database_session,
) -> None:
    """VAL-REVIEW-037: compromising the bearer must not decrypt evidence."""

    from agent_challenge.core.models import ReviewEvidenceObject
    from agent_challenge.review.evidence import (
        ReviewEvidenceError,
        load_review_evidence_object,
        store_review_evidence_objects,
    )

    now = datetime.now(UTC)
    artifact_bytes = b"evidence-key-split-zip"
    submission = AgentSubmission(
        miner_hotkey="evidence-key-split-miner",
        name="evidence-key-split-agent",
        agent_hash=hashlib.sha256(b"evidence-key-split").hexdigest(),
        artifact_uri="/tmp/evidence-key-split.zip",
        artifact_path="/tmp/evidence-key-split.zip",
        zip_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        zip_size_bytes=len(artifact_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )
    settings = _settings_with_evidence_key(
        shared_token="internal-bearer-secret",
        evidence_key="only-evidence-can-decrypt",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_bytes,
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        descriptors = await store_review_evidence_objects(
            session,
            assignment=created.assignment,
            settings=settings,
            objects={
                "planned_request": b'{"schema_version":1,"method":"POST"}',
                "request_body": b'{"model":"x-ai/grok-4.5"}',
                "response_body": b'{"id":"or-1"}',
            },
        )
        await session.commit()
        object_ref = str(descriptors["response_body"]["object_ref"])
        row = await session.scalar(
            select(ReviewEvidenceObject).where(ReviewEvidenceObject.object_ref == object_ref)
        )
        assert row is not None
        # Dedicated evidence key material decrypts correctly.
        _loaded, plaintext = await load_review_evidence_object(
            session,
            review_session=created.session,
            object_ref=object_ref,
            settings=settings,
        )
        assert plaintext == b'{"id":"or-1"}'
        # Compromised bearer-only settings cannot derive the evidence Fernet key.
        bearer_only = ChallengeSettings(shared_token="internal-bearer-secret")
        with pytest.raises(ReviewEvidenceError):
            await load_review_evidence_object(
                session,
                review_session=created.session,
                object_ref=object_ref,
                settings=bearer_only,
            )
        wrong_evidence_key = _settings_with_evidence_key(
            shared_token="internal-bearer-secret",
            evidence_key="attacker-evidence-key",
        )
        with pytest.raises(ReviewEvidenceError):
            await load_review_evidence_object(
                session,
                review_session=created.session,
                object_ref=object_ref,
                settings=wrong_evidence_key,
            )


@pytest.mark.asyncio
async def test_initial_report_receipt_commits_before_verification_and_requires_evidence(
    database_session,
) -> None:
    """VAL-REVIEW-041/042/059: receipt+evidence are durable before verification crashes."""

    from agent_challenge.core.models import ReviewAssignment
    from agent_challenge.review.report import ReviewReportError

    (
        _review_session,
        assignment,
        settings,
        envelope,
        allowlist,
        evidence,
        now,
    ) = await _durable_report_fixture(database_session, label="commit-boundary")
    assignment_pk = assignment.id
    receipt_time = now + timedelta(seconds=1)

    # Initial receipts without evidence are rejected before any durable state.
    async with database_session() as session:
        locked = await session.get(ReviewAssignment, assignment_pk)
        assert locked is not None
        with pytest.raises(ReviewReportError, match="evidence is required"):
            await submit_review_report(
                session,
                assignment=locked,
                envelope=envelope,
                evidence_objects=None,
                evidence_settings=settings,
                quote_verifier=_UnavailableVerifier(),
                allowlist=allowlist,
                now=receipt_time,
            )
        await session.rollback()

    import agent_challenge.review.report as report_module

    def _crash_after_receipt(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated post-receipt verification crash")

    original_verify = report_module.verify_review_envelope
    report_module.verify_review_envelope = _crash_after_receipt  # type: ignore[assignment]
    try:
        async with database_session() as session:
            locked = await session.get(ReviewAssignment, assignment_pk)
            assert locked is not None
            with pytest.raises(RuntimeError, match="post-receipt verification crash"):
                await submit_review_report(
                    session,
                    assignment=locked,
                    envelope=envelope,
                    evidence_objects=evidence,
                    evidence_settings=settings,
                    quote_verifier=StaticQuoteVerifier(),
                    allowlist=allowlist,
                    now=receipt_time,
                )
            # The function committed the receipt boundary before verification, so
            # rolling back this session must not erase the durable exact bytes.
            await session.rollback()
    finally:
        report_module.verify_review_envelope = original_verify  # type: ignore[assignment]

    async with database_session() as session:
        durable = await session.get(ReviewAssignment, assignment_pk)
        assert durable is not None
        assert durable.review_report_envelope_json is not None
        assert durable.review_report_sha256 is not None
        assert durable.review_report_received_at.replace(tzinfo=UTC) == receipt_time
        assert durable.review_evidence_descriptor_json is not None
        assert durable.phase == "review_verifying"
        # Outcome was not durable: verification crashed after receipt commit.
        assert durable.review_verification_outcome_json is None

        recovered = await recover_pending_review_reports(
            session,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=allowlist,
            now=now + timedelta(hours=1),
            evidence_settings=settings,
        )
        await session.commit()
        ready = await session.get(ReviewAssignment, assignment_pk)
        assert recovered == 1
        assert ready is not None
        assert ready.phase == "review_allowed"
        assert ready.review_report_received_at.replace(tzinfo=UTC) == receipt_time


@pytest.mark.asyncio
async def test_recovery_skips_incomplete_or_non_transient_receipts_and_audit_retryable_matches(
    database_session,
) -> None:
    """VAL-REVIEW-042/059: recovery is receipt/evidence aware; audit retryable matches outcome."""

    from agent_challenge.core.models import ReviewAssignment
    from agent_challenge.review.canonical import canonical_json_v1

    (
        review_session,
        assignment,
        settings,
        envelope,
        allowlist,
        evidence,
        now,
    ) = await _durable_report_fixture(database_session, label="recovery-aware")
    assignment_pk = assignment.id
    receipt_time = now + timedelta(seconds=2)

    async with database_session() as session:
        # Incomplete staged envelope without evidence descriptor is not recoverable.
        locked = await session.get(ReviewAssignment, assignment_pk)
        assert locked is not None
        locked.phase = "review_verifying"
        locked.review_report_envelope_json = json.dumps(envelope, separators=(",", ":"))
        locked.review_report_sha256 = hashlib.sha256(
            locked.review_report_envelope_json.encode()
        ).hexdigest()
        locked.review_report_received_at = receipt_time
        await session.commit()

        recovered = await recover_pending_review_reports(
            session,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=allowlist,
            now=now + timedelta(hours=1),
            evidence_settings=settings,
        )
        assert recovered == 0

        # Park a genuine verifier-unavailable outcome after full evidence receipt.
        locked = await session.get(ReviewAssignment, assignment_pk)
        assert locked is not None
        locked.review_report_envelope_json = None
        locked.review_report_sha256 = None
        locked.review_report_received_at = None
        locked.review_evidence_descriptor_json = None
        locked.phase = "review_cvm_running"
        await session.flush()
        parked = await submit_review_report(
            session,
            assignment=locked,
            envelope=envelope,
            evidence_objects=evidence,
            evidence_settings=settings,
            quote_verifier=_UnavailableVerifier(),
            allowlist=allowlist,
            now=receipt_time,
        )
        assert parked.status == "verifier_unavailable"
        await session.commit()

    async with database_session() as session:
        durable_session = await session.get(ReviewSession, review_session.id)
        assert durable_session is not None
        internal = await review_audit_page(
            session,
            session_row=durable_session,
            cursor=None,
            limit=10,
            internal=True,
        )
        assert internal["items"][0]["phase"] == "review_verifying"
        assert internal["items"][0]["retryable"] is True
        assert internal["items"][0]["verification_outcome"]["retryable"] is True
        assert internal["items"][0]["verification_outcome"]["status"] == "verifier_unavailable"

        # A definitive failure outcome already stamped must not be re-accepted as allow.
        row = await session.get(ReviewAssignment, assignment_pk)
        assert row is not None
        row.review_verification_outcome_json = canonical_json_v1(
            {
                "status": "trust_failed",
                "terminal": True,
                "retryable": False,
                "reason_code": "review_binding_invalid",
                "nonce_consumed": True,
                "verified_at_ms": int(receipt_time.timestamp() * 1000),
                "measurement_allowlisted": False,
                "report_data_matched": False,
            }
        ).decode("utf-8")
        # Keep phase verifying to ensure recovery consults the durable outcome, not phase alone.
        row.phase = "review_verifying"
        await session.commit()

        recovered = await recover_pending_review_reports(
            session,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=allowlist,
            now=now + timedelta(hours=1),
            evidence_settings=settings,
        )
        assert recovered == 0
        stayed = await session.get(ReviewAssignment, assignment_pk)
        assert stayed is not None
        assert stayed.phase == "review_verifying"
        outcome = json.loads(stayed.review_verification_outcome_json or "{}")
        assert outcome["status"] == "trust_failed"
