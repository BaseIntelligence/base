"""Strict review-domain report construction and validator verification.

Review reports are intentionally separate from score ``ExecutionProof`` objects.
The core is hashed first, then a second small binding is hashed into the 64-byte
TDX ``report_data`` field.  This avoids a self-referential envelope hash while
still binding the validator-issued session and purpose-scoped nonce.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from agent_challenge.core.models import ReviewAssignment
    from agent_challenge.sdk.config import ChallengeSettings

from .canonical import CanonicalJsonError, canonical_json_v1, parse_json_object
from .policy import MAX_REVIEW_DECISION_ENTRIES
from .schemas import (
    REVIEW_MODEL,
    AssignmentSchemaError,
    _require_id,
    _require_positive_int,
    _require_sha256,
    _require_time_ms,
    validate_review_assignment,
)

REVIEW_REPORT_SCHEMA_VERSION = 1
# report_data preimage schema is independent of review_core schema_version
# (library/ac-attestation.md freeze: explicit issued_at/received_at echoes).
REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_VERSION = 2
REVIEW_REPORT_DOMAIN = "base-agent-challenge-review-v1"
REPORT_DATA_HEX_LENGTH = 128
MAX_REVIEW_EVENT_LOG_ENTRIES = 4096
MAX_REVIEW_QUOTE_BYTES = 65_536
MAX_REVIEW_EVENT_LOG_BYTES = 2_097_152
MAX_REVIEW_VM_CONFIG_BYTES = 65_536

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ReviewReportError(ValueError):
    """A review core, envelope, or validator binding is malformed."""


class ReviewReportConflict(ReviewReportError):
    """A report conflicts with a durable receipt or assignment state."""


class ReviewVerifierUnavailable(RuntimeError):
    """The external quote verifier is temporarily unavailable."""


@dataclass
class DcapReviewQuoteVerifier:
    """Map DCAP verifier outages to the review's retryable disposition."""

    verifier: _ReviewQuoteVerifier | None = None

    def verify(self, quote_hex: str) -> Any:
        from agent_challenge.keyrelease.quote import (
            DcapQvlVerifier,
            QuoteVerificationError,
            QuoteVerifierUnavailable,
        )

        verifier = self.verifier or DcapQvlVerifier()
        try:
            return verifier.verify(quote_hex)
        except QuoteVerifierUnavailable as exc:
            raise ReviewVerifierUnavailable("review quote verifier unavailable") from exc
        except QuoteVerificationError:
            raise


class _ReviewQuoteVerifier(Protocol):
    def verify(self, quote_hex: str) -> Any:  # pragma: no cover - protocol
        """Return a quote verdict or raise a review verifier exception."""


@dataclass(frozen=True)
class ReviewMeasurementAllowlist:
    """Validator-owned, rotating review measurement allowlist.

    An empty allowlist matches nothing.  Construction accepts only the canonical
    six-field measurement record, so callers cannot select a subset or smuggle
    runtime-only fields into trust membership.
    """

    entries: tuple[dict[str, str], ...] = ()

    @classmethod
    def from_measurements(
        cls, measurements: Iterable[Mapping[str, Any]]
    ) -> ReviewMeasurementAllowlist:
        return cls(tuple(_canonical_allowlist_entry(item) for item in measurements))

    def __bool__(self) -> bool:
        return bool(self.entries)

    def contains(self, measurement: Mapping[str, Any]) -> bool:
        try:
            candidate = _measurement_for_allowlist(measurement)
        except ReviewReportError:
            return False
        return any(candidate == item for item in self.entries)


@dataclass(frozen=True)
class ReviewVerificationOutcome:
    """Durable disposition returned after one report verification attempt."""

    status: Literal[
        "verified_allow",
        "verified_reject",
        "verified_escalate",
        "trust_failed",
        "verifier_unavailable",
    ]
    terminal: bool
    retryable: bool
    reason_code: str
    nonce_consumed: bool
    measurement_allowlisted: bool
    report_data_matched: bool
    verified_at_ms: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "terminal": self.terminal,
            "retryable": self.retryable,
            "reason_code": self.reason_code,
            "nonce_consumed": self.nonce_consumed,
            "verified_at_ms": self.verified_at_ms,
            "measurement_allowlisted": self.measurement_allowlisted,
            "report_data_matched": self.report_data_matched,
        }


def validate_review_core(value: object, *, settings: object | None = None) -> bytes:
    """Validate Review core v1 and return exactly its canonical bytes."""

    error = ReviewReportError
    if not isinstance(value, Mapping):
        raise error("review core must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "session_id",
            "assignment_id",
            "assignment_digest",
            "submission_id",
            "artifact_observation",
            "rules_observation",
            "policy_observation",
            "openrouter_observation",
            "decision",
            "times",
            "review_nonce",
        },
        error,
    )
    if value["schema_version"] != REVIEW_REPORT_SCHEMA_VERSION:
        raise error("unsupported review core schema version")
    for field in ("session_id", "assignment_id", "submission_id", "review_nonce"):
        _require_id(value[field], field, error)
    _require_sha256(value["assignment_digest"], "assignment_digest", error)
    _validate_artifact_observation(value["artifact_observation"], error)
    _validate_rules_observation(value["rules_observation"], error)
    _validate_policy_observation(value["policy_observation"], error)
    _validate_openrouter_observation(value["openrouter_observation"], error)
    limits = _review_resource_limits(settings)
    _validate_decision(
        value["decision"],
        error,
        maximum_items=limits["reason_evidence_items"],
    )
    _validate_times(value["times"], error)
    return _canonical_bytes(value, error)


def review_digest(review_core: Mapping[str, Any], *, settings: object | None = None) -> str:
    """Return SHA-256 of the strict, canonical non-circular review core."""

    return sha256(validate_review_core(review_core, settings=settings)).hexdigest()


def review_report_data_preimage(review_core: Mapping[str, Any]) -> dict[str, Any]:
    """Build the only second-layer review-domain report-data object (v2 times).

    Cryptographically binds challenge/validator-domain ``issued_at_ms`` and
    product ``received_at_ms`` (submission/send receive) into the preimage that
    is hashed into the 64-byte TDX ``report_data`` field (VAL-ACAT-007/008/021/022).
    """

    from .attested_times import extract_bound_times_from_core, review_report_data_preimage_v2

    validate_review_core(review_core)
    issued_at_ms, received_at_ms = extract_bound_times_from_core(review_core)
    return review_report_data_preimage_v2(
        review_digest=review_digest(review_core),
        session_id=str(review_core["session_id"]),
        review_nonce=str(review_core["review_nonce"]),
        issued_at_ms=issued_at_ms,
        received_at_ms=received_at_ms,
    )


def review_report_data_hex(review_core: Mapping[str, Any]) -> str:
    """Derive the 64-byte, left-aligned review-domain TDX report-data field."""

    digest = sha256(canonical_json_v1(review_report_data_preimage(review_core))).hexdigest()
    return digest + ("00" * 32)


def extract_bound_attestation_times(review_core: Mapping[str, Any]) -> dict[str, int]:
    """Return bound issued/received times from validated review materials."""

    from .attested_times import extract_bound_times_from_core

    validate_review_core(review_core)
    issued, received = extract_bound_times_from_core(review_core)
    return {"issued_at_ms": issued, "received_at_ms": received}


def reverify_bound_attestation_times(
    *,
    review_core: Mapping[str, Any],
    report_data_hex: str,
) -> dict[str, int]:
    """Re-verify report_data and return only cryptographically bound times."""

    from .attested_times import reverify_extract_bound_times

    validate_review_core(review_core)
    return reverify_extract_bound_times(
        review_core=review_core,
        report_data_hex=report_data_hex,
        review_digest=review_digest(review_core),
    )


def build_review_envelope(
    *,
    review_core: Mapping[str, Any],
    tdx_quote_hex: str,
    event_log: list[Mapping[str, Any]],
    measurement: Mapping[str, Any],
    vm_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the strict outer Review envelope after requesting the quote."""

    envelope = {
        "schema_version": REVIEW_REPORT_SCHEMA_VERSION,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_core": dict(review_core),
        "review_digest": review_digest(review_core),
        "report_data_hex": review_report_data_hex(review_core),
        "attestation": {
            "tdx_quote_hex": tdx_quote_hex,
            "event_log": [dict(item) for item in event_log],
            "measurement": dict(measurement),
            "vm_config": dict(vm_config),
        },
    }
    validate_review_envelope(envelope)
    return envelope


def validate_review_envelope(value: object, *, settings: object | None = None) -> bytes:
    """Validate Review envelope v1 without trusting its quoted measurements."""

    error = ReviewReportError
    if not isinstance(value, Mapping):
        raise error("review envelope must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "domain",
            "review_core",
            "review_digest",
            "report_data_hex",
            "attestation",
        },
        error,
    )
    if value["schema_version"] != REVIEW_REPORT_SCHEMA_VERSION:
        raise error("unsupported review envelope schema version")
    if value["domain"] != REVIEW_REPORT_DOMAIN:
        raise error("review envelope domain is invalid")
    core = value["review_core"]
    if not isinstance(core, Mapping):
        raise error("review envelope core is invalid")
    digest = review_digest(core, settings=settings)
    if value["review_digest"] != digest:
        raise error("review envelope digest mismatches review core")
    expected_data = review_report_data_hex(core)
    _require_report_data(value["report_data_hex"], error)
    if value["report_data_hex"] != expected_data:
        raise error("review envelope report data mismatches review core")
    _validate_attestation(value["attestation"], error, settings=settings)
    _validate_quote_envelope_binding(
        attestation=value["attestation"],
        expected_report_data_hex=expected_data,
        error=error,
    )
    return _canonical_bytes(value, error)


def verify_review_envelope(
    *,
    envelope: Mapping[str, Any],
    assignment: Mapping[str, Any],
    quote_verifier: _ReviewQuoteVerifier,
    allowlist: ReviewMeasurementAllowlist,
    received_at_ms: int,
    settings: object | None = None,
) -> ReviewVerificationOutcome:
    """Run every definitive review trust check without mutating persistence."""

    from agent_challenge.keyrelease.quote import (
        QuoteStructureError,
        QuoteVerificationError,
        QuoteVerifierUnavailable,
        os_image_hash_from_registers,
        parse_tdx_quote_v4,
        replay_rtmr3,
        validate_rtmr3_event_log,
    )

    try:
        limits = _review_resource_limits(settings)
        validate_review_envelope(envelope, settings=settings)
        assignment_for_schema = {
            key: value for key, value in assignment.items() if key != "model_call_started_json"
        }
        validate_review_assignment(assignment_for_schema)
        _validate_core_assignment_binding(envelope["review_core"], assignment)
        _validate_attestation_assignment_binding(envelope["attestation"], assignment)
        _validate_receipt_time(assignment, received_at_ms)
        _validate_timeline_against_receipt(envelope["review_core"], received_at_ms)
        attestation = envelope["attestation"]
        assert isinstance(attestation, Mapping)  # covered by validate_review_envelope
        quote = str(attestation["tdx_quote_hex"])
        report = parse_tdx_quote_v4(quote)

        try:
            # Apply the configured verification budget so a hung DCAP tool cannot
            # monopolize validator capacity (attestation_verification_timeout_seconds).
            from concurrent.futures import ThreadPoolExecutor
            from concurrent.futures import TimeoutError as FuturesTimeout

            deadline = float(
                getattr(settings, "attestation_verification_timeout_seconds", 60.0)
                if settings is not None
                else 60.0
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(quote_verifier.verify, quote)
                try:
                    verdict = future.result(timeout=deadline)
                except FuturesTimeout as exc:
                    raise ReviewVerifierUnavailable("review quote verification timed out") from exc
        except ReviewVerifierUnavailable:
            return _verifier_unavailable()
        except QuoteVerifierUnavailable:
            return _verifier_unavailable()
        except (QuoteVerificationError, QuoteStructureError):
            return _trust_failed("review_quote_invalid")
        except Exception:
            return _trust_failed("review_quote_invalid")
        if getattr(verdict, "tcb_status", None) != "UpToDate":
            return _trust_failed("review_tcb_unacceptable")

        event_log = attestation["event_log"]
        assert isinstance(event_log, list)
        validated_log = validate_rtmr3_event_log(event_log, max_entries=limits["event_log_entries"])
        replay = replay_rtmr3(validated_log)
        if replay.rtmr3 != report.rtmr3 or replay.compose_hash is None:
            return _trust_failed("review_event_log_mismatch")
        replay_key_provider = _decode_key_provider(replay.key_provider)

        measurement = attestation["measurement"]
        assert isinstance(measurement, Mapping)
        signed_measurement = {
            "mrtd": report.mrtd,
            "rtmr0": report.rtmr0,
            "rtmr1": report.rtmr1,
            "rtmr2": report.rtmr2,
            "rtmr3": report.rtmr3,
            "compose_hash": replay.compose_hash,
            "os_image_hash": os_image_hash_from_registers(report.mrtd, report.rtmr1, report.rtmr2),
            "key_provider": replay_key_provider,
            "vm_shape": measurement["vm_shape"],
        }
        if dict(measurement) != signed_measurement:
            return _trust_failed("review_measurement_mismatch")
        if not allowlist or not allowlist.contains(signed_measurement):
            return _trust_failed("review_measurement_unallowlisted")

        # Live production admission: re-verify bound times + outcome + OR digests
        # from report_data materials (not guest clock / unattested DB alone /
        # plain status). VAL-ACAT-007/008/021–024; feature wire requirement.
        from .or_outcome_bind import (
            ReviewOrOutcomeError,
            admit_production_from_bound_outcome,
        )

        try:
            production = admit_production_from_bound_outcome(
                review_core=envelope["review_core"],
                reported_report_data_hex=str(envelope.get("report_data_hex")),
                plain_status_allow=None,
                require_or_digests=True,
            )
        except ReviewOrOutcomeError:
            return _trust_failed("review_binding_invalid")
        if production.status == "verified_allow":
            return _verified("verified_allow", received_at_ms)
        if production.status == "verified_reject":
            return _verified("verified_reject", received_at_ms)
        if production.status == "verified_escalate":
            return _verified("verified_escalate", received_at_ms)
        return _trust_failed("review_binding_invalid")
    except ReviewVerifierUnavailable:
        return _verifier_unavailable()
    except QuoteVerifierUnavailable:
        return _verifier_unavailable()
    except (
        AssignmentSchemaError,
        CanonicalJsonError,
        QuoteStructureError,
        QuoteVerificationError,
        ReviewReportError,
        ValueError,
        TypeError,
    ):
        return _trust_failed("review_binding_invalid")


async def submit_review_report(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
    envelope: Mapping[str, Any],
    evidence_objects: Mapping[str, bytes] | None = None,
    evidence_settings: ChallengeSettings | None = None,
    quote_verifier: _ReviewQuoteVerifier,
    allowlist: ReviewMeasurementAllowlist,
    now: datetime,
) -> ReviewVerificationOutcome:
    """Durably receipt then verify one immutable review report.

    Exact report bytes, digest, identity, receipt time, and evidence descriptors
    commit before verification so a crash/outage after the receipt boundary can
    resume the same timely bytes. A schema-valid timely receipt can resume after
    expiry. Transient verifier outage leaves the nonce active; every valid or
    definitive-invalid result atomically terminalizes and consumes it once.
    """

    from sqlalchemy import select

    from agent_challenge.core.models import ReviewAssignment, ReviewNonce, ReviewSession
    from agent_challenge.review.sessions import record_review_submission_status

    now_ms = int(_as_utc(now).timestamp() * 1000)
    envelope_bytes = validate_review_envelope(envelope, settings=evidence_settings)
    envelope_json = envelope_bytes.decode("utf-8")
    envelope_sha256 = sha256(envelope_bytes).hexdigest()
    locked_assignment = await session.get(ReviewAssignment, assignment.id, with_for_update=True)
    if locked_assignment is None:
        raise ReviewReportConflict("review assignment does not exist")
    assignment = locked_assignment

    existing_json = assignment.review_report_envelope_json
    first_receipt = existing_json is None
    if existing_json is not None:
        if assignment.review_report_sha256 != envelope_sha256 or existing_json != envelope_json:
            raise ReviewReportConflict("review report conflicts with durable receipt")
        if assignment.review_verification_outcome_json is not None:
            existing = _outcome_from_json(assignment.review_verification_outcome_json)
            if existing.terminal:
                return existing
            if existing.status != "verifier_unavailable" or not existing.retryable:
                # Definitive invalid trust/policy already parked: never reopen.
                raise ReviewReportConflict("review report outcome is not retryable")
        if assignment.review_report_received_at is None:
            raise ReviewReportConflict("review report receipt time is missing")
        receipt_ms = _to_ms(assignment.review_report_received_at)
    elif now_ms >= _to_ms(assignment.expires_at):
        raise ReviewReportConflict("review report first receipt is expired")
    else:
        # Initial receipts must present evidence at the call boundary. Exact
        # report bytes commit independently so they survive later evidence or
        # verification failures before descriptors/outcomes are written.
        if evidence_objects is None or not evidence_objects:
            raise ReviewReportError("review evidence is required for a new report receipt")
        assignment.review_report_envelope_json = envelope_json
        assignment.review_report_sha256 = envelope_sha256
        assignment.review_digest = str(envelope["review_digest"])
        assignment.review_report_data_hex = str(envelope["report_data_hex"])
        assignment.review_report_received_at = _as_utc(now)
        receipt_ms = now_ms

    if assignment.phase not in {
        "review_cvm_running",
        "review_provider_standby",
        "review_verifying",
    }:
        raise ReviewReportConflict("review report assignment is not accepting reports")
    if assignment.phase != "review_verifying":
        assignment.phase = "review_verifying"
        review_session = await session.get(ReviewSession, assignment.session_id)
        if review_session is None:
            raise ReviewReportConflict("review session does not exist")
        await record_review_submission_status(
            session,
            review_session=review_session,
            assignment=assignment,
            raw_status="review_verifying",
            reason="review_report_receipted",
        )

    if first_receipt:
        # Boundary 1: exact report identity/digest/time is durable before later
        # evidence storage or verification work can fail or crash.
        await session.flush()
        await session.commit()
        assignment = await session.get(ReviewAssignment, assignment.id, with_for_update=True)
        if assignment is None:
            raise ReviewReportConflict("review assignment does not exist")
        if assignment.review_report_received_at is None:
            raise ReviewReportConflict("review report receipt time is missing")
        receipt_ms = _to_ms(assignment.review_report_received_at)

    if evidence_objects:
        if evidence_settings is None:
            raise ReviewReportError("review evidence storage settings are required")
        descriptor = await _store_and_describe_evidence(
            session,
            assignment=assignment,
            envelope=envelope,
            objects=evidence_objects,
            settings=evidence_settings,
        )
        descriptor_json = canonical_json_v1(descriptor).decode("utf-8")
        if (
            assignment.review_evidence_descriptor_json is not None
            and assignment.review_evidence_descriptor_json != descriptor_json
        ):
            raise ReviewReportConflict("review evidence conflicts with durable receipt")
        assignment.review_evidence_descriptor_json = descriptor_json
    elif assignment.review_evidence_descriptor_json is None:
        raise ReviewReportError("review evidence is required for a new report receipt")

    # Boundary 2: evidence descriptors (when newly written) are committed before
    # verification so recovery resumes exactly the immutable receipt+evidence.
    await session.flush()
    await session.commit()
    assignment = await session.get(ReviewAssignment, assignment.id, with_for_update=True)
    if assignment is None:
        raise ReviewReportConflict("review assignment does not exist")
    if assignment.review_report_received_at is None:
        raise ReviewReportConflict("review report receipt time is missing")
    if assignment.review_evidence_descriptor_json is None:
        raise ReviewReportError("review evidence is required for a new report receipt")
    receipt_ms = _to_ms(assignment.review_report_received_at)

    assignment_data = parse_json_object(assignment.assignment_bytes)
    assignment_data["model_call_started_json"] = assignment.model_call_started_json
    outcome = verify_review_envelope(
        envelope=envelope,
        assignment=assignment_data,
        quote_verifier=quote_verifier,
        allowlist=allowlist,
        received_at_ms=receipt_ms,
        settings=evidence_settings,
    )
    if outcome.status == "verifier_unavailable":
        assignment.review_verification_outcome_json = canonical_json_v1(outcome.as_dict()).decode(
            "utf-8"
        )
        assignment.phase = "review_verifying"
        assignment.reason_code = outcome.reason_code
        return outcome

    # Definitive allow/reject/escalate/trust-failed consume the nonce exactly once
    # with the outcome row so terminalization stays atomic under retry/race.
    nonce = await session.scalar(
        select(ReviewNonce).where(ReviewNonce.assignment_id == assignment.id).with_for_update()
    )
    if nonce is None or nonce.state != "active":
        if assignment.review_verification_outcome_json is not None:
            existing = _outcome_from_json(assignment.review_verification_outcome_json)
            if existing.terminal:
                return existing
        raise ReviewReportConflict("review nonce is no longer active")

    assignment.review_verification_outcome_json = canonical_json_v1(outcome.as_dict()).decode(
        "utf-8"
    )
    nonce.state = "consumed"
    nonce.consumed_at = _as_utc(now)
    assignment.capability_state = "revoked"
    assignment.active_key = None
    assignment.finished_at = _as_utc(now)
    assignment.reason_code = outcome.reason_code
    if outcome.status == "verified_allow":
        assignment.phase = "review_allowed"
    elif outcome.status == "verified_reject":
        assignment.phase = "review_rejected"
    elif outcome.status == "verified_escalate":
        assignment.phase = "review_escalated"
    else:
        assignment.phase = "review_error"
    assignment.review_public_projection_json = canonical_json_v1(
        _public_projection(envelope=envelope, outcome=outcome)
    ).decode("utf-8")
    review_session = await session.get(
        ReviewSession,
        assignment.session_id,
        with_for_update=True,
    )
    if review_session is None:
        raise ReviewReportConflict("review session does not exist")
    if outcome.status == "verified_allow":
        if review_session.authorizing_assignment_id not in {None, assignment.assignment_id}:
            raise ReviewReportConflict("review authorizing assignment conflicts")
        review_session.authorizing_assignment_id = assignment.assignment_id
    await record_review_submission_status(
        session,
        review_session=review_session,
        assignment=assignment,
        raw_status=assignment.phase,
        reason=f"review_{outcome.status}",
    )
    return outcome


async def _store_and_describe_evidence(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
    envelope: Mapping[str, Any],
    objects: Mapping[str, bytes],
    settings: ChallengeSettings,
) -> dict[str, object]:
    """Persist the only raw review bytes and bind every descriptor field."""

    from .evidence import (
        REVIEW_EVIDENCE_ENCRYPTION_PROFILE,
        store_review_evidence_objects,
    )
    from .schemas import (
        validate_observed_openrouter_transport,
        validate_planned_openrouter_request,
    )

    core = envelope["review_core"]
    if not isinstance(core, Mapping):
        raise ReviewReportError("review evidence has no valid report core")
    openrouter = core["openrouter_observation"]
    if not isinstance(openrouter, Mapping):
        raise ReviewReportError("review evidence has no OpenRouter observation")
    expected_keys = {"planned_request", "transport_observation", "request_body", "response_body"}
    metadata_sha256 = openrouter["metadata_sha256"]
    if metadata_sha256 is not None:
        expected_keys.add("metadata")
    if set(objects) != expected_keys:
        raise ReviewReportError("review evidence object set is invalid")
    planned_bytes = objects["planned_request"]
    observed_bytes = objects["transport_observation"]
    request_body = objects["request_body"]
    response_body = objects["response_body"]
    try:
        planned = parse_json_object(planned_bytes)
        observed = parse_json_object(observed_bytes)
        if validate_planned_openrouter_request(planned) != planned_bytes:
            raise ReviewReportError("planned review evidence is not canonical")
        if validate_observed_openrouter_transport(observed) != observed_bytes:
            raise ReviewReportError("transport review evidence is not canonical")
    except (CanonicalJsonError, ValueError) as exc:
        raise ReviewReportError("review evidence is malformed") from exc
    marker = parse_json_object(assignment.model_call_started_json or "")
    if (
        sha256(planned_bytes).hexdigest() != openrouter["planned_request_sha256"]
        or sha256(observed_bytes).hexdigest() != openrouter["transport_observation_sha256"]
        or sha256(request_body).hexdigest() != openrouter["request_body_sha256"]
        or len(request_body) != openrouter["request_body_length"]
        or sha256(response_body).hexdigest() != openrouter["response_body_sha256"]
        or len(response_body) != openrouter["response_body_length"]
        or planned["body_sha256"] != openrouter["request_body_sha256"]
        or planned["body_length"] != openrouter["request_body_length"]
        or observed["planned_request_sha256"] != openrouter["planned_request_sha256"]
        or observed["response_body_sha256"] != openrouter["response_body_sha256"]
        or observed["response_body_length"] != openrouter["response_body_length"]
        or marker["planned_request_sha256"] != openrouter["planned_request_sha256"]
        or marker["request_body_sha256"] != openrouter["request_body_sha256"]
        or marker["request_body_length"] != openrouter["request_body_length"]
    ):
        raise ReviewReportError("review evidence does not bind report and marker")
    if metadata_sha256 is not None and sha256(objects["metadata"]).hexdigest() != metadata_sha256:
        raise ReviewReportError("review metadata evidence does not bind report")
    # ReviewEvidenceError is a bare ValueError subclass (not ReviewReportError).
    # Wrap store failures so /report maps closed detail.code instead of raw HTTP 500
    # after first-receipt commit (Mode B residual: missing evidence encryption key).
    from .evidence import ReviewEvidenceError

    try:
        descriptors = await store_review_evidence_objects(
            session,
            assignment=assignment,
            settings=settings,
            objects=objects,
        )
    except ReviewEvidenceError as exc:
        text = str(exc).lower()
        if any(
            token in text
            for token in (
                "encryption key",
                "unavailable",
                "not configured",
            )
        ):
            raise ReviewReportError("review evidence encryption key is unavailable") from exc
        raise ReviewReportError("review evidence is invalid") from exc
    return {
        "schema_version": 1,
        "planned_request_object_ref": descriptors["planned_request"]["object_ref"],
        "planned_request_sha256": descriptors["planned_request"]["sha256"],
        "transport_object_ref": descriptors["transport_observation"]["object_ref"],
        "transport_observation_sha256": descriptors["transport_observation"]["sha256"],
        "request_object_ref": descriptors["request_body"]["object_ref"],
        "request_body_sha256": descriptors["request_body"]["sha256"],
        "request_body_length": descriptors["request_body"]["length"],
        "response_object_ref": descriptors["response_body"]["object_ref"],
        "response_body_sha256": descriptors["response_body"]["sha256"],
        "response_body_length": descriptors["response_body"]["length"],
        "metadata_object_ref": (
            descriptors["metadata"]["object_ref"] if "metadata" in descriptors else None
        ),
        "metadata_sha256": descriptors["metadata"]["sha256"] if "metadata" in descriptors else None,
        "encryption_profile": REVIEW_EVIDENCE_ENCRYPTION_PROFILE,
        "retention_class": "review-trust-evidence-v1",
    }


def _public_projection(
    *,
    envelope: Mapping[str, Any],
    outcome: ReviewVerificationOutcome,
) -> dict[str, object]:
    """Derive the sole deterministic public report projection from trusted rows."""

    core = envelope["review_core"]
    assert isinstance(core, Mapping)
    artifact = core["artifact_observation"]
    rules = core["rules_observation"]
    policy = core["policy_observation"]
    transport = core["openrouter_observation"]
    decision = core["decision"]
    times = core["times"]
    attestation = envelope["attestation"]
    assert all(
        isinstance(value, Mapping)
        for value in (artifact, rules, policy, transport, decision, times, attestation)
    )
    return {
        "schema_version": 1,
        "session_id": core["session_id"],
        "assignment_id": core["assignment_id"],
        "submission_id": core["submission_id"],
        "assignment_digest": core["assignment_digest"],
        "agent_hash": artifact["agent_hash"],
        "zip_sha256": artifact["zip_sha256"],
        "rules_snapshot_sha256": rules["snapshot_sha256"],
        "prompt_sha256": policy["prompt_sha256"],
        "tool_schema_sha256": policy["tool_schema_sha256"],
        "model": policy["model"],
        "routing_sha256": policy["routing_sha256"],
        "request_body_sha256": transport["request_body_sha256"],
        "response_body_sha256": transport["response_body_sha256"],
        "verifier_sha256": policy["verifier_sha256"],
        "verifier_output_sha256": decision["verifier_output_sha256"],
        "review_digest": envelope["review_digest"],
        "verdict": decision["verdict"],
        "reason_codes": decision["reason_codes"],
        "issued_at_ms": times["issued_at_ms"],
        "started_at_ms": times["started_at_ms"],
        "finished_at_ms": times["report_finished_at_ms"],
        "quote_fingerprint_sha256": sha256(
            bytes.fromhex(str(attestation["tdx_quote_hex"]))
        ).hexdigest(),
        "measurement_allowlisted": outcome.measurement_allowlisted,
        "attestation_verified": outcome.status.startswith("verified_"),
    }


def _validate_artifact_observation(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, Mapping):
        raise error("artifact observation must be an object")
    _require_exact_keys(
        value,
        {
            "agent_hash",
            "zip_sha256",
            "zip_size_bytes",
            "manifest_sha256",
            "manifest_entries_sha256",
        },
        error,
    )
    for field in ("agent_hash", "zip_sha256", "manifest_sha256", "manifest_entries_sha256"):
        _require_sha256(value[field], field, error)
    _require_positive_int(value["zip_size_bytes"], "zip_size_bytes", error)


def _validate_rules_observation(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, Mapping):
        raise error("rules observation must be an object")
    _require_exact_keys(value, {"snapshot_sha256", "revision_id"}, error)
    _require_sha256(value["snapshot_sha256"], "snapshot_sha256", error)
    _require_id(value["revision_id"], "revision_id", error)


def _validate_policy_observation(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, Mapping):
        raise error("policy observation must be an object")
    _require_exact_keys(
        value,
        {
            "model",
            "routing_sha256",
            "prompt_version",
            "prompt_sha256",
            "tool_schema_version",
            "tool_schema_sha256",
            "verifier_version",
            "verifier_sha256",
        },
        error,
    )
    if value["model"] != REVIEW_MODEL:
        raise error("review model is not exact")
    for field in ("routing_sha256", "prompt_sha256", "tool_schema_sha256", "verifier_sha256"):
        _require_sha256(value[field], field, error)
    for field in ("prompt_version", "tool_schema_version", "verifier_version"):
        _require_id(value[field], field, error)


def _validate_openrouter_observation(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, Mapping):
        raise error("OpenRouter observation must be an object")
    _require_exact_keys(
        value,
        {
            "planned_request_sha256",
            "transport_observation_sha256",
            "request_body_sha256",
            "request_body_length",
            "response_status",
            "response_content_encoding",
            "response_body_sha256",
            "response_body_length",
            "response_id",
            "returned_model",
            "metadata_sha256",
            "observed_provider",
            "provider_provenance",
            "cache_hit",
        },
        error,
    )
    for field in (
        "planned_request_sha256",
        "transport_observation_sha256",
        "request_body_sha256",
        "response_body_sha256",
    ):
        _require_sha256(value[field], field, error)
    if value["metadata_sha256"] is not None:
        _require_sha256(value["metadata_sha256"], "metadata_sha256", error)
    _require_positive_int(value["request_body_length"], "request_body_length", error)
    _require_positive_int(value["response_body_length"], "response_body_length", error)
    status = value["response_status"]
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise error("response_status is invalid")
    if value["response_content_encoding"] != "identity":
        raise error("response content encoding is invalid")
    _require_id(value["response_id"], "response_id", error)
    from .schemas import is_pinned_review_model

    if not is_pinned_review_model(value["returned_model"]):
        raise error("returned model is not exact")
    if value["provider_provenance"] not in {"openrouter_metadata", "unavailable"}:
        raise error("provider provenance is invalid")
    if value["provider_provenance"] == "openrouter_metadata":
        _require_id(value["observed_provider"], "observed_provider", error)
    elif value["observed_provider"] is not None:
        raise error("unavailable provider provenance must not name a provider")
    if value["cache_hit"] is not False:
        raise error("review must not use a provider cache")


def _validate_decision(
    value: object,
    error: type[ValueError],
    *,
    maximum_items: int = MAX_REVIEW_DECISION_ENTRIES,
) -> None:
    if not isinstance(value, Mapping):
        raise error("review decision must be an object")
    _require_exact_keys(
        value,
        {
            "static_findings_sha256",
            "parsed_output_sha256",
            "verifier_input_sha256",
            "verifier_output_sha256",
            "verifier_result",
            "verdict",
            "reason_codes",
            "evidence_digests",
        },
        error,
    )
    for field in (
        "static_findings_sha256",
        "parsed_output_sha256",
        "verifier_input_sha256",
        "verifier_output_sha256",
    ):
        _require_sha256(value[field], field, error)
    if value["verifier_result"] not in {"pass", "reject", "escalate", "error"}:
        raise error("verifier result is invalid")
    if value["verdict"] not in {"allow", "reject", "escalate"}:
        raise error("review verdict is invalid")
    if value["verdict"] == "allow" and value["verifier_result"] != "pass":
        raise error("allow requires a passed deterministic verifier")
    _require_sorted_set(
        value["reason_codes"],
        "reason_codes",
        _require_reason_code,
        error,
        maximum_items=maximum_items,
    )
    _require_sorted_set(
        value["evidence_digests"],
        "evidence_digests",
        _require_digest_item,
        error,
        maximum_items=maximum_items,
    )


def _validate_times(value: object, error: type[ValueError]) -> None:
    """Validate closed review times including attestation-bound submission receive.

    Product freeze requires ``submission_received_at_ms`` so ``received_at``
    cannot live as a DB-only column for age/freshness (VAL-ACAT-008/022).

    Ordering (non-strict among middle leaves, strict terminal):
      issued_at_ms ≤ started_at_ms ≤ model_call_marked_at_ms ≤ request_started_at_ms
        ≤ request_finished_at_ms ≤ verifier_finished_at_ms ≤ report_finished_at_ms
        < expires_at_ms
      and issued_at_ms ≤ submission_received_at_ms ≤ expires_at_ms
    """

    if not isinstance(value, Mapping):
        raise error("review times must be an object")
    timeline_names = (
        "issued_at_ms",
        "started_at_ms",
        "model_call_marked_at_ms",
        "request_started_at_ms",
        "request_finished_at_ms",
        "verifier_finished_at_ms",
        "report_finished_at_ms",
        "expires_at_ms",
    )
    # Closed exact key set includes attested submission receive.
    names = (*timeline_names, "submission_received_at_ms")
    _require_exact_keys(value, set(names), error)
    for name in names:
        _require_time_ms(value[name], name, error)
    sequence = [value[name] for name in timeline_names]
    if sequence != sorted(sequence) or sequence[-2] >= sequence[-1]:
        raise error("review timestamps are not strictly valid")
    # submission_received_at_ms is challenge-domain ZIP/send receive; it may
    # pretangle assignment issue (submit-first path) or follow it (self-deploy
    # then admit). Only require it inside the broad time range; order for age
    # is adjudicated at freshness admission against bound preimage values.
    submission_received = value["submission_received_at_ms"]
    if not isinstance(submission_received, int) or isinstance(submission_received, bool):
        raise error("submission_received_at_ms is invalid")


def _review_resource_limits(settings: object | None = None) -> dict[str, int]:
    """Resolve review-report resource caps from ChallengeSettings when present."""

    defaults = {
        "quote_bytes": MAX_REVIEW_QUOTE_BYTES,
        "event_log_bytes": MAX_REVIEW_EVENT_LOG_BYTES,
        "event_log_entries": MAX_REVIEW_EVENT_LOG_ENTRIES,
        "vm_config_bytes": MAX_REVIEW_VM_CONFIG_BYTES,
        "reason_evidence_items": MAX_REVIEW_DECISION_ENTRIES,
    }
    if settings is None:
        return defaults
    return {
        "quote_bytes": int(getattr(settings, "review_max_quote_bytes", defaults["quote_bytes"])),
        "event_log_bytes": int(
            getattr(settings, "review_max_event_log_bytes", defaults["event_log_bytes"])
        ),
        "event_log_entries": int(
            getattr(settings, "review_max_event_log_entries", defaults["event_log_entries"])
        ),
        "vm_config_bytes": int(
            getattr(settings, "review_max_vm_config_bytes", defaults["vm_config_bytes"])
        ),
        "reason_evidence_items": int(
            getattr(settings, "review_max_reason_evidence_items", defaults["reason_evidence_items"])
        ),
    }


def _validate_attestation(
    value: object,
    error: type[ValueError],
    *,
    settings: object | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise error("review attestation must be an object")
    limits = _review_resource_limits(settings)
    _require_exact_keys(value, {"tdx_quote_hex", "event_log", "measurement", "vm_config"}, error)
    _require_lower_hex(
        value["tdx_quote_hex"],
        "tdx_quote_hex",
        error,
        minimum_bytes=1,
        maximum_bytes=limits["quote_bytes"],
    )
    _validate_event_log(
        value["event_log"],
        error,
        max_entries=limits["event_log_entries"],
        max_bytes=limits["event_log_bytes"],
    )
    _validate_measurement(value["measurement"], error)
    _validate_vm_config(
        value["vm_config"],
        error,
        max_bytes=limits["vm_config_bytes"],
    )


def _validate_event_log(
    value: object,
    error: type[ValueError],
    *,
    max_entries: int = MAX_REVIEW_EVENT_LOG_ENTRIES,
    max_bytes: int = MAX_REVIEW_EVENT_LOG_BYTES,
) -> None:
    from agent_challenge.keyrelease.quote import (
        COMPOSE_HASH_EVENT,
        KEY_PROVIDER_EVENT,
        QuoteStructureError,
        QuoteVerificationError,
        runtime_event_digest,
        validate_rtmr3_event_log,
    )

    encoded = canonical_json_v1(value) if isinstance(value, (list, Mapping)) else b""
    if len(encoded) > max_bytes:
        raise error("review event log exceeds configured byte bound")
    from agent_challenge.keyrelease.quote import DSTACK_RUNTIME_EVENT_TYPE

    try:
        validated = validate_rtmr3_event_log(value, max_entries=max_entries)
    except (QuoteStructureError, QuoteVerificationError, ValueError, TypeError) as exc:
        raise error("review event log is invalid") from exc
    # Only dstack runtime events recompute digest(event∥payload). Firmware /
    # early IMR0-2 entries carry platform digests that are not recomputed here.
    for entry in validated:
        if entry.get("event_type") != DSTACK_RUNTIME_EVENT_TYPE:
            continue
        payload = bytes.fromhex(entry["event_payload"])
        if runtime_event_digest(entry["event"], payload).hex() != entry["digest"]:
            raise error("review event digest mismatches payload")
    # Identity events are guaranteed by validate_rtmr3_event_log; normalize the
    # key-provider payload (plain id or live dstack KMS JSON) before id checks.
    provider_entry = next(item for item in validated if item["event"] == KEY_PROVIDER_EVENT)
    try:
        provider = _decode_key_provider(provider_entry["event_payload"])
    except ReviewReportError as exc:
        raise error(str(exc)) from exc
    _require_id(provider, "key_provider", error)
    compose_entry = next(item for item in validated if item["event"] == COMPOSE_HASH_EVENT)
    _require_lower_hex(
        compose_entry["event_payload"],
        "compose hash event payload",
        error,
        exact_bytes=32,
    )


def _validate_measurement(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, Mapping):
        raise error("review measurement must be an object")
    _require_exact_keys(
        value,
        {
            "mrtd",
            "rtmr0",
            "rtmr1",
            "rtmr2",
            "rtmr3",
            "compose_hash",
            "os_image_hash",
            "key_provider",
            "vm_shape",
        },
        error,
    )
    for field in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"):
        _require_lower_hex(value[field], field, error, exact_bytes=48)
    for field in ("compose_hash", "os_image_hash"):
        _require_lower_hex(value[field], field, error, exact_bytes=32)
    _require_id(value["key_provider"], "key_provider", error)
    _require_id(value["vm_shape"], "vm_shape", error)


def _validate_vm_config(
    value: object,
    error: type[ValueError],
    *,
    max_bytes: int = MAX_REVIEW_VM_CONFIG_BYTES,
) -> None:
    if not isinstance(value, Mapping):
        raise error("review vm_config must be an object")
    encoded = canonical_json_v1(value)
    if len(encoded) > max_bytes:
        raise error("review vm_config exceeds configured byte bound")
    _require_exact_keys(value, {"vcpu", "memory_mb", "os_image_hash"}, error)
    _require_positive_int(value["vcpu"], "vcpu", error)
    _require_positive_int(value["memory_mb"], "memory_mb", error)
    if value["os_image_hash"] is not None:
        _require_sha256(value["os_image_hash"], "vm_config os_image_hash", error)


def _validate_core_assignment_binding(review_core: object, assignment: Mapping[str, Any]) -> None:
    if not isinstance(review_core, Mapping):
        raise ReviewReportError("review core is invalid")
    assignment_core = assignment["assignment_core"]
    if not isinstance(assignment_core, Mapping):
        raise ReviewReportError("assignment core is invalid")
    expected = {
        "session_id": assignment_core["session_id"],
        "assignment_id": assignment_core["assignment_id"],
        "assignment_digest": assignment["assignment_digest"],
        "submission_id": assignment_core["submission_id"],
        "review_nonce": assignment_core["review_nonce"],
    }
    if any(review_core[name] != item for name, item in expected.items()):
        raise ReviewReportError("review core identity does not match assignment")
    times = review_core["times"]
    if not isinstance(times, Mapping) or (
        times["issued_at_ms"] != assignment_core["issued_at_ms"]
        or times["expires_at_ms"] != assignment_core["expires_at_ms"]
    ):
        raise ReviewReportError("review timeline does not match assignment")
    for group, fields in (
        (
            "artifact_observation",
            (
                "agent_hash",
                "zip_sha256",
                "zip_size_bytes",
                "manifest_sha256",
                "manifest_entries_sha256",
            ),
        ),
        ("rules_observation", ("snapshot_sha256", "revision_id")),
        (
            "policy_observation",
            (
                "model",
                "routing_sha256",
                "prompt_version",
                "prompt_sha256",
                "tool_schema_version",
                "tool_schema_sha256",
                "verifier_version",
                "verifier_sha256",
            ),
        ),
    ):
        observed = review_core[group]
        assignment_group = {
            "artifact_observation": "artifact",
            "rules_observation": "rules",
            "policy_observation": "policy",
        }[group]
        expected_group = assignment_core[assignment_group]
        if not isinstance(observed, Mapping) or any(
            observed[name] != expected_group[name] for name in fields
        ):
            raise ReviewReportError(f"{group} does not match assignment")
    transport = review_core["openrouter_observation"]
    if not isinstance(transport, Mapping):
        raise ReviewReportError("OpenRouter observation is invalid")
    from .schemas import is_pinned_review_model

    # Requested pin is exact; returned may be the pin or OpenRouter's dated
    # canonical snapshot for the same pin (YYYYMMDD suffix).
    if assignment_core["policy"]["model"] != REVIEW_MODEL or not is_pinned_review_model(
        transport["returned_model"]
    ):
        raise ReviewReportError("returned model does not match assignment")
    marker_json = assignment.get("model_call_started_json")
    if not isinstance(marker_json, str):
        raise ReviewReportError("review report has no durable model call marker")
    marker = parse_json_object(marker_json)
    if (
        marker["planned_request_sha256"] != transport["planned_request_sha256"]
        or marker["request_body_sha256"] != transport["request_body_sha256"]
        or marker["request_body_length"] != transport["request_body_length"]
    ):
        raise ReviewReportError("review report transport does not match durable marker")


def _validate_receipt_time(assignment: Mapping[str, Any], received_at_ms: int) -> None:
    core = assignment["assignment_core"]
    if not isinstance(received_at_ms, int) or received_at_ms < 0:
        raise ReviewReportError("receipt time is invalid")
    if received_at_ms < core["issued_at_ms"]:
        raise ReviewReportError("receipt precedes assignment issue")


def _validate_timeline_against_receipt(review_core: object, received_at_ms: int) -> None:
    """Reject future-dated or post-receipt report timelines at verification time.

    Schema validation already enforces internal ordering and the assignment-bound
    `issued_at_ms`/`expires_at_ms`.  Receipt boundary is a trust decision that
    only exists once the durable receipt moment is known.
    """

    if not isinstance(review_core, Mapping):
        raise ReviewReportError("review core is invalid")
    times = review_core.get("times")
    if not isinstance(times, Mapping):
        raise ReviewReportError("review times are invalid")
    report_finished_at_ms = times.get("report_finished_at_ms")
    if not isinstance(report_finished_at_ms, int):
        raise ReviewReportError("report finished time is invalid")
    if report_finished_at_ms > received_at_ms:
        raise ReviewReportError("report timeline is future or post-receipt")
    # Every other core timestamp is ordered leafward of report_finished_at_ms by
    # `_validate_times`, so binding the final leaf to the receipt is sufficient.


def _validate_attestation_assignment_binding(
    attestation: object, assignment: Mapping[str, Any]
) -> None:
    if not isinstance(attestation, Mapping):
        raise ReviewReportError("review attestation is invalid")
    measurement = attestation["measurement"]
    if not isinstance(measurement, Mapping):
        raise ReviewReportError("review measurement is invalid")
    review_app = assignment["assignment_core"]["review_app"]
    expected_static = review_app["measurement"]
    if not isinstance(expected_static, Mapping):
        raise ReviewReportError("assignment review measurement is invalid")
    expected = {
        **dict(expected_static),
        "compose_hash": review_app["compose_hash"],
    }
    fields = (
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "compose_hash",
        "os_image_hash",
        "key_provider",
        "vm_shape",
    )
    if any(measurement[field] != expected[field] for field in fields):
        raise ReviewReportError("quote measurement does not match assignment")


def _validate_quote_envelope_binding(
    *,
    attestation: object,
    expected_report_data_hex: str,
    error: type[ValueError],
) -> None:
    from agent_challenge.keyrelease.quote import (
        QuoteStructureError,
        QuoteVerificationError,
        os_image_hash_from_registers,
        parse_tdx_quote_v4,
        replay_rtmr3,
        validate_rtmr3_event_log,
    )

    assert isinstance(attestation, Mapping)
    try:
        report = parse_tdx_quote_v4(str(attestation["tdx_quote_hex"]))
        if report.report_data.hex() != expected_report_data_hex:
            raise error("quote report data does not match envelope")
        event_log = attestation["event_log"]
        assert isinstance(event_log, list)
        validated_log = validate_rtmr3_event_log(event_log)
        replay = replay_rtmr3(validated_log)
        if replay.rtmr3 != report.rtmr3 or replay.compose_hash is None:
            raise error("quote event log does not reproduce signed RTMR3")
        measurement = attestation["measurement"]
        assert isinstance(measurement, Mapping)
        signed_measurement = {
            "mrtd": report.mrtd,
            "rtmr0": report.rtmr0,
            "rtmr1": report.rtmr1,
            "rtmr2": report.rtmr2,
            "rtmr3": report.rtmr3,
            "compose_hash": replay.compose_hash,
            "os_image_hash": os_image_hash_from_registers(
                report.mrtd,
                report.rtmr1,
                report.rtmr2,
            ),
            "key_provider": _decode_key_provider(replay.key_provider),
            "vm_shape": measurement["vm_shape"],
        }
        if dict(measurement) != signed_measurement:
            raise error("quote measurement does not match envelope")
        vm_config = attestation["vm_config"]
        assert isinstance(vm_config, Mapping)
        if vm_config["os_image_hash"] not in {None, measurement["os_image_hash"]}:
            raise error("vm config os image hash does not match measurement")
    except (QuoteStructureError, QuoteVerificationError, UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, ReviewReportError):
            raise
        raise error("quote envelope binding is invalid") from exc


def _decode_key_provider(value: str | None) -> str:
    """Normalize the RTMR3 key-provider payload into the measurement id.

    Offline fixtures and simple providers emit a short UTF-8 identifier (``phala``).
    Live dstack KMS emits a JSON object such as ``{"name":"kms","id":"..."}``.
    Collapse the latter onto the stable sealed review id ``phala`` so the report
    measurement remains a bounded ASCII identifier.
    """

    if not isinstance(value, str):
        raise ReviewReportError("key provider event is missing")
    try:
        decoded = bytes.fromhex(value).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReviewReportError("key provider event is invalid") from exc
    text = decoded.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReviewReportError("key provider event is invalid") from exc
        if not isinstance(payload, Mapping):
            raise ReviewReportError("key provider event is invalid")
        name = str(payload.get("name") or payload.get("type") or "").strip().lower()
        # Live dstack KMS payload is JSON; map the KMS family onto the sealed
        # review measurement id "phala" used by assignment pins.
        if name in {"kms", "phala", "phala-kms"}:
            text = "phala"
        elif name:
            text = name
        else:
            raise ReviewReportError("key provider event is invalid")
    _require_id(text, "key_provider", ReviewReportError)
    return text


def _canonical_allowlist_entry(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ReviewReportError("review allowlist entry must be an object")
    expected = {"mrtd", "rtmr0", "rtmr1", "rtmr2", "compose_hash", "os_image_hash"}
    _require_exact_keys(value, expected, ReviewReportError)
    normalized = {name: str(value[name]) for name in expected}
    for name in ("mrtd", "rtmr0", "rtmr1", "rtmr2"):
        _require_lower_hex(normalized[name], name, ReviewReportError, exact_bytes=48)
    for name in ("compose_hash", "os_image_hash"):
        _require_lower_hex(normalized[name], name, ReviewReportError, exact_bytes=32)
    return {name: normalized[name] for name in sorted(expected)}


def _measurement_for_allowlist(value: Mapping[str, Any]) -> dict[str, str]:
    expected = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "compose_hash", "os_image_hash")
    if not isinstance(value, Mapping) or any(name not in value for name in expected):
        raise ReviewReportError("review measurement is incomplete")
    return _canonical_allowlist_entry({name: value[name] for name in expected})


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], error: type[ValueError]
) -> None:
    if set(value) != expected:
        raise error("schema keys differ")


def _require_lower_hex(
    value: object,
    name: str,
    error: type[ValueError],
    *,
    exact_bytes: int | None = None,
    minimum_bytes: int | None = None,
    maximum_bytes: int | None = None,
) -> None:
    if not isinstance(value, str) or not _HEX_RE.fullmatch(value) or len(value) % 2:
        raise error(f"{name} must be lowercase even-length hex")
    size = len(value) // 2
    if exact_bytes is not None and size != exact_bytes:
        raise error(f"{name} has invalid fixed width")
    if minimum_bytes is not None and size < minimum_bytes:
        raise error(f"{name} is empty")
    if maximum_bytes is not None and size > maximum_bytes:
        raise error(f"{name} exceeds configured byte bound")


def _require_report_data(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, str) or len(value) != REPORT_DATA_HEX_LENGTH:
        raise error("report_data_hex must be 64-byte lowercase hex")
    _require_lower_hex(value, "report_data_hex", error, exact_bytes=64)


def _require_sorted_set(
    value: object,
    name: str,
    validator: Any,
    error: type[ValueError],
    *,
    maximum_items: int | None = None,
) -> None:
    if not isinstance(value, list) or value != sorted(value) or len(set(value)) != len(value):
        raise error(f"{name} must be sorted and unique")
    if maximum_items is not None and len(value) > maximum_items:
        raise error(f"{name} exceeds aggregate 256-entry bound")
    for item in value:
        validator(item, error)


def _require_reason_code(value: object, error: type[ValueError]) -> None:
    if not isinstance(value, str) or not _REASON_CODE_RE.fullmatch(value):
        raise error("reason code is invalid")


def _require_digest_item(value: object, error: type[ValueError]) -> None:
    _require_sha256(value, "evidence digest", error)


def _canonical_bytes(value: Mapping[str, Any], error: type[ValueError]) -> bytes:
    try:
        return canonical_json_v1(dict(value))
    except CanonicalJsonError as exc:
        raise error(str(exc)) from exc


def _verified(
    status: Literal["verified_allow", "verified_reject", "verified_escalate"], now_ms: int
) -> ReviewVerificationOutcome:
    return ReviewVerificationOutcome(
        status=status,
        terminal=True,
        retryable=False,
        reason_code="review_verified",
        nonce_consumed=True,
        measurement_allowlisted=True,
        report_data_matched=True,
        verified_at_ms=now_ms,
    )


def _trust_failed(
    reason_code: str, *, report_data_matched: bool = True
) -> ReviewVerificationOutcome:
    return ReviewVerificationOutcome(
        status="trust_failed",
        terminal=True,
        retryable=False,
        reason_code=reason_code,
        nonce_consumed=True,
        measurement_allowlisted=False,
        report_data_matched=report_data_matched,
        verified_at_ms=None,
    )


def _verifier_unavailable() -> ReviewVerificationOutcome:
    return ReviewVerificationOutcome(
        status="verifier_unavailable",
        terminal=False,
        retryable=True,
        reason_code="review_verifier_unavailable",
        nonce_consumed=False,
        measurement_allowlisted=False,
        report_data_matched=False,
        verified_at_ms=None,
    )


def _outcome_from_json(value: str) -> ReviewVerificationOutcome:
    try:
        parsed = parse_json_object(value)
        return ReviewVerificationOutcome(
            status=parsed["status"],
            terminal=parsed["terminal"],
            retryable=parsed["retryable"],
            reason_code=parsed["reason_code"],
            nonce_consumed=parsed["nonce_consumed"],
            measurement_allowlisted=parsed["measurement_allowlisted"],
            report_data_matched=parsed["report_data_matched"],
            verified_at_ms=parsed["verified_at_ms"],
        )
    except (CanonicalJsonError, KeyError, TypeError) as exc:  # pragma: no cover
        raise ReviewReportConflict("stored review outcome is corrupt") from exc


def _as_utc(value: datetime) -> datetime:
    from datetime import UTC

    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _to_ms(value: datetime) -> int:
    return int(_as_utc(value).timestamp() * 1000)


__all__ = [
    "REVIEW_REPORT_DOMAIN",
    "REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_VERSION",
    "REVIEW_REPORT_SCHEMA_VERSION",
    "DcapReviewQuoteVerifier",
    "ReviewMeasurementAllowlist",
    "ReviewReportConflict",
    "ReviewReportError",
    "ReviewVerificationOutcome",
    "ReviewVerifierUnavailable",
    "build_review_envelope",
    "extract_bound_attestation_times",
    "reverify_bound_attestation_times",
    "review_digest",
    "review_report_data_hex",
    "review_report_data_preimage",
    "submit_review_report",
    "validate_review_core",
    "validate_review_envelope",
    "verify_review_envelope",
]
