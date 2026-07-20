"""Durable key-release registration, receipt, grant, and nonce disposition.

Encodes VAL-KEY-004/005:

* Registration requires exactly one matching outstanding key-release nonce from
  the immutable Eval plan.
* Missing, mismatched, consumed, revoked, or expired nonces cannot register.
* The exact schema-valid request is receipted before expensive verification.
* Transient and unexpected verifier failures retain a retryable receipt without
  consuming the nonce.
* Definitive invalid/valid outcomes terminalize once; valid grant and nonce
  consumption commit atomically.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from agent_challenge.core.models import (
    AgentSubmission,
    EvalNonce,
    EvalRun,
    ReviewAssignment,
    ReviewSession,
)
from agent_challenge.evaluation.authorization import (
    EvalAuthorizationConflict,
    create_eval_run,
    mark_eval_key_granted,
    mark_eval_key_release_denied,
    mark_eval_key_release_retryable,
    receipt_eval_key_release,
    register_eval_key_release,
)
from agent_challenge.keyrelease.allowlist import CanonicalEntry, MeasurementAllowlist
from agent_challenge.keyrelease.client import key_release_report_data
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    QuoteVerdict,
    QuoteVerificationError,
    QuoteVerifierUnavailable,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
)
from agent_challenge.keyrelease.server import (
    REASON_GOLDEN_KEY_UNAVAILABLE,
    REASON_MEASUREMENT_NOT_ALLOWLISTED,
    REASON_VERIFIER_UNAVAILABLE,
    KeyReleaseService,
    build_frame,
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
    review_report_data_hex,
    sha256_hex,
)
from agent_challenge.review.or_outcome_bind import (
    review_digest as bound_review_digest,
)
from agent_challenge.sdk.config import ChallengeSettings

MRTD = "11" * 48
RTMR0 = "22" * 48
RTMR1 = "33" * 48
RTMR2 = "44" * 48
COMPOSE_HASH = "ab" * 32
KEY_PROVIDER_PAYLOAD = b'{"name":"kms","id":"kms-1"}'
GOLDEN_KEY = bytes(range(32))
MEASUREMENT = {
    "mrtd": MRTD,
    "rtmr0": RTMR0,
    "rtmr1": RTMR1,
    "rtmr2": RTMR2,
    "os_image_hash": os_image_hash_from_registers(MRTD, RTMR1, RTMR2),
    "key_provider": "validator-kms",
    "vm_shape": "tdx-small",
}


def _settings() -> ChallengeSettings:
    return ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        eval_app_image_ref="registry.example/eval@sha256:" + "a" * 64,
        eval_app_compose_hash=COMPOSE_HASH,
        eval_app_identity="agent-challenge-eval-v1",
        eval_app_kms_public_key_hex="07" * 32,
        eval_app_measurement=MEASUREMENT,
        eval_app_measurement_allowlist=(
            {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "compose_hash": COMPOSE_HASH,
                "os_image_hash": MEASUREMENT["os_image_hash"],
            },
        ),
        eval_key_release_endpoint="validator.example:8701",
        eval_k=1,
        evaluation_task_count=1,
    )


def _patch_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "aa" * 32},
                },
            )()
        ],
    )


_SUBMISSION_SEQ = 0
_T0 = 1_700_000_000_000
_ROUTE = sha256_hex(b'{"order":["dur-key"]}')
_BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
_BODY_SHA = sha256_hex(_BODY)
_RESP = b'{"id":"gen-dur","model":"x-ai/grok-4.5","choices":[]}'
_RESP_SHA = sha256_hex(_RESP)
_META = sha256_hex(b"meta-dur")


def _fresh_review_envelope(*, seq: int) -> tuple[str, str, str]:
    """Receipted allow envelope so create_eval_run re-verifies (not cache-only)."""

    planned = build_planned_openrouter_request(
        body_sha256=_BODY_SHA,
        body_length=len(_BODY),
        routing_sha256=_ROUTE,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_observed_openrouter_transport(
        planned_request_sha256_=p_digest,
        response_body_sha256=_RESP_SHA,
        response_body_length=len(_RESP),
        metadata_sha256=_META,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=_BODY_SHA,
        request_body_length=len(_BODY),
        response_id=f"gen-dur-{seq}",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-dur",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-dur",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-dur",
        routing_sha256=_ROUTE,
    )
    sid = f"rs-dur-{seq}"
    aid = f"ra-dur-{seq}"
    core = build_review_core_minimal(
        session_id=sid,
        assignment_id=aid,
        submission_id=f"sub-dur-{seq}",
        review_nonce=f"nonce-dur-{seq}",
        assignment_digest="13" * 32,
        rules_observation={
            "rules_version": "rules-v1",
            "rules_bundle_sha256": "11" * 32,
            "rules_files": [".rules/acceptance.md"],
            "rules_file_digests": {".rules/acceptance.md": "22" * 32},
            "rules_policy_text_sha256": "33" * 32,
        },
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict="allow"),
        times={
            "issued_at_ms": _T0,
            "started_at_ms": _T0,
            "model_call_marked_at_ms": _T0 + 1,
            "request_started_at_ms": _T0 + 2,
            "request_finished_at_ms": _T0 + 3,
            "verifier_finished_at_ms": _T0 + 4,
            "report_finished_at_ms": _T0 + 5,
            "expires_at_ms": _T0 + 3_600_000,
            "submission_received_at_ms": _T0 + 60_000,
        },
    )
    digest = bound_review_digest(core)
    rd = review_report_data_hex(core)
    env = {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": digest,
        "report_data_hex": rd,
        "review_core": core,
    }
    return json.dumps(env, sort_keys=True, separators=(",", ":")), rd, digest


async def _authorized_submission(database_session) -> int:
    global _SUBMISSION_SEQ
    _SUBMISSION_SEQ += 1
    salt = f"dur-key-{_SUBMISSION_SEQ}".encode()
    envelope_json, report_data_hex, digest = _fresh_review_envelope(seq=_SUBMISSION_SEQ)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"dur-key-miner-{_SUBMISSION_SEQ}",
            name=f"dur-key-agent-{_SUBMISSION_SEQ}",
            agent_hash=hashlib.sha256(b"agent-" + salt).hexdigest(),
            artifact_uri=f"/tmp/agent-{_SUBMISSION_SEQ}.zip",
            artifact_path=f"/tmp/agent-{_SUBMISSION_SEQ}.zip",
            zip_sha256=hashlib.sha256(b"zip-" + salt).hexdigest(),
            zip_size_bytes=3,
            raw_status="review_allowed",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id=f"review-dur-{submission.id}",
            submission_id=submission.id,
            artifact_sha256=submission.zip_sha256,
            artifact_size_bytes=3,
            manifest_sha256="11" * 32,
            manifest_entries_sha256="12" * 32,
            authorizing_assignment_id=f"assign-dur-{submission.id}",
            current_assignment_id=f"assign-dur-{submission.id}",
        )
        session.add(review_session)
        await session.flush()
        assignment = ReviewAssignment(
            session_id=review_session.id,
            assignment_id=f"assign-dur-{submission.id}",
            attempt=1,
            assignment_bytes="{}",
            assignment_digest="13" * 32,
            artifact_sha256=submission.zip_sha256,
            rules_snapshot_sha256="14" * 32,
            rules_revision_id="rules-1",
            review_nonce=f"review-nonce-{submission.id}",
            session_token_sha256="15" * 32,
            capability_state="revoked",
            phase="review_allowed",
            issued_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
            # Full receipted envelope: create_eval_run re-verifies (VAL-ACAT-028/029).
            review_report_envelope_json=envelope_json,
            review_report_data_hex=report_data_hex,
            review_digest=digest,
            review_verification_outcome_json=(
                '{"status":"verified_allow","terminal":true,"retryable":false,'
                '"nonce_consumed":true}'
            ),
        )
        session.add(assignment)
        await session.commit()
        return submission.id


async def _create_run(database_session, monkeypatch: pytest.MonkeyPatch):
    submission_id = await _authorized_submission(database_session)
    _patch_tasks(monkeypatch)
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        created = await create_eval_run(session, submission, settings=_settings())
        await session.commit()
        return created.run.eval_run_id, created.plan, submission_id


async def _key_nonce(session, eval_run_id: str) -> EvalNonce:
    run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
    assert run is not None
    nonce = await session.scalar(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "key_release",
        )
    )
    assert nonce is not None
    return nonce


def _event_log() -> tuple[list[dict[str, object]], str]:
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex(COMPOSE_HASH)),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
        ]
    )


def _canonical_entry() -> CanonicalEntry:
    return CanonicalEntry(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        compose_hash=COMPOSE_HASH,
        os_image_hash=os_image_hash_from_registers(MRTD, RTMR1, RTMR2),
        # Live/KMS JSON payloads decode to the stable pin "phala".
        key_provider="phala",
    )


def _ratls_cert() -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=hashlib.sha512(b"ratls-cert:" + public_key).digest(),
    )

    def octet_string(value: bytes) -> bytes:
        length = len(value)
        if length < 128:
            return b"\x04" + bytes([length]) + value
        width = (length.bit_length() + 7) // 8
        return b"\x04" + bytes([0x80 | width]) + length.to_bytes(width, "big") + value

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ratls-client")])
    event_bytes = __import__("json").dumps(event_log, separators=(",", ":")).encode()
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(minutes=5))
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=True,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier("1.3.6.1.4.1.62397.1.1"),
                octet_string(bytes.fromhex(quote)),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier("1.3.6.1.4.1.62397.1.2"),
                octet_string(event_bytes),
            ),
            critical=False,
        )
    )
    return builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


def _frame_for(
    *,
    eval_run_id: str,
    nonce: str,
    certificate: bytes,
    bad_measurement: bool = False,
) -> bytes:
    from agent_challenge.keyrelease.server import spki_sha256_from_certificate

    spki = spki_sha256_from_certificate(certificate)
    event_log, rtmr3 = _event_log()
    report_data = key_release_report_data(
        "",
        b"",
        eval_run_id=eval_run_id,
        key_release_nonce=nonce,
        ra_tls_spki_digest=spki,
    )
    quote = build_tdx_quote(
        mrtd="ff" * 48 if bad_measurement else MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )
    request = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "nonce": nonce,
        "quote_hex": quote,
        "event_log": event_log,
    }
    return build_frame(request)[4:]


# --------------------------------------------------------------------------- #
# VAL-KEY-004 -- purpose-typed nonce registration from the immutable plan
# --------------------------------------------------------------------------- #
async def test_register_requires_exactly_one_matching_outstanding_key_nonce(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
    async with database_session() as session:
        run = await register_eval_key_release(session, eval_run_id=eval_run_id)
        assert run.eval_run_id == eval_run_id
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.nonce == plan["key_release_nonce"]
        assert nonce.state == "outstanding"
        assert nonce.purpose == "key_release"
        # The score nonce is never accepted as a key-release outstanding match.
        score = await session.scalar(
            select(EvalNonce).where(
                EvalNonce.purpose == "score",
                EvalNonce.nonce == plan["score_nonce"],
            )
        )
        assert score is not None
        assert score.state == "outstanding"
        assert plan["key_release_nonce"] != plan["score_nonce"]


@pytest.mark.parametrize(
    "mutate",
    ("missing", "mismatched", "consumed", "revoked", "expired"),
)
async def test_register_rejects_non_outstanding_key_nonces(
    database_session,
    monkeypatch,
    mutate: str,
) -> None:
    eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        nonce = await _key_nonce(session, eval_run_id)
        if mutate == "missing":
            await session.delete(nonce)
        elif mutate == "mismatched":
            nonce.nonce = "attacker-chosen-nonce-value"
        elif mutate == "consumed":
            nonce.state = "consumed"
            nonce.consumed_at = datetime.now(UTC)
        elif mutate == "revoked":
            nonce.state = "revoked"
            nonce.consumed_at = datetime.now(UTC)
        else:
            nonce.state = "expired"
            nonce.consumed_at = datetime.now(UTC)
        await session.commit()

    async with database_session() as session:
        with pytest.raises(EvalAuthorizationConflict) as exc:
            await register_eval_key_release(session, eval_run_id=eval_run_id)
        assert "outstanding" in str(exc.value).lower() or "not eligible" in str(exc.value).lower()
        # Plan bytes still hold the original reserved nonce (spent/miss values cannot grant).
        reloaded = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert reloaded is not None
        assert reloaded.key_release_state is None
        assert reloaded.key_release_receipt_sha256 is None


# --------------------------------------------------------------------------- #
# VAL-KEY-005 -- receipt digest, retryable, terminal consume, atomic grant
# --------------------------------------------------------------------------- #
async def test_first_schema_valid_request_is_receipted_before_verification(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, _plan, _ = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b'{"release":"exact-bytes"}').hexdigest()
    async with database_session() as session:
        run, should_verify = await receipt_eval_key_release(
            session,
            eval_run_id=eval_run_id,
            body_sha256=digest,
        )
        assert should_verify is True
        assert run.key_release_receipt_sha256 == digest
        assert run.key_release_state == "verifying"
        assert run.key_release_receipt_received_at is not None
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "outstanding"
        await session.commit()

    async with database_session() as session:
        # Identical in-flight / concurrent ingest cannot start a second verification.
        run, should_verify = await receipt_eval_key_release(
            session,
            eval_run_id=eval_run_id,
            body_sha256=digest,
        )
        assert should_verify is False
        assert run.key_release_state == "verifying"
        # Conflicting bytes never replace the durable receipt.
        with pytest.raises(EvalAuthorizationConflict, match="conflict"):
            await receipt_eval_key_release(
                session,
                eval_run_id=eval_run_id,
                body_sha256=hashlib.sha256(b'{"release":"other"}').hexdigest(),
            )


async def test_verifier_unavailable_and_unexpected_keep_retryable_unconsumed(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, _plan, _ = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b"retryable-payload").hexdigest()
    async with database_session() as session:
        await receipt_eval_key_release(session, eval_run_id=eval_run_id, body_sha256=digest)
        await mark_eval_key_release_retryable(
            session,
            eval_run_id=eval_run_id,
            body_sha256=digest,
            reason_code="verifier_unavailable",
        )
        await session.commit()

    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        assert run.key_release_state == "retryable"
        assert run.key_release_reason == "verifier_unavailable"
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "outstanding"
        # Same encrypted request may resume verification ownership.
        run, should_verify = await receipt_eval_key_release(
            session,
            eval_run_id=eval_run_id,
            body_sha256=digest,
        )
        assert should_verify is True
        assert run.key_release_state == "verifying"


async def test_definitive_deny_terminalizes_and_consumes_once(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, _plan, _ = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b"deny-payload").hexdigest()
    async with database_session() as session:
        await receipt_eval_key_release(session, eval_run_id=eval_run_id, body_sha256=digest)
        denied = await mark_eval_key_release_denied(
            session,
            eval_run_id=eval_run_id,
            body_sha256=digest,
            reason_code=REASON_MEASUREMENT_NOT_ALLOWLISTED,
        )
        assert denied.key_release_state == "denied"
        assert denied.retryable is False
        assert denied.phase == "eval_error"
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "consumed"
        await session.commit()

    async with database_session() as session:
        with pytest.raises(EvalAuthorizationConflict):
            await mark_eval_key_release_denied(
                session,
                eval_run_id=eval_run_id,
                body_sha256=digest,
                reason_code=REASON_MEASUREMENT_NOT_ALLOWLISTED,
            )
        with pytest.raises(EvalAuthorizationConflict):
            await mark_eval_key_granted(session, eval_run_id=eval_run_id)
        with pytest.raises(EvalAuthorizationConflict):
            await register_eval_key_release(session, eval_run_id=eval_run_id)


async def test_grant_consumes_nonce_atomically_and_blocks_replacement(
    database_session,
    monkeypatch,
) -> None:
    from agent_challenge.evaluation.authorization import cancel_eval_run, retry_eval_run

    eval_run_id, _plan, submission_id = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b"grant-payload").hexdigest()
    async with database_session() as session:
        await receipt_eval_key_release(session, eval_run_id=eval_run_id, body_sha256=digest)
        granted = await mark_eval_key_granted(session, eval_run_id=eval_run_id)
        assert granted.key_release_state == "granted"
        assert granted.key_granted_at is not None
        assert granted.retryable is False
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "consumed"
        # Second grant is idempotent: no double consume and no phase flip.
        again = await mark_eval_key_granted(session, eval_run_id=eval_run_id)
        assert again.key_granted_at == granted.key_granted_at
        await session.commit()

    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        with pytest.raises(EvalAuthorizationConflict):
            await cancel_eval_run(session, submission, eval_run_id)
        with pytest.raises(EvalAuthorizationConflict):
            await retry_eval_run(
                session,
                submission,
                expected_run_id=eval_run_id,
                settings=_settings(),
            )
        # Receipt of new bytes on a granted run is rejected.
        with pytest.raises(EvalAuthorizationConflict):
            await receipt_eval_key_release(
                session,
                eval_run_id=eval_run_id,
                body_sha256=hashlib.sha256(b"different").hexdigest(),
            )


# --------------------------------------------------------------------------- #
# End-to-end framed path with session-backed durable disposition.
# --------------------------------------------------------------------------- #
class _BoomVerifier:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def verify(self, quote_hex: str) -> QuoteVerdict:
        self.calls += 1
        if self.mode == "unavailable":
            raise QuoteVerifierUnavailable("dcap-qvl offline")
        if self.mode == "unexpected":
            raise RuntimeError("unexpected verifier crash")
        if self.mode == "invalid":
            raise QuoteVerificationError("bad signature")
        return QuoteVerdict(tcb_status="UpToDate")


async def test_framed_path_receipts_before_expensive_verification(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
    certificate = _ratls_cert()
    payload = _frame_for(
        eval_run_id=eval_run_id,
        nonce=plan["key_release_nonce"],
        certificate=certificate,
    )
    body_digest = hashlib.sha256(payload).hexdigest()

    class _CountingVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify(self, quote_hex: str) -> QuoteVerdict:
            self.calls += 1
            return QuoteVerdict(tcb_status="UpToDate")

    verifier = _CountingVerifier()
    service = KeyReleaseService(
        allowlist=MeasurementAllowlist([_canonical_entry()]),
        verifier=verifier,
        golden_key_loader=lambda: GOLDEN_KEY,
        session_context_factory=database_session,
    )
    key, reason, _detail = await service.authorize_framed_request(
        payload,
        peer_certificate_der=certificate,
    )
    assert reason is None
    assert key == GOLDEN_KEY
    # Certificate + payload quote each go through verify after receipt commit.
    assert verifier.calls >= 1

    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        assert run.key_release_receipt_sha256 == body_digest
        assert run.key_release_state == "granted"
        assert run.key_granted_at is not None
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "consumed"
        prior_consumed = nonce.consumed_at

    # Second identical framed request returns the same grant without re-consume.
    key2, reason2, _detail2 = await service.authorize_framed_request(
        payload,
        peer_certificate_der=certificate,
    )
    assert reason2 is None
    assert key2 == GOLDEN_KEY
    async with database_session() as session:
        nonce2 = await _key_nonce(session, eval_run_id)
        assert nonce2.state == "consumed"
        assert nonce2.consumed_at == prior_consumed


async def test_framed_transient_and_unexpected_keep_receipt_retryable(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
    certificate = _ratls_cert()
    payload = _frame_for(
        eval_run_id=eval_run_id,
        nonce=plan["key_release_nonce"],
        certificate=certificate,
    )
    digest = hashlib.sha256(payload).hexdigest()

    for mode in ("unavailable", "unexpected"):
        # Fresh run per mode.
        if mode == "unexpected":
            eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
            payload = _frame_for(
                eval_run_id=eval_run_id,
                nonce=plan["key_release_nonce"],
                certificate=certificate,
            )
            digest = hashlib.sha256(payload).hexdigest()
        service = KeyReleaseService(
            allowlist=MeasurementAllowlist([_canonical_entry()]),
            verifier=_BoomVerifier(mode=mode),
            golden_key_loader=lambda: GOLDEN_KEY,
            session_context_factory=database_session,
        )
        key, reason, _detail = await service.authorize_framed_request(
            payload,
            peer_certificate_der=certificate,
        )
        assert key is None
        assert reason == REASON_VERIFIER_UNAVAILABLE
        async with database_session() as session:
            run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
            assert run is not None
            assert run.key_release_receipt_sha256 == digest
            assert run.key_release_state == "retryable"
            nonce = await _key_nonce(session, eval_run_id)
            assert nonce.state == "outstanding"
            assert run.key_granted_at is None


async def test_framed_golden_key_unavailable_stays_retryable_unconsumed(
    database_session,
    monkeypatch,
) -> None:
    """Post-receipt golden-key loader outage must park, not terminal-deny.

    Discriminator against marking golden_key_unavailable through
    mark_eval_key_release_denied (which would consume the purpose-typed nonce).
    """

    eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
    certificate = _ratls_cert()
    payload = _frame_for(
        eval_run_id=eval_run_id,
        nonce=plan["key_release_nonce"],
        certificate=certificate,
    )
    digest = hashlib.sha256(payload).hexdigest()

    def _missing_key() -> bytes:
        raise OSError("golden key file missing")

    service = KeyReleaseService(
        allowlist=MeasurementAllowlist([_canonical_entry()]),
        verifier=StaticQuoteVerifier(tcb_status="UpToDate"),
        golden_key_loader=_missing_key,
        session_context_factory=database_session,
    )
    key, reason, _detail = await service.authorize_framed_request(
        payload,
        peer_certificate_der=certificate,
    )
    assert key is None
    # Wire vocabulary folds golden_key_unavailable into verifier_unavailable.
    assert reason == REASON_VERIFIER_UNAVAILABLE
    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        assert run.key_release_receipt_sha256 == digest
        assert run.key_release_state == "retryable"
        assert run.key_release_reason in {
            REASON_GOLDEN_KEY_UNAVAILABLE,
            REASON_VERIFIER_UNAVAILABLE,
        }
        assert run.key_granted_at is None
        assert run.key_release_completed_at is None
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "outstanding"
        assert nonce.consumed_at is None

    # Identical framed retry after park resumes verification (nonce still live).
    async with database_session() as session:
        run, should_verify = await receipt_eval_key_release(
            session,
            eval_run_id=eval_run_id,
            body_sha256=digest,
        )
        assert should_verify is True
        assert run.key_release_state == "verifying"
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "outstanding"


async def test_framed_definitive_deny_consumes_once_and_blocks_second_grant(
    database_session,
    monkeypatch,
) -> None:
    eval_run_id, plan, _ = await _create_run(database_session, monkeypatch)
    certificate = _ratls_cert()
    # Valid signature, wrong measurement → definitive deny after receipt.
    payload = _frame_for(
        eval_run_id=eval_run_id,
        nonce=plan["key_release_nonce"],
        certificate=certificate,
        bad_measurement=True,
    )
    digest = hashlib.sha256(payload).hexdigest()
    service = KeyReleaseService(
        allowlist=MeasurementAllowlist([_canonical_entry()]),
        verifier=StaticQuoteVerifier(tcb_status="UpToDate"),
        golden_key_loader=lambda: GOLDEN_KEY,
        session_context_factory=database_session,
    )
    key, reason, _detail = await service.authorize_framed_request(
        payload,
        peer_certificate_der=certificate,
    )
    assert key is None
    assert reason is not None
    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        assert run.key_release_receipt_sha256 == digest
        assert run.key_release_state == "denied"
        nonce = await _key_nonce(session, eval_run_id)
        assert nonce.state == "consumed"
    # Second present of the same or different payload cannot produce a grant.
    key2, reason2, _detail2 = await service.authorize_framed_request(
        payload,
        peer_certificate_der=certificate,
    )
    assert key2 is None
    assert reason2 is not None


async def test_authorize_release_unexpected_is_retryable_not_destinative() -> None:
    """Discriminator: a bare Exception from the verifier must not terminalize."""

    class _Unexpected:
        def verify(self, quote_hex: str) -> QuoteVerdict:
            raise RuntimeError("disk full mid-verify")

    service = KeyReleaseService(
        allowlist=MeasurementAllowlist([_canonical_entry()]),
        verifier=_Unexpected(),
        golden_key_loader=lambda: GOLDEN_KEY,
    )
    nonce = service.issue_nonce()
    event_log, rtmr3 = _event_log()
    report_data = key_release_report_data(nonce, b"peer-key")
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )
    outcome = service.authorize_release(
        nonce=nonce,
        quote_hex=quote,
        ra_tls_pubkey_hex=b"peer-key".hex(),
        event_log=event_log,
        session_peer_pubkey=b"peer-key",
    )
    assert outcome.released is False
    assert outcome.reason == REASON_VERIFIER_UNAVAILABLE


async def test_authorize_release_golden_unavailable_is_retryable() -> None:
    service = KeyReleaseService(
        allowlist=MeasurementAllowlist([_canonical_entry()]),
        verifier=StaticQuoteVerifier(tcb_status="UpToDate"),
        golden_key_loader=lambda: (_ for _ in ()).throw(OSError("key file missing")),
    )
    # Issue anonymous nonce path for the pure decision core.
    nonce = service.issue_nonce()
    event_log, rtmr3 = _event_log()
    report_data = key_release_report_data(nonce, b"peer-key")
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )
    outcome = service.authorize_release(
        nonce=nonce,
        quote_hex=quote,
        ra_tls_pubkey_hex=b"peer-key".hex(),
        event_log=event_log,
        session_peer_pubkey=b"peer-key",
    )
    assert outcome.released is False
    assert outcome.reason in {REASON_GOLDEN_KEY_UNAVAILABLE, REASON_VERIFIER_UNAVAILABLE}
