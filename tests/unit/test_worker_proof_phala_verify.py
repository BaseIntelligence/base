"""Phala-tier quote verifier for ExecutionProof (M4, VAL-VERIFY-001..014).

Black-box behavioral tests for the validator/master Phala-tier verifier added to
``verify_execution_proof``: it accepts a wholly-valid attested envelope and
rejects every tamper / mismatch / stale / repurposed case, keeps the tier-0
worker-signature + cross-unit-replay checks in force on the Phala tier, and
PARKS (raises, does not accept nor fraud-reject) when the quote-verification
dependency is transiently unavailable.

Quotes are assembled offline with the base ``phala_quote`` helpers (the inverse
of the parser); the DCAP signature/TCB layer is modeled with an injectable
:class:`StaticQuoteVerifier`, and a fake-runner :class:`DcapQvlVerifier` test
pins the CLI accept/reject/park mapping. A real ``dcap-qvl`` run is an M6 live
assertion.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from base.schemas.worker import (
    PHALA_TDX_TIER,
    ExecutionProof,
    PhalaAttestation,
    PhalaMeasurement,
    WorkerSignature,
)
from base.validator.agent.adapters.agent_challenge import rebind_worker_signature
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.phala_quote import (
    DcapQvlVerifier,
    QuoteVerificationError,
    StaticQuoteVerifier,
    VerifierUnavailableError,
    build_rtmr3_event_log,
    build_tdx_quote,
)
from base.worker.phala_verify import (
    InMemoryNonceValidator,
    MeasurementAllowlist,
    PhalaBinding,
)
from base.worker.proof import (
    PHALA_REPORT_DATA_TAG,
    build_phala_execution_proof,
    phala_report_data_hex,
    verify_execution_proof,
)

MANIFEST = "a" * 64
UNIT_ID = "submission-verify-1"

MRTD = "a1" * 48
RTMR0 = "b0" * 48
RTMR1 = "b1" * 48
RTMR2 = "b2" * 48
COMPOSE_PAYLOAD = bytes.fromhex("c3" * 32)

AGENT_HASH = "f0" * 32
TASK_IDS = ("task-b", "task-a", "task-c")
SCORES_DIGEST = "9a" * 32


def _signer(uri: str = "//WorkerVerify") -> KeypairRequestSigner:
    import bittensor as bt

    return KeypairRequestSigner(bt.Keypair.create_from_uri(uri))


def _os_image_hash(mrtd: str, rtmr1: str, rtmr2: str) -> str:
    preimage = bytes.fromhex(mrtd) + bytes.fromhex(rtmr1) + bytes.fromhex(rtmr2)
    return hashlib.sha256(preimage).hexdigest()


def _build_attestation(
    *,
    agent_hash: str = AGENT_HASH,
    task_ids: tuple[str, ...] = TASK_IDS,
    scores_digest: str = SCORES_DIGEST,
    validator_nonce: str,
    compose_payload: bytes = COMPOSE_PAYLOAD,
    mrtd: str = MRTD,
    rtmr0: str = RTMR0,
    rtmr1: str = RTMR1,
    rtmr2: str = RTMR2,
    report_data_hex: str | None = None,
) -> tuple[PhalaAttestation, dict[str, str]]:
    """A self-consistent Phala attestation + its reconstructed canonical measurement."""

    event_log, rtmr3 = build_rtmr3_event_log([("compose-hash", compose_payload)])
    compose_hash = compose_payload.hex()
    os_image_hash = _os_image_hash(mrtd, rtmr1, rtmr2)
    measurement = {
        "mrtd": mrtd,
        "rtmr0": rtmr0,
        "rtmr1": rtmr1,
        "rtmr2": rtmr2,
        "compose_hash": compose_hash,
        "os_image_hash": os_image_hash,
    }
    if report_data_hex is None:
        report_data_hex = phala_report_data_hex(
            canonical_measurement=measurement,
            agent_hash=agent_hash,
            task_ids=task_ids,
            scores_digest=scores_digest,
            validator_nonce=validator_nonce,
        )
    quote = build_tdx_quote(
        mrtd=mrtd,
        rtmr0=rtmr0,
        rtmr1=rtmr1,
        rtmr2=rtmr2,
        rtmr3=rtmr3,
        report_data=report_data_hex,
    )
    attestation = PhalaAttestation(
        tdx_quote=quote,
        event_log=event_log,
        report_data=report_data_hex,
        measurement=PhalaMeasurement(
            mrtd=mrtd,
            rtmr0=rtmr0,
            rtmr1=rtmr1,
            rtmr2=rtmr2,
            rtmr3=rtmr3,
            compose_hash=compose_hash,
            os_image_hash=os_image_hash,
        ),
        vm_config={"vcpu": 1, "memory_mb": 2048},
    )
    return attestation, measurement


def _allowlist(measurement: dict[str, str]) -> MeasurementAllowlist:
    return MeasurementAllowlist.from_measurements([measurement])


def _fixture(
    *,
    unit_id: str = UNIT_ID,
    agent_hash: str = AGENT_HASH,
    task_ids: tuple[str, ...] = TASK_IDS,
    scores_digest: str = SCORES_DIGEST,
):
    """A fully-valid (proof, binding, verifier, allowlist, nonce store) tuple."""

    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    attestation, measurement = _build_attestation(
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=scores_digest,
        validator_nonce=nonce,
    )
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=unit_id,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=scores_digest,
        validator_nonce=nonce,
    )
    return proof, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces


def _verify(proof, binding, verifier, allowlist, nonces, *, unit_id: str = UNIT_ID):
    return verify_execution_proof(
        proof,
        unit_id=unit_id,
        expected_binding=binding,
        quote_verifier=verifier,
        allowlist=allowlist,
        nonce_validator=nonces,
    )


# --- VAL-VERIFY-001: fully valid attested result is ACCEPTED ----------------


def test_valid_attested_result_accepts() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    assert _verify(proof, binding, verifier, allowlist, nonces) is True


def test_backward_compatible_tier0_only_without_binding() -> None:
    # No expected_binding => existing tier-0 signature semantics, unchanged.
    proof, _binding, _verifier, _allowlist, _nonces = _fixture()
    assert verify_execution_proof(proof, unit_id=UNIT_ID) is True
    assert verify_execution_proof(proof, unit_id="other") is False


# --- VAL-VERIFY-002: invalid/forged quote signature is REJECTED -------------


def test_forged_quote_signature_rejected() -> None:
    proof, binding, _verifier, allowlist, nonces = _fixture()
    invalid = StaticQuoteVerifier(valid=False)
    assert _verify(proof, binding, invalid, allowlist, nonces) is False


def test_dcap_qvl_nonzero_exit_is_reject_not_park() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, returncode=1, stdout="", stderr="bad sig"
        )

    verifier = DcapQvlVerifier(runner=runner)
    with pytest.raises(QuoteVerificationError):
        verifier.verify("00" * 8)


# --- VAL-VERIFY-003: out-of-date / invalid TCB is REJECTED ------------------


def test_out_of_date_tcb_rejected() -> None:
    proof, binding, _verifier, allowlist, nonces = _fixture()
    stale_tcb = StaticQuoteVerifier(tcb_status="OutOfDate")
    assert _verify(proof, binding, stale_tcb, allowlist, nonces) is False


def test_dcap_qvl_reports_tcb_status() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, returncode=0, stdout=json.dumps({"status": "OutOfDate"}), stderr=""
        )

    verdict = DcapQvlVerifier(runner=runner).verify("00" * 8)
    assert verdict.tcb_status == "OutOfDate"


# --- VAL-VERIFY-004: measurement not in the validator allowlist is REJECTED --


def test_measurement_not_in_allowlist_rejected_then_accepted() -> None:
    proof, binding, verifier, _allowlist, nonces = _fixture()
    empty = MeasurementAllowlist()
    assert _verify(proof, binding, verifier, empty, nonces) is False

    # The identical quote accepted once its measurement is added to the allowlist.
    _p2, _b2, _v2, allowlist_ok, _n2 = _fixture()
    assert _verify(_p2, _b2, _v2, allowlist_ok, _n2) is True


def test_measurement_single_register_mismatch_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    # Allowlist a measurement whose MRTD differs by one register.
    other = MeasurementAllowlist.from_measurements(
        [
            {
                "mrtd": "ff" * 48,
                "rtmr0": RTMR0,
                "rtmr1": RTMR1,
                "rtmr2": RTMR2,
                "compose_hash": COMPOSE_PAYLOAD.hex(),
                "os_image_hash": _os_image_hash(MRTD, RTMR1, RTMR2),
            }
        ]
    )
    assert _verify(proof, binding, verifier, other, nonces) is False


# --- VAL-VERIFY-005: event-log / RTMR3 that does not replay is REJECTED ------


def test_event_log_replay_mismatch_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    att = PhalaAttestation.model_validate(proof.attestation)
    # Swap in an event log that replays to a DIFFERENT RTMR3 than the quote binds.
    other_log, _other_rtmr3 = build_rtmr3_event_log(
        [("compose-hash", bytes.fromhex("dd" * 32))]
    )
    att.event_log = other_log
    proof.attestation = att.model_dump(mode="json")
    assert _verify(proof, binding, verifier, allowlist, nonces) is False


def test_event_log_internally_inconsistent_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    att = PhalaAttestation.model_validate(proof.attestation)
    # Mutate a payload so its logged digest no longer matches (self-inconsistent).
    att.event_log[0]["event_payload"] = "ee" * 32
    proof.attestation = att.model_dump(mode="json")
    assert _verify(proof, binding, verifier, allowlist, nonces) is False


# --- VAL-VERIFY-006: wrong agent_hash is REJECTED ---------------------------


def test_wrong_agent_hash_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    wrong = PhalaBinding(
        agent_hash="00" * 32,
        task_ids=binding.task_ids,
        scores_digest=binding.scores_digest,
        validator_nonce=binding.validator_nonce,
    )
    assert _verify(proof, wrong, verifier, allowlist, nonces) is False


# --- VAL-VERIFY-007: wrong task set REJECTED; reorder still ACCEPTED ---------


def test_wrong_task_set_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    wrong = PhalaBinding(
        agent_hash=binding.agent_hash,
        task_ids=("task-a", "task-b"),
        scores_digest=binding.scores_digest,
        validator_nonce=binding.validator_nonce,
    )
    assert _verify(proof, wrong, verifier, allowlist, nonces) is False


def test_task_set_reordered_still_accepted() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    reordered = PhalaBinding(
        agent_hash=binding.agent_hash,
        task_ids=("task-c", "task-b", "task-a"),
        scores_digest=binding.scores_digest,
        validator_nonce=binding.validator_nonce,
    )
    assert _verify(proof, reordered, verifier, allowlist, nonces) is True


# --- VAL-VERIFY-008: wrong scores is REJECTED -------------------------------


def test_wrong_scores_digest_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    wrong = PhalaBinding(
        agent_hash=binding.agent_hash,
        task_ids=binding.task_ids,
        scores_digest="00" * 32,
        validator_nonce=binding.validator_nonce,
    )
    assert _verify(proof, wrong, verifier, allowlist, nonces) is False


# --- VAL-VERIFY-009: wrong domain-separation tag is REJECTED ----------------


def test_wrong_domain_tag_rejected() -> None:
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    # Build a report_data whose preimage uses a DIFFERENT tag but is otherwise
    # correctly bound; the verifier recomputes with PHALA_REPORT_DATA_TAG.
    os_image_hash = _os_image_hash(MRTD, RTMR1, RTMR2)
    measurement = {
        "mrtd": MRTD,
        "rtmr0": RTMR0,
        "rtmr1": RTMR1,
        "rtmr2": RTMR2,
        "compose_hash": COMPOSE_PAYLOAD.hex(),
        "os_image_hash": os_image_hash,
    }
    preimage = {
        "tag": "some-other-protocol-v9",
        "canonical_measurement": measurement,
        "agent_hash": AGENT_HASH,
        "task_ids": sorted(TASK_IDS),
        "scores_digest": SCORES_DIGEST,
        "validator_nonce": nonce,
    }
    digest = hashlib.sha256(
        json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    report_data_hex = digest.ljust(64, b"\x00").hex()
    assert PHALA_REPORT_DATA_TAG not in preimage["tag"]

    attestation, _m = _build_attestation(
        validator_nonce=nonce, report_data_hex=report_data_hex
    )
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )
    assert (
        _verify(proof, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces)
        is False
    )


# --- VAL-VERIFY-010: stale or replayed validator nonce is REJECTED ----------


def test_fresh_nonce_first_use_accepts_then_reuse_rejected() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    assert _verify(proof, binding, verifier, allowlist, nonces) is True
    # Same nonce reused (consumed) => rejected.
    assert _verify(proof, binding, verifier, allowlist, nonces) is False


def test_never_issued_nonce_rejected() -> None:
    nonces = InMemoryNonceValidator()
    attestation, measurement = _build_attestation(validator_nonce="never-issued-nonce")
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce="never-issued-nonce",
    )
    assert (
        _verify(proof, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces)
        is False
    )


def test_expired_nonce_rejected() -> None:
    clock = {"t": 1000.0}
    nonces = InMemoryNonceValidator(ttl_seconds=120, clock=lambda: clock["t"])
    nonce = nonces.issue()
    attestation, measurement = _build_attestation(validator_nonce=nonce)
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )
    clock["t"] = 2000.0  # advance well past the TTL
    assert (
        _verify(proof, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces)
        is False
    )


# --- VAL-VERIFY-011: absent nonce in report_data is REJECTED ----------------


def test_absent_nonce_rejected() -> None:
    nonces = InMemoryNonceValidator()
    attestation, measurement = _build_attestation(validator_nonce="")
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce="",
    )
    assert (
        _verify(proof, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces)
        is False
    )


# --- VAL-VERIFY-012: genuine quote repurposed from a different submission ----


def test_repurposed_quote_rejected_for_other_submission() -> None:
    # Submission B: a genuine, self-consistent, valid attested envelope.
    nonces = InMemoryNonceValidator()
    nonce_b = nonces.issue()
    attestation_b, measurement = _build_attestation(
        agent_hash="bb" * 32,
        task_ids=("b-task-1", "b-task-2"),
        scores_digest="cc" * 32,
        validator_nonce=nonce_b,
    )
    proof_b = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id="submission-B",
        attestation=attestation_b,
    )
    # Present B's envelope as submission A's result (A's expected binding).
    nonce_a = nonces.issue()
    binding_a = PhalaBinding(
        agent_hash="aa" * 32,
        task_ids=("a-task-1", "a-task-2"),
        scores_digest="dd" * 32,
        validator_nonce=nonce_a,
    )
    proof_a = rebind_worker_signature(proof_b, signer=_signer(), unit_id="submission-A")
    assert (
        verify_execution_proof(
            proof_a,
            unit_id="submission-A",
            expected_binding=binding_a,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=_allowlist(measurement),
            nonce_validator=nonces,
        )
        is False
    )


# --- VAL-VERIFY-013: tier-0 worker signature + unit binding still enforced ---


def test_cross_unit_replay_rejected_on_phala_tier() -> None:
    proof, binding, verifier, allowlist, nonces = _fixture()
    # Valid for the signed unit, rejected when presented for another unit.
    assert (
        _verify(proof, binding, verifier, allowlist, nonces, unit_id="other-unit")
        is False
    )


def test_placeholder_worker_signature_rejected() -> None:
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    attestation, measurement = _build_attestation(validator_nonce=nonce)
    # The M1 image emits an EMPTY placeholder worker_signature.
    proof = ExecutionProof(
        version=1,
        tier=PHALA_TDX_TIER,
        manifest_sha256=MANIFEST,
        worker_signature=WorkerSignature(worker_pubkey="", sig=""),
        attestation=attestation.model_dump(mode="json"),
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )
    assert (
        _verify(proof, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces)
        is False
    )


def test_rebind_makes_placeholder_envelope_verifiable() -> None:
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    attestation, measurement = _build_attestation(validator_nonce=nonce)
    placeholder = ExecutionProof(
        version=1,
        tier=PHALA_TDX_TIER,
        manifest_sha256=MANIFEST,
        worker_signature=WorkerSignature(worker_pubkey="", sig=""),
        attestation=attestation.model_dump(mode="json"),
    )
    signer = _signer("//Validator")
    bound = rebind_worker_signature(placeholder, signer=signer, unit_id=UNIT_ID)
    assert bound.worker_signature.worker_pubkey == signer.hotkey
    assert bound.tier == PHALA_TDX_TIER
    assert bound.attestation == placeholder.attestation

    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )
    assert (
        _verify(bound, binding, StaticQuoteVerifier(), _allowlist(measurement), nonces)
        is True
    )
    # Re-bound to UNIT_ID => cross-unit replay still rejected.
    nonces2 = InMemoryNonceValidator()
    nonce2 = nonces2.issue()
    att2, m2 = _build_attestation(validator_nonce=nonce2)
    placeholder2 = ExecutionProof(
        version=1,
        tier=PHALA_TDX_TIER,
        manifest_sha256=MANIFEST,
        worker_signature=WorkerSignature(worker_pubkey="", sig=""),
        attestation=att2.model_dump(mode="json"),
    )
    bound2 = rebind_worker_signature(placeholder2, signer=signer, unit_id=UNIT_ID)
    binding2 = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce2,
    )
    assert (
        verify_execution_proof(
            bound2,
            unit_id="different-unit",
            expected_binding=binding2,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=_allowlist(m2),
            nonce_validator=nonces2,
        )
        is False
    )


def test_non_phala_tier_with_binding_rejected() -> None:
    # A tier-0-only proof presented where a Phala attestation is expected.
    from base.worker.proof import build_execution_proof

    proof = build_execution_proof(
        signer=_signer(), manifest_sha256=MANIFEST, unit_id=UNIT_ID
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce="n",
    )
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=binding,
            quote_verifier=StaticQuoteVerifier(),
            allowlist=MeasurementAllowlist(),
            nonce_validator=InMemoryNonceValidator(),
        )
        is False
    )


# --- VAL-VERIFY-014: verifier transient unavailability/timeout PARKS ---------


def test_verifier_outage_parks_not_accepts_nor_rejects() -> None:
    proof, binding, _verifier, allowlist, nonces = _fixture()
    outage = StaticQuoteVerifier(unavailable=True)
    with pytest.raises(VerifierUnavailableError):
        _verify(proof, binding, outage, allowlist, nonces)


def test_parked_result_accepts_once_verifier_restored() -> None:
    proof, binding, _verifier, allowlist, nonces = _fixture()
    outage = StaticQuoteVerifier(unavailable=True)
    with pytest.raises(VerifierUnavailableError):
        _verify(proof, binding, outage, allowlist, nonces)
    # The park did not consume the nonce nor fraud-reject: a later pass accepts.
    assert _verify(proof, binding, StaticQuoteVerifier(), allowlist, nonces) is True


def test_dcap_qvl_timeout_is_park_not_reject() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=1.0)

    verifier = DcapQvlVerifier(runner=runner)
    with pytest.raises(VerifierUnavailableError):
        verifier.verify("00" * 8)


def test_dcap_qvl_missing_binary_is_park_not_reject() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("dcap-qvl not found")

    verifier = DcapQvlVerifier(runner=runner)
    with pytest.raises(VerifierUnavailableError):
        verifier.verify("00" * 8)


# --- VAL-VERIFY-025: empty/missing/unparseable allowlist fails closed --------


def test_empty_or_missing_allowlist_fails_closed_then_populated_accepts() -> None:
    # A genuine, fully-valid attested envelope that WOULD verify against a
    # populated allowlist. An empty/missing allowlist must reject it (never
    # accept-any) WITHOUT consuming the nonce, so the same valid result still
    # accepts once the allowlist is populated.
    proof, binding, verifier, populated, nonces = _fixture()

    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=binding,
            quote_verifier=verifier,
            allowlist=MeasurementAllowlist(),
            nonce_validator=nonces,
        )
        is False
    )
    # Absent/unconfigured allowlist (None) also fails closed.
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=binding,
            quote_verifier=verifier,
            allowlist=None,
            nonce_validator=nonces,
        )
        is False
    )
    # The SAME valid result now accepts against the populated allowlist.
    assert _verify(proof, binding, verifier, populated, nonces) is True


def test_unparseable_or_missing_allowlist_config_fails_closed(tmp_path) -> None:
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    attestation, measurement = _build_attestation(validator_nonce=nonce)
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )
    verifier = StaticQuoteVerifier()

    # Unparseable config => empty allowlist => reject (nonce not consumed).
    unparseable = MeasurementAllowlist.from_json("{ not valid json ]")
    assert bool(unparseable) is False
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=binding,
            quote_verifier=verifier,
            allowlist=unparseable,
            nonce_validator=nonces,
        )
        is False
    )
    # Missing config file => empty allowlist => reject (nonce not consumed).
    missing = MeasurementAllowlist.from_file(tmp_path / "absent.json")
    assert bool(missing) is False
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=binding,
            quote_verifier=verifier,
            allowlist=missing,
            nonce_validator=nonces,
        )
        is False
    )
    # A valid config carrying the measurement accepts the SAME valid result.
    config = tmp_path / "allowlist.json"
    config.write_text(json.dumps({"entries": [measurement]}), encoding="utf-8")
    populated = MeasurementAllowlist.from_file(config)
    assert bool(populated) is True
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=binding,
            quote_verifier=verifier,
            allowlist=populated,
            nonce_validator=nonces,
        )
        is True
    )


# --- VAL-VERIFY-027: multi-entry allowlist (image rotation) ------------------


def _rotation_case(
    nonces: InMemoryNonceValidator,
    *,
    unit_id: str,
    mrtd: str,
    compose_payload: bytes,
):
    """A self-consistent (proof, binding, measurement) for a given image."""

    nonce = nonces.issue()
    attestation, measurement = _build_attestation(
        validator_nonce=nonce, mrtd=mrtd, compose_payload=compose_payload
    )
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=unit_id,
        attestation=attestation,
    )
    binding = PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )
    return proof, binding, measurement


def test_two_entry_allowlist_accepts_either_rejects_third() -> None:
    nonces = InMemoryNonceValidator()
    old_proof, old_binding, old_m = _rotation_case(
        nonces,
        unit_id="unit-old",
        mrtd="a1" * 48,
        compose_payload=bytes.fromhex("c3" * 32),
    )
    new_proof, new_binding, new_m = _rotation_case(
        nonces,
        unit_id="unit-new",
        mrtd="b2" * 48,
        compose_payload=bytes.fromhex("d4" * 32),
    )
    third_proof, third_binding, _third_m = _rotation_case(
        nonces,
        unit_id="unit-third",
        mrtd="c3" * 48,
        compose_payload=bytes.fromhex("e5" * 32),
    )
    assert old_m != new_m

    rotation = MeasurementAllowlist.from_measurements([old_m, new_m])
    verifier = StaticQuoteVerifier()

    # Both the outgoing (old) and incoming (new) images verify against the
    # two-entry allowlist during a rotation window.
    assert (
        verify_execution_proof(
            old_proof,
            unit_id="unit-old",
            expected_binding=old_binding,
            quote_verifier=verifier,
            allowlist=rotation,
            nonce_validator=nonces,
        )
        is True
    )
    assert (
        verify_execution_proof(
            new_proof,
            unit_id="unit-new",
            expected_binding=new_binding,
            quote_verifier=verifier,
            allowlist=rotation,
            nonce_validator=nonces,
        )
        is True
    )
    # A third image, listed under neither entry, is rejected.
    assert (
        verify_execution_proof(
            third_proof,
            unit_id="unit-third",
            expected_binding=third_binding,
            quote_verifier=verifier,
            allowlist=rotation,
            nonce_validator=nonces,
        )
        is False
    )
