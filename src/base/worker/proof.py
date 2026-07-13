"""ExecutionProof construction and verification (architecture.md sec 3.4).

Every worker result carries an ``ExecutionProof`` envelope. The tier-0 core is a
deterministic ``manifest_sha256`` plus the worker's sr25519 signature binding
that hash to the work unit. The signed message format is PINNED identically to
prism (VAL-AGENT-008 / VAL-PRISM-006): the signature is over
``sha256(manifest_sha256 + ":" + unit_id)`` -- the sha256 digest of the UTF-8
bytes of the string ``{manifest_sha256}:{unit_id}`` -- so a proof produced by the
worker plane verifies with the same code as one produced by prism, and a proof
cannot be replayed across units.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from base.challenge_sdk.proof import (
    EXECUTION_PROOF_VERSION,
    execution_proof_signing_payload,
)
from base.schemas.worker import (
    PHALA_TDX_TIER,
    ExecutionProof,
    PhalaAttestation,
    PhalaMeasurement,
    ProviderInfo,
    WorkerSignature,
)
from base.security.miner_auth import SignatureVerifier, verify_substrate_signature
from base.validator.agent.signing import RequestSigner
from base.worker.phala_quote import (
    QuoteStructureError,
    QuoteVerificationError,
    QuoteVerifier,
    os_image_hash_from_registers,
    parse_td_report,
    replay_rtmr3,
)
from base.worker.phala_verify import (
    MeasurementAllowlist,
    NonceState,
    NonceValidator,
    PhalaBinding,
)

#: TCB statuses the Phala-tier verifier accepts by default (architecture sec 7).
#: A quote whose collateral reports any other posture is rejected.
ACCEPTABLE_TCB_DEFAULT: tuple[str, ...] = ("UpToDate",)

#: Result-payload key carrying the serialized :class:`ExecutionProof`.
PROOF_PAYLOAD_KEY = "execution_proof"
#: Result-payload key carrying the deterministic prism manifest hash.
MANIFEST_SHA256_PAYLOAD_KEY = "manifest_sha256"

#: Domain-separation tag bound into a Phala attestation's ``report_data``
#: (architecture.md sec 6). Prevents a quote minted for this purpose from being
#: repurposed under a different protocol tag.
PHALA_REPORT_DATA_TAG = "base-agent-challenge-v1"
#: Byte width of the TDX ``report_data`` field a quote carries.
PHALA_REPORT_DATA_BYTES = 64


def build_execution_proof(
    *,
    signer: RequestSigner,
    manifest_sha256: str,
    unit_id: str,
    tier: int | str = 0,
    provider: ProviderInfo | None = None,
    image_digest: str | None = None,
    attestation: dict[str, Any] | None = None,
) -> ExecutionProof:
    """Build and sign an ExecutionProof for ``unit_id`` under ``signer``.

    ``signer`` is the WORKER keypair; its public identity becomes
    ``worker_signature.worker_pubkey`` and it signs the pinned message.
    """

    signature = signer.sign(
        execution_proof_signing_payload(
            manifest_sha256=manifest_sha256, unit_id=unit_id
        )
    )
    return ExecutionProof(
        version=EXECUTION_PROOF_VERSION,
        tier=tier,
        manifest_sha256=manifest_sha256,
        image_digest=image_digest,
        provider=provider,
        worker_signature=WorkerSignature(worker_pubkey=signer.hotkey, sig=signature),
        attestation=attestation,
    )


def verify_execution_proof(
    proof: ExecutionProof,
    *,
    unit_id: str,
    expected_binding: PhalaBinding | None = None,
    quote_verifier: QuoteVerifier | None = None,
    allowlist: MeasurementAllowlist | None = None,
    nonce_validator: NonceValidator | None = None,
    acceptable_tcb: Collection[str] = ACCEPTABLE_TCB_DEFAULT,
    signature_verifier: SignatureVerifier = verify_substrate_signature,
) -> bool:
    """Whether ``proof`` verifies for ``unit_id``.

    Without ``expected_binding`` this is the tier-0 check only: the worker sr25519
    signature over ``sha256(f"{manifest_sha256}:{unit_id}")`` must verify, and a
    proof signed for a DIFFERENT unit is rejected (no cross-unit replay). This is
    the unchanged behavior every existing (tier 0/1/2) caller relies on.

    With ``expected_binding`` the full **Phala-tier** verification runs on top of
    the tier-0 check (architecture sec 4 C4 / 6 / 7): the proof must be the Phala
    tier with an attestation whose TDX quote (a) is DCAP-valid with an acceptable
    TCB, (b) reconstructs a measurement in the validator ``allowlist``, (c) whose
    ``report_data`` binds exactly ``expected_binding``, and (d) whose nonce is a
    fresh, validator-issued, unconsumed one. The tier-0 worker signature is STILL
    required, so the attested envelope's trust root is the quote while a real
    (validator-rebound) signature prevents cross-unit replay -- an absent/wrong
    signature is rejected even for a valid quote (VAL-VERIFY-013).

    Returns ``True`` to ACCEPT and ``False`` to REJECT. When the quote-verification
    dependency is transiently unavailable it raises
    :class:`base.worker.phala_quote.VerifierUnavailableError` so the caller PARKS
    the result rather than accepting or fraud-rejecting it (VAL-VERIFY-014).
    """

    payload = execution_proof_signing_payload(
        manifest_sha256=proof.manifest_sha256, unit_id=unit_id
    )
    signature_ok = signature_verifier(
        proof.worker_signature.worker_pubkey, payload, proof.worker_signature.sig
    )
    if expected_binding is None:
        return signature_ok

    if not signature_ok:
        return False
    if proof.tier != PHALA_TDX_TIER or proof.attestation is None:
        return False
    if quote_verifier is None or allowlist is None or nonce_validator is None:
        return False
    try:
        attestation = PhalaAttestation.model_validate(proof.attestation)
    except ValidationError:
        return False
    return _verify_phala_attestation(
        attestation,
        expected_binding=expected_binding,
        quote_verifier=quote_verifier,
        allowlist=allowlist,
        nonce_validator=nonce_validator,
        acceptable_tcb=acceptable_tcb,
    )


def _verify_phala_attestation(
    attestation: PhalaAttestation,
    *,
    expected_binding: PhalaBinding,
    quote_verifier: QuoteVerifier,
    allowlist: MeasurementAllowlist,
    nonce_validator: NonceValidator,
    acceptable_tcb: Collection[str],
) -> bool:
    """Verify a Phala attestation against the validator's expectations.

    Fail-closed and conjunctive: every check must pass. The trust root is the
    hardware-signed quote -- the measurement and ``report_data`` are read from the
    parsed TD report (not the untrusted attestation block), and the event log must
    replay to the quote's signed RTMR3. Raises
    :class:`base.worker.phala_quote.VerifierUnavailableError` (park) if the quote
    verifier is transiently unavailable.
    """

    if not allowlist:
        return False
    if not expected_binding.nonce:
        return False

    try:
        report = parse_td_report(attestation.tdx_quote)
    except QuoteStructureError:
        return False

    try:
        verdict = quote_verifier.verify(attestation.tdx_quote)
    except QuoteVerificationError:
        return False
    if verdict.tcb_status not in acceptable_tcb:
        return False

    try:
        replay = replay_rtmr3(attestation.event_log)
    except QuoteVerificationError:
        return False
    if replay.rtmr3 != report.rtmr3 or replay.compose_hash is None:
        return False

    measurement = {
        "mrtd": report.mrtd,
        "rtmr0": report.rtmr0,
        "rtmr1": report.rtmr1,
        "rtmr2": report.rtmr2,
        "compose_hash": replay.compose_hash,
        "os_image_hash": os_image_hash_from_registers(
            report.mrtd, report.rtmr1, report.rtmr2
        ),
    }
    if not allowlist.contains(measurement):
        return False

    binding_args: dict[str, str | None] = (
        {
            "eval_run_id": expected_binding.eval_run_id,
            "score_nonce": expected_binding.score_nonce,
        }
        if expected_binding.is_eval_v2
        else {"validator_nonce": expected_binding.validator_nonce}
    )
    expected_report_data = phala_report_data_hex(
        canonical_measurement=measurement,
        agent_hash=expected_binding.agent_hash,
        task_ids=expected_binding.task_ids,
        scores_digest=expected_binding.scores_digest,
        **binding_args,
    )
    if report.report_data != bytes.fromhex(expected_report_data):
        return False

    return nonce_validator.consume(expected_binding.nonce) is NonceState.OK


def _canonical_measurement_mapping(
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
) -> dict[str, str]:
    """The static, allowlist-pinnable measurement subset (excludes ``rtmr3``)."""

    if isinstance(canonical_measurement, PhalaMeasurement):
        return canonical_measurement.canonical()
    static_fields = (
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "compose_hash",
        "os_image_hash",
    )
    return {field: str(canonical_measurement[field]) for field in static_fields}


def _is_visible_id(value: object) -> bool:
    """Whether an Eval wire identifier is one to 128 visible ASCII bytes."""

    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all("!" <= char <= "~" for char in value)
    )


def _is_lower_hex(value: object, width: int) -> bool:
    """Whether ``value`` has an exact-width lowercase hexadecimal encoding."""

    return (
        isinstance(value, str)
        and len(value) == width
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _schema_v2_measurement(
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
) -> dict[str, str]:
    """Validate and return the strict static Eval measurement object."""

    try:
        measurement = _canonical_measurement_mapping(canonical_measurement)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "schema-version-2 canonical measurement is incomplete"
        ) from exc

    register_fields = ("mrtd", "rtmr0", "rtmr1", "rtmr2")
    if any(not _is_lower_hex(measurement[field], 96) for field in register_fields):
        raise ValueError("schema-version-2 measurement registers must be 96 hex chars")
    if not _is_lower_hex(measurement["compose_hash"], 64) or not _is_lower_hex(
        measurement["os_image_hash"], 64
    ):
        raise ValueError("schema-version-2 measurement hashes must be 64 hex chars")
    return measurement


def phala_report_data(
    *,
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
    agent_hash: str,
    task_ids: Iterable[str],
    scores_digest: str,
    validator_nonce: str | None = None,
    eval_run_id: str | None = None,
    score_nonce: str | None = None,
) -> bytes:
    """The 32-byte ``report_data`` digest binding a Phala run (architecture sec 6).

    The schema-version-2 Eval boundary hashes canonical JSON over
    ``{agent_hash, canonical_measurement, domain, eval_run_id, schema_version,
    score_nonce, scores_digest, task_ids}``; its sorted task IDs are a
    duplicate-free set and its measurement excludes runtime ``rtmr3``. Legacy
    callers that still supply only ``validator_nonce`` retain the original v1
    derivation until they migrate to the direct Eval boundary.

    This is the single source of truth for the derivation shared by the image
    emitter (M1) and the validator/master verifier (M4): both MUST call this
    function rather than re-implementing sec 6.
    """

    supplied_tasks = tuple(task_ids)
    if (eval_run_id is None) != (score_nonce is None):
        raise ValueError("eval_run_id and score_nonce must be supplied together")

    if eval_run_id is not None and score_nonce is not None:
        if validator_nonce is not None:
            raise ValueError("schema-version-2 binding does not use validator_nonce")
        if (
            not _is_visible_id(eval_run_id)
            or not _is_visible_id(score_nonce)
            or any(not _is_visible_id(task_id) for task_id in supplied_tasks)
        ):
            raise ValueError("schema-version-2 ids must be visible ASCII")
        if not _is_lower_hex(agent_hash, 64) or not _is_lower_hex(scores_digest, 64):
            raise ValueError("schema-version-2 digests must be 64 lowercase hex chars")
        task_set = tuple(sorted(supplied_tasks))
        if len(task_set) != len(set(task_set)):
            raise ValueError("task_ids must be unique")
        if supplied_tasks != task_set:
            raise ValueError("schema-version-2 task_ids must be sorted")
        preimage = {
            "agent_hash": agent_hash,
            "canonical_measurement": _schema_v2_measurement(canonical_measurement),
            "domain": PHALA_REPORT_DATA_TAG,
            "eval_run_id": eval_run_id,
            "schema_version": 2,
            "score_nonce": score_nonce,
            "scores_digest": scores_digest,
            "task_ids": list(supplied_tasks),
        }
    else:
        if validator_nonce is None:
            raise ValueError("validator_nonce is required for legacy bindings")
        preimage = {
            "tag": PHALA_REPORT_DATA_TAG,
            "canonical_measurement": _canonical_measurement_mapping(
                canonical_measurement
            ),
            "agent_hash": agent_hash,
            "task_ids": sorted(supplied_tasks),
            "scores_digest": scores_digest,
            "validator_nonce": validator_nonce,
        }
    encoded = json.dumps(
        preimage,
        ensure_ascii=eval_run_id is None,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).digest()


def phala_report_data_hex(
    *,
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
    agent_hash: str,
    task_ids: Iterable[str],
    scores_digest: str,
    validator_nonce: str | None = None,
    eval_run_id: str | None = None,
    score_nonce: str | None = None,
) -> str:
    """``report_data`` as a 64-byte TDX field (128 hex chars, left-aligned).

    The 32-byte :func:`phala_report_data` digest occupies the leading bytes and
    the trailing bytes are zero, matching Phala's observed left-aligned zero-pad
    round-trip for the fixed-width quote field.
    """

    digest = phala_report_data(
        canonical_measurement=canonical_measurement,
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=scores_digest,
        validator_nonce=validator_nonce,
        eval_run_id=eval_run_id,
        score_nonce=score_nonce,
    )
    return digest.ljust(PHALA_REPORT_DATA_BYTES, b"\x00").hex()


def build_phala_execution_proof(
    *,
    signer: RequestSigner,
    manifest_sha256: str,
    unit_id: str,
    attestation: PhalaAttestation | Mapping[str, Any],
    provider: ProviderInfo | None = None,
    image_digest: str | None = None,
) -> ExecutionProof:
    """Build a Phala-tier ExecutionProof carrying a TDX attestation payload.

    The envelope keeps the tier-0 worker sr25519 signature over the pinned
    ``sha256(f"{manifest_sha256}:{unit_id}")`` message, so
    :func:`verify_execution_proof` still validates the worker-signature layer;
    ``tier`` is set to :data:`PHALA_TDX_TIER` and ``attestation`` carries the
    serialized :class:`PhalaAttestation`. Cryptographic quote verification is
    layered on top by the validator/master (milestone M4); this helper does not
    fork a parallel envelope.
    """

    payload = (
        attestation
        if isinstance(attestation, PhalaAttestation)
        else PhalaAttestation.model_validate(dict(attestation))
    )
    return build_execution_proof(
        signer=signer,
        manifest_sha256=manifest_sha256,
        unit_id=unit_id,
        tier=PHALA_TDX_TIER,
        provider=provider,
        image_digest=image_digest,
        attestation=payload.model_dump(mode="json"),
    )


__all__ = [
    "ACCEPTABLE_TCB_DEFAULT",
    "EXECUTION_PROOF_VERSION",
    "MANIFEST_SHA256_PAYLOAD_KEY",
    "PHALA_REPORT_DATA_BYTES",
    "PHALA_REPORT_DATA_TAG",
    "PHALA_TDX_TIER",
    "PROOF_PAYLOAD_KEY",
    "build_execution_proof",
    "build_phala_execution_proof",
    "execution_proof_signing_payload",
    "phala_report_data",
    "phala_report_data_hex",
    "verify_execution_proof",
]
