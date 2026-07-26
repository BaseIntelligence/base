"""Base->prism result ingestion: verify the ExecutionProof, then finalize idempotently.

This is the prism half of the base worker-plane accept path (architecture.md 3.3/3.5). After the
base master reconciles a gpu work unit's R=2 replicas it forwards the accepted result here (the
``HttpChallengeResultForwarder`` counterpart) with the pinned body ``{work_unit_id, submission_ref,
result}``. Before anything is scored, the forwarded :class:`~prism_challenge.proof.ExecutionProof`
is verified:

* **shape** (VAL-PRISM-018): the envelope must exist, be ``version == 1``, carry a 64-char
  lowercase-hex ``manifest_sha256`` and a ``worker_signature`` with both ``worker_pubkey`` and
  ``sig``; each failure carries a distinguishable reason and NOTHING is scored;
* **integrity** (VAL-PRISM-007): the sr25519 signature must verify against the pinned message for
  this unit, and -- when the run manifest is forwarded -- its content must hash back to the signed
  ``manifest_sha256`` (a tampered manifest or a forged digest is rejected);
* **tier** (VAL-PRISM-019): the claimed tier is downgraded to its verified EFFECTIVE tier and the
  downgrade recorded, so the audit scheduler samples at the honest rate.

Only a fully verified result is finalized, and finalization is idempotent: the first accepted
delivery finalizes the submission from the FORWARDED, verified+reconciled manifest via the
CAS-guarded :meth:`PrismWorker.finalize_worker_result` path (worker plane on) -- scoring the run
WITHOUT re-executing the evaluator, since the heavy GPU work already ran on the miner-funded worker
(architecture.md 4). A duplicate delivery of the same manifest is a no-op and a CONFLICTING delivery
(a different ``manifest_sha256`` for an already-accepted unit) is rejected -- never overwriting the
stored score (VAL-PRISM-017). With the worker plane OFF, finalization falls back to the legacy
in-process re-execution path (:meth:`PrismWorker.process_submission`), byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .audit import AuditSampler, effective_tier
from .auth import verify_hotkey_signature
from .breakglass import (
    BreakGlassAuditLog,
    BreakGlassRequest,
    evaluate_break_glass,
    fault_class_of,
)
from .constation import (
    INFRA_FAULT_CONSTATION_UNAVAILABLE,
    INFRA_FAULT_RETRY_EXHAUSTED,
    AllowlistChecker,
    ConstationBundle,
    ConstationResult,
    NonceChecker,
    classify_constation_fault,
    constation_ok,
    infra_fault_reason,
    miner_fault_reason,
)
from .constation import (
    SignatureVerifier as ConstationSignatureVerifier,
)
from .plausibility import check_manifest_plausibility
from .proof import (
    ATTESTATION_MODE_V1,
    EXECUTION_PROOF_VERSION,
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    ExecutionProof,
    attestation_mode_of,
    compute_manifest_sha256,
    verify_execution_proof,
)
from .queue import PrismWorker, WorkerFinalizationError

logger = logging.getLogger(__name__)

#: A verified 64-char lowercase-hex manifest digest (VAL-PRISM-018).
_MANIFEST_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Bounded retry before discarding an infra-fault run (todo 22).
DEFAULT_MAX_CONSTATION_ATTEMPTS = 3

SignatureVerifier = Callable[[str, bytes, str], bool]


def _as_int(value: object, default: int) -> int:
    """Coerce a persisted (sqlite ``object``-typed) tier column back to ``int``."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


class ResultIngestionError(Exception):
    """A forwarded result is rejected before scoring; ``reason`` is a stable machine code.

    Reason codes (all distinct from plausibility/scoring failure modes):

    * ``result_malformed`` -- the ``result`` field is not an object;
    * ``proof_missing`` -- no ExecutionProof envelope in the result (VAL-PRISM-018a);
    * ``proof_bad_version`` -- ``version`` is not ``1`` (VAL-PRISM-018b);
    * ``proof_bad_manifest_hash`` -- ``manifest_sha256`` is not 64-char lowercase hex
      (VAL-PRISM-018c);
    * ``proof_missing_signature`` -- ``worker_signature`` lacks ``worker_pubkey`` or ``sig``
      (VAL-PRISM-018d);
    * ``proof_malformed`` -- the envelope fails to parse into an :class:`ExecutionProof`;
    * ``manifest_tampered`` -- the forwarded manifest does not hash to ``manifest_sha256``
      (VAL-PRISM-007a);
    * ``signature_invalid`` -- the worker signature does not verify for this unit
      (VAL-PRISM-007b/c);
    * ``manifest_missing`` -- the worker plane is on but no run manifest was forwarded, so there is
      nothing to finalize from without re-executing (VAL-FINAL-001);
    * ``finalization_failed`` -- worker-plane finalization failed for an internal/transient reason
      (source-static derivation error): the submission is reverted to pending and NOTHING is
      recorded, so the forwarded result is retried rather than sealed as a clean finalize.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True)
class IngestionOutcome:
    """The observable outcome of ingesting one forwarded result."""

    status: str  # "accepted" | "conflict" | "rejected"
    work_unit_id: str
    submission_id: str
    claimed_tier: int
    effective_tier: int
    tier_downgraded: bool
    idempotent: bool
    finalized: bool
    submission_status: str | None = None
    audit_sampled: bool | None = None
    audit_unit_id: str | None = None
    reason: str | None = None
    attestation_mode: str | None = None
    break_glass_admitted: bool = False
    score_written: bool = False

    def to_response(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "work_unit_id": self.work_unit_id,
            "submission_id": self.submission_id,
            "claimed_tier": self.claimed_tier,
            "effective_tier": self.effective_tier,
            "tier_downgraded": self.tier_downgraded,
            "idempotent": self.idempotent,
            "finalized": self.finalized,
            "submission_status": self.submission_status,
            "score_written": self.score_written,
            "break_glass_admitted": self.break_glass_admitted,
        }
        if self.audit_sampled is not None:
            payload["audit_sampled"] = self.audit_sampled
        if self.audit_unit_id is not None:
            payload["audit_unit_id"] = self.audit_unit_id
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.attestation_mode is not None:
            payload["attestation_mode"] = self.attestation_mode
        return payload


def parse_execution_proof(result: Mapping[str, Any]) -> ExecutionProof:
    """Validate the raw proof envelope shape and return a typed :class:`ExecutionProof`.

    Rejects (before any typed coercion so the reason is precise) a missing envelope, a wrong
    version, a non-64-hex manifest hash, or a signature block missing either field (VAL-PRISM-018).
    """

    raw = result.get(PROOF_PAYLOAD_KEY)
    if raw is None:
        raise ResultIngestionError("proof_missing", "result has no execution_proof envelope")
    if not isinstance(raw, Mapping):
        raise ResultIngestionError("proof_missing", "execution_proof must be an object")
    if raw.get("version") != EXECUTION_PROOF_VERSION:
        raise ResultIngestionError(
            "proof_bad_version", f"unsupported execution_proof version {raw.get('version')!r}"
        )
    manifest_sha256 = raw.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not _MANIFEST_SHA256_RE.fullmatch(manifest_sha256):
        raise ResultIngestionError(
            "proof_bad_manifest_hash", "manifest_sha256 must be 64-char lowercase hex"
        )
    signature = raw.get("worker_signature")
    if (
        not isinstance(signature, Mapping)
        or not signature.get("worker_pubkey")
        or not signature.get("sig")
    ):
        raise ResultIngestionError(
            "proof_missing_signature", "worker_signature must include worker_pubkey and sig"
        )
    try:
        return ExecutionProof.model_validate(dict(raw))
    except Exception as exc:  # noqa: BLE001 - normalise any pydantic error to a stable reason
        raise ResultIngestionError("proof_malformed", str(exc)) from exc


def verify_proof_integrity(
    proof: ExecutionProof,
    *,
    unit_id: str,
    manifest: Mapping[str, Any] | None = None,
    verify: SignatureVerifier = verify_hotkey_signature,
) -> None:
    """Reject a tampered manifest or a forged/invalid signature for ``unit_id`` (VAL-PRISM-007).

    When the run manifest is forwarded, its canonical content MUST hash to the signed
    ``manifest_sha256`` (catches a manifest mutated after signing). The sr25519 signature MUST then
    verify against the worker pubkey over the pinned ``sha256(manifest_sha256:unit_id)`` message
    (catches a rewritten digest whose signature was not re-issued, or corrupted signature bytes).
    """

    if manifest is not None and compute_manifest_sha256(manifest) != proof.manifest_sha256:
        raise ResultIngestionError(
            "manifest_tampered", "forwarded manifest does not hash to manifest_sha256"
        )
    if not verify_execution_proof(proof, unit_id=unit_id, verify=verify):
        raise ResultIngestionError(
            "signature_invalid", "worker signature does not verify for this unit"
        )


async def ingest_work_unit_result(
    *,
    worker: PrismWorker,
    work_unit_id: str,
    submission_ref: str,
    result: Mapping[str, Any],
    pinned_image_digest: str | None = None,
    audit_sampler: AuditSampler | None = None,
    verify: SignatureVerifier = verify_hotkey_signature,
    constation_bundle: ConstationBundle | None = None,
    check_allowlist: AllowlistChecker | None = None,
    check_nonce: NonceChecker | None = None,
    verify_constation_signature: ConstationSignatureVerifier | None = None,
    constation_infra_fault: str | None = None,
    constation_attempt: int = 1,
    max_constation_attempts: int = DEFAULT_MAX_CONSTATION_ATTEMPTS,
    break_glass: BreakGlassRequest | None = None,
    break_glass_audit_log: BreakGlassAuditLog | None = None,
) -> IngestionOutcome:
    """Verify a forwarded worker result and finalize under strict fail-closed constation (P1).

    ``work_unit_id`` is prism's stable unit id (``== submission_id``). Verification (shape ->
    integrity) runs BEFORE any scoring; a rejected result raises :class:`ResultIngestionError` and
    leaves the submission untouched (eligible for retry).

    **P1 fail-closed (todo 22):** no valid constation bundle ⇒ **no** ``final_score`` row is
    written at all. Failures carry ``miner_fault:*`` or ``infra_fault:*`` reason codes.
    Infra-fault runs may be admitted only via an audited operator break-glass (todo 23);
    miner-fault runs can never be override-admitted. Bounded retry applies to infra faults
    before discard.

    Effective tier is granted only via ``constation_ok`` (M14 / todo 21); max tier is 1.
    Claimed attestation / TEE fields never elevate. Self-reported ``PRISM_IMAGE_DIGEST`` is
    telemetry only.
    """
    del submission_ref  # reserved for future cross-checks

    if not isinstance(result, Mapping):
        raise ResultIngestionError("result_malformed", "result must be an object")

    proof = parse_execution_proof(result)
    raw_manifest = result.get(MANIFEST_PAYLOAD_KEY)
    manifest = raw_manifest if isinstance(raw_manifest, Mapping) else None
    verify_proof_integrity(proof, unit_id=work_unit_id, manifest=manifest, verify=verify)

    claimed_tier = int(proof.tier)
    submission_id = work_unit_id
    replication = _as_int(result.get("replication"), 2)
    repository = worker.repository
    att_mode = attestation_mode_of(proof) or ATTESTATION_MODE_V1

    existing = await repository.get_work_unit_result(work_unit_id)
    if existing is not None:
        recorded_hash = str(existing.get("manifest_sha256"))
        if recorded_hash == proof.manifest_sha256:
            return IngestionOutcome(
                status="accepted",
                work_unit_id=work_unit_id,
                submission_id=submission_id,
                claimed_tier=_as_int(existing.get("claimed_tier"), claimed_tier),
                effective_tier=_as_int(existing.get("effective_tier"), 0),
                tier_downgraded=bool(existing.get("tier_downgraded", True)),
                idempotent=True,
                finalized=False,
                submission_status=await repository.submission_status(submission_id),
                attestation_mode=att_mode,
                score_written=True,
            )
        logger.warning(
            "rejecting conflicting result delivery for finalized work unit %s "
            "(delivered %s != accepted %s); stored score left untouched",
            work_unit_id,
            proof.manifest_sha256,
            recorded_hash,
        )
        return IngestionOutcome(
            status="conflict",
            work_unit_id=work_unit_id,
            submission_id=submission_id,
            claimed_tier=claimed_tier,
            effective_tier=0,
            tier_downgraded=True,
            idempotent=False,
            finalized=False,
            submission_status=await repository.submission_status(submission_id),
            reason="manifest_conflict",
            attestation_mode=att_mode,
            score_written=True,
        )

    # --- Constation gate (P1) -----------------------------------------------------------------
    gate = _evaluate_constation_gate(
        bundle=constation_bundle,
        check_allowlist=check_allowlist,
        check_nonce=check_nonce,
        verify_constation_signature=verify_constation_signature,
        constation_infra_fault=constation_infra_fault,
        constation_attempt=constation_attempt,
        max_constation_attempts=max_constation_attempts,
    )
    break_glass_admitted = False
    if not gate.admit:
        if gate.retryable:
            raise ResultIngestionError(
                gate.reason or "infra_fault:constation_retry",
                gate.message or "constation infra fault; retryable",
            )
        # Optional break-glass for infra_fault only.
        if break_glass is not None and fault_class_of(gate.reason or "") == "infra_fault":
            decision = evaluate_break_glass(
                break_glass,
                fault_reason=gate.reason or infra_fault_reason(INFRA_FAULT_CONSTATION_UNAVAILABLE),
                audit_log=break_glass_audit_log,
            )
            if decision.admitted:
                break_glass_admitted = True
                logger.warning(
                    "break-glass admitted infra-fault run %s by operator %s",
                    work_unit_id,
                    break_glass.operator_id,
                )
            else:
                return await _reject_no_score(
                    work_unit_id=work_unit_id,
                    submission_id=submission_id,
                    claimed_tier=claimed_tier,
                    reason=gate.reason or miner_fault_reason("missing_constation_bundle"),
                    attestation_mode=att_mode,
                    repository=repository,
                )
        else:
            if break_glass is not None and fault_class_of(gate.reason or "") == "miner_fault":
                # Explicit refuse path for attempted miner_fault override (todo 23).
                evaluate_break_glass(
                    break_glass,
                    fault_reason=gate.reason or miner_fault_reason("unknown"),
                    audit_log=break_glass_audit_log,
                )
            return await _reject_no_score(
                work_unit_id=work_unit_id,
                submission_id=submission_id,
                claimed_tier=claimed_tier,
                reason=gate.reason or miner_fault_reason("missing_constation_bundle"),
                attestation_mode=att_mode,
                repository=repository,
            )

    # Break-glass admits a score at tier 0 only — elevation still requires real constation_ok.
    tier = effective_tier(
        proof,
        pinned_image_digest=pinned_image_digest,
        constation_ok_result=bool(gate.constation_ok) and not break_glass_admitted,
    )
    downgraded = claimed_tier != tier

    if downgraded:
        logger.warning(
            "downgrading unverifiable tier claim for work unit %s: claimed %d -> effective %d",
            work_unit_id,
            claimed_tier,
            tier,
        )

    if manifest is not None:
        check_manifest_plausibility(
            manifest,
            wall_clock_budget_seconds=float(worker.settings.base_eval_hard_timeout_seconds),
        )

    if worker.settings.worker_plane.enabled:
        if manifest is None:
            raise ResultIngestionError(
                "manifest_missing",
                "worker-plane finalization requires the forwarded run manifest",
            )
        try:
            result_id = await worker.finalize_worker_result(
                submission_id,
                dict(manifest),
            )
        except WorkerFinalizationError as exc:
            raise ResultIngestionError(
                "finalization_failed",
                f"worker-plane finalization failed transiently and is retryable: {exc}",
            ) from exc
    else:
        result_id = await worker.process_submission(submission_id)
    submission_status = await repository.submission_status(submission_id)
    await repository.record_work_unit_result(
        work_unit_id=work_unit_id,
        submission_id=submission_id,
        manifest_sha256=proof.manifest_sha256,
        claimed_tier=claimed_tier,
        effective_tier=tier,
        tier_downgraded=downgraded,
        worker_pubkey=proof.worker_signature.worker_pubkey,
    )
    audit_sampled: bool | None = None
    audit_unit_id: str | None = None
    if audit_sampler is not None:
        audit_sampled = audit_sampler.should_sample(work_unit_id=work_unit_id, effective_tier=tier)
        if audit_sampled:
            audit_unit_id = await repository.create_audit_unit(
                submission_id=submission_id,
                origin_work_unit_id=work_unit_id,
                audited_manifest_sha256=proof.manifest_sha256,
                effective_tier=tier,
                replication=replication,
            )
    return IngestionOutcome(
        status="accepted",
        work_unit_id=work_unit_id,
        submission_id=submission_id,
        claimed_tier=claimed_tier,
        effective_tier=tier,
        tier_downgraded=downgraded,
        idempotent=False,
        finalized=result_id is not None,
        submission_status=submission_status,
        audit_sampled=audit_sampled,
        audit_unit_id=audit_unit_id,
        attestation_mode=att_mode,
        break_glass_admitted=break_glass_admitted,
        score_written=result_id is not None,
        reason=gate.reason if break_glass_admitted else None,
    )


@dataclass(frozen=True)
class _ConstationGate:
    admit: bool
    constation_ok: bool
    reason: str | None = None
    message: str | None = None
    retryable: bool = False


def _evaluate_constation_gate(
    *,
    bundle: ConstationBundle | None,
    check_allowlist: AllowlistChecker | None,
    check_nonce: NonceChecker | None,
    verify_constation_signature: ConstationSignatureVerifier | None,
    constation_infra_fault: str | None,
    constation_attempt: int,
    max_constation_attempts: int,
) -> _ConstationGate:
    """Decide whether scoring may proceed under P1 fail-closed policy."""
    if constation_infra_fault:
        reason = infra_fault_reason(constation_infra_fault)
        if constation_attempt < max_constation_attempts:
            return _ConstationGate(
                admit=False,
                constation_ok=False,
                reason=reason,
                message=f"infra fault attempt {constation_attempt}/{max_constation_attempts}",
                retryable=True,
            )
        return _ConstationGate(
            admit=False,
            constation_ok=False,
            reason=infra_fault_reason(INFRA_FAULT_RETRY_EXHAUSTED)
            if constation_attempt >= max_constation_attempts
            else reason,
            message="infra fault retry budget exhausted; no score",
            retryable=False,
        )

    if bundle is None:
        return _ConstationGate(
            admit=False,
            constation_ok=False,
            reason=miner_fault_reason("missing_constation_bundle"),
            message="no constation bundle; no score (P1)",
        )

    if check_allowlist is None or check_nonce is None or verify_constation_signature is None:
        # Bundle present but checkers unavailable → infra (cannot evaluate).
        reason = infra_fault_reason(INFRA_FAULT_CONSTATION_UNAVAILABLE)
        if constation_attempt < max_constation_attempts:
            return _ConstationGate(
                admit=False,
                constation_ok=False,
                reason=reason,
                message="constation checkers unavailable; retryable",
                retryable=True,
            )
        return _ConstationGate(
            admit=False,
            constation_ok=False,
            reason=infra_fault_reason(INFRA_FAULT_RETRY_EXHAUSTED),
            message="constation checkers unavailable; retries exhausted",
        )

    result: ConstationResult = constation_ok(
        bundle,
        check_allowlist=check_allowlist,
        check_nonce=check_nonce,
        verify_signature=verify_constation_signature,
    )
    if result.ok:
        return _ConstationGate(admit=True, constation_ok=True, reason="ok")

    fault = classify_constation_fault(result)
    return _ConstationGate(
        admit=False,
        constation_ok=False,
        reason=fault,
        message=f"constation_ok failed: {result.reason.value}",
    )


async def _reject_no_score(
    *,
    work_unit_id: str,
    submission_id: str,
    claimed_tier: int,
    reason: str,
    attestation_mode: str,
    repository: Any,
) -> IngestionOutcome:
    """Return a rejected outcome without writing a score row (P1)."""
    logger.warning(
        "fail-closed: refusing score for work unit %s reason=%s",
        work_unit_id,
        reason,
    )
    status = await repository.submission_status(submission_id)
    return IngestionOutcome(
        status="rejected",
        work_unit_id=work_unit_id,
        submission_id=submission_id,
        claimed_tier=claimed_tier,
        effective_tier=0,
        tier_downgraded=claimed_tier != 0,
        idempotent=False,
        finalized=False,
        submission_status=status,
        reason=reason,
        attestation_mode=attestation_mode,
        score_written=False,
    )


__all__ = [
    "DEFAULT_MAX_CONSTATION_ATTEMPTS",
    "IngestionOutcome",
    "ResultIngestionError",
    "ingest_work_unit_result",
    "parse_execution_proof",
    "verify_proof_integrity",
]
