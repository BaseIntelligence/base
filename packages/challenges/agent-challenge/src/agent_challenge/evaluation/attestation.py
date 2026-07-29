"""Host-trust result admission shim (T40 — Phala AttestationGate removed).

Product path is unattested/host-trust only. This module keeps a minimal
decision type so direct_result / validator_executor can admit plan-backed
results without TDX quote verification. Never claims TEE / independent
verification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AttestationOutcome(StrEnum):
    """Admission outcome labels (host-trust; names kept for wire/DB compat)."""

    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AttestationDecision:
    outcome: AttestationOutcome
    reason: str | None = None

    @classmethod
    def of(cls, outcome: AttestationOutcome, reason: str | None = None) -> AttestationDecision:
        return cls(outcome=outcome, reason=reason)

    @property
    def accepted(self) -> bool:
        return self.outcome is AttestationOutcome.VERIFIED


class ResultMeasurementAllowlist:
    """No-op allowlist stub — measurement allowlists removed with Phala."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def allows(self, *args: Any, **kwargs: Any) -> bool:
        return False


class AttestationGate:
    """Host-trust gate: admit plan-matching software checks only (no TDX).

    Quote verification always fails closed. Callers under unattested mode
    should use :meth:`decide_host_trust` instead of quote paths.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.quote_verifier = kwargs.get("quote_verifier")

    def decide(self, *args: Any, **kwargs: Any) -> AttestationDecision:
        return AttestationDecision.of(
            AttestationOutcome.VERIFICATION_FAILED,
            reason="phala_attestation_removed_use_host_trust",
        )

    def decide_eval_result(
        self,
        validated: Mapping[str, Any],
        *,
        eval_plan: Mapping[str, Any] | None = None,
        expected_agent_hash: str | None = None,
        nonce_outstanding: bool = False,
        key_granted: bool = False,
        endpoint_rebound: bool = False,
        rebound_worker_signature: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> AttestationDecision:
        """Host-trust admit: plan agent_hash match + outstanding nonce.

        Never verifies TDX quotes. Marks path as host-trust only.
        """

        if not nonce_outstanding:
            return AttestationDecision.of(
                AttestationOutcome.VERIFICATION_FAILED,
                reason="score_nonce_not_outstanding",
            )
        plan = eval_plan or {}
        plan_hash = str(plan.get("agent_hash") or "").strip()
        expected = (expected_agent_hash or plan_hash).strip()
        got = ""
        if isinstance(validated, Mapping):
            proof = validated.get("execution_proof")
            if isinstance(proof, Mapping):
                got = str(proof.get("agent_hash") or "").strip()
            if not got:
                got = str(validated.get("agent_hash") or "").strip()
        if expected and got and expected != got:
            return AttestationDecision.of(
                AttestationOutcome.VERIFICATION_FAILED,
                reason="agent_hash_mismatch",
            )
        # Host-trust software admit (unattested). Callers must mark envelopes.
        return AttestationDecision.of(
            AttestationOutcome.VERIFIED,
            reason="host_trust_plan_admit_unattested",
        )

    def decide_host_trust(self, *args: Any, **kwargs: Any) -> AttestationDecision:
        return self.decide_eval_result(*args, **kwargs)


def execution_proof_signing_payload(*, manifest_sha256: str, unit_id: str) -> bytes:
    """Canonical bytes signed by the endpoint worker for result rebound."""

    man = (manifest_sha256 or "").strip().lower()
    uid = (unit_id or "").strip()
    return f"agent-challenge-exec-proof-v1|{man}|{uid}".encode()


def verify_worker_signature(*args: Any, **kwargs: Any) -> bool:
    """Signature verify stub — TEE worker signatures not product path after T40."""

    return False


def failclosed_gate() -> AttestationGate:
    """Return a gate that never verifies TEE quotes (host-trust only)."""

    return AttestationGate()


__all__ = [
    "AttestationDecision",
    "AttestationGate",
    "AttestationOutcome",
    "ResultMeasurementAllowlist",
    "execution_proof_signing_payload",
    "failclosed_gate",
    "verify_worker_signature",
]
