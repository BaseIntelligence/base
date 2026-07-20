"""Challenge-side acceptance gate for Phala-attested task results (M4).

When the Phala attestation flag is ON the decentralized validator must not write
a task's score unless that task's result carries a Phala-tier attestation the
validator can VERIFY (architecture.md sec 4 C4). This module is the
self-contained, validator-owned verifier + decision layer the acceptance gate in
:mod:`agent_challenge.evaluation.validator_executor` consults:

* :func:`extract_attestation_envelope` pulls the additive ``execution_proof`` +
  ``attestation_binding`` blocks off a task's ``BASE_BENCHMARK_RESULT=`` line
  (emitted by :mod:`agent_challenge.canonical.attested_result`), or reports the
  result unattested.
* :class:`AttestationGate` verifies that envelope against the VALIDATOR's own
  expectations -- fail-closed and conjunctive, mirroring base's Phala-tier
  ``verify_execution_proof``: the TDX quote must be DCAP-valid with an acceptable
  TCB, its event log must replay to the quote's signed RTMR3 yielding the
  canonical ``compose_hash``, the reconstructed measurement must be in the
  validator allowlist, ``report_data`` must equal the architecture-sec-6 binding
  over ``(measurement, submission agent_hash, task_ids, scores_digest, nonce)``,
  and the nonce must be a fresh, validator-issued, unconsumed one.
* The gate returns an :class:`AttestationDecision` carrying a distinguishable,
  retrievable reason so a non-accepted result is observable (unattested vs
  verification-failed vs verifier-unavailable/retryable), never a silent no-op.

The trust root is the hardware-signed quote: the measurement and ``report_data``
are read from the parsed TD report (not the untrusted attestation block). base's
models are not importable in this repo's venv, so the derivation is the shared
single-source :mod:`agent_challenge.canonical.report_data` (byte-identical to
base's ``phala_report_data``) and the quote primitives are agent-challenge's own
:mod:`agent_challenge.keyrelease.quote`.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from agent_challenge.canonical import report_data as rd
from agent_challenge.canonical.attested_result import (
    ATTESTATION_BINDING_RESULT_KEY,
    EXECUTION_PROOF_RESULT_KEY,
    PHALA_TDX_TIER,
)
from agent_challenge.canonical.measurement import CANONICAL_MEASUREMENT_FIELDS
from agent_challenge.evaluation.own_runner.result_schema import RESULT_LINE_PREFIX
from agent_challenge.keyrelease.nonce import NonceState
from agent_challenge.keyrelease.quote import (
    QuoteStructureError,
    QuoteVerificationError,
    QuoteVerifier,
    QuoteVerifierUnavailable,
    decode_key_provider,
    os_image_hash_from_registers,
    parse_td_report,
    parse_tdx_quote_v4,
    replay_rtmr3,
    validate_rtmr3_event_log,
)

#: TCB statuses the acceptance gate accepts by default (architecture sec 7).
ACCEPTABLE_TCB_DEFAULT: frozenset[str] = frozenset({"UpToDate"})

# --------------------------------------------------------------------------- #
# Retrievable, distinguishable acceptance-outcome reasons (VAL-VERIFY-026).
# --------------------------------------------------------------------------- #
#: The result carried no (Phala-tier) attestation at all.
ATTESTATION_MISSING = "attestation_missing"
#: An attestation was present but its verification failed (permanent for this
#: result: forged/invalid quote, bad TCB, non-allowlisted measurement, bad
#: event-log replay, mis-bound report_data, or stale/unknown nonce).
ATTESTATION_VERIFICATION_FAILED = "attestation_verification_failed"
#: The quote-verification dependency was transiently unavailable/timed out; the
#: result is PARKED (retryable), neither accepted nor permanently rejected.
ATTESTATION_VERIFIER_UNAVAILABLE = "attestation_verifier_unavailable"


class AttestationVerifierUnavailable(Exception):
    """Signals a transient quote-verifier outage so the result is parked (retryable).

    A :class:`~agent_challenge.keyrelease.quote.QuoteVerifier` raises this (rather
    than :class:`~agent_challenge.keyrelease.quote.QuoteVerificationError`) when it
    cannot reach its collateral/dependency, distinguishing a "cannot verify right
    now" outage from a "quote is invalid" cryptographic rejection.
    """


class AttestationOutcome(Enum):
    """The gate's verdict for a task result under the Phala flag."""

    #: Attestation present and fully verified -> the score may be written.
    VERIFIED = "verified"
    #: No Phala-tier attestation on the result -> reject/park (not scored).
    UNATTESTED = "unattested"
    #: Attestation present but verification failed -> reject/park (not scored).
    VERIFICATION_FAILED = "verification_failed"
    #: Quote verifier transiently unavailable -> park (retryable, not scored).
    VERIFIER_UNAVAILABLE = "verifier_unavailable"


_REASON_FOR: dict[AttestationOutcome, str | None] = {
    AttestationOutcome.VERIFIED: None,
    AttestationOutcome.UNATTESTED: ATTESTATION_MISSING,
    AttestationOutcome.VERIFICATION_FAILED: ATTESTATION_VERIFICATION_FAILED,
    AttestationOutcome.VERIFIER_UNAVAILABLE: ATTESTATION_VERIFIER_UNAVAILABLE,
}


@dataclass(frozen=True)
class AttestationDecision:
    """The gate's decision: the outcome and a retrievable, distinguishable reason."""

    outcome: AttestationOutcome
    reason: str | None

    @property
    def accepted(self) -> bool:
        """Whether the result's attestation verified (its score may be written)."""

        return self.outcome is AttestationOutcome.VERIFIED

    @property
    def retryable(self) -> bool:
        """Whether the non-acceptance is a transient (retryable) park, not permanent."""

        return self.outcome is AttestationOutcome.VERIFIER_UNAVAILABLE

    @classmethod
    def of(cls, outcome: AttestationOutcome) -> AttestationDecision:
        return cls(outcome=outcome, reason=_REASON_FOR[outcome])


# --------------------------------------------------------------------------- #
# Validator-owned measurement allowlist (canonical 6-register subset).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResultMeasurementAllowlist:
    """A validator-owned set of canonical measurements a result quote must match.

    Matching is exact across ALL canonical registers
    (``mrtd, rtmr0, rtmr1, rtmr2, compose_hash, os_image_hash``; ``rtmr3`` is
    runtime and excluded). An EMPTY allowlist matches nothing (fail closed) -- an
    unconfigured validator never accepts a quote, never accept-any.
    """

    entries: tuple[dict[str, str], ...] = ()

    @classmethod
    def from_measurements(
        cls, measurements: Iterable[Mapping[str, Any]]
    ) -> ResultMeasurementAllowlist:
        return cls(tuple(_canonical_measurement_mapping(m) for m in measurements))

    def __bool__(self) -> bool:
        return bool(self.entries)

    def contains(self, measurement: Mapping[str, Any]) -> bool:
        candidate = _canonical_measurement_mapping(measurement)
        return any(candidate == entry for entry in self.entries)


def _canonical_measurement_mapping(measurement: Mapping[str, Any]) -> dict[str, str]:
    return {field: str(measurement[field]) for field in CANONICAL_MEASUREMENT_FIELDS}


# --------------------------------------------------------------------------- #
# Validator-issued, single-use, TTL-bounded nonce ledger.
# --------------------------------------------------------------------------- #
@runtime_checkable
class NonceConsumer(Protocol):
    """Consumes a validator-issued nonce, reporting its freshness state.

    ``consume`` is single-use: the first consume of a known, unexpired nonce
    returns :attr:`~agent_challenge.keyrelease.nonce.NonceState.OK`; any later
    consume returns :attr:`~agent_challenge.keyrelease.nonce.NonceState.CONSUMED`.
    """

    def consume(self, nonce: str) -> NonceState:  # pragma: no cover - protocol
        ...


def execution_proof_signing_payload(*, manifest_sha256: str, unit_id: str) -> bytes:
    """Return the pinned tier-0 worker signature payload."""

    return hashlib.sha256(f"{manifest_sha256}:{unit_id}".encode()).digest()


def verify_worker_signature(
    worker_pubkey: str,
    payload: bytes,
    signature: str,
) -> bool:
    """Verify the endpoint-owned tier-0 signature without trusting caller state."""

    try:
        import bittensor as bt

        return bool(bt.Keypair(ss58_address=worker_pubkey).verify(payload, signature))
    except Exception:  # noqa: BLE001 - malformed/unavailable signer fails closed
        return False


@dataclass
class InMemoryNonceLedger:
    """A validator-local, single-use, TTL-bounded :class:`NonceConsumer`.

    An empty ledger consumes nothing (every nonce is UNKNOWN), so an unconfigured
    validator fails closed. ``clock`` is injectable for deterministic expiry.
    """

    ttl_seconds: float = 120.0
    clock: Callable[[], float] = time.monotonic
    _issued: dict[str, float] = field(default_factory=dict)
    _consumed: set[str] = field(default_factory=set)

    def issue(self, nonce: str | None = None) -> str:
        value = nonce if nonce is not None else secrets.token_urlsafe(32)
        self._issued[value] = self.clock()
        return value

    def is_outstanding(self, nonce: str) -> bool:
        if not nonce or nonce in self._consumed or nonce not in self._issued:
            return False
        return (self.clock() - self._issued[nonce]) <= self.ttl_seconds

    def consume(self, nonce: str) -> NonceState:
        if not nonce or nonce not in self._issued:
            return NonceState.UNKNOWN
        if nonce in self._consumed:
            return NonceState.CONSUMED
        if (self.clock() - self._issued[nonce]) > self.ttl_seconds:
            del self._issued[nonce]
            self._consumed.add(nonce)
            return NonceState.EXPIRED
        del self._issued[nonce]
        self._consumed.add(nonce)
        return NonceState.OK


# --------------------------------------------------------------------------- #
# Envelope extraction off the BASE_BENCHMARK_RESULT= line.
# --------------------------------------------------------------------------- #
def parse_benchmark_result_payload(stdout: str) -> dict[str, Any] | None:
    """Return the parsed ``BASE_BENCHMARK_RESULT=`` JSON payload, or ``None``.

    Scans (last-wins, matching the runner's own parser) for the single result
    line and parses its JSON object. Returns ``None`` when no parseable result
    line is present.
    """

    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_LINE_PREFIX):
            try:
                parsed = json.loads(line[len(RESULT_LINE_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def extract_attestation_envelope(
    stdout: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Extract ``(execution_proof, attestation_binding)`` from a result's stdout.

    Returns ``None`` when the result carries no Phala-tier attestation (no result
    line, no ``execution_proof`` block, a non-Phala tier, or a missing binding),
    so the caller treats it as unattested. Only a well-formed Phala-tier envelope
    with both additive blocks is returned for verification.
    """

    payload = parse_benchmark_result_payload(stdout)
    if payload is None:
        return None
    execution_proof = payload.get(EXECUTION_PROOF_RESULT_KEY)
    binding = payload.get(ATTESTATION_BINDING_RESULT_KEY)
    if not isinstance(execution_proof, Mapping) or not isinstance(binding, Mapping):
        return None
    if execution_proof.get("tier") != PHALA_TDX_TIER:
        return None
    return dict(execution_proof), dict(binding)


# --------------------------------------------------------------------------- #
# The acceptance gate.
# --------------------------------------------------------------------------- #
@dataclass
class AttestationGate:
    """Verifies a task result's Phala attestation against validator expectations.

    Fail-closed and conjunctive: any missing dependency (no quote verifier, empty
    allowlist, no nonce ledger) or any failing check yields a non-accepting
    decision, so an unconfigured or misconfigured validator never accepts an
    attested-looking result. A transient quote-verifier outage parks (retryable).
    """

    quote_verifier: QuoteVerifier | None = None
    allowlist: ResultMeasurementAllowlist = field(default_factory=ResultMeasurementAllowlist)
    nonce_validator: NonceConsumer | None = None
    acceptable_tcb: frozenset[str] = ACCEPTABLE_TCB_DEFAULT

    def decide(self, stdout: str, *, expected_agent_hash: str) -> AttestationDecision:
        """Decide whether ``stdout``'s attested result may be written as a score."""

        envelope = extract_attestation_envelope(stdout)
        if envelope is None:
            return AttestationDecision.of(AttestationOutcome.UNATTESTED)
        execution_proof, binding = envelope
        outcome = self._verify(execution_proof, binding, expected_agent_hash)
        return AttestationDecision.of(outcome)

    def decide_eval_result(
        self,
        result_request: Mapping[str, Any],
        *,
        eval_plan: Mapping[str, Any],
        expected_agent_hash: str,
        nonce_outstanding: bool,
        key_granted: bool,
        consume_nonce: bool = False,
        endpoint_rebound: bool = False,
        rebound_worker_signature: Mapping[str, Any] | None = None,
    ) -> AttestationDecision:
        """Verify one schema-v1 direct Eval result against immutable plan state.

        This is the production-result counterpart to :meth:`decide`.  It never
        derives expected identity from the submitted proof and does not use the
        legacy stdout/binding envelope.  Database nonce consumption is normally
        performed by the surrounding transaction after this method returns
        ``VERIFIED``.  ``consume_nonce`` remains available for offline
        single-use ledgers and is deliberately false for the HTTP route so the
        receipt, nonce, and score transaction has one owner.
        """

        if endpoint_rebound:
            proof = result_request.get("execution_proof")
            if not isinstance(proof, Mapping):
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
            signature = proof.get("worker_signature")
            if not isinstance(signature, Mapping) or signature != {
                "worker_pubkey": "",
                "sig": "",
            }:
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
            if not isinstance(rebound_worker_signature, Mapping):
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
            rebound_pubkey = rebound_worker_signature.get("worker_pubkey")
            rebound_sig = rebound_worker_signature.get("sig")
            if (
                not isinstance(rebound_pubkey, str)
                or not rebound_pubkey
                or not isinstance(rebound_sig, str)
                or not rebound_sig
            ):
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        try:
            from agent_challenge.evaluation.plan_scoring import validate_eval_result_from_plan

            request = validate_eval_result_from_plan(eval_plan, result_request)
        except (ValueError, KeyError, TypeError):
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)

        if request["agent_hash"] != expected_agent_hash or not key_granted:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        proof = request["execution_proof"]
        if endpoint_rebound:
            signature = {
                "worker_pubkey": str(rebound_worker_signature["worker_pubkey"]),
                "sig": str(rebound_worker_signature["sig"]),
            }
            proof = dict(proof)
            proof["worker_signature"] = signature
            request = dict(request)
            request["execution_proof"] = proof
            payload = execution_proof_signing_payload(
                manifest_sha256=proof["manifest_sha256"],
                unit_id=eval_plan["eval_run_id"],
            )
            if not verify_worker_signature(signature["worker_pubkey"], payload, signature["sig"]):
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        attestation = proof["attestation"]
        quote = attestation["tdx_quote"]

        try:
            report = parse_tdx_quote_v4(quote)
            validate_rtmr3_event_log(attestation["event_log"])
            verdict = self.quote_verifier.verify(quote) if self.quote_verifier else None
        except (QuoteVerifierUnavailable, AttestationVerifierUnavailable):
            return AttestationDecision.of(AttestationOutcome.VERIFIER_UNAVAILABLE)
        except (QuoteStructureError, QuoteVerificationError, ValueError, TypeError):
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        except Exception:  # noqa: BLE001 - unexpected verifier/backend failures are retryable
            return AttestationDecision.of(AttestationOutcome.VERIFIER_UNAVAILABLE)

        if verdict is None or verdict.tcb_status not in self.acceptable_tcb:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)

        try:
            replay = replay_rtmr3(attestation["event_log"])
        except (QuoteVerificationError, ValueError, TypeError):
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if replay.rtmr3 != report.rtmr3 or replay.compose_hash is None:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if attestation["measurement"]["rtmr3"] != report.rtmr3:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if attestation["measurement"]["compose_hash"] != replay.compose_hash:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)

        expected = eval_plan["eval_app"]["measurement"]
        measurement = {
            "mrtd": report.mrtd,
            "rtmr0": report.rtmr0,
            "rtmr1": report.rtmr1,
            "rtmr2": report.rtmr2,
            "compose_hash": replay.compose_hash,
            "os_image_hash": os_image_hash_from_registers(report.mrtd, report.rtmr1, report.rtmr2),
        }
        expected_measurement = {
            "mrtd": expected["mrtd"],
            "rtmr0": expected["rtmr0"],
            "rtmr1": expected["rtmr1"],
            "rtmr2": expected["rtmr2"],
            "compose_hash": eval_plan["eval_app"]["compose_hash"],
            "os_image_hash": expected["os_image_hash"],
        }
        if measurement != expected_measurement or not self.allowlist.contains(measurement):
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        # Live dstack RTMR3 key-provider events carry JSON (e.g. {"name":"kms",...})
        # as hex. Collapse onto the allowlist pin via the shared KR decoder so the
        # score gate matches host KR / plan pin ``phala`` (not raw hex equality).
        try:
            decoded_provider = decode_key_provider(replay.key_provider)
        except QuoteVerificationError:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if decoded_provider != str(expected["key_provider"]):
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        vm_image_hash = attestation["vm_config"]["os_image_hash"]
        if vm_image_hash is not None and vm_image_hash != expected_measurement["os_image_hash"]:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if report.report_data.hex() != attestation["report_data"]:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if not nonce_outstanding:
            return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        if consume_nonce:
            if self.nonce_validator is None:
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
            if self.nonce_validator.consume(eval_plan["score_nonce"]) is not NonceState.OK:
                return AttestationDecision.of(AttestationOutcome.VERIFICATION_FAILED)
        return AttestationDecision.of(AttestationOutcome.VERIFIED)

    def _verify(
        self,
        execution_proof: Mapping[str, Any],
        binding: Mapping[str, Any],
        expected_agent_hash: str,
    ) -> AttestationOutcome:
        attestation = execution_proof.get("attestation")
        if not isinstance(attestation, Mapping):
            return AttestationOutcome.VERIFICATION_FAILED

        quote = attestation.get("tdx_quote")
        event_log = attestation.get("event_log")
        if not isinstance(quote, str) or not quote or not isinstance(event_log, list):
            return AttestationOutcome.VERIFICATION_FAILED

        # -- the run identity the VALIDATOR expects (never trusted blindly) ---- #
        agent_hash = binding.get("agent_hash")
        task_ids = binding.get("task_ids")
        scores = binding.get("scores")
        scores_digest = binding.get("scores_digest")
        nonce = binding.get("validator_nonce")
        if (
            not isinstance(agent_hash, str)
            or not isinstance(task_ids, list)
            or not isinstance(scores, Mapping)
            or not isinstance(scores_digest, str)
            or not isinstance(nonce, str)
            or not nonce
        ):
            return AttestationOutcome.VERIFICATION_FAILED

        # scores_digest must be the digest of the scores the result reports, so a
        # score cannot be altered without breaking the report_data binding.
        if rd.scores_digest(scores) != scores_digest:
            return AttestationOutcome.VERIFICATION_FAILED
        # the attested agent must be the submission actually under evaluation.
        if agent_hash != expected_agent_hash:
            return AttestationOutcome.VERIFICATION_FAILED

        # dependencies missing -> fail closed (never accept-any).
        if self.quote_verifier is None or not self.allowlist or self.nonce_validator is None:
            return AttestationOutcome.VERIFICATION_FAILED

        try:
            report = parse_td_report(quote)
        except QuoteStructureError:
            return AttestationOutcome.VERIFICATION_FAILED

        # cryptographic signature/cert-chain + TCB posture (trust root).
        try:
            verdict = self.quote_verifier.verify(quote)
        except (AttestationVerifierUnavailable, QuoteVerifierUnavailable):
            return AttestationOutcome.VERIFIER_UNAVAILABLE
        except QuoteVerificationError:
            return AttestationOutcome.VERIFICATION_FAILED
        except Exception:  # noqa: BLE001 - any verifier error fails closed
            return AttestationOutcome.VERIFICATION_FAILED
        if verdict.tcb_status not in self.acceptable_tcb:
            return AttestationOutcome.VERIFICATION_FAILED

        # RTMR3 validated by content: replay the event log to the signed RTMR3.
        try:
            replay = replay_rtmr3(event_log)
        except QuoteVerificationError:
            return AttestationOutcome.VERIFICATION_FAILED
        if replay.rtmr3 != report.rtmr3 or replay.compose_hash is None:
            return AttestationOutcome.VERIFICATION_FAILED

        # measurement reconstructed from the SIGNED registers must be allowlisted.
        measurement = {
            "mrtd": report.mrtd,
            "rtmr0": report.rtmr0,
            "rtmr1": report.rtmr1,
            "rtmr2": report.rtmr2,
            "compose_hash": replay.compose_hash,
            "os_image_hash": os_image_hash_from_registers(report.mrtd, report.rtmr1, report.rtmr2),
        }
        if not self.allowlist.contains(measurement):
            return AttestationOutcome.VERIFICATION_FAILED

        # report_data must bind exactly (measurement, agent, task set, scores, nonce).
        expected_report_data = rd.report_data_hex(
            canonical_measurement=measurement,
            agent_hash=agent_hash,
            task_ids=task_ids,
            scores_digest=scores_digest,
            validator_nonce=nonce,
        )
        if report.report_data.hex() != expected_report_data:
            return AttestationOutcome.VERIFICATION_FAILED

        # nonce consumed LAST (single-use); a rejection above never burns it.
        if self.nonce_validator.consume(nonce) is not NonceState.OK:
            return AttestationOutcome.VERIFICATION_FAILED

        return AttestationOutcome.VERIFIED


def failclosed_gate() -> AttestationGate:
    """A default, fully fail-closed gate (accepts nothing).

    Used when the Phala flag is ON but no gate was injected: with no quote
    verifier, an empty allowlist, and an empty nonce ledger, every attested
    result is rejected -- an unconfigured validator never writes an unverified
    score.
    """

    return AttestationGate(
        quote_verifier=None,
        allowlist=ResultMeasurementAllowlist(),
        nonce_validator=InMemoryNonceLedger(),
    )


__all__ = [
    "ACCEPTABLE_TCB_DEFAULT",
    "ATTESTATION_MISSING",
    "ATTESTATION_VERIFICATION_FAILED",
    "ATTESTATION_VERIFIER_UNAVAILABLE",
    "AttestationDecision",
    "AttestationGate",
    "AttestationOutcome",
    "AttestationVerifierUnavailable",
    "InMemoryNonceLedger",
    "NonceConsumer",
    "ResultMeasurementAllowlist",
    "extract_attestation_envelope",
    "execution_proof_signing_payload",
    "failclosed_gate",
    "parse_benchmark_result_payload",
    "verify_worker_signature",
]
