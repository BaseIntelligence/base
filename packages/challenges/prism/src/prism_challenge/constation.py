"""Sole elevation predicate for Prism image-constation (M14).

``constation_ok(bundle)`` is the **only** function in Prism that may authorize
tier elevation from a constation bundle. No other module, path, or helper may
grant a tier from attestation, allowlist, nonce, signature, or Lium data alone
(M14). Callers that need an effective tier must route through this predicate
(wiring lands in a later todo — this module deliberately exposes **no**
``effective_tier`` / ``grant_tier`` API).

Honesty constraints (binding):

* A True result is **tamper-evidence**, not tamper-prevention. The miner rents
  and controls the pod.
* Corroboration (Lium-declared digest vs sidecar digest) is a **negative-only**
  signal: mismatch fails; agreement alone never elevates.
* Signature validity proves only that an entity holding the in-image secret
  responded — never sufficient alone (B3).
* Dependencies are injected (allowlist / nonce / signature checkers) so this
  module stays pure and testable without live Lium.

Required conjunction (all must pass):

1. Allowlist hit for ``(commit_sha, tree_sha, variant, digest)``
2. Nonce valid (BASE-issued, unexpired, single-use, binding match)
3. Attestation signature valid
4. Sealed manifest hashes match (expected vs reported)
5. Corroboration not contradicting (negative only)
6. No constation gap beyond budget
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

# Allowlist miss reason strings — must match base.compute.digest_allowlist.AllowlistMissReason
_ALLOWLIST_UNKNOWN: Final[str] = "unknown_digest"
_ALLOWLIST_VARIANT: Final[str] = "variant_mismatch"
_ALLOWLIST_COMMIT: Final[str] = "commit_mismatch"
_ALLOWLIST_REVOKED: Final[str] = "revoked"

# Nonce miss reason strings — must match base.compute.attestation_nonce.NonceConsumeReason
_NONCE_ALREADY: Final[str] = "already_consumed"
_NONCE_UNKNOWN: Final[str] = "unknown_nonce"
_NONCE_EXPIRED: Final[str] = "expired"
_NONCE_WORK_UNIT: Final[str] = "work_unit_mismatch"
_NONCE_HOTKEY: Final[str] = "miner_hotkey_mismatch"
_NONCE_POD: Final[str] = "pod_mismatch"


class ConstationFailReason(StrEnum):
    """Machine-consumed outcome codes for ``constation_ok``.

    Callers must not string-match prose. Allowlist/nonce detail codes mirror
    base miss enums so adapters can pass reasons through without remapping
    tables beyond the thin maps below.
    """

    OK = "ok"
    ALLOWLIST_UNKNOWN_DIGEST = "allowlist_unknown_digest"
    ALLOWLIST_VARIANT_MISMATCH = "allowlist_variant_mismatch"
    ALLOWLIST_COMMIT_MISMATCH = "allowlist_commit_mismatch"
    ALLOWLIST_REVOKED = "allowlist_revoked"
    ALLOWLIST_FAILED = "allowlist_failed"
    NONCE_ALREADY_CONSUMED = "nonce_already_consumed"
    NONCE_UNKNOWN = "nonce_unknown"
    NONCE_EXPIRED = "nonce_expired"
    NONCE_WORK_UNIT_MISMATCH = "nonce_work_unit_mismatch"
    NONCE_MINER_HOTKEY_MISMATCH = "nonce_miner_hotkey_mismatch"
    NONCE_POD_MISMATCH = "nonce_pod_mismatch"
    NONCE_FAILED = "nonce_failed"
    SIGNATURE_INVALID = "signature_invalid"
    SEALED_MANIFEST_MISMATCH = "sealed_manifest_mismatch"
    CORROBORATION_MISMATCH = "corroboration_mismatch"
    CONSTATION_GAP = "constation_gap"


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """Structured result from an injected checker (allowlist / nonce / signature).

    ``reason`` is a machine code. For allowlist/nonce, prefer the base miss
    reason value (e.g. ``unknown_digest``); ``constation_ok`` maps it onto
    :class:`ConstationFailReason`.
    """

    ok: bool
    reason: str = "ok"


@dataclass(frozen=True, slots=True)
class ConstationBundle:
    """Inputs required to evaluate the six-mechanism elevation conjunction.

    Field meanings (tamper-evidence only):

    * Identity: ``commit_sha``, ``tree_sha``, ``variant``, ``digest``
    * Nonce binding: ``work_unit_id``, ``miner_hotkey``, ``pod_id``, ``nonce``
    * ``signed_attestation`` — opaque object passed to ``verify_signature``
    * Sealed surface: ``expected_sealed_manifest_hashes`` (build-time) vs
      ``reported_sealed_manifest_hashes`` (sidecar self-measure)
    * ``lium_declared_digest`` — same-account corroboration channel (optional;
      absence is not a contradiction; mismatch is fatal)
    * Gap budget: ``constation_gap_budget_seconds`` vs
      ``constation_observed_max_gap_seconds``
    """

    commit_sha: str
    tree_sha: str
    variant: str
    digest: str
    work_unit_id: str
    miner_hotkey: str
    pod_id: str
    nonce: str
    signed_attestation: object
    expected_sealed_manifest_hashes: Mapping[str, str]
    reported_sealed_manifest_hashes: Mapping[str, str]
    lium_declared_digest: str | None
    constation_gap_budget_seconds: float
    constation_observed_max_gap_seconds: float


@dataclass(frozen=True, slots=True)
class ConstationResult:
    """Structured predicate outcome. ``ok`` never implies hardware trust."""

    ok: bool
    reason: ConstationFailReason
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.ok


class AllowlistChecker(Protocol):
    """Protocol matching ``DigestAllowlist.lookup`` miss-reason surface."""

    def __call__(
        self,
        *,
        digest: str,
        commit_sha: str,
        tree_sha: str,
        variant: str,
    ) -> CheckOutcome: ...


class NonceChecker(Protocol):
    """Protocol matching single-use nonce consume surface."""

    def __call__(
        self,
        *,
        nonce: str,
        work_unit_id: str,
        miner_hotkey: str,
        pod_id: str,
    ) -> CheckOutcome: ...


class SignatureVerifier(Protocol):
    """Protocol matching attestation payload verify surface."""

    def __call__(self, signed: object) -> CheckOutcome: ...


_ALLOWLIST_MAP: Final[dict[str, ConstationFailReason]] = {
    _ALLOWLIST_UNKNOWN: ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST,
    _ALLOWLIST_VARIANT: ConstationFailReason.ALLOWLIST_VARIANT_MISMATCH,
    _ALLOWLIST_COMMIT: ConstationFailReason.ALLOWLIST_COMMIT_MISMATCH,
    _ALLOWLIST_REVOKED: ConstationFailReason.ALLOWLIST_REVOKED,
}

_NONCE_MAP: Final[dict[str, ConstationFailReason]] = {
    _NONCE_ALREADY: ConstationFailReason.NONCE_ALREADY_CONSUMED,
    _NONCE_UNKNOWN: ConstationFailReason.NONCE_UNKNOWN,
    _NONCE_EXPIRED: ConstationFailReason.NONCE_EXPIRED,
    _NONCE_WORK_UNIT: ConstationFailReason.NONCE_WORK_UNIT_MISMATCH,
    _NONCE_HOTKEY: ConstationFailReason.NONCE_MINER_HOTKEY_MISMATCH,
    _NONCE_POD: ConstationFailReason.NONCE_POD_MISMATCH,
}


def constation_ok(
    bundle: ConstationBundle,
    *,
    check_allowlist: AllowlistChecker,
    check_nonce: NonceChecker,
    verify_signature: SignatureVerifier,
) -> ConstationResult:
    """Return whether ``bundle`` satisfies every elevation requirement.

    **M14 — sole elevation predicate.** No other Prism path may grant a tier.
    This function returns a structured :class:`ConstationResult`; it does not
    itself assign tiers. Downstream ``effective_tier`` wiring (separate todo)
    must call this predicate and must not bypass it.

    Check order (first failure wins, distinct reason each):

    1. allowlist hit
    2. nonce valid
    3. signature valid
    4. sealed manifest hashes match
    5. corroboration not contradicting (negative only)
    6. constation gap within budget

    All checkers are injected so unit tests never need live Lium, BASE DB, or
    network I/O.
    """
    allow = check_allowlist(
        digest=bundle.digest,
        commit_sha=bundle.commit_sha,
        tree_sha=bundle.tree_sha,
        variant=bundle.variant,
    )
    if not allow.ok:
        mapped = _ALLOWLIST_MAP.get(allow.reason, ConstationFailReason.ALLOWLIST_FAILED)
        return ConstationResult(ok=False, reason=mapped, detail=allow.reason)

    nonce = check_nonce(
        nonce=bundle.nonce,
        work_unit_id=bundle.work_unit_id,
        miner_hotkey=bundle.miner_hotkey,
        pod_id=bundle.pod_id,
    )
    if not nonce.ok:
        mapped = _NONCE_MAP.get(nonce.reason, ConstationFailReason.NONCE_FAILED)
        return ConstationResult(ok=False, reason=mapped, detail=nonce.reason)

    sig = verify_signature(bundle.signed_attestation)
    if not sig.ok:
        return ConstationResult(
            ok=False,
            reason=ConstationFailReason.SIGNATURE_INVALID,
            detail=sig.reason,
        )

    if not _manifests_match(
        bundle.expected_sealed_manifest_hashes,
        bundle.reported_sealed_manifest_hashes,
    ):
        return ConstationResult(
            ok=False,
            reason=ConstationFailReason.SEALED_MANIFEST_MISMATCH,
        )

    # Corroboration is negative-only: mismatch fails; absence or agreement
    # does not grant elevation by itself (other checks already required).
    if bundle.lium_declared_digest is not None:
        declared = bundle.lium_declared_digest.strip().lower()
        sidecar = bundle.digest.strip().lower()
        if declared != sidecar:
            return ConstationResult(
                ok=False,
                reason=ConstationFailReason.CORROBORATION_MISMATCH,
                detail=f"lium={declared} sidecar={sidecar}",
            )

    if bundle.constation_observed_max_gap_seconds > bundle.constation_gap_budget_seconds:
        return ConstationResult(
            ok=False,
            reason=ConstationFailReason.CONSTATION_GAP,
            detail=(
                f"observed={bundle.constation_observed_max_gap_seconds}"
                f" budget={bundle.constation_gap_budget_seconds}"
            ),
        )

    return ConstationResult(ok=True, reason=ConstationFailReason.OK)


def _manifests_match(
    expected: Mapping[str, str],
    reported: Mapping[str, str],
) -> bool:
    """Exact path→hash equality (order-insensitive). Empty expected is not a free pass."""
    if not expected:
        return False
    exp = {str(k): str(v).strip().lower() for k, v in expected.items()}
    rep = {str(k): str(v).strip().lower() for k, v in reported.items()}
    return exp == rep


def adapt_allowlist_lookup(lookup_result: object) -> CheckOutcome:
    """Thin adapter: base ``AllowlistHit`` / ``AllowlistMiss`` → :class:`CheckOutcome`.

    Accepts any object with optional ``reason`` attribute (miss) or hit without
    a failing reason. Does not import base so prism stays runnable against older
    wheels; pass the live lookup result from ``DigestAllowlist.lookup``.
    """
    reason_obj = getattr(lookup_result, "reason", None)
    if reason_obj is None:
        # Hit (or unknown success shape)
        return CheckOutcome(ok=True, reason="ok")
    reason = getattr(reason_obj, "value", None) or str(reason_obj)
    return CheckOutcome(ok=False, reason=str(reason))


def adapt_nonce_consume(consume_result: object) -> CheckOutcome:
    """Thin adapter: base ``NonceConsumeHit`` / ``NonceConsumeMiss`` → CheckOutcome."""
    reason_obj = getattr(consume_result, "reason", None)
    if reason_obj is None:
        return CheckOutcome(ok=True, reason="ok")
    reason = getattr(reason_obj, "value", None) or str(reason_obj)
    return CheckOutcome(ok=False, reason=str(reason))


def adapt_attestation_verify(verify_result: object) -> CheckOutcome:
    """Thin adapter: base ``AttestationVerifyResult`` → CheckOutcome."""
    ok = bool(getattr(verify_result, "ok", False))
    reason_obj = getattr(verify_result, "reason", "ok")
    reason = getattr(reason_obj, "value", None) or str(reason_obj)
    return CheckOutcome(ok=ok, reason=str(reason))


# --- Fault attribution (todo 22) ---------------------------------------------------------------

#: ConstationFailReason values that are miner-attributable.
_MINER_FAIL_REASONS: Final[frozenset[ConstationFailReason]] = frozenset(
    {
        ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST,
        ConstationFailReason.ALLOWLIST_VARIANT_MISMATCH,
        ConstationFailReason.ALLOWLIST_COMMIT_MISMATCH,
        ConstationFailReason.ALLOWLIST_REVOKED,
        ConstationFailReason.ALLOWLIST_FAILED,
        ConstationFailReason.NONCE_ALREADY_CONSUMED,
        ConstationFailReason.NONCE_UNKNOWN,
        ConstationFailReason.NONCE_EXPIRED,
        ConstationFailReason.NONCE_WORK_UNIT_MISMATCH,
        ConstationFailReason.NONCE_MINER_HOTKEY_MISMATCH,
        ConstationFailReason.NONCE_POD_MISMATCH,
        ConstationFailReason.NONCE_FAILED,
        ConstationFailReason.SIGNATURE_INVALID,
        ConstationFailReason.SEALED_MANIFEST_MISMATCH,
        ConstationFailReason.CORROBORATION_MISMATCH,
        ConstationFailReason.CONSTATION_GAP,  # sidecar gap is miner-side
    }
)

#: Stable bare codes for infra faults (no ConstationFailReason enum value).
INFRA_FAULT_CONSTATION_UNAVAILABLE = "constation_unavailable"
INFRA_FAULT_LIUM_5XX = "lium_5xx"
INFRA_FAULT_NETWORK_PARTITION = "network_partition"
INFRA_FAULT_RETRY_EXHAUSTED = "constation_retry_exhausted"

#: Map fail reasons → bare miner_fault codes used in ingestion reason strings.
_MINER_BARE_CODES: Final[dict[ConstationFailReason, str]] = {
    ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST: "unknown_digest",
    ConstationFailReason.ALLOWLIST_VARIANT_MISMATCH: "variant_mismatch",
    ConstationFailReason.ALLOWLIST_COMMIT_MISMATCH: "commit_mismatch",
    ConstationFailReason.ALLOWLIST_REVOKED: "revoked_digest",
    ConstationFailReason.ALLOWLIST_FAILED: "allowlist_failed",
    ConstationFailReason.NONCE_ALREADY_CONSUMED: "replayed_nonce",
    ConstationFailReason.NONCE_UNKNOWN: "unknown_nonce",
    ConstationFailReason.NONCE_EXPIRED: "expired_nonce",
    ConstationFailReason.NONCE_WORK_UNIT_MISMATCH: "nonce_work_unit_mismatch",
    ConstationFailReason.NONCE_MINER_HOTKEY_MISMATCH: "nonce_hotkey_mismatch",
    ConstationFailReason.NONCE_POD_MISMATCH: "nonce_pod_mismatch",
    ConstationFailReason.NONCE_FAILED: "nonce_failed",
    ConstationFailReason.SIGNATURE_INVALID: "signature_invalid",
    ConstationFailReason.SEALED_MANIFEST_MISMATCH: "manifest_mismatch",
    ConstationFailReason.CORROBORATION_MISMATCH: "corroboration_mismatch",
    ConstationFailReason.CONSTATION_GAP: "constation_gap",
}


def classify_constation_fault(result: ConstationResult) -> str:
    """Return ``miner_fault:<code>`` or ``infra_fault:<code>`` for a failed result.

    A successful result returns ``ok``. Missing-bundle / service-down cases are
    classified by callers via :func:`infra_fault_reason` / :func:`miner_fault_reason`.
    """
    if result.ok:
        return "ok"
    bare = _MINER_BARE_CODES.get(result.reason, str(result.reason.value))
    if result.reason in _MINER_FAIL_REASONS:
        return f"miner_fault:{bare}"
    return f"infra_fault:{bare}"


def miner_fault_reason(code: str) -> str:
    bare = code.removeprefix("miner_fault:").strip() or "unknown"
    return f"miner_fault:{bare}"


def infra_fault_reason(code: str) -> str:
    bare = code.removeprefix("infra_fault:").strip() or "unknown"
    return f"infra_fault:{bare}"


def constation_bundle_to_dict(bundle: ConstationBundle) -> dict[str, object]:
    """Serialize a :class:`ConstationBundle` for wire / result payload embedding."""
    return {
        "commit_sha": bundle.commit_sha,
        "tree_sha": bundle.tree_sha,
        "variant": bundle.variant,
        "digest": bundle.digest,
        "work_unit_id": bundle.work_unit_id,
        "miner_hotkey": bundle.miner_hotkey,
        "pod_id": bundle.pod_id,
        "nonce": bundle.nonce,
        "signed_attestation": bundle.signed_attestation,
        "expected_sealed_manifest_hashes": dict(bundle.expected_sealed_manifest_hashes),
        "reported_sealed_manifest_hashes": dict(bundle.reported_sealed_manifest_hashes),
        "lium_declared_digest": bundle.lium_declared_digest,
        "constation_gap_budget_seconds": float(bundle.constation_gap_budget_seconds),
        "constation_observed_max_gap_seconds": float(bundle.constation_observed_max_gap_seconds),
    }


def constation_bundle_from_dict(raw: object) -> ConstationBundle:
    """Parse a wire dict into :class:`ConstationBundle` (boundary validation)."""
    if not isinstance(raw, dict):
        raise ValueError("constation_bundle must be an object")
    required = (
        "commit_sha",
        "tree_sha",
        "variant",
        "digest",
        "work_unit_id",
        "miner_hotkey",
        "pod_id",
        "nonce",
        "signed_attestation",
        "expected_sealed_manifest_hashes",
        "reported_sealed_manifest_hashes",
        "constation_gap_budget_seconds",
        "constation_observed_max_gap_seconds",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"constation_bundle missing fields: {missing}")
    exp = raw["expected_sealed_manifest_hashes"]
    rep = raw["reported_sealed_manifest_hashes"]
    if not isinstance(exp, Mapping) or not isinstance(rep, Mapping):
        raise ValueError("sealed manifest hashes must be objects")
    lium = raw.get("lium_declared_digest")
    if lium is not None and not isinstance(lium, str):
        raise ValueError("lium_declared_digest must be string or null")
    return ConstationBundle(
        commit_sha=str(raw["commit_sha"]),
        tree_sha=str(raw["tree_sha"]),
        variant=str(raw["variant"]),
        digest=str(raw["digest"]),
        work_unit_id=str(raw["work_unit_id"]),
        miner_hotkey=str(raw["miner_hotkey"]),
        pod_id=str(raw["pod_id"]),
        nonce=str(raw["nonce"]),
        signed_attestation=raw["signed_attestation"],
        expected_sealed_manifest_hashes={str(k): str(v) for k, v in exp.items()},
        reported_sealed_manifest_hashes={str(k): str(v) for k, v in rep.items()},
        lium_declared_digest=lium,
        constation_gap_budget_seconds=float(raw["constation_gap_budget_seconds"]),
        constation_observed_max_gap_seconds=float(raw["constation_observed_max_gap_seconds"]),
    )


__all__ = [
    "AllowlistChecker",
    "CheckOutcome",
    "ConstationBundle",
    "ConstationFailReason",
    "ConstationResult",
    "NonceChecker",
    "SignatureVerifier",
    "adapt_allowlist_lookup",
    "adapt_attestation_verify",
    "adapt_nonce_consume",
    "constation_ok",
    "classify_constation_fault",
    "miner_fault_reason",
    "infra_fault_reason",
    "INFRA_FAULT_CONSTATION_UNAVAILABLE",
    "INFRA_FAULT_LIUM_5XX",
    "INFRA_FAULT_NETWORK_PARTITION",
    "INFRA_FAULT_RETRY_EXHAUSTED",
    "constation_bundle_to_dict",
    "constation_bundle_from_dict",
]
