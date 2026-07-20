"""Eval CVM launch gate: fresh re-verified review-domain attestation only.

Product freeze (library/ac-attestation.md, VAL-ACAT-010 / 028 / 029):

- Eval CVM / key-material readiness for scored spend launches **only** when a
  recent (≤24h) **re-verified** review-domain attestation admits with measured
  OpenRouter digests and bound ``allow`` (or a defined measured successor).
- Reject / stale / missing / wrong domain / failed re-verify must **not** start
  Eval CVM resources that would consume miner spend on the production path.
- Cached DB bits (``phase=review_allowed``, ``status=verified_allow``) alone are
  **insufficient** — launch re-runs ``admit_production_from_bound_outcome``
  (report_data + bound times + ≤24h + OR digests + decision).

Does **not** restore Base LLM gateway. Does **not** trust guest wall clock.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    ReviewOrOutcomeError,
    admit_production_from_bound_outcome,
)

# Stable refuse codes (library/ac-attestation.md)
REFUSE_EVAL_CVM = "eval_cvm_refused_no_fresh_review"
REFUSE_ATTESTATION_MISSING = "review_attestation_missing"
REFUSE_REVERIFY_FAILED = "review_attestation_reverify_failed"
REFUSE_STALE = "attestation_stale_over_24h"
REFUSE_WRONG_DOMAIN = "review_domain_mismatch"
REFUSE_CACHED_ALLOW_ONLY = "eval_cvm_refused_cached_allow_only"


class EvalCvmFreshReviewError(PermissionError):
    """Fail-closed Eval CVM launch refuse with a stable reason code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class EvalCvmFreshAdmission:
    """Decision for whether Eval CVM lifecycle / spend may start."""

    may_launch: bool
    reason_code: str
    review_digest: str | None = None
    reverify_exercised: bool = False
    production_status: str | None = None
    verdict: str | None = None
    bound_issued_at_ms: int | None = None
    bound_received_at_ms: int | None = None
    report_data_hex: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "may_launch": self.may_launch,
            "reason_code": self.reason_code,
            "review_digest": self.review_digest,
            "reverify_exercised": self.reverify_exercised,
            "production_status": self.production_status,
            "verdict": self.verdict,
            "bound_issued_at_ms": self.bound_issued_at_ms,
            "bound_received_at_ms": self.bound_received_at_ms,
            "report_data_hex": self.report_data_hex,
        }


def _refuse(
    code: str,
    *,
    reverify_exercised: bool = False,
    review_digest: str | None = None,
) -> EvalCvmFreshAdmission:
    return EvalCvmFreshAdmission(
        may_launch=False,
        reason_code=code,
        review_digest=review_digest,
        reverify_exercised=reverify_exercised,
    )


def _parse_envelope(envelope: Mapping[str, Any] | str | bytes | None) -> Mapping[str, Any] | None:
    if envelope is None:
        return None
    if isinstance(envelope, (str, bytes)):
        try:
            raw = envelope if isinstance(envelope, str) else envelope.decode("utf-8")
            parsed = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(envelope, Mapping):
        return envelope
    return None


def _extract_core_and_report_data(
    envelope: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    """Return (review_core, report_data_hex, domain)."""

    domain = envelope.get("domain")
    domain_s = str(domain) if isinstance(domain, str) else None
    core = envelope.get("review_core")
    if not isinstance(core, Mapping):
        # Some storage shapes nest under "envelope" or keep only report pieces.
        nested = envelope.get("envelope")
        if isinstance(nested, Mapping):
            return _extract_core_and_report_data(nested)
        return None, None, domain_s
    report_data = envelope.get("report_data_hex")
    report_s = str(report_data) if isinstance(report_data, str) else None
    return core, report_s, domain_s


def admit_eval_cvm_fresh_review(
    *,
    envelope: Mapping[str, Any] | str | bytes | None = None,
    review_core: Mapping[str, Any] | None = None,
    report_data_hex: str | None = None,
    domain: str | None = None,
    cached_phase: str | None = None,
    cached_outcome_status: str | None = None,
    cached_review_digest: str | None = None,
    plain_status_allow: bool | None = None,
    # Optional full quote re-verify hook (DCAP/trust roots). When provided and
    # envelope contains materials, it must return True for launch. Absence of
    # this hook does **not** fall back to cache-only allow (report_data re-admit
    # still required). When provided and returns False → refuse.
    quote_reverify: Callable[[Mapping[str, Any]], bool] | None = None,
    db_issued_at_ms: object | None = None,
    db_received_at_ms: object | None = None,
    client_bag: Mapping[str, Any] | None = None,
    http_date_ms: object | None = None,
) -> EvalCvmFreshAdmission:
    """Admit Eval CVM launch only on fresh re-verified review materials.

    Cached ``verified_allow`` / ``review_allowed`` alone always fails closed when
    materials are missing or when re-verify does not produce ``allow``.
    """

    cached_allow = (
        cached_outcome_status == "verified_allow"
        or cached_phase == "review_allowed"
        or plain_status_allow is True
    )

    env = _parse_envelope(envelope)
    core = review_core
    rd = report_data_hex
    dom = domain

    if env is not None:
        env_core, env_rd, env_dom = _extract_core_and_report_data(env)
        if core is None:
            core = env_core
        if rd is None:
            rd = env_rd
        if dom is None:
            dom = env_dom

    # Cache allow bit alone is never enough (VAL-ACAT-029).
    if core is None or not rd:
        if cached_allow:
            return _refuse(
                REFUSE_CACHED_ALLOW_ONLY,
                reverify_exercised=False,
                review_digest=cached_review_digest,
            )
        return _refuse(
            REFUSE_ATTESTATION_MISSING,
            reverify_exercised=False,
            review_digest=cached_review_digest,
        )

    if dom is not None and dom != REVIEW_REPORT_DOMAIN:
        return _refuse(
            REFUSE_WRONG_DOMAIN,
            reverify_exercised=True,
            review_digest=cached_review_digest,
        )

    # Optional hardware/quote re-verify path when caller supplies it.
    if quote_reverify is not None:
        try:
            materials = env if env is not None else {"review_core": core, "report_data_hex": rd}
            if not quote_reverify(dict(materials)):
                return _refuse(
                    REFUSE_REVERIFY_FAILED,
                    reverify_exercised=True,
                    review_digest=cached_review_digest,
                )
        except Exception:
            return _refuse(
                REFUSE_REVERIFY_FAILED,
                reverify_exercised=True,
                review_digest=cached_review_digest,
            )

    # Core production re-admit: bound decision + OR digests + report_data + ≤24h.
    # Do **not** force plain_status_allow from cache bits — cache is only informative
    # for the missing-materials path above. Binding must speak for itself.
    try:
        production = admit_production_from_bound_outcome(
            review_core=core,
            reported_report_data_hex=rd,
            plain_status_allow=plain_status_allow,
            require_or_digests=True,
            db_issued_at_ms=db_issued_at_ms,
            db_received_at_ms=db_received_at_ms,
            client_bag=client_bag,
            http_date_ms=http_date_ms,
        )
    except ReviewOrOutcomeError as exc:
        code = exc.code
        if code == REFUSE_STALE or code == "attestation_stale_over_24h":
            return _refuse(
                REFUSE_STALE,
                reverify_exercised=True,
                review_digest=cached_review_digest,
            )
        if code in {
            "attestation_time_order_invalid",
            "attestation_times_missing",
            "attestation_times_invalid",
        }:
            return _refuse(
                code,
                reverify_exercised=True,
                review_digest=cached_review_digest,
            )
        # plain_status vs bound verdict shear is not a launchable admit.
        if code == "review_outcome_status_shear":
            return _refuse(
                REFUSE_EVAL_CVM,
                reverify_exercised=True,
                review_digest=cached_review_digest,
            )
        # Unbound decision / report_data mismatch / missing OR digests → re-verify
        # failed (crypto money trail incomplete for launch).
        return _refuse(
            REFUSE_REVERIFY_FAILED,
            reverify_exercised=True,
            review_digest=cached_review_digest,
        )

    if not production.admitted or production.verdict != "allow":
        return EvalCvmFreshAdmission(
            may_launch=False,
            reason_code=REFUSE_EVAL_CVM,
            review_digest=production.review_digest,
            reverify_exercised=True,
            production_status=production.status,
            verdict=production.verdict,
            report_data_hex=production.report_data_hex,
        )

    # Bound times for evidence / operators (already re-verified inside admit).
    bound_issued: int | None = None
    bound_received: int | None = None
    try:
        from agent_challenge.review.attested_times import extract_bound_times_from_core

        bound_issued, bound_received = extract_bound_times_from_core(core)
    except Exception:
        # Production admit already enforced times; extract is diagnostic only.
        pass

    return EvalCvmFreshAdmission(
        may_launch=True,
        reason_code="review_verified",
        review_digest=production.review_digest,
        reverify_exercised=True,
        production_status=production.status,
        verdict=production.verdict,
        bound_issued_at_ms=bound_issued,
        bound_received_at_ms=bound_received,
        report_data_hex=production.report_data_hex,
    )


def admit_eval_cvm_launch_from_assignment(assignment: Any) -> EvalCvmFreshAdmission:
    """Eval-launch gate from a durable ReviewAssignment row (or duck-type).

    Loads receipted envelope materials and re-verifies; never trusts phase /
    outcome JSON alone.
    """

    phase = getattr(assignment, "phase", None)
    outcome_json = getattr(assignment, "review_verification_outcome_json", None)
    outcome_status: str | None = None
    if isinstance(outcome_json, str) and outcome_json:
        try:
            parsed = json.loads(outcome_json)
            if isinstance(parsed, dict):
                status = parsed.get("status")
                outcome_status = str(status) if isinstance(status, str) else None
        except (TypeError, ValueError):
            outcome_status = None

    envelope_json = getattr(assignment, "review_report_envelope_json", None)
    report_data_col = getattr(assignment, "review_report_data_hex", None)
    review_digest = getattr(assignment, "review_digest", None)

    return admit_eval_cvm_fresh_review(
        envelope=envelope_json,
        report_data_hex=str(report_data_col) if isinstance(report_data_col, str) else None,
        cached_phase=str(phase) if phase is not None else None,
        cached_outcome_status=outcome_status,
        cached_review_digest=str(review_digest) if isinstance(review_digest, str) else None,
    )


def require_eval_cvm_fresh_review(
    *,
    envelope: Mapping[str, Any] | str | bytes | None = None,
    review_core: Mapping[str, Any] | None = None,
    report_data_hex: str | None = None,
    domain: str | None = None,
    cached_phase: str | None = None,
    cached_outcome_status: str | None = None,
    cached_review_digest: str | None = None,
    quote_reverify: Callable[[Mapping[str, Any]], bool] | None = None,
) -> EvalCvmFreshAdmission:
    """Fail closed on refuse; return admission when launch may proceed."""

    decision = admit_eval_cvm_fresh_review(
        envelope=envelope,
        review_core=review_core,
        report_data_hex=report_data_hex,
        domain=domain,
        cached_phase=cached_phase,
        cached_outcome_status=cached_outcome_status,
        cached_review_digest=cached_review_digest,
        quote_reverify=quote_reverify,
    )
    if not decision.may_launch:
        raise EvalCvmFreshReviewError(decision.reason_code)
    return decision


__all__ = [
    "EvalCvmFreshAdmission",
    "EvalCvmFreshReviewError",
    "REFUSE_ATTESTATION_MISSING",
    "REFUSE_CACHED_ALLOW_ONLY",
    "REFUSE_EVAL_CVM",
    "REFUSE_REVERIFY_FAILED",
    "REFUSE_STALE",
    "REFUSE_WRONG_DOMAIN",
    "admit_eval_cvm_fresh_review",
    "admit_eval_cvm_launch_from_assignment",
    "require_eval_cvm_fresh_review",
]
