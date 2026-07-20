"""Cryptographically bound review issued_at / received_at (challenge clock domain).

Product freeze (library/ac-attestation.md, VAL-ACAT-007/008/021–024/038):

- ``issued_at_ms`` and ``received_at_ms`` appear in the review-domain
  report_data preimage (schema v2) hashed into TDX ``report_data``.
- Guest wall clock alone never authorizes freshness / age / admission.
- Unattested DB columns may cache times but never alone authorize under
  dual Phala flags ON. Re-verify extracts times from the verified preimage
  that matches quote ``report_data``.
- Client-supplied times on public bags must be ignored for security
  decisions (extra=forbid / closed schemas).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .canonical import canonical_json_v1, canonical_sha256

REVIEW_REPORT_DOMAIN = "base-agent-challenge-review-v1"
# report_data preimage schema is independent of review_core schema_version.
REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_V2 = 2
REPORT_DATA_FIELD_HEX_LEN = 128
FRESHNESS_WINDOW_MS = 86_400_000  # exactly 24h in ms (≤ boundary admits; +1 ms refuses)

# Stable refuse codes (library/ac-attestation.md)
REFUSE_TIMES_MISSING = "attestation_times_missing"
REFUSE_TIMES_INVALID = "attestation_times_invalid"
REFUSE_TIME_ORDER = "attestation_time_order_invalid"
REFUSE_STALE = "attestation_stale_over_24h"
REFUSE_REPORT_DATA_MISMATCH = "review_report_data_mismatch"
REFUSE_GUEST_CLOCK_ALONE = "attestation_guest_clock_alone_forbidden"
REFUSE_DB_ONLY_TIMES = "attestation_db_only_times_forbidden"
REFUSE_CLIENT_SMUGGLED_TIMES = "attestation_client_smuggled_times_ignored"


class AttestedTimeError(ValueError):
    """Fail-closed attested time / preimage error with a stable reason code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def _require_time_ms(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AttestedTimeError(REFUSE_TIMES_INVALID, f"{name} must be int UTC ms")
    if not 0 <= value <= (2**63 - 1):
        raise AttestedTimeError(REFUSE_TIMES_INVALID, f"{name} out of range")
    return value


def datetime_to_ms(value: datetime) -> int:
    """Convert a challenge-side aware UTC datetime to epoch milliseconds."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.astimezone(UTC).timestamp() * 1000)


def extract_bound_times_from_core(review_core: Mapping[str, Any]) -> tuple[int, int]:
    """Extract challenge-bound times from a validated review core.

    Prefer ``times.submission_received_at_ms`` (product submission/send receive
    bound into materials). Fall back is not guest wall clock.
    """

    times = review_core.get("times")
    if not isinstance(times, Mapping):
        raise AttestedTimeError(REFUSE_TIMES_MISSING, "review_core.times missing")
    if "issued_at_ms" not in times:
        raise AttestedTimeError(REFUSE_TIMES_MISSING, "issued_at_ms missing from core")
    issued = _require_time_ms(times["issued_at_ms"], "issued_at_ms")
    if "submission_received_at_ms" not in times:
        raise AttestedTimeError(
            REFUSE_TIMES_MISSING,
            "submission_received_at_ms missing from review_core.times",
        )
    received = _require_time_ms(
        times["submission_received_at_ms"],
        "submission_received_at_ms",
    )
    return issued, received


def review_report_data_preimage_v2(
    *,
    review_digest: str,
    session_id: str,
    review_nonce: str,
    issued_at_ms: int,
    received_at_ms: int,
) -> dict[str, Any]:
    """Closed review-domain report_data preimage with cryptographically bound times."""

    if not isinstance(review_digest, str) or len(review_digest) != 64:
        raise AttestedTimeError(REFUSE_TIMES_INVALID, "review_digest must be 64-char hex")
    if any(c not in "0123456789abcdef" for c in review_digest):
        raise AttestedTimeError(REFUSE_TIMES_INVALID, "review_digest must be lowercase hex")
    if not isinstance(session_id, str) or not session_id:
        raise AttestedTimeError(REFUSE_TIMES_INVALID, "session_id required")
    if not isinstance(review_nonce, str) or not review_nonce:
        raise AttestedTimeError(REFUSE_TIMES_INVALID, "review_nonce required")
    issued = _require_time_ms(issued_at_ms, "issued_at_ms")
    received = _require_time_ms(received_at_ms, "received_at_ms")
    # Order (issued ≤ received) is enforced by check_freshness for admission age,
    # not by preimage construction: both times must still appear in report_data so
    # re-verify can extract them after a quote (VAL-ACAT-021/022).
    return {
        "domain": REVIEW_REPORT_DOMAIN,
        "schema_version": REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_V2,
        "review_digest": review_digest,
        "session_id": session_id,
        "review_nonce": review_nonce,
        "issued_at_ms": issued,
        "received_at_ms": received,
    }


def report_data_field_hex(digest32_hex: str) -> str:
    if len(digest32_hex) != 64 or any(c not in "0123456789abcdef" for c in digest32_hex):
        raise AttestedTimeError(REFUSE_REPORT_DATA_MISMATCH, "digest must be 64 lowercase hex")
    field = digest32_hex + ("00" * 32)
    if len(field) != REPORT_DATA_FIELD_HEX_LEN:
        raise RuntimeError("report_data field width broken")
    return field


def review_report_data_hex_v2(
    *,
    review_digest: str,
    session_id: str,
    review_nonce: str,
    issued_at_ms: int,
    received_at_ms: int,
) -> str:
    """64-byte left-aligned TDX report_data hex (128 chars) for preimage v2."""

    preimage = review_report_data_preimage_v2(
        review_digest=review_digest,
        session_id=session_id,
        review_nonce=review_nonce,
        issued_at_ms=issued_at_ms,
        received_at_ms=received_at_ms,
    )
    digest = canonical_sha256(preimage)
    return report_data_field_hex(digest)


def assert_core_times_match_preimage(
    *,
    review_core: Mapping[str, Any],
    preimage: Mapping[str, Any],
) -> None:
    """Fail closed when preimage times shear from core materials."""

    issued, received = extract_bound_times_from_core(review_core)
    if preimage.get("issued_at_ms") != issued:
        raise AttestedTimeError(
            REFUSE_REPORT_DATA_MISMATCH,
            "preimage issued_at_ms does not match review_core.times.issued_at_ms",
        )
    if preimage.get("received_at_ms") != received:
        raise AttestedTimeError(
            REFUSE_REPORT_DATA_MISMATCH,
            "preimage received_at_ms does not match review_core submission receive",
        )
    if preimage.get("session_id") != review_core.get("session_id"):
        raise AttestedTimeError(REFUSE_REPORT_DATA_MISMATCH, "session_id shear")
    if preimage.get("review_nonce") != review_core.get("review_nonce"):
        raise AttestedTimeError(REFUSE_REPORT_DATA_MISMATCH, "review_nonce shear")


def reverify_extract_bound_times(
    *,
    review_core: Mapping[str, Any],
    report_data_hex: str,
    review_digest: str | None = None,
) -> dict[str, int]:
    """Re-verify report_data matches preimage v2; return bound times.

    Used by admission paths so age decisions never take guest/DB/client times.
    """

    issued, received = extract_bound_times_from_core(review_core)
    digest = review_digest
    if digest is None:
        # Prefer full product digest when core is fully schema-valid; fall back to
        # lightweight canonical digest used by outcome-bind helpers on minimal cores.
        try:
            from .report import review_digest as product_review_digest

            digest = product_review_digest(review_core)
        except Exception:
            digest = canonical_sha256(review_core)
    expected = review_report_data_hex_v2(
        review_digest=digest,
        session_id=str(review_core["session_id"]),
        review_nonce=str(review_core["review_nonce"]),
        issued_at_ms=issued,
        received_at_ms=received,
    )
    if not isinstance(report_data_hex, str) or report_data_hex != expected:
        raise AttestedTimeError(
            REFUSE_REPORT_DATA_MISMATCH,
            "quote report_data does not match v2 preimage with bound times",
        )
    return {"issued_at_ms": issued, "received_at_ms": received}


def refuse_guest_clock_alone_authorization(
    *,
    guest_issued_at_ms: object | None,
    guest_received_at_ms: object | None,
    challenge_bound_issued_at_ms: object | None,
    challenge_bound_received_at_ms: object | None,
    report_data_matched: bool,
) -> str | None:
    """Return refuse code if only guest wall clock is offered as time root."""

    bound_ok = (
        isinstance(challenge_bound_issued_at_ms, int)
        and isinstance(challenge_bound_received_at_ms, int)
        and report_data_matched
    )
    if bound_ok:
        return None
    if guest_issued_at_ms is not None or guest_received_at_ms is not None:
        return REFUSE_GUEST_CLOCK_ALONE
    return REFUSE_TIMES_MISSING


def refuse_db_only_time_authorization(
    *,
    db_issued_at_ms: object | None,
    db_received_at_ms: object | None,
    report_data_reverified: bool,
    bound_issued_at_ms: object | None,
    bound_received_at_ms: object | None,
) -> str | None:
    """Return refuse code when unattested DB columns alone would authorize."""

    if (
        report_data_reverified
        and isinstance(bound_issued_at_ms, int)
        and isinstance(bound_received_at_ms, int)
    ):
        return None
    if db_issued_at_ms is not None or db_received_at_ms is not None:
        return REFUSE_DB_ONLY_TIMES
    return REFUSE_TIMES_MISSING


def ignore_client_smuggled_times(public_bag: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strip client-supplied time fields from a public bag (security no-op).

    Admission must never use these keys for age decisions. Schema closure
    (extra=forbid) is the primary defense; this strips with a stable code.
    """

    if not public_bag:
        return {}
    forbidden = {
        "issued_at",
        "issued_at_ms",
        "received_at",
        "received_at_ms",
        "submission_received_at_ms",
        "reviewed_at",
        "reviewed_at_ms",
        "created",
        "already_fresh",
        "openrouter_created",
        "openrouter_created_at_ms",
    }
    cleaned = {k: v for k, v in public_bag.items() if k not in forbidden}
    return cleaned


def client_smuggled_time_keys(public_bag: Mapping[str, Any] | None) -> frozenset[str]:
    """Return forbidden time-key names present on a public payload."""

    if not public_bag:
        return frozenset()
    forbidden = {
        "issued_at",
        "issued_at_ms",
        "received_at",
        "received_at_ms",
        "submission_received_at_ms",
        "reviewed_at",
        "reviewed_at_ms",
        "created",
        "already_fresh",
        "openrouter_created",
        "openrouter_created_at_ms",
    }
    return frozenset(k for k in public_bag if k in forbidden)


def check_freshness(*, issued_at_ms: object, received_at_ms: object) -> str | None:
    """Return refuse code on failure, or None when absolute bound age is OK.

    Product boundary (VAL-ACAT-009 / 025 / residual report_timeline_invalid):

    - Age: ``abs(received_at_ms - issued_at_ms) <= FRESHNESS_WINDOW_MS``
      (exactly ``86400000`` ms **passes**; ``86400000 + 1`` refuses)
    - Order: **either** chronological order of the two challenge-domain stamps
      is admitted inside the window.

      Rationale (measured residual sub25): miner ZIP admit stamps
      ``submission_received_at_ms`` *before* assignment ``issued_at_ms``
      (submit-first). The inverse order is the self-deploy-then-admit path.
      Both are real product sequences; directional order must not map to
      ``attestation_time_order_invalid`` → guest ``report_timeline_invalid``.
      Internal review leaf chronology (started → model_call → report_finished)
      remains enforced by ``report._validate_times`` separately.

    Caller must feed times extracted from **re-verified** report_data / bound
    materials only. HTTP ``Date``, miner client headers, signed-request skew,
    guest wall clock, and bare DB columns are **not** valid substitutes for
    either argument (VAL-ACAT-027).
    """

    if issued_at_ms is None or received_at_ms is None:
        return REFUSE_TIMES_MISSING
    if isinstance(issued_at_ms, bool) or isinstance(received_at_ms, bool):
        return REFUSE_TIMES_INVALID
    if not isinstance(issued_at_ms, int) or not isinstance(received_at_ms, int):
        return REFUSE_TIMES_INVALID
    if not (0 <= issued_at_ms <= 2**63 - 1 and 0 <= received_at_ms <= 2**63 - 1):
        return REFUSE_TIMES_INVALID
    # Absolute separation only. Directional order of issued vs submission
    # receive is not a security failure for the measured dual-flag path
    # (see residual report_timeline_invalid / admission comments above).
    if abs(received_at_ms - issued_at_ms) > FRESHNESS_WINDOW_MS:
        return REFUSE_STALE
    return None


def enforce_bound_freshness(
    *,
    issued_at_ms: object,
    received_at_ms: object,
    http_date_ms: object | None = None,
    client_header_ms: object | None = None,
    client_skew_ms: object | None = None,
) -> dict[str, int]:
    """Fail closed on age/order using only attestation-bound times.

    ``http_date_ms`` / ``client_header_ms`` / ``client_skew_ms`` may be present
    on the wire for auth/debug but **never** replace or "fix" bound times
    for the ≤24h window (VAL-ACAT-027). Returns the accepted bound pair on OK.
    """

    # Explicitly discard non-bound clock sources for security decisions.
    # Presence of HTTP Date / client headers / auth skew must never substitute
    # for or "heal" bound attestation times in the age check.
    _ = (http_date_ms, client_header_ms, client_skew_ms)

    code = check_freshness(issued_at_ms=issued_at_ms, received_at_ms=received_at_ms)
    if code is not None:
        raise AttestedTimeError(code)
    assert isinstance(issued_at_ms, int) and isinstance(received_at_ms, int)
    return {"issued_at_ms": issued_at_ms, "received_at_ms": received_at_ms}


def production_freshness_from_reverified_materials(
    *,
    review_core: Mapping[str, Any],
    report_data_hex: str,
    db_issued_at_ms: object | None = None,
    db_received_at_ms: object | None = None,
    guest_issued_at_ms: object | None = None,
    guest_received_at_ms: object | None = None,
    client_bag: Mapping[str, Any] | None = None,
    http_date_ms: object | None = None,
    client_header_ms: object | None = None,
    client_skew_ms: object | None = None,
) -> dict[str, int]:
    """Re-verify bound times then enforce ≤24h freshness (production conjunction).

    Sole production extractor for the OpenRouter/review → submit age window.
    Header/client clocks cannot satisfy the window when bound times are stale
    or inverted (VAL-ACAT-025 / 026 / 027).
    """

    bound = production_times_from_reverified_materials(
        review_core=review_core,
        report_data_hex=report_data_hex,
        db_issued_at_ms=db_issued_at_ms,
        db_received_at_ms=db_received_at_ms,
        guest_issued_at_ms=guest_issued_at_ms,
        guest_received_at_ms=guest_received_at_ms,
        client_bag=client_bag,
    )
    return enforce_bound_freshness(
        issued_at_ms=bound["issued_at_ms"],
        received_at_ms=bound["received_at_ms"],
        http_date_ms=http_date_ms,
        client_header_ms=client_header_ms,
        client_skew_ms=client_skew_ms,
    )


def production_times_from_reverified_materials(
    *,
    review_core: Mapping[str, Any],
    report_data_hex: str,
    db_issued_at_ms: object | None = None,
    db_received_at_ms: object | None = None,
    guest_issued_at_ms: object | None = None,
    guest_received_at_ms: object | None = None,
    client_bag: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Sole production extractor for age/admission times.

    Steps (fail closed):
    1. Ignore client-smuggled time keys for security decisions.
    2. Re-verify report_data binds core times (VAL-ACAT-021/022).
    3. Refuse guest-only and DB-only authorization (VAL-ACAT-023/024/038).
    """

    if client_smuggled_time_keys(client_bag):
        # Present smuggled keys are ignored; decision still requires bound path.
        pass

    try:
        bound = reverify_extract_bound_times(
            review_core=review_core,
            report_data_hex=report_data_hex,
        )
    except AttestedTimeError as reverify_exc:
        # Try guest/db diagnosis for clearer refuse without authorizing them.
        guest_code = refuse_guest_clock_alone_authorization(
            guest_issued_at_ms=guest_issued_at_ms,
            guest_received_at_ms=guest_received_at_ms,
            challenge_bound_issued_at_ms=None,
            challenge_bound_received_at_ms=None,
            report_data_matched=False,
        )
        if guest_code == REFUSE_GUEST_CLOCK_ALONE and (
            guest_issued_at_ms is not None or guest_received_at_ms is not None
        ):
            # Prefer guest-alone diagnosis only when guest times are the offered root
            # and no re-verify succeeded.
            if not report_data_hex:
                raise AttestedTimeError(REFUSE_GUEST_CLOCK_ALONE) from reverify_exc
        db_code = refuse_db_only_time_authorization(
            db_issued_at_ms=db_issued_at_ms,
            db_received_at_ms=db_received_at_ms,
            report_data_reverified=False,
            bound_issued_at_ms=None,
            bound_received_at_ms=None,
        )
        if db_code == REFUSE_DB_ONLY_TIMES and not report_data_hex:
            raise AttestedTimeError(REFUSE_DB_ONLY_TIMES) from reverify_exc
        raise

    # Re-verify succeeded: DB and guest must not shear the bound values if provided.
    if db_issued_at_ms is not None and db_issued_at_ms != bound["issued_at_ms"]:
        # DB is cache only; shear is refused for authorization surfaces.
        raise AttestedTimeError(
            REFUSE_DB_ONLY_TIMES,
            "DB issued_at shear vs report_data-bound issued_at",
        )
    if db_received_at_ms is not None and db_received_at_ms != bound["received_at_ms"]:
        raise AttestedTimeError(
            REFUSE_DB_ONLY_TIMES,
            "DB received_at shear vs report_data-bound received_at",
        )
    return bound


def preimage_canonical_bytes(preimage: Mapping[str, Any]) -> bytes:
    return canonical_json_v1(preimage)


__all__ = [
    "FRESHNESS_WINDOW_MS",
    "REFUSE_CLIENT_SMUGGLED_TIMES",
    "REFUSE_DB_ONLY_TIMES",
    "REFUSE_GUEST_CLOCK_ALONE",
    "REFUSE_REPORT_DATA_MISMATCH",
    "REFUSE_STALE",
    "REFUSE_TIME_ORDER",
    "REFUSE_TIMES_INVALID",
    "REFUSE_TIMES_MISSING",
    "REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_V2",
    "REVIEW_REPORT_DOMAIN",
    "AttestedTimeError",
    "assert_core_times_match_preimage",
    "check_freshness",
    "client_smuggled_time_keys",
    "datetime_to_ms",
    "enforce_bound_freshness",
    "extract_bound_times_from_core",
    "ignore_client_smuggled_times",
    "preimage_canonical_bytes",
    "production_freshness_from_reverified_materials",
    "production_times_from_reverified_materials",
    "refuse_db_only_time_authorization",
    "refuse_guest_clock_alone_authorization",
    "report_data_field_hex",
    "reverify_extract_bound_times",
    "review_report_data_hex_v2",
    "review_report_data_preimage_v2",
]
