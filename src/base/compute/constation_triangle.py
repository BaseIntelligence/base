"""Three-way digest triangle — required / Lium-declared / sidecar (fail-closed).

Compares BASE allowlist **required** digest against Lium **declared** and
sidecar **actual**. Negative-only: any divergence fails with a typed code;
agreement never elevates trust (tamper-evidence only; prism ``constation_ok``
remains the sole elevation path).
"""

from __future__ import annotations

from dataclasses import dataclass

from base.compute.constation_types import ConstationFailCode


def _normalize(value: str | None) -> str | None:
    """Strip + casefold; blank becomes None."""
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    if not normalized:
        return None
    return normalized


@dataclass(frozen=True, slots=True)
class DigestTriangleResult:
    """Outcome of required / Lium / sidecar digest agreement (ok/fail only)."""

    ok: bool
    fail_code: ConstationFailCode | None

    def __bool__(self) -> bool:
        return self.ok


def evaluate_digest_triangle(
    *,
    required: str | None,
    lium_declared: str | None,
    sidecar: str | None,
) -> DigestTriangleResult:
    """Fail-closed three-way digest check (precedence order is binding).

    a. required blank/None → REQUIRED_DIGEST_MISMATCH
    b. sidecar blank/None → SIDECAR_RESPONSE_INVALID
    c. lium_declared blank/None → LIUM_DIGEST_ABSENT
    d. sidecar != required → REQUIRED_DIGEST_MISMATCH
    e. lium_declared != required → CORROBORATION_MISMATCH
    f. otherwise ok=True
    """
    req = _normalize(required)
    if req is None:
        return DigestTriangleResult(
            ok=False,
            fail_code=ConstationFailCode.REQUIRED_DIGEST_MISMATCH,
        )

    side = _normalize(sidecar)
    if side is None:
        return DigestTriangleResult(
            ok=False,
            fail_code=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
        )

    declared = _normalize(lium_declared)
    if declared is None:
        return DigestTriangleResult(
            ok=False,
            fail_code=ConstationFailCode.LIUM_DIGEST_ABSENT,
        )

    if side != req:
        return DigestTriangleResult(
            ok=False,
            fail_code=ConstationFailCode.REQUIRED_DIGEST_MISMATCH,
        )

    if declared != req:
        return DigestTriangleResult(
            ok=False,
            fail_code=ConstationFailCode.CORROBORATION_MISMATCH,
        )

    return DigestTriangleResult(ok=True, fail_code=None)


__all__ = [
    "DigestTriangleResult",
    "evaluate_digest_triangle",
]
