"""Same-account digest corroboration — negative-only signal (todo 17 / B2).

This module compares the Lium API **declared** template digest (configuration
recorded at rent time via ``get_pod_raw`` / ``get_template_raw``) against the
sidecar **reported** digest.

Honesty constraints (binding)
-----------------------------
* This is **corroboration** only — both channels share one principal
  (not a second root of trust). The miner supplies the Lium API key and
  owns the pod — one principal controls both channels.
* A **mismatch** forces failure (``corroboration_mismatch`` / miner_fault).
* **Agreement alone never elevates.** It contributes nothing toward tier grant;
  prism ``constation_ok`` still requires allowlist, nonce, signature, sealed
  manifest, and gap budget. Callers must not treat ``agree`` as sufficient.
* Absence of a Lium-declared digest is **not** a contradiction (channel optional).
"""

from __future__ import annotations

from dataclasses import dataclass

from base.compute.constation_types import (
    ConstationFailCode,
    ConstationVerdict,
    CorroborationStatus,
    FaultClass,
)


@dataclass(frozen=True, slots=True)
class CorroborationOutcome:
    """Result of same-account digest corroboration (negative-only)."""

    status: CorroborationStatus
    verdict: ConstationVerdict
    lium_declared_digest: str | None
    sidecar_digest: str

    @property
    def ok(self) -> bool:
        return self.verdict.ok


def evaluate_corroboration(
    *,
    lium_declared_digest: str | None,
    sidecar_digest: str,
) -> CorroborationOutcome:
    """Compare Lium-declared vs sidecar-reported digests (corroboration only).

    Returns:
        * ``absent`` + ok when Lium declared digest is missing/blank
        * ``agree`` + ok when normalized digests match (does **not** elevate)
        * ``mismatch`` + fail when both present and differ
    """
    sidecar = sidecar_digest.strip().lower()
    if not sidecar:
        raise ValueError("sidecar_digest must be a non-empty string")

    if lium_declared_digest is None or not str(lium_declared_digest).strip():
        return CorroborationOutcome(
            status=CorroborationStatus.ABSENT,
            verdict=ConstationVerdict(ok=True, reason=ConstationFailCode.OK),
            lium_declared_digest=None,
            sidecar_digest=sidecar,
        )

    declared = str(lium_declared_digest).strip().lower()
    if declared == sidecar:
        return CorroborationOutcome(
            status=CorroborationStatus.AGREE,
            verdict=ConstationVerdict(ok=True, reason=ConstationFailCode.OK),
            lium_declared_digest=declared,
            sidecar_digest=sidecar,
        )

    return CorroborationOutcome(
        status=CorroborationStatus.MISMATCH,
        verdict=ConstationVerdict(
            ok=False,
            reason=ConstationFailCode.CORROBORATION_MISMATCH,
            fault_class=FaultClass.MINER,
            detail=f"lium={declared} sidecar={sidecar}",
        ),
        lium_declared_digest=declared,
        sidecar_digest=sidecar,
    )


__all__ = [
    "CorroborationOutcome",
    "evaluate_corroboration",
]
