"""Shared types for the BASE constation service (todos 15–17).

Outputs are shaped so prism ``constation_ok`` can consume gap budget fields and
``lium_declared_digest`` without this package importing prism.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class FaultClass(StrEnum):
    """Attribution for fail-closed constation outcomes."""

    MINER = "miner_fault"
    INFRA = "infra_fault"


class ConstationFailCode(StrEnum):
    """Machine-consumed constation runner / poller / corroboration codes."""

    OK = "ok"
    LIUM_AUTH_REVOKED = "lium_auth_revoked"
    LIUM_RATE_LIMITED = "lium_rate_limited"
    NETWORK_PARTITION = "network_partition"
    CONSTATION_GAP = "constation_gap"
    CORROBORATION_MISMATCH = "corroboration_mismatch"
    POLL_CAP_EXCEEDED = "poll_cap_exceeded"
    COST_CAP_EXCEEDED = "cost_cap_exceeded"
    PROBE_FAILED = "probe_failed"
    KEY_NOT_REGISTERED = "key_not_registered"
    SIDECAR_ATTEST_FAILED = "sidecar_attest_failed"
    RUN_INCOMPLETE = "run_incomplete"


class CorroborationStatus(StrEnum):
    """Same-account corroboration channel status (never 'independent')."""

    AGREE = "agree"
    MISMATCH = "mismatch"
    ABSENT = "absent"
    NOT_EVALUATED = "not_evaluated"


_FAULT_BY_CODE: Final[dict[ConstationFailCode, FaultClass | None]] = {
    ConstationFailCode.OK: None,
    ConstationFailCode.LIUM_AUTH_REVOKED: FaultClass.MINER,
    ConstationFailCode.LIUM_RATE_LIMITED: FaultClass.INFRA,
    ConstationFailCode.NETWORK_PARTITION: FaultClass.INFRA,
    ConstationFailCode.CONSTATION_GAP: FaultClass.MINER,
    ConstationFailCode.CORROBORATION_MISMATCH: FaultClass.MINER,
    ConstationFailCode.POLL_CAP_EXCEEDED: FaultClass.INFRA,
    ConstationFailCode.COST_CAP_EXCEEDED: FaultClass.INFRA,
    ConstationFailCode.PROBE_FAILED: FaultClass.MINER,
    ConstationFailCode.KEY_NOT_REGISTERED: FaultClass.MINER,
    ConstationFailCode.SIDECAR_ATTEST_FAILED: FaultClass.MINER,
    ConstationFailCode.RUN_INCOMPLETE: FaultClass.INFRA,
}


def fault_class_for(code: ConstationFailCode) -> FaultClass | None:
    """Return miner vs infra attribution for ``code``."""
    return _FAULT_BY_CODE[code]


@dataclass(frozen=True, slots=True)
class ConstationVerdict:
    """Structured fail-closed outcome from custody, poller, or runner."""

    ok: bool
    reason: ConstationFailCode
    fault_class: FaultClass | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.fault_class is None and self.reason is not ConstationFailCode.OK:
            object.__setattr__(self, "fault_class", fault_class_for(self.reason))

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True, slots=True)
class PollSample:
    """One successful constation poll observation."""

    at_monotonic: float
    phase: str
    sidecar_digest: str
    lium_declared_digest: str | None


@dataclass(frozen=True, slots=True)
class ConstationRunRecord:
    """Runner output bundle fragment for prism ``constation_ok`` consumers.

    Elevation still requires the full six-mechanism conjunction in prism; this
    record only supplies gap metrics, digests, and corroboration status.
    """

    ok: bool
    reason: ConstationFailCode
    fault_class: FaultClass | None
    miner_hotkey: str
    work_unit_id: str
    pod_id: str
    sidecar_digest: str | None
    lium_declared_digest: str | None
    constation_gap_budget_seconds: float
    constation_observed_max_gap_seconds: float
    corroboration_status: CorroborationStatus
    samples: tuple[PollSample, ...]
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.ok


__all__ = [
    "ConstationFailCode",
    "ConstationRunRecord",
    "ConstationVerdict",
    "CorroborationStatus",
    "FaultClass",
    "PollSample",
    "fault_class_for",
]
