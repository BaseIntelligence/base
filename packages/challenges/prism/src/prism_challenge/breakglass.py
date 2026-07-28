"""Audited operator break-glass for infra-fault attestation failures (todo 23).

P1 fail-closed scoring erases runs without a valid constation bundle. When the
failure is classified as BASE/infra fault (constation service down, Lium 5xx,
network partition), an explicit operator override may admit the run. Overrides
are never automatic, always attributable to an operator identity, and always
written to an audit log. Miner-fault runs can never be admitted this way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

FaultClass = Literal["miner_fault", "infra_fault"]

MINER_FAULT = "miner_fault"
INFRA_FAULT = "infra_fault"


@dataclass(frozen=True, slots=True)
class BreakGlassRequest:
    """Operator request to admit an infra-fault run."""

    operator_id: str
    reason: str
    work_unit_id: str
    fault_code: str  # e.g. infra_fault:constation_unavailable


@dataclass(frozen=True, slots=True)
class BreakGlassDecision:
    """Outcome of evaluating a break-glass request."""

    admitted: bool
    reason: str
    audit_entry: dict[str, Any] | None = None


@dataclass
class BreakGlassAuditLog:
    """In-memory / injectable audit sink for break-glass decisions."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.entries)


def fault_class_of(reason_code: str) -> FaultClass:
    """Return miner_fault or infra_fault from a dotted/colon reason code."""
    code = (reason_code or "").strip().lower()
    if code.startswith("infra_fault") or code.startswith(f"{INFRA_FAULT}:"):
        return INFRA_FAULT
    if code.startswith("miner_fault") or code.startswith(f"{MINER_FAULT}:"):
        return MINER_FAULT
    # bare codes
    if code in {
        "constation_unavailable",
        "lium_5xx",
        "network_partition",
        "constation_retry_exhausted",
        "lium_auth_revoked_infra",
    }:
        return INFRA_FAULT
    return MINER_FAULT


def format_fault_reason(fault_class: FaultClass, code: str) -> str:
    """Normalize to ``miner_fault:<code>`` / ``infra_fault:<code>``."""
    bare = code
    for prefix in (f"{MINER_FAULT}:", f"{INFRA_FAULT}:"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    bare = bare.strip() or "unknown"
    return f"{fault_class}:{bare}"


def evaluate_break_glass(
    request: BreakGlassRequest,
    *,
    fault_reason: str,
    audit_log: BreakGlassAuditLog | None = None,
) -> BreakGlassDecision:
    """Admit only infra_fault runs; refuse miner_fault. Always audit."""
    operator = (request.operator_id or "").strip()
    if not operator:
        entry = _entry(request, fault_reason, admitted=False, detail="missing_operator_id")
        if audit_log is not None:
            audit_log.append(entry)
        return BreakGlassDecision(
            admitted=False, reason="breakglass_missing_operator", audit_entry=entry
        )

    cls = fault_class_of(fault_reason)
    if cls != INFRA_FAULT:
        entry = _entry(
            request,
            fault_reason,
            admitted=False,
            detail="miner_fault_override_refused",
        )
        if audit_log is not None:
            audit_log.append(entry)
        return BreakGlassDecision(
            admitted=False,
            reason="breakglass_refused_miner_fault",
            audit_entry=entry,
        )

    entry = _entry(request, fault_reason, admitted=True, detail="infra_fault_admitted")
    if audit_log is not None:
        audit_log.append(entry)
    return BreakGlassDecision(
        admitted=True, reason="breakglass_admitted", audit_entry=entry
    )


def _entry(
    request: BreakGlassRequest,
    fault_reason: str,
    *,
    admitted: bool,
    detail: str,
) -> dict[str, Any]:
    return {
        "event": "break_glass",
        "admitted": admitted,
        "detail": detail,
        "operator_id": request.operator_id,
        "operator_reason": request.reason,
        "work_unit_id": request.work_unit_id,
        "fault_reason": fault_reason,
        "fault_class": fault_class_of(fault_reason),
        "requested_fault_code": request.fault_code,
        "ts": datetime.now(UTC).isoformat(),
    }


__all__ = [
    "BreakGlassAuditLog",
    "BreakGlassDecision",
    "BreakGlassRequest",
    "INFRA_FAULT",
    "MINER_FAULT",
    "evaluate_break_glass",
    "fault_class_of",
    "format_fault_reason",
]
