"""Attested-result emission removed (T40). Host-trust only."""

from __future__ import annotations

from typing import Any

# Stable result-line key retained so host-trust tests can assert absence.
EXECUTION_PROOF_RESULT_KEY = "execution_proof"

# Legacy fail-closed reason code (wire/DB compat); not a TEE claim.
PHALA_ATTESTATION_FAILED_REASON = "phala_attestation_failed"


class AttestationEmissionError(RuntimeError):
    """TEE attested emission is unavailable after Phala removal."""


class DstackQuoteProvider:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AttestationEmissionError("DstackQuoteProvider removed with Phala TEE (T40)")

    def get_quote(self, *args: Any, **kwargs: Any) -> Any:
        raise AttestationEmissionError("DstackQuoteProvider removed with Phala TEE (T40)")


def emit_attested_eval_result_from_plan(*args: Any, **kwargs: Any) -> Any:
    raise AttestationEmissionError(
        "emit_attested_eval_result_from_plan removed with Phala TEE (T40); "
        "use host-trust unattested path (mark_result_unattested)"
    )


def emit_attested_result(*args: Any, **kwargs: Any) -> Any:
    raise AttestationEmissionError("emit_attested_result removed with Phala TEE (T40)")


def build_phala_attestation(*args: Any, **kwargs: Any) -> Any:
    raise AttestationEmissionError("build_phala_attestation removed with Phala TEE (T40)")


def emit_failclosed_result(*, total: int = 0, **kwargs: Any) -> None:
    """Print a failed BASE_BENCHMARK_RESULT line (Phala emit path removed)."""
    from agent_challenge.evaluation.own_runner.result_schema import (
        build_benchmark_result,
        emit_benchmark_result_line,
    )

    payload = build_benchmark_result(
        status="failed",
        score=0.0,
        resolved=0,
        total=int(total),
        reason_code=PHALA_ATTESTATION_FAILED_REASON,
    )
    emit_benchmark_result_line(payload)


__all__ = [
    "AttestationEmissionError",
    "DstackQuoteProvider",
    "EXECUTION_PROOF_RESULT_KEY",
    "PHALA_ATTESTATION_FAILED_REASON",
    "build_phala_attestation",
    "emit_attested_eval_result_from_plan",
    "emit_attested_result",
    "emit_failclosed_result",
]
