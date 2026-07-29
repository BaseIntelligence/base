"""Attested-result emission removed (T40). Host-trust only."""

from __future__ import annotations

from typing import Any


class AttestationEmissionError(RuntimeError):
    """TEE attested emission is unavailable after Phala removal."""


# Stable result-line key retained so host-trust tests can assert absence.
EXECUTION_PROOF_RESULT_KEY = "execution_proof"


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


__all__ = [
    "AttestationEmissionError",
    "EXECUTION_PROOF_RESULT_KEY",
    "DstackQuoteProvider",
    "build_phala_attestation",
    "emit_attested_eval_result_from_plan",
    "emit_attested_result",
]
