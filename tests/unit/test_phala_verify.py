"""Unit tests for the validator-owned allowlist + nonce primitives (M4).

Pins measurement allowlist membership + fail-closed empties, and the single-use /
TTL nonce lifecycle the Phala-tier verifier relies on.
"""

from __future__ import annotations

from base.schemas.worker import PhalaMeasurement
from base.worker.phala_verify import (
    InMemoryNonceValidator,
    MeasurementAllowlist,
    NonceState,
    canonical_measurement_mapping,
)

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}


def _phala_measurement() -> PhalaMeasurement:
    return PhalaMeasurement(rtmr3="d" * 96, **MEASUREMENT)


def test_canonical_mapping_from_model_excludes_rtmr3() -> None:
    mapping = canonical_measurement_mapping(_phala_measurement())
    assert mapping == MEASUREMENT
    assert "rtmr3" not in mapping


def test_allowlist_contains_exact_match() -> None:
    allowlist = MeasurementAllowlist.from_measurements([MEASUREMENT])
    assert allowlist.contains(MEASUREMENT) is True
    assert allowlist.contains(_phala_measurement()) is True


def test_allowlist_rejects_single_register_mismatch() -> None:
    allowlist = MeasurementAllowlist.from_measurements([MEASUREMENT])
    off = {**MEASUREMENT, "compose_hash": "0" * 64}
    assert allowlist.contains(off) is False


def test_empty_allowlist_fails_closed() -> None:
    empty = MeasurementAllowlist()
    assert bool(empty) is False
    assert empty.contains(MEASUREMENT) is False


def test_nonce_single_use_lifecycle() -> None:
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    assert nonces.is_outstanding(nonce) is True
    assert nonces.consume(nonce) is NonceState.OK
    assert nonces.is_outstanding(nonce) is False
    assert nonces.consume(nonce) is NonceState.CONSUMED


def test_nonce_unknown_and_empty() -> None:
    nonces = InMemoryNonceValidator()
    assert nonces.consume("never-issued") is NonceState.UNKNOWN
    assert nonces.consume("") is NonceState.UNKNOWN
    assert nonces.is_outstanding("") is False


def test_nonce_expiry() -> None:
    clock = {"t": 0.0}
    nonces = InMemoryNonceValidator(ttl_seconds=10, clock=lambda: clock["t"])
    nonce = nonces.issue()
    clock["t"] = 100.0
    assert nonces.is_outstanding(nonce) is False
    assert nonces.consume(nonce) is NonceState.EXPIRED


def test_issue_accepts_explicit_value() -> None:
    nonces = InMemoryNonceValidator()
    assert nonces.issue("fixed-nonce") == "fixed-nonce"
    assert nonces.consume("fixed-nonce") is NonceState.OK
