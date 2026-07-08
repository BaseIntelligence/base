"""Unit tests for the validator-owned allowlist + nonce primitives (M4).

Pins measurement allowlist membership + fail-closed empties, and the single-use /
TTL nonce lifecycle the Phala-tier verifier relies on.
"""

from __future__ import annotations

import json

from base.schemas.worker import PhalaMeasurement
from base.worker.phala_verify import (
    MEASUREMENT_ALLOWLIST_FILE_ENV,
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

#: A second, distinct canonical measurement (image rotation: old + new).
OTHER_MEASUREMENT = {
    "mrtd": "f" * 96,
    "rtmr0": "a0" * 48,
    "rtmr1": "a1" * 48,
    "rtmr2": "a2" * 48,
    "compose_hash": "d" * 64,
    "os_image_hash": "9" * 64,
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


# --- VAL-VERIFY-027: multiple canonical entries (image rotation) -------------


def test_allowlist_holds_multiple_entries_matches_any() -> None:
    rotation = MeasurementAllowlist.from_measurements([MEASUREMENT, OTHER_MEASUREMENT])
    assert len(rotation.entries) == 2
    # Either allowlisted entry (outgoing + incoming image) matches.
    assert rotation.contains(MEASUREMENT) is True
    assert rotation.contains(OTHER_MEASUREMENT) is True
    # A measurement listed under neither entry is rejected.
    third = {**MEASUREMENT, "compose_hash": "1" * 64}
    assert rotation.contains(third) is False


# --- VAL-VERIFY-025: fail-closed loading (empty / missing / unparseable) -----


def test_from_json_parses_entries_object_and_bare_list() -> None:
    obj = MeasurementAllowlist.from_json(json.dumps({"entries": [MEASUREMENT]}))
    assert obj.contains(MEASUREMENT) is True
    bare = MeasurementAllowlist.from_json(json.dumps([MEASUREMENT, OTHER_MEASUREMENT]))
    assert bare.contains(MEASUREMENT) is True
    assert bare.contains(OTHER_MEASUREMENT) is True


def test_from_json_unparseable_fails_closed() -> None:
    broken = MeasurementAllowlist.from_json("{ this is : not json ]")
    assert bool(broken) is False
    assert broken.contains(MEASUREMENT) is False


def test_from_json_wrong_json_shape_fails_closed() -> None:
    assert bool(MeasurementAllowlist.from_json(json.dumps(42))) is False
    assert bool(MeasurementAllowlist.from_json(json.dumps("nope"))) is False
    assert bool(MeasurementAllowlist.from_json(json.dumps({"other": []}))) is False


def test_from_json_malformed_entry_fails_closed() -> None:
    missing_register = {k: v for k, v in MEASUREMENT.items() if k != "mrtd"}
    assert bool(MeasurementAllowlist.from_json(json.dumps([missing_register]))) is False
    assert bool(MeasurementAllowlist.from_json(json.dumps(["not-a-mapping"]))) is False


def test_from_file_missing_fails_closed(tmp_path) -> None:
    absent = MeasurementAllowlist.from_file(tmp_path / "does-not-exist.json")
    assert bool(absent) is False


def test_from_file_reads_entries(tmp_path) -> None:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"entries": [MEASUREMENT]}), encoding="utf-8")
    loaded = MeasurementAllowlist.from_file(path)
    assert loaded.contains(MEASUREMENT) is True


def test_from_env_unconfigured_fails_closed() -> None:
    assert bool(MeasurementAllowlist.from_env(env={})) is False
    assert (
        bool(MeasurementAllowlist.from_env(env={MEASUREMENT_ALLOWLIST_FILE_ENV: ""}))
        is False
    )


def test_from_env_reads_configured_file(tmp_path) -> None:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps([MEASUREMENT, OTHER_MEASUREMENT]), encoding="utf-8")
    loaded = MeasurementAllowlist.from_env(
        env={MEASUREMENT_ALLOWLIST_FILE_ENV: str(path)}
    )
    assert loaded.contains(MEASUREMENT) is True
    assert loaded.contains(OTHER_MEASUREMENT) is True


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
