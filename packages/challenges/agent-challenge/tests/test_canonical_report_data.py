"""Behavioral tests for the architecture-sec-6 ``report_data`` binding (M1).

Fulfils VAL-IMG-012..018:
  * VAL-IMG-012 derivation matches architecture sec 6 exactly (pinned golden vector)
  * VAL-IMG-013 task_ids are sorted before hashing (order-independent binding)
  * VAL-IMG-014 report_data is sensitive to every bound component
  * VAL-IMG-015 a fresh validator nonce changes report_data (anti-replay)
  * VAL-IMG-016 oversized preimage is SHA-256'd to 32 bytes; no >64-byte value
    reaches get_quote; malformed values are rejected
  * VAL-IMG-017 report_data_hex is a 64-byte field whose leading 32 bytes are the digest
  * VAL-IMG-018 scores_digest binds the actual reported scores

The derivation MUST be byte-identical to base's canonical helper
``src/base/worker/proof.py::phala_report_data`` (tag ``base-agent-challenge-v1``,
sorted-key JSON preimage, ``sorted(task_ids)``, ``rtmr3`` excluded). Because the
base package installed in this repo's venv is pinned to origin/main and does not
ship that helper, the algorithm is replicated self-contained here and pinned to a
shared cross-repo golden vector so drift between the two implementations is
caught: :data:`GOLDEN_DIGEST_HEX` / :data:`GOLDEN_FIELD_HEX` are asserted here AND
in ``base/tests/unit/test_worker_proof_phala.py`` against base's helper.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from agent_challenge.canonical import report_data as rd
from agent_challenge.canonical.measurement import CanonicalMeasurement

# --------------------------------------------------------------------------- #
# Shared cross-repo golden vector (fixed inputs -> expected digest/field).
# These exact inputs and expected outputs are also asserted in base against
# base.worker.proof.phala_report_data / phala_report_data_hex. Do NOT change one
# side without changing the other -- that is the whole point of pinning them.
# --------------------------------------------------------------------------- #
GOLDEN_MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}
GOLDEN_AGENT_HASH = "f" * 64
GOLDEN_TASK_IDS = ["task-b", "task-a", "task-c"]
GOLDEN_SCORES_DIGEST = "9" * 64
GOLDEN_NONCE = "nonce-123"

GOLDEN_DIGEST_HEX = "dd2c57688b55e25df20e292b71e1cb97d8501e9280e1dd3475b3e61c30e38cc2"
GOLDEN_FIELD_HEX = GOLDEN_DIGEST_HEX + "00" * 32


def _kwargs(**overrides: object) -> dict[str, object]:
    base = dict(
        canonical_measurement=dict(GOLDEN_MEASUREMENT),
        agent_hash=GOLDEN_AGENT_HASH,
        task_ids=list(GOLDEN_TASK_IDS),
        scores_digest=GOLDEN_SCORES_DIGEST,
        validator_nonce=GOLDEN_NONCE,
    )
    base.update(overrides)
    return base


def _sec6_digest(
    *,
    tag: str,
    measurement: dict[str, str],
    agent_hash: str,
    task_ids: list[str],
    scores_digest: str,
    validator_nonce: str,
) -> bytes:
    """Independent architecture-sec-6 recomputation (mirrors the spec verbatim)."""

    preimage = {
        "tag": tag,
        "canonical_measurement": {
            field: str(measurement[field])
            for field in (
                "mrtd",
                "rtmr0",
                "rtmr1",
                "rtmr2",
                "compose_hash",
                "os_image_hash",
            )
        },
        "agent_hash": agent_hash,
        "task_ids": sorted(task_ids),
        "scores_digest": scores_digest,
        "validator_nonce": validator_nonce,
    }
    encoded = json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).digest()


# --- VAL-IMG-012: golden vector + determinism ------------------------------ #


def test_report_data_matches_pinned_golden_vector() -> None:
    digest = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    assert isinstance(digest, bytes)
    assert len(digest) == 32
    assert digest.hex() == GOLDEN_DIGEST_HEX


def test_report_data_field_matches_pinned_golden_vector() -> None:
    assert rd.report_data_hex(**_kwargs()) == GOLDEN_FIELD_HEX  # type: ignore[arg-type]


def test_report_data_matches_independent_sec6_computation() -> None:
    expected = _sec6_digest(
        tag=rd.PHALA_REPORT_DATA_TAG,
        measurement=GOLDEN_MEASUREMENT,
        agent_hash=GOLDEN_AGENT_HASH,
        task_ids=GOLDEN_TASK_IDS,
        scores_digest=GOLDEN_SCORES_DIGEST,
        validator_nonce=GOLDEN_NONCE,
    )
    assert rd.report_data(**_kwargs()) == expected  # type: ignore[arg-type]


def test_report_data_is_deterministic() -> None:
    assert rd.report_data(**_kwargs()) == rd.report_data(**_kwargs())  # type: ignore[arg-type]


def test_tag_constant_is_base_agent_challenge_v1() -> None:
    assert rd.PHALA_REPORT_DATA_TAG == "base-agent-challenge-v1"


# --- VAL-IMG-013: task_ids order independence ------------------------------ #


def test_task_ids_order_independent() -> None:
    forward = rd.report_data(**_kwargs(task_ids=["task-a", "task-b", "task-c"]))  # type: ignore[arg-type]
    shuffled = rd.report_data(**_kwargs(task_ids=["task-c", "task-a", "task-b"]))  # type: ignore[arg-type]
    assert forward == shuffled


# --- VAL-IMG-014: sensitivity to every bound component --------------------- #


def test_tag_is_bound() -> None:
    real = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    off_tag = _sec6_digest(
        tag="some-other-tag",
        measurement=GOLDEN_MEASUREMENT,
        agent_hash=GOLDEN_AGENT_HASH,
        task_ids=GOLDEN_TASK_IDS,
        scores_digest=GOLDEN_SCORES_DIGEST,
        validator_nonce=GOLDEN_NONCE,
    )
    assert real != off_tag


def test_measurement_is_bound() -> None:
    base = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    changed = rd.report_data(  # type: ignore[arg-type]
        **_kwargs(canonical_measurement=dict(GOLDEN_MEASUREMENT, compose_hash="0" * 64))
    )
    assert changed != base


def test_agent_hash_is_bound() -> None:
    base = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    changed = rd.report_data(**_kwargs(agent_hash="0" * 64))  # type: ignore[arg-type]
    assert changed != base


def test_task_ids_set_is_bound() -> None:
    base = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    changed = rd.report_data(**_kwargs(task_ids=["task-a", "task-b"]))  # type: ignore[arg-type]
    assert changed != base


def test_scores_digest_is_bound() -> None:
    base = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    changed = rd.report_data(**_kwargs(scores_digest="0" * 64))  # type: ignore[arg-type]
    assert changed != base


# --- VAL-IMG-015: fresh nonce changes report_data -------------------------- #


def test_fresh_nonce_changes_report_data() -> None:
    a = rd.report_data(**_kwargs(validator_nonce="nonce-A"))  # type: ignore[arg-type]
    b = rd.report_data(**_kwargs(validator_nonce="nonce-B"))  # type: ignore[arg-type]
    assert a != b


# --- VAL-IMG-016: oversized preimage + malformed rejection ----------------- #


def test_oversized_payload_is_sha256_reduced_not_truncated() -> None:
    payload = b"x" * 200
    field_hex = rd.to_report_data_field(payload)
    field = bytes.fromhex(field_hex)
    assert len(field) == 64
    assert field[:32] == hashlib.sha256(payload).digest()
    assert field[32:] == b"\x00" * 32


def test_value_handed_to_get_quote_never_exceeds_64_bytes() -> None:
    captured: dict[str, bytes] = {}

    def fake_get_quote(report_data: bytes) -> str:
        captured["value"] = report_data
        return "quote"

    # emitter path: sanitize before ever calling get_quote.
    field = bytes.fromhex(rd.to_report_data_field(b"y" * 500))
    fake_get_quote(field)
    assert len(captured["value"]) <= 64


def test_small_payload_is_zero_padded_not_hashed() -> None:
    payload = bytes.fromhex("abcd")
    field = bytes.fromhex(rd.to_report_data_field(payload))
    assert field[:2] == payload
    assert field[2:] == b"\x00" * 62


def test_report_data_field_accepts_the_sec6_digest_unchanged() -> None:
    digest = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    assert rd.to_report_data_field(digest) == GOLDEN_FIELD_HEX


@pytest.mark.parametrize("bad", ["zz" * 4, "abc", "not-hex"])
def test_malformed_hex_value_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        rd.to_report_data_field(bad)


def test_non_bytes_non_str_value_is_rejected() -> None:
    with pytest.raises(TypeError):
        rd.to_report_data_field(12345)  # type: ignore[arg-type]


# --- VAL-IMG-017: report_data_hex field shape ------------------------------ #


def test_report_data_hex_is_128_char_zero_padded_field() -> None:
    digest = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    hex_field = rd.report_data_hex(**_kwargs())  # type: ignore[arg-type]
    assert len(hex_field) == 128
    assert re.fullmatch(r"[0-9a-f]{128}", hex_field)
    field = bytes.fromhex(hex_field)
    assert len(field) == 64
    assert field[:32] == digest
    assert field[32:] == b"\x00" * 32


# --- VAL-IMG-018: scores_digest binds the actual reported scores ----------- #


def test_scores_digest_is_deterministic_and_order_independent() -> None:
    scores = {"task-a": 1.0, "task-b": 0.0, "task-c": 0.5}
    reordered = {"task-c": 0.5, "task-a": 1.0, "task-b": 0.0}
    d1 = rd.scores_digest(scores)
    d2 = rd.scores_digest(reordered)
    assert d1 == d2
    assert re.fullmatch(r"[0-9a-f]{64}", d1)


def test_report_data_binds_the_actual_scores() -> None:
    scores = {"task-a": 1.0, "task-b": 0.0, "task-c": 0.5}
    bound = rd.report_data(**_kwargs(scores_digest=rd.scores_digest(scores)))  # type: ignore[arg-type]

    # Recomputing scores_digest from the reported scores reproduces the binding.
    recomputed = rd.report_data(**_kwargs(scores_digest=rd.scores_digest(scores)))  # type: ignore[arg-type]
    assert recomputed == bound

    # Altering any reported score changes the digest and therefore report_data.
    altered = dict(scores, **{"task-a": 0.0})
    tampered = rd.report_data(**_kwargs(scores_digest=rd.scores_digest(altered)))  # type: ignore[arg-type]
    assert tampered != bound


# --- measurement input shapes --------------------------------------------- #


def test_accepts_canonical_measurement_dataclass() -> None:
    cm = CanonicalMeasurement(**GOLDEN_MEASUREMENT)
    from_dataclass = rd.report_data(**_kwargs(canonical_measurement=cm))  # type: ignore[arg-type]
    from_mapping = rd.report_data(**_kwargs())  # type: ignore[arg-type]
    assert from_dataclass == from_mapping
    assert from_dataclass.hex() == GOLDEN_DIGEST_HEX


def test_mapping_measurement_ignores_runtime_rtmr3() -> None:
    with_rtmr3 = rd.report_data(  # type: ignore[arg-type]
        **_kwargs(canonical_measurement=dict(GOLDEN_MEASUREMENT, rtmr3="d" * 96))
    )
    other_rtmr3 = rd.report_data(  # type: ignore[arg-type]
        **_kwargs(canonical_measurement=dict(GOLDEN_MEASUREMENT, rtmr3="7" * 96))
    )
    assert with_rtmr3 == other_rtmr3 == rd.report_data(**_kwargs())  # type: ignore[arg-type]
