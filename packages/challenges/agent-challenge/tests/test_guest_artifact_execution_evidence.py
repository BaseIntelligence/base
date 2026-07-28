"""Guest must recompute download + executed artifact hashes (no caller echo).

The TEE guest proves the bytes it downloaded and the bytes it handed to the
orchestrator both match the immutable plan agent_hash. Evidence is a structured
serializable object for a later attestation-envelope fold — not free-text logs.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from agent_challenge.canonical.eval_wire import agent_artifact_sha256_hex, canonical_json_v1
from agent_challenge.evaluation.guest_execution_evidence import (
    GuestArtifactExecutionEvidence,
    prove_guest_artifact_execution,
    prove_guest_artifact_execution_from_path,
    serialize_guest_artifact_execution_evidence,
)
from agent_challenge.evaluation.own_runner_backend import (
    assert_agent_artifact_matches_plan,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# S1 — matching artifact → match=True, execution path proceeds
# --------------------------------------------------------------------------- #


def test_matching_bytes_produce_match_true_evidence() -> None:
    """Given honest download+executed bytes equal to plan, When prove, Then match."""

    payload = b"PK\x03\x04honest-agent-zip-bytes-v1"
    expected = _sha(payload)

    evidence = prove_guest_artifact_execution(
        plan_agent_hash=expected,
        download_bytes=payload,
        executed_bytes=payload,
    )

    assert isinstance(evidence, GuestArtifactExecutionEvidence)
    assert evidence.match is True
    assert evidence.expected_hash == expected
    assert evidence.download_hash == expected
    assert evidence.executed_hash == expected
    assert evidence.byte_size == len(payload)
    assert evidence.download_hash == agent_artifact_sha256_hex(payload)


def test_matching_on_disk_path_proceeds_and_returns_evidence(tmp_path: Path) -> None:
    """Given on-disk ZIP matching plan, When path prove + assert, Then both succeed."""

    payload = b"PK\x03\x04path-mounted-agent"
    expected = _sha(payload)
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(payload)

    evidence = prove_guest_artifact_execution_from_path(
        plan_agent_hash=expected,
        artifact_path=zip_path,
    )
    assert evidence.match is True
    assert (
        assert_agent_artifact_matches_plan(
            artifact_path=zip_path,
            plan_agent_hash=expected,
        )
        == expected
    )


# --------------------------------------------------------------------------- #
# S2 — tamper between download and exec → HARD FAIL
# --------------------------------------------------------------------------- #


def test_tamper_between_download_and_exec_hard_fails() -> None:
    """Given download≠executed, When prove, Then raise and do not return match=True."""

    download = b"PK\x03\x04download-bytes"
    executed = b"PK\x03\x04EXECUTED-SWAPPED-bytes"
    plan = _sha(download)

    with pytest.raises(ValueError, match=r"executed|download|mismatch|artifact"):
        prove_guest_artifact_execution(
            plan_agent_hash=plan,
            download_bytes=download,
            executed_bytes=executed,
        )


def test_path_tamper_after_first_observation_hard_fails(tmp_path: Path) -> None:
    """Given path rewritten after download observation, When dual-read prove, Then fail."""

    honest = b"PK\x03\x04first-observation"
    swapped = b"PK\x03\x04second-observation-SWAP"
    plan = _sha(honest)
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(honest)

    download_bytes = zip_path.read_bytes()
    zip_path.write_bytes(swapped)
    executed_bytes = zip_path.read_bytes()

    with pytest.raises(ValueError, match=r"executed|download|mismatch|artifact"):
        prove_guest_artifact_execution(
            plan_agent_hash=plan,
            download_bytes=download_bytes,
            executed_bytes=executed_bytes,
        )


# --------------------------------------------------------------------------- #
# S3 — plan hash mismatch → HARD FAIL
# --------------------------------------------------------------------------- #


def test_plan_hash_mismatch_hard_fails() -> None:
    """Given bytes that do not match plan agent_hash, When prove, Then HARD FAIL."""

    payload = b"PK\x03\x04real-bytes"
    wrong_plan = _sha(b"other-agent")

    with pytest.raises(ValueError, match=r"plan|agent_hash|mismatch|artifact"):
        prove_guest_artifact_execution(
            plan_agent_hash=wrong_plan,
            download_bytes=payload,
            executed_bytes=payload,
        )


def test_assert_agent_artifact_still_fails_on_plan_mismatch(tmp_path: Path) -> None:
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"agent-a")
    with pytest.raises(ValueError, match="agent artifact"):
        assert_agent_artifact_matches_plan(
            artifact_path=zip_path,
            plan_agent_hash=_sha(b"agent-b"),
        )


# --------------------------------------------------------------------------- #
# S4 — deterministic serialization
# --------------------------------------------------------------------------- #


def test_evidence_serializes_deterministically() -> None:
    """Given same inputs, When serialize twice, Then byte-identical canonical JSON."""

    payload = b"PK\x03\x04serialize-me"
    expected = _sha(payload)
    evidence = prove_guest_artifact_execution(
        plan_agent_hash=expected,
        download_bytes=payload,
        executed_bytes=payload,
    )

    first = serialize_guest_artifact_execution_evidence(evidence)
    second = serialize_guest_artifact_execution_evidence(evidence)
    assert first == second
    assert isinstance(first, (bytes, bytearray))
    # Independent canonicalization of the public dict shape must match.
    assert first == canonical_json_v1(evidence.to_dict())
    # Same logical inputs → same bytes even via a fresh evidence instance.
    again = prove_guest_artifact_execution(
        plan_agent_hash=expected,
        download_bytes=payload,
        executed_bytes=payload,
    )
    assert serialize_guest_artifact_execution_evidence(again) == first


# --------------------------------------------------------------------------- #
# S5 — no caller-supplied hash substitutes for guest computation
# --------------------------------------------------------------------------- #


def test_caller_supplied_download_hash_kwarg_is_rejected() -> None:
    """Given a download_hash kwarg, When prove, Then TypeError (not trusted)."""

    payload = b"PK\x03\x04bytes"
    plan = _sha(payload)
    with pytest.raises(TypeError):
        prove_guest_artifact_execution(  # type: ignore[call-arg]
            plan_agent_hash=plan,
            download_bytes=payload,
            executed_bytes=payload,
            download_hash=plan,
        )


def test_caller_supplied_executed_hash_kwarg_is_rejected() -> None:
    payload = b"PK\x03\x04bytes"
    plan = _sha(payload)
    with pytest.raises(TypeError):
        prove_guest_artifact_execution(  # type: ignore[call-arg]
            plan_agent_hash=plan,
            download_bytes=payload,
            executed_bytes=payload,
            executed_hash=plan,
        )


def test_prove_signature_has_no_hash_input_parameters() -> None:
    """Static guard: prove_* must not accept precomputed digest parameters."""

    sig = inspect.signature(prove_guest_artifact_execution)
    forbidden = {"download_hash", "executed_hash", "match", "declared_agent_hash"}
    assert forbidden.isdisjoint(sig.parameters.keys())
    # Only plan expectation + raw byte buffers.
    assert "download_bytes" in sig.parameters
    assert "executed_bytes" in sig.parameters
    assert "plan_agent_hash" in sig.parameters


def test_hashes_are_computed_from_real_bytes_not_echo() -> None:
    """Mutating only the bytes changes the evidence hashes (proves computation)."""

    a = b"PK\x03\x04aaa"
    b = b"PK\x03\x04bbb"
    ea = prove_guest_artifact_execution(
        plan_agent_hash=_sha(a),
        download_bytes=a,
        executed_bytes=a,
    )
    eb = prove_guest_artifact_execution(
        plan_agent_hash=_sha(b),
        download_bytes=b,
        executed_bytes=b,
    )
    assert ea.download_hash != eb.download_hash
    assert ea.executed_hash != eb.executed_hash
    assert ea.download_hash == _sha(a)
    assert eb.download_hash == _sha(b)


def test_missing_plan_hash_hard_fails() -> None:
    with pytest.raises(ValueError, match=r"agent_hash|missing|plan"):
        prove_guest_artifact_execution(
            plan_agent_hash="",
            download_bytes=b"x",
            executed_bytes=b"x",
        )


def test_missing_bytes_hard_fails() -> None:
    with pytest.raises(ValueError, match=r"bytes|unavailable|artifact"):
        prove_guest_artifact_execution(
            plan_agent_hash="a" * 64,
            download_bytes=b"",
            executed_bytes=b"",
        )


def test_evidence_field_shape_is_stable() -> None:
    """Later attestation fold codes against these exact field names."""

    payload = b"PK\x03\x04shape"
    evidence = prove_guest_artifact_execution(
        plan_agent_hash=_sha(payload),
        download_bytes=payload,
        executed_bytes=payload,
    )
    d = evidence.to_dict()
    assert set(d.keys()) == {
        "schema_version",
        "expected_hash",
        "download_hash",
        "executed_hash",
        "byte_size",
        "match",
    }
    assert d["schema_version"] == 1
    assert isinstance(d["byte_size"], int)
    assert isinstance(d["match"], bool)
