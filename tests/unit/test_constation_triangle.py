"""TDD tests for three-way digest triangle (required / Lium / sidecar)."""

from __future__ import annotations

from base.compute.constation_triangle import (
    DigestTriangleResult,
    evaluate_digest_triangle,
)
from base.compute.constation_types import (
    ConstationFailCode,
    FaultClass,
    fault_class_for,
)

DIGEST_A = "sha256:" + ("a" * 64)
DIGEST_B = "sha256:" + ("b" * 64)

_NEW_FAIL_CODES: tuple[ConstationFailCode, ...] = (
    ConstationFailCode.REQUIRED_DIGEST_MISMATCH,
    ConstationFailCode.LIUM_DIGEST_ABSENT,
    ConstationFailCode.SIDECAR_PORT_UNPUBLISHED,
    ConstationFailCode.POD_HOTKEY_MISMATCH,
    ConstationFailCode.POD_NOT_RUNNING,
    ConstationFailCode.SIDECAR_RESPONSE_INVALID,
    ConstationFailCode.BUNDLE_INCOMPLETE,
)

_NEW_MINER_CODES: frozenset[ConstationFailCode] = frozenset(
    {
        ConstationFailCode.REQUIRED_DIGEST_MISMATCH,
        ConstationFailCode.LIUM_DIGEST_ABSENT,
        ConstationFailCode.SIDECAR_PORT_UNPUBLISHED,
        ConstationFailCode.POD_HOTKEY_MISMATCH,
        ConstationFailCode.POD_NOT_RUNNING,
        ConstationFailCode.SIDECAR_RESPONSE_INVALID,
    }
)


def test_triangle_all_equal_ok() -> None:
    # Given three channels report the same digest
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared=DIGEST_A,
        sidecar=DIGEST_A,
    )
    # Then agreement is ok and never elevates (ok/fail only)
    assert isinstance(result, DigestTriangleResult)
    assert result.ok is True
    assert result.fail_code is None


def test_triangle_mismatch_fail_closed() -> None:
    # Given sidecar diverges from required
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared=DIGEST_A,
        sidecar=DIGEST_B,
    )
    # Then fail closed with REQUIRED_DIGEST_MISMATCH
    assert result.ok is False
    assert result.fail_code is ConstationFailCode.REQUIRED_DIGEST_MISMATCH


def test_triangle_lium_divergence_is_corroboration_mismatch() -> None:
    # Given Lium declared diverges from required (sidecar matches required)
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared=DIGEST_B,
        sidecar=DIGEST_A,
    )
    # Then reuse existing CORROBORATION_MISMATCH
    assert result.ok is False
    assert result.fail_code is ConstationFailCode.CORROBORATION_MISMATCH


def test_triangle_absent_lium_fails() -> None:
    # Given Lium declared digest is blank/None
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared=None,
        sidecar=DIGEST_A,
    )
    # Then LIUM_DIGEST_ABSENT
    assert result.ok is False
    assert result.fail_code is ConstationFailCode.LIUM_DIGEST_ABSENT

    blank = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared="   ",
        sidecar=DIGEST_A,
    )
    assert blank.ok is False
    assert blank.fail_code is ConstationFailCode.LIUM_DIGEST_ABSENT


def test_triangle_absent_sidecar_fails() -> None:
    # Given sidecar digest is blank/None
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared=DIGEST_A,
        sidecar=None,
    )
    # Then SIDECAR_RESPONSE_INVALID
    assert result.ok is False
    assert result.fail_code is ConstationFailCode.SIDECAR_RESPONSE_INVALID

    blank = evaluate_digest_triangle(
        required=DIGEST_A,
        lium_declared=DIGEST_A,
        sidecar="  ",
    )
    assert blank.ok is False
    assert blank.fail_code is ConstationFailCode.SIDECAR_RESPONSE_INVALID


def test_triangle_absent_required_fails() -> None:
    # Given required digest is blank/None
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=None,
        lium_declared=DIGEST_A,
        sidecar=DIGEST_A,
    )
    # Then REQUIRED_DIGEST_MISMATCH (we must always know what we require)
    assert result.ok is False
    assert result.fail_code is ConstationFailCode.REQUIRED_DIGEST_MISMATCH

    blank = evaluate_digest_triangle(
        required="",
        lium_declared=DIGEST_A,
        sidecar=DIGEST_A,
    )
    assert blank.ok is False
    assert blank.fail_code is ConstationFailCode.REQUIRED_DIGEST_MISMATCH


def test_triangle_case_and_whitespace_insensitive() -> None:
    # Given digests differ only by case and surrounding whitespace
    upper = "SHA256:" + ("AB" * 32)
    lower_padded = " sha256:" + ("ab" * 32) + " "
    # When evaluate_digest_triangle runs
    result = evaluate_digest_triangle(
        required=upper,
        lium_declared=lower_padded,
        sidecar=lower_padded,
    )
    # Then they compare equal
    assert result.ok is True
    assert result.fail_code is None


def test_new_fail_codes_have_fault_classification() -> None:
    # Given every newly added fail code
    # When fault_class_for is consulted
    # Then each code is classified (miner vs infra)
    for code in _NEW_FAIL_CODES:
        fault = fault_class_for(code)
        if code is ConstationFailCode.BUNDLE_INCOMPLETE:
            assert fault is FaultClass.INFRA, code
        else:
            assert code in _NEW_MINER_CODES
            assert fault is FaultClass.MINER, code
