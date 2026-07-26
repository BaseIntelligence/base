"""TDD tests for constation_ok — sole elevation predicate (checkbox 12 / M14).

constation_ok is the ONLY path that may authorize tier elevation. Each required
mechanism is tested in isolation: omitting or breaking any one yields False with
a distinct machine reason code. Dependencies are injected so tests never need
live Lium or network.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from prism_challenge.constation import (
    CheckOutcome,
    ConstationBundle,
    ConstationFailReason,
    ConstationResult,
    constation_ok,
)

COMMIT = "a" * 40
TREE = "c" * 40
DIGEST = "sha256:" + ("1" * 64)
DIGEST_OTHER = "sha256:" + ("2" * 64)
NONCE = "550e8400-e29b-41d4-a716-446655440000"
POD = "pod_test_001"
HOTKEY = "5FakeHotkeyForUnitTestsOnly000000000000000"
WORK_UNIT = "wu-unit-001"
VARIANT = "cuda"
MANIFEST: dict[str, str] = {
    "src/prism_recipe/harness.py": "a" * 64,
    "src/prism_recipe/gpu_train.py": "b" * 64,
}
GAP_BUDGET = 30.0


def _ok() -> CheckOutcome:
    return CheckOutcome(ok=True, reason="ok")


def _miss(reason: str) -> CheckOutcome:
    return CheckOutcome(ok=False, reason=reason)


def _bundle(**overrides: Any) -> ConstationBundle:
    fields: dict[str, Any] = {
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "variant": VARIANT,
        "digest": DIGEST,
        "work_unit_id": WORK_UNIT,
        "miner_hotkey": HOTKEY,
        "pod_id": POD,
        "nonce": NONCE,
        "signed_attestation": {"schema": "test", "signature": "deadbeef"},
        "expected_sealed_manifest_hashes": dict(MANIFEST),
        "reported_sealed_manifest_hashes": dict(MANIFEST),
        "lium_declared_digest": DIGEST,
        "constation_gap_budget_seconds": GAP_BUDGET,
        "constation_observed_max_gap_seconds": 5.0,
    }
    fields.update(overrides)
    return ConstationBundle(**fields)


@dataclass(frozen=True, slots=True)
class _Injected:
    """Injectable checker outcomes for a single constation_ok call."""

    allowlist: CheckOutcome = CheckOutcome(ok=True, reason="ok")
    nonce: CheckOutcome = CheckOutcome(ok=True, reason="ok")
    signature: CheckOutcome = CheckOutcome(ok=True, reason="ok")


def _run(
    bundle: ConstationBundle,
    inj: _Injected | None = None,
) -> ConstationResult:
    inj = inj or _Injected()

    def check_allowlist(
        *,
        digest: str,
        commit_sha: str,
        tree_sha: str,
        variant: str,
    ) -> CheckOutcome:
        del digest, commit_sha, tree_sha, variant
        return inj.allowlist

    def check_nonce(
        *,
        nonce: str,
        work_unit_id: str,
        miner_hotkey: str,
        pod_id: str,
    ) -> CheckOutcome:
        del nonce, work_unit_id, miner_hotkey, pod_id
        return inj.nonce

    def verify_signature(signed: object) -> CheckOutcome:
        del signed
        return inj.signature

    return constation_ok(
        bundle,
        check_allowlist=check_allowlist,
        check_nonce=check_nonce,
        verify_signature=verify_signature,
    )


def test_complete_valid_bundle_returns_true() -> None:
    """S1 happy: all six mechanisms pass → ok=True reason=ok."""
    result = _run(_bundle())

    assert result.ok is True
    assert result.reason is ConstationFailReason.OK
    assert bool(result) is True


def test_allowlist_miss_unknown_digest() -> None:
    result = _run(
        _bundle(),
        _Injected(allowlist=_miss("unknown_digest")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST
    assert result.detail == "unknown_digest"


def test_allowlist_miss_variant_mismatch() -> None:
    result = _run(
        _bundle(),
        _Injected(allowlist=_miss("variant_mismatch")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.ALLOWLIST_VARIANT_MISMATCH


def test_allowlist_miss_commit_mismatch() -> None:
    result = _run(
        _bundle(),
        _Injected(allowlist=_miss("commit_mismatch")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.ALLOWLIST_COMMIT_MISMATCH


def test_allowlist_miss_revoked() -> None:
    result = _run(
        _bundle(),
        _Injected(allowlist=_miss("revoked")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.ALLOWLIST_REVOKED


def test_nonce_already_consumed() -> None:
    result = _run(
        _bundle(),
        _Injected(nonce=_miss("already_consumed")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.NONCE_ALREADY_CONSUMED


def test_nonce_unknown() -> None:
    result = _run(
        _bundle(),
        _Injected(nonce=_miss("unknown_nonce")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.NONCE_UNKNOWN


def test_nonce_expired() -> None:
    result = _run(
        _bundle(),
        _Injected(nonce=_miss("expired")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.NONCE_EXPIRED


def test_signature_mismatch() -> None:
    result = _run(
        _bundle(),
        _Injected(signature=_miss("signature_mismatch")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.SIGNATURE_INVALID
    assert result.detail == "signature_mismatch"


def test_sealed_manifest_mismatch() -> None:
    bad_manifest = {**MANIFEST, "src/prism_recipe/harness.py": "f" * 64}
    result = _run(
        _bundle(reported_sealed_manifest_hashes=bad_manifest),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.SEALED_MANIFEST_MISMATCH


def test_corroboration_mismatch_fails() -> None:
    """Negative-only: Lium-declared digest disagrees with sidecar digest."""
    result = _run(_bundle(lium_declared_digest=DIGEST_OTHER))

    assert result.ok is False
    assert result.reason is ConstationFailReason.CORROBORATION_MISMATCH


def test_corroboration_agreement_alone_insufficient() -> None:
    """Agreement contributes nothing: allowlist miss still fails."""
    result = _run(
        _bundle(lium_declared_digest=DIGEST),
        _Injected(allowlist=_miss("unknown_digest")),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST


def test_constation_gap_beyond_budget() -> None:
    result = _run(
        _bundle(
            constation_gap_budget_seconds=10.0,
            constation_observed_max_gap_seconds=10.0001,
        ),
    )

    assert result.ok is False
    assert result.reason is ConstationFailReason.CONSTATION_GAP


def test_gap_within_budget_ok() -> None:
    result = _run(
        _bundle(
            constation_gap_budget_seconds=10.0,
            constation_observed_max_gap_seconds=10.0,
        ),
    )

    assert result.ok is True
    assert result.reason is ConstationFailReason.OK


@pytest.mark.parametrize(
    ("label", "bundle_kw", "inj", "expected"),
    [
        (
            "allowlist",
            {},
            _Injected(allowlist=_miss("unknown_digest")),
            ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST,
        ),
        (
            "nonce",
            {},
            _Injected(nonce=_miss("already_consumed")),
            ConstationFailReason.NONCE_ALREADY_CONSUMED,
        ),
        (
            "signature",
            {},
            _Injected(signature=_miss("signature_mismatch")),
            ConstationFailReason.SIGNATURE_INVALID,
        ),
        (
            "sealed_manifest",
            {
                "reported_sealed_manifest_hashes": {
                    **MANIFEST,
                    "src/prism_recipe/harness.py": "0" * 64,
                }
            },
            _Injected(),
            ConstationFailReason.SEALED_MANIFEST_MISMATCH,
        ),
        (
            "corroboration",
            {"lium_declared_digest": DIGEST_OTHER},
            _Injected(),
            ConstationFailReason.CORROBORATION_MISMATCH,
        ),
        (
            "constation_gap",
            {
                "constation_gap_budget_seconds": 1.0,
                "constation_observed_max_gap_seconds": 2.0,
            },
            _Injected(),
            ConstationFailReason.CONSTATION_GAP,
        ),
    ],
    ids=[
        "allowlist",
        "nonce",
        "signature",
        "sealed_manifest",
        "corroboration",
        "constation_gap",
    ],
)
def test_each_single_mechanism_omission_fails_with_distinct_reason(
    label: str,
    bundle_kw: dict[str, Any],
    inj: _Injected,
    expected: ConstationFailReason,
) -> None:
    """Parameterized: each single-field/mechanism break → False + distinct reason."""
    del label
    result = _run(_bundle(**bundle_kw), inj)

    assert result.ok is False
    assert result.reason is expected
    assert result.reason is not ConstationFailReason.OK


def test_omission_reasons_are_pairwise_distinct() -> None:
    """The six mechanism failure reasons used by the param table must all differ."""
    reasons = [
        ConstationFailReason.ALLOWLIST_UNKNOWN_DIGEST,
        ConstationFailReason.NONCE_ALREADY_CONSUMED,
        ConstationFailReason.SIGNATURE_INVALID,
        ConstationFailReason.SEALED_MANIFEST_MISMATCH,
        ConstationFailReason.CORROBORATION_MISMATCH,
        ConstationFailReason.CONSTATION_GAP,
    ]
    assert len(reasons) == len(set(reasons))


def test_module_docstring_states_sole_elevation_predicate_m14() -> None:
    import prism_challenge.constation as mod

    doc = (mod.__doc__ or "") + (constation_ok.__doc__ or "")
    lowered = doc.lower()
    assert "sole" in lowered or "only" in lowered
    assert "tier" in lowered
    assert "m14" in lowered or "no other" in lowered


def test_module_exposes_no_tier_grant_api() -> None:
    import prism_challenge.constation as mod

    forbidden = {
        "effective_tier",
        "grant_tier",
        "elevate_tier",
        "set_tier",
        "compute_tier",
    }
    names = {n for n in dir(mod) if not n.startswith("_")}
    assert names.isdisjoint(forbidden)


def test_checkers_receive_bundle_fields() -> None:
    """Injected checkers are called with the bundle's identity fields."""
    seen: dict[str, Any] = {}

    def check_allowlist(
        *,
        digest: str,
        commit_sha: str,
        tree_sha: str,
        variant: str,
    ) -> CheckOutcome:
        seen["allowlist"] = (digest, commit_sha, tree_sha, variant)
        return _ok()

    def check_nonce(
        *,
        nonce: str,
        work_unit_id: str,
        miner_hotkey: str,
        pod_id: str,
    ) -> CheckOutcome:
        seen["nonce"] = (nonce, work_unit_id, miner_hotkey, pod_id)
        return _ok()

    def verify_signature(signed: object) -> CheckOutcome:
        seen["sig"] = signed
        return _ok()

    bundle = _bundle()
    result = constation_ok(
        bundle,
        check_allowlist=check_allowlist,
        check_nonce=check_nonce,
        verify_signature=verify_signature,
    )

    assert result.ok is True
    assert seen["allowlist"] == (DIGEST, COMMIT, TREE, VARIANT)
    assert seen["nonce"] == (NONCE, WORK_UNIT, HOTKEY, POD)
    assert seen["sig"] == bundle.signed_attestation


def test_missing_lium_corroboration_is_not_contradiction() -> None:
    """Negative-only: absent Lium channel does not fail (not a positive grant)."""
    result = _run(_bundle(lium_declared_digest=None))

    assert result.ok is True
    assert result.reason is ConstationFailReason.OK


def test_result_is_structured_and_frozen() -> None:
    result = _run(_bundle(), _Injected(allowlist=_miss("revoked")))
    assert isinstance(result, ConstationResult)
    assert result.ok is False
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        result.ok = True  # type: ignore[misc]
