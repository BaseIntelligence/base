"""Wire serdes for ConstationBundle (production HTTP path)."""

from __future__ import annotations

import pytest

from prism_challenge.constation import (
    ConstationBundle,
    constation_bundle_from_dict,
    constation_bundle_to_dict,
)


def _bundle() -> ConstationBundle:
    man = {"h.py": "a" * 64}
    digest = "sha256:" + ("1" * 64)
    return ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest=digest,
        work_unit_id="wu-1",
        miner_hotkey="hk",
        pod_id="pod",
        nonce="nonce-1",
        signed_attestation={"sig": "x"},
        expected_sealed_manifest_hashes=dict(man),
        reported_sealed_manifest_hashes=dict(man),
        lium_declared_digest=digest,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
    )


def test_roundtrip() -> None:
    b = _bundle()
    again = constation_bundle_from_dict(constation_bundle_to_dict(b))
    assert again == b


def test_missing_field_raises() -> None:
    raw = constation_bundle_to_dict(_bundle())
    del raw["nonce"]
    with pytest.raises(ValueError, match="missing"):
        constation_bundle_from_dict(raw)


def test_non_object_raises() -> None:
    with pytest.raises(ValueError, match="object"):
        constation_bundle_from_dict([])
