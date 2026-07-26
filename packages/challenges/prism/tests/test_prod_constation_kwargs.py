"""Production constation kwargs: deserialize bundle + attach checkers."""

from __future__ import annotations

from prism_challenge.app import _constation_ingest_kwargs
from prism_challenge.config import PrismSettings
from prism_challenge.constation import ConstationBundle, constation_bundle_to_dict


def _bundle_dict() -> dict:
    man = {"h.py": "a" * 64}
    digest = "sha256:" + ("1" * 64)
    b = ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest=digest,
        work_unit_id="wu-1",
        miner_hotkey="hk",
        pod_id="pod",
        nonce="n",
        signed_attestation={"sig": "x"},
        expected_sealed_manifest_hashes=dict(man),
        reported_sealed_manifest_hashes=dict(man),
        lium_declared_digest=digest,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
    )
    return constation_bundle_to_dict(b)


def test_prod_missing_bundle_returns_empty() -> None:
    settings = PrismSettings(
        allow_insecure_signatures=False,
        constation_base_url="http://base.test",
        constation_internal_token="tok",
    )
    assert _constation_ingest_kwargs(settings, {"executed": 1}) == {}


def test_prod_with_bundle_and_checkers() -> None:
    settings = PrismSettings(
        allow_insecure_signatures=False,
        constation_base_url="http://base.test",
        constation_internal_token="tok",
    )
    kwargs = _constation_ingest_kwargs(settings, {"constation_bundle": _bundle_dict()})
    assert "constation_bundle" in kwargs
    assert kwargs["check_allowlist"] is not None
    assert kwargs["check_nonce"] is not None
    assert kwargs["verify_constation_signature"] is not None


def test_insecure_seam_still_injects_without_bundle() -> None:
    settings = PrismSettings(allow_insecure_signatures=True)
    kwargs = _constation_ingest_kwargs(settings, {})
    # Test seam injects synthetic bundle
    assert "constation_bundle" in kwargs
