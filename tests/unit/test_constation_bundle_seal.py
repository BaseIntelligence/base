"""TDD: pure seal_constation_bundle → prism ConstationBundle wire dict."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from base.compute.constation_types import (
    ConstationFailCode,
    ConstationRunRecord,
    CorroborationStatus,
    FaultClass,
)
from base.compute.digest_allowlist import DigestRecord, ImageVariant
from base.master.constation.bundle_seal import seal_constation_bundle

COMMIT = "a" * 40
TREE = "b" * 40
DIGEST = "sha256:" + ("1" * 64)
MANIFEST = {"src/harness.py": "c" * 64}
WIRE_MANIFEST = {"src/harness.py": "d" * 64}

_PRISM_WIRE_KEYS = frozenset(
    {
        "commit_sha",
        "tree_sha",
        "variant",
        "digest",
        "work_unit_id",
        "miner_hotkey",
        "pod_id",
        "nonce",
        "signed_attestation",
        "expected_sealed_manifest_hashes",
        "reported_sealed_manifest_hashes",
        "lium_declared_digest",
        "constation_gap_budget_seconds",
        "constation_observed_max_gap_seconds",
    }
)


def _allowlist(
    *,
    sealed: Mapping[str, str] | None = None,
) -> DigestRecord:
    return DigestRecord(
        commit_sha=COMMIT,
        tree_sha=TREE,
        variant=ImageVariant.CUDA,
        digest=DIGEST,
        sealed_manifest_hashes=dict(sealed if sealed is not None else MANIFEST),
    )


def _run(
    *,
    work_unit_id: str = "wu-1",
    miner_hotkey: str = "hk-miner",
    pod_id: str = "pod-9",
    lium: str | None = DIGEST,
    gap_budget: float = 30.0,
    gap_obs: float = 1.5,
) -> ConstationRunRecord:
    return ConstationRunRecord(
        ok=True,
        reason=ConstationFailCode.OK,
        fault_class=None,
        miner_hotkey=miner_hotkey,
        work_unit_id=work_unit_id,
        pod_id=pod_id,
        sidecar_digest=DIGEST,
        lium_declared_digest=lium,
        constation_gap_budget_seconds=gap_budget,
        constation_observed_max_gap_seconds=gap_obs,
        corroboration_status=CorroborationStatus.AGREE,
        samples=(),
    )


def _sidecar_wire(
    *,
    sealed: Mapping[str, str] | None = WIRE_MANIFEST,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "digest": DIGEST,
        "nonce": "poll-n",
        "pod_id": "pod-9",
        "variant": "cuda",
        "build_secret_response": "ab" * 32,
    }
    if sealed is not None:
        payload["sealed_manifest_hashes"] = dict(sealed)
    return {
        "payload": payload,
        "signature": "ef" * 32,
        "algorithm": "hmac-sha256",
        "schema_version": "prism_attestation_payload.v1",
        "phase": "end",
    }


def _prism_from_dict():
    """Load constation_bundle_from_dict when prism package is on path."""
    try:
        from prism_challenge.constation import constation_bundle_from_dict

        return constation_bundle_from_dict
    except ImportError:
        pass
    root = Path(__file__).resolve().parents[2]
    prism_src = root / "packages" / "challenges" / "prism" / "src"
    if not prism_src.is_dir():
        return None
    inserted = str(prism_src)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    try:
        from prism_challenge.constation import constation_bundle_from_dict

        return constation_bundle_from_dict
    except ImportError:
        return None


def test_seal_happy_path_wire_fields() -> None:
    # Given allowlist identity, run record, end-phase nonce, last sidecar wire
    allow = _allowlist()
    run = _run()
    wire = _sidecar_wire()
    nonce = "end-phase-nonce-abc"

    # When seal_constation_bundle assembles the prism wire dict
    out = seal_constation_bundle(
        allowlist_record=allow,
        run_record=run,
        nonce=nonce,
        signed_attestation=wire,
    )

    # Then identity + binding + gap fields match sources; keys are prism wire
    assert set(out) == _PRISM_WIRE_KEYS
    assert out["commit_sha"] == COMMIT
    assert out["tree_sha"] == TREE
    assert out["variant"] == "cuda"
    assert out["digest"] == DIGEST
    assert out["work_unit_id"] == "wu-1"
    assert out["miner_hotkey"] == "hk-miner"
    assert out["pod_id"] == "pod-9"
    assert out["nonce"] == nonce
    assert out["signed_attestation"] == wire
    assert out["expected_sealed_manifest_hashes"] == dict(MANIFEST)
    assert out["reported_sealed_manifest_hashes"] == dict(WIRE_MANIFEST)
    assert out["lium_declared_digest"] == DIGEST
    assert out["constation_gap_budget_seconds"] == 30.0
    assert out["constation_observed_max_gap_seconds"] == 1.5


def test_seal_reported_hashes_fallback_to_expected_when_absent_on_wire() -> None:
    # Given sidecar wire without sealed_manifest_hashes
    allow = _allowlist()
    run = _run()
    wire = _sidecar_wire(sealed=None)
    assert "sealed_manifest_hashes" not in wire["payload"]

    # When sealed
    out = seal_constation_bundle(
        allowlist_record=allow,
        run_record=run,
        nonce="n1",
        signed_attestation=wire,
    )

    # Then reported copies expected (allowlist sealed surface)
    assert out["reported_sealed_manifest_hashes"] == dict(MANIFEST)
    assert out["expected_sealed_manifest_hashes"] == dict(MANIFEST)


def test_seal_missing_signed_attestation_raises_bundle_incomplete() -> None:
    # Given no last good sidecar wire
    # When seal runs
    # Then fail-closed with BUNDLE_INCOMPLETE
    with pytest.raises(ValueError, match="BUNDLE_INCOMPLETE"):
        seal_constation_bundle(
            allowlist_record=_allowlist(),
            run_record=_run(),
            nonce="n1",
            signed_attestation=None,
        )


def test_seal_empty_nonce_raises_bundle_incomplete() -> None:
    with pytest.raises(ValueError, match="BUNDLE_INCOMPLETE"):
        seal_constation_bundle(
            allowlist_record=_allowlist(),
            run_record=_run(),
            nonce="   ",
            signed_attestation=_sidecar_wire(),
        )


def test_seal_blank_binding_fields_raise() -> None:
    with pytest.raises(ValueError, match="BUNDLE_INCOMPLETE"):
        seal_constation_bundle(
            allowlist_record=_allowlist(),
            run_record=_run(work_unit_id=""),
            nonce="n1",
            signed_attestation=_sidecar_wire(),
        )


def test_seal_lium_declared_digest_none_allowed() -> None:
    out = seal_constation_bundle(
        allowlist_record=_allowlist(),
        run_record=_run(lium=None),
        nonce="n1",
        signed_attestation=_sidecar_wire(),
    )
    assert out["lium_declared_digest"] is None


def test_seal_does_not_mutate_inputs() -> None:
    allow = _allowlist()
    run = _run()
    wire = _sidecar_wire()
    wire_before = {
        "payload": dict(wire["payload"]),
        "signature": wire["signature"],
        "algorithm": wire["algorithm"],
        "schema_version": wire["schema_version"],
        "phase": wire["phase"],
    }
    out = seal_constation_bundle(
        allowlist_record=allow,
        run_record=run,
        nonce="n1",
        signed_attestation=wire,
    )
    out["nonce"] = "mutated"
    nested = cast(dict[str, object], out["expected_sealed_manifest_hashes"])
    nested["x"] = "y"
    assert wire == wire_before
    assert "x" not in allow.sealed_manifest_hashes


def test_seal_roundtrip_prism_constation_bundle_from_dict() -> None:
    # Given a sealed wire dict
    from_dict = _prism_from_dict()
    if from_dict is None:
        pytest.skip("prism_challenge not importable")

    out = seal_constation_bundle(
        allowlist_record=_allowlist(),
        run_record=_run(),
        nonce="roundtrip-nonce",
        signed_attestation=_sidecar_wire(),
    )

    # When parsed by prism boundary
    bundle = from_dict(out)

    # Then all identity/binding/gap fields survive
    assert bundle.commit_sha == COMMIT
    assert bundle.tree_sha == TREE
    assert bundle.variant == "cuda"
    assert bundle.digest == DIGEST
    assert bundle.work_unit_id == "wu-1"
    assert bundle.miner_hotkey == "hk-miner"
    assert bundle.pod_id == "pod-9"
    assert bundle.nonce == "roundtrip-nonce"
    assert bundle.signed_attestation == out["signed_attestation"]
    assert dict(bundle.expected_sealed_manifest_hashes) == dict(MANIFEST)
    assert dict(bundle.reported_sealed_manifest_hashes) == dict(WIRE_MANIFEST)
    assert bundle.lium_declared_digest == DIGEST
    assert bundle.constation_gap_budget_seconds == 30.0
    assert bundle.constation_observed_max_gap_seconds == 1.5


def test_bundle_incomplete_code_exists_for_callers() -> None:
    # Adjacent: fail code available for infra attribution if callers catch
    assert ConstationFailCode.BUNDLE_INCOMPLETE.value == "bundle_incomplete"
    assert FaultClass.INFRA.value == "infra_fault"
