"""Pure sealer: assemble prism ``ConstationBundle`` wire dict (no I/O).

Does **not** issue or consume nonces. Callers pass the end-phase nonce and the
last good sidecar signed wire; this module only maps allowlist + run record +
those inputs onto the prism wire field names.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Final

from base.compute.constation_types import ConstationRunRecord
from base.compute.digest_allowlist import DigestRecord

_BUNDLE_INCOMPLETE: Final[str] = "BUNDLE_INCOMPLETE"


def seal_constation_bundle(
    *,
    allowlist_record: DigestRecord,
    run_record: ConstationRunRecord,
    nonce: str,
    signed_attestation: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Build a prism-compatible constation bundle wire dict.

    Field sources:

    * Identity (``commit_sha``, ``tree_sha``, ``variant``, ``digest``,
      ``expected_sealed_manifest_hashes``) — ``allowlist_record``
    * Binding (``work_unit_id``, ``miner_hotkey``, ``pod_id``) — ``run_record``
    * ``nonce`` — caller-supplied end-phase nonce (not issued here)
    * ``signed_attestation`` — last good sidecar answer wire (opaque object)
    * ``reported_sealed_manifest_hashes`` — from sidecar wire payload when
      present and non-empty; otherwise a copy of expected
    * ``lium_declared_digest``, gap budget/observed — ``run_record``

    Raises:
        ValueError: fail-closed when a required field is missing/blank
            (message includes ``BUNDLE_INCOMPLETE``).
    """
    nonce_s = _require_nonblank("nonce", nonce)
    work_unit_id = _require_nonblank("work_unit_id", run_record.work_unit_id)
    miner_hotkey = _require_nonblank("miner_hotkey", run_record.miner_hotkey)
    pod_id = _require_nonblank("pod_id", run_record.pod_id)

    if signed_attestation is None:
        raise ValueError(f"{_BUNDLE_INCOMPLETE}: missing signed_attestation")
    if not isinstance(signed_attestation, Mapping):
        raise ValueError(
            f"{_BUNDLE_INCOMPLETE}: signed_attestation must be a mapping wire object"
        )

    expected = {
        str(path): str(digest)
        for path, digest in allowlist_record.sealed_manifest_hashes.items()
    }
    if not expected:
        raise ValueError(
            f"{_BUNDLE_INCOMPLETE}: expected_sealed_manifest_hashes must be non-empty"
        )

    reported = _reported_sealed_manifest_hashes(signed_attestation, expected)
    wire_attestation: dict[str, object] = copy.deepcopy(dict(signed_attestation))

    variant = allowlist_record.variant
    variant_s = variant.value if hasattr(variant, "value") else str(variant)

    return {
        "commit_sha": str(allowlist_record.commit_sha),
        "tree_sha": str(allowlist_record.tree_sha),
        "variant": variant_s,
        "digest": str(allowlist_record.digest),
        "work_unit_id": work_unit_id,
        "miner_hotkey": miner_hotkey,
        "pod_id": pod_id,
        "nonce": nonce_s,
        "signed_attestation": wire_attestation,
        "expected_sealed_manifest_hashes": dict(expected),
        "reported_sealed_manifest_hashes": dict(reported),
        "lium_declared_digest": run_record.lium_declared_digest,
        "constation_gap_budget_seconds": float(
            run_record.constation_gap_budget_seconds
        ),
        "constation_observed_max_gap_seconds": float(
            run_record.constation_observed_max_gap_seconds
        ),
    }


def _require_nonblank(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{_BUNDLE_INCOMPLETE}: {name} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{_BUNDLE_INCOMPLETE}: {name} must be a non-empty string")
    return stripped


def _reported_sealed_manifest_hashes(
    wire: Mapping[str, Any],
    expected: Mapping[str, str],
) -> dict[str, str]:
    """Prefer sidecar payload hashes; fall back to expected allowlist surface."""
    raw: object | None = None
    payload = wire.get("payload")
    if isinstance(payload, Mapping):
        raw = payload.get("sealed_manifest_hashes")
    if raw is None:
        raw = wire.get("sealed_manifest_hashes")
    parsed = _as_str_str_map(raw)
    if parsed:
        return parsed
    return dict(expected)


def _as_str_str_map(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, Mapping) or not raw:
        return None
    out: dict[str, str] = {}
    for key, value in raw.items():
        path = str(key).strip()
        digest = str(value).strip()
        if not path or not digest:
            return None
        out[path] = digest
    return out or None


__all__ = ["seal_constation_bundle"]
