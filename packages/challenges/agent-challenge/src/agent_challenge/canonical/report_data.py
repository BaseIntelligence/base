"""Architecture-sec-6 ``report_data`` binding for the canonical Phala eval image.

The attested-result envelope binds the exact run into the TDX quote's
``report_data`` field (architecture.md sec 6)::

    report_data = SHA256( tag ∥ canonical_measurement ∥ agent_hash ∥
                          sorted(task_ids) ∥ scores_digest ∥ validator_nonce )

with ``tag == "base-agent-challenge-v1"``. Changing any bound component changes
the digest; ``task_ids`` are sorted so the binding is order-independent; the
measurement contributes only its static, allowlist-pinnable subset (``rtmr3`` is
runtime and excluded); and a fresh validator nonce defeats quote replay.

This derivation is the single source of truth in base
(``src/base/worker/proof.py::phala_report_data``). The base package installed in
this repo's venv is pinned to origin/main and does not ship that helper, so the
algorithm is replicated here **byte-for-byte identically** -- same domain tag,
same canonical sorted-key/compact JSON preimage, same ``sorted(task_ids)``, same
``rtmr3``-excluded measurement subset, same 64-byte left-aligned zero-padded
field. A shared cross-repo golden vector (fixed input -> expected 64-byte hex) is
asserted against BOTH this module and base's helper so the two implementations
cannot silently drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from agent_challenge.canonical.measurement import (
    CANONICAL_MEASUREMENT_FIELDS,
    CanonicalMeasurement,
)

#: Domain-separation tag bound into ``report_data`` (architecture.md sec 6).
#: MUST equal base's ``PHALA_REPORT_DATA_TAG`` so a quote's binding verifies with
#: base's single-source helper.
PHALA_REPORT_DATA_TAG = "base-agent-challenge-v1"

#: Byte width of the TDX ``report_data`` field a quote carries.
PHALA_REPORT_DATA_BYTES = 64

#: Byte width of the SHA-256 sec-6 digest occupying the field's leading bytes.
REPORT_DATA_DIGEST_BYTES = 32


def _canonical_measurement_mapping(
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
) -> dict[str, str]:
    """The static, allowlist-pinnable measurement subset (excludes ``rtmr3``)."""

    if isinstance(canonical_measurement, CanonicalMeasurement):
        source: Mapping[str, Any] = canonical_measurement.as_dict()
    elif isinstance(canonical_measurement, Mapping):
        source = canonical_measurement
    else:
        raise TypeError(
            "canonical_measurement must be a CanonicalMeasurement or mapping, "
            f"not {type(canonical_measurement).__name__}"
        )
    return {field: str(source[field]) for field in CANONICAL_MEASUREMENT_FIELDS}


def report_data(
    *,
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
    agent_hash: str,
    task_ids: Iterable[str],
    scores_digest: str,
    validator_nonce: str | None = None,
    eval_run_id: str | None = None,
    score_nonce: str | None = None,
) -> bytes:
    """The 32-byte ``report_data`` digest binding a Phala run (architecture sec 6).

    Legacy callers retain the original v1 binding with ``validator_nonce``.
    Schema-version-2 Eval callers instead supply the paired ``eval_run_id`` and
    ``score_nonce``.  That strict boundary uses the closed object mandated by
    architecture §6.2 and is byte-identical to BASE's ``phala_report_data``.
    """

    supplied_tasks = list(task_ids)
    if (eval_run_id is None) != (score_nonce is None):
        raise ValueError("eval_run_id and score_nonce must be supplied together")

    if eval_run_id is not None and score_nonce is not None:
        if validator_nonce is not None:
            raise ValueError("schema-version-2 bindings do not use validator_nonce")
        from agent_challenge.canonical.eval_wire import (
            EvalWireError,
            build_score_binding,
            canonical_json_v1,
        )

        try:
            binding = build_score_binding(
                canonical_measurement=_canonical_measurement_mapping(canonical_measurement),
                agent_hash=agent_hash,
                eval_run_id=eval_run_id,
                score_nonce=score_nonce,
                scores_digest=scores_digest,
                task_ids=supplied_tasks,
            )
        except EvalWireError as exc:
            raise ValueError(str(exc)) from exc
        return hashlib.sha256(canonical_json_v1(binding)).digest()

    if validator_nonce is None:
        raise ValueError("validator_nonce is required for legacy report_data bindings")
    preimage = {
        "tag": PHALA_REPORT_DATA_TAG,
        "canonical_measurement": _canonical_measurement_mapping(canonical_measurement),
        "agent_hash": agent_hash,
        "task_ids": sorted(supplied_tasks),
        "scores_digest": scores_digest,
        "validator_nonce": validator_nonce,
    }
    encoded = json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).digest()


def report_data_hex(
    *,
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
    agent_hash: str,
    task_ids: Iterable[str],
    scores_digest: str,
    validator_nonce: str | None = None,
    eval_run_id: str | None = None,
    score_nonce: str | None = None,
) -> str:
    """``report_data`` as a 64-byte TDX field (128 hex chars, left-aligned).

    The 32-byte :func:`report_data` digest occupies the leading bytes and the
    trailing bytes are zero, matching Phala's observed left-aligned zero-pad
    round-trip for the fixed-width quote field. Byte-identical to base's
    ``phala_report_data_hex``.
    """

    digest = report_data(
        canonical_measurement=canonical_measurement,
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=scores_digest,
        validator_nonce=validator_nonce,
        eval_run_id=eval_run_id,
        score_nonce=score_nonce,
    )
    return to_report_data_field(digest)


def to_report_data_field(value: bytes | str) -> str:
    """Normalize an arbitrary ``report_data`` payload to the 64-byte field (hex).

    Guarantees a value handed to ``get_quote`` is never larger than the 64-byte
    TDX field: a raw payload exceeding 64 bytes is SHA-256-reduced to 32 bytes
    (never truncated), and any value is left-aligned zero-padded to exactly
    64 bytes. ``value`` may be raw ``bytes`` or a hex string; a malformed hex
    string (non-hex or odd length) is rejected rather than emitted.
    """

    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"report_data hex is malformed: {exc}") from exc
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise TypeError(f"report_data must be bytes or a hex string, not {type(value).__name__}")

    if len(raw) > PHALA_REPORT_DATA_BYTES:
        raw = hashlib.sha256(raw).digest()
    return raw.ljust(PHALA_REPORT_DATA_BYTES, b"\x00").hex()


def scores_digest(scores: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 (hex) of the canonical per-task scores.

    Hashes the sorted-key, compact JSON serialization of the reported per-task
    scores, so recomputing the digest from the scores an envelope actually
    reports reproduces the value bound into :func:`report_data` -- a score cannot
    be altered without breaking the binding (architecture.md sec 6, VAL-IMG-018).
    """

    if not isinstance(scores, Mapping):
        raise TypeError(f"scores must be a mapping, not {type(scores).__name__}")
    encoded = json.dumps(scores, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PHALA_REPORT_DATA_BYTES",
    "PHALA_REPORT_DATA_TAG",
    "REPORT_DATA_DIGEST_BYTES",
    "report_data",
    "report_data_hex",
    "scores_digest",
    "to_report_data_field",
]
