"""Deterministic measurement reproduction + allowlist verdict for the miner CLI.

The validator pins a canonical eval image by its measurement record
``{mrtd, rtmr0, rtmr1, rtmr2, compose_hash, os_image_hash}`` (architecture §6/§7).
The miner reproduces the *same* record from the same pinned image + compose so
both sides agree on the allowlist (VAL-DEPLOY-003/004), and the CLI reports a
run's measurement together with a correct in-allowlist verdict (VAL-DEPLOY-012).

Reproduction wraps :mod:`agent_challenge.canonical.measurement` (``dstack-mr`` +
normalized compose-hash); the verdict compares the canonical six-field subset
against a validator-owned allowlist (a JSON list of entries, or an entries file in
the key-release allowlist format), ignoring any extra register such as
``key_provider`` that the run measurement does not carry.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_challenge.canonical.measurement import (
    CANONICAL_MEASUREMENT_FIELDS,
    CanonicalMeasurement,
    build_canonical_measurement,
    measurement_uses_product_os_identity,
    product_os_image_hash,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class MeasurementError(ValueError):
    """A measurement input (record or allowlist) is malformed."""


class ProvisionOsIdentityError(MeasurementError):
    """Phala provision OS field fails the product identity policy."""


def verify_provision_os_identity(
    *,
    measurement: Mapping[str, Any],
    provision_os: Any,
    mismatch_message: str,
) -> None:
    """Bind Phala provision OS against the sealed measurement without overloading.

    Product Mode B residual (bd369a catalog vs 5c6d register formula):

    * When the sealed ``measurement.os_image_hash`` is the **product formula**
      of ``measurement`` registers (``sha256(MRTD||RTMR1||RTMR2)``), Phala
      provision may return a *different* dstack/teepod **catalog** digest on the
      wire field also named ``os_image_hash``. Do not reject that residual by
      equating catalog to the product seal.
    * If the seal also declares optional ``dstack_mr_image`` (allowlisted catalog
      pin), provision catalog must equal that pin (no invent loophole).
    * When the seal is a legacy pin that still overloads catalog as
      ``os_image_hash`` (not product-formula-consistent), keep fail-closed
      equality vs provision (pre-fix residual behaviour).
    """

    expected_os = measurement.get("os_image_hash")
    if not isinstance(expected_os, str) or not expected_os:
        return
    if not isinstance(provision_os, str) or not provision_os:
        raise ProvisionOsIdentityError(mismatch_message)

    expected_norm = expected_os.strip().lower()
    provision_norm = provision_os.strip().lower()
    if provision_norm == expected_norm:
        return

    # Optional allowlisted catalog digest (never invent when pin names one).
    catalog_pin = measurement.get("dstack_mr_image")
    if isinstance(catalog_pin, str) and catalog_pin.strip():
        pin_norm = catalog_pin.strip().lower()
        if _SHA256_HEX_RE.fullmatch(pin_norm) is None:
            raise ProvisionOsIdentityError(
                "measurement dstack_mr_image catalog pin is not a 64-char hex digest"
            )
        if provision_norm != pin_norm:
            raise ProvisionOsIdentityError(
                "Phala provision os_image_hash mismatches allowlisted dstack_mr_image catalog pin"
            )
        # Catalog matched pin; product formula seal need not equal provision.
        if measurement_uses_product_os_identity(measurement):
            return
        # Pin named a catalog without product-consistent seal: still accept
        # only when catalog pin matched (already verified). Fail if product
        # seal is neither formula nor equal catalog (handled below for legacy).
        return

    if measurement_uses_product_os_identity(measurement):
        # Product seal is guest-bind identity; provision catalog is observed only.
        if _SHA256_HEX_RE.fullmatch(provision_norm) is None:
            raise ProvisionOsIdentityError(
                "Phala provision os_image_hash is not a 64-char hex digest"
            )
        return

    raise ProvisionOsIdentityError(mismatch_message)


def reproduce_measurement(
    *,
    metadata_path: Path | str,
    cpu: int,
    memory: int | str,
    compose: Mapping[str, Any] | str,
    dstack_mr_bin: str | None = None,
) -> CanonicalMeasurement:
    """Recompute the canonical measurement record for a pinned image + compose.

    Deterministic: the same inputs always yield the same record, and
    ``.to_json()`` is a byte-stable serialization a validator can pin verbatim
    (VAL-DEPLOY-003).
    """

    return build_canonical_measurement(
        metadata_path=metadata_path,
        cpu=cpu,
        memory=memory,
        compose=compose,
        dstack_mr_bin=dstack_mr_bin,
    )


def canonical_measurement_subset(measurement: Mapping[str, Any]) -> dict[str, str]:
    """Extract the six canonical (allowlist-pinnable) fields; fail closed if absent.

    ``rtmr3`` and other runtime/extra registers (e.g. ``key_provider``) are
    excluded — the canonical set is exactly :data:`CANONICAL_MEASUREMENT_FIELDS`.
    """

    if not isinstance(measurement, Mapping):
        raise MeasurementError("measurement must be a mapping")
    subset: dict[str, str] = {}
    for field in CANONICAL_MEASUREMENT_FIELDS:
        value = measurement.get(field)
        if not isinstance(value, str) or not value:
            raise MeasurementError(f"measurement is missing/invalid canonical field {field!r}")
        subset[field] = value.strip().lower()
    return subset


def load_allowlist_entries(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Load a validator allowlist as a list of canonical six-field entries.

    Accepts a JSON file path, a JSON string, or an already-parsed iterable of
    mappings. A top-level ``{"entries": [...]}`` wrapper (the key-release allowlist
    file format) is unwrapped. Each entry must carry the six canonical fields; any
    extra register is ignored.
    """

    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8") if _looks_like_path(source) else str(source)
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MeasurementError(f"allowlist is not valid JSON: {exc}") from exc
    else:
        data = list(source)

    if isinstance(data, Mapping):
        data = data.get("entries", [])
    if not isinstance(data, list):
        raise MeasurementError("allowlist must be a list of entries or {'entries': [...]}")
    return [canonical_measurement_subset(entry) for entry in data]


def _looks_like_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    stripped = source.strip()
    # A JSON document starts with '[' or '{'; anything else is treated as a path.
    return not stripped.startswith(("[", "{"))


@dataclass(frozen=True)
class AllowlistVerdict:
    """The reported measurement plus its in-allowlist decision (VAL-DEPLOY-012)."""

    measurement: dict[str, str]
    in_allowlist: bool
    matched_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "measurement": self.measurement,
            "in_allowlist": self.in_allowlist,
            "verdict": "IN-LIST" if self.in_allowlist else "NOT-IN-LIST",
            "matched_index": self.matched_index,
        }


def allowlist_verdict(
    measurement: Mapping[str, Any],
    allowlist: str | Path | Iterable[Mapping[str, Any]],
) -> AllowlistVerdict:
    """Report a measurement's six-field subset and whether it is in the allowlist.

    A measurement matching an allowlist entry on ALL six canonical fields is
    IN-LIST; any single-field difference is NOT-IN-LIST (VAL-DEPLOY-012). An empty
    allowlist matches nothing (fail closed).
    """

    subset = canonical_measurement_subset(measurement)
    entries = load_allowlist_entries(allowlist)
    for index, entry in enumerate(entries):
        if entry == subset:
            return AllowlistVerdict(measurement=subset, in_allowlist=True, matched_index=index)
    return AllowlistVerdict(measurement=subset, in_allowlist=False, matched_index=None)


def domain_allowlist_verdict(
    *,
    domain: str,
    measurement: Mapping[str, Any],
    review_allowlist: str | Path | Iterable[Mapping[str, Any]] | None = None,
    eval_allowlist: str | Path | Iterable[Mapping[str, Any]] | None = None,
) -> AllowlistVerdict:
    """Evaluate one measurement only against its validator-owned app domain."""

    if domain not in {"review", "eval"}:
        raise MeasurementError("measurement domain must be review or eval")
    source = review_allowlist if domain == "review" else eval_allowlist
    if source is None:
        raise MeasurementError(f"{domain} validator allowlist is required")
    return allowlist_verdict(measurement, source)


def short_measurement_hex_prefix(value: object, *, length: int = 12) -> str | None:
    """Return a truncated lowercase hex prefix for operator diagnostics.

    Never returns the full digest when the source is longer than ``length``.
    Rejects non-hex input so callers never echo secrets as opaque blobs.
    """

    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned or any(ch not in "0123456789abcdef" for ch in cleaned):
        return None
    if length < 1:
        return None
    return cleaned[:length]


def format_eval_shape_mismatch_error(
    *,
    plan_instance_type: str,
    requested_instance_type: str,
    plan_vm_shape: str | None = None,
    plan_rtmr0: str | None = None,
) -> str:
    """Loud, operator-facing message for plan vs CLI eval shape mismatch.

    Names ``vm_shape`` / ``instance_type``, both shapes, the rtmr0 allowlist
    footgun, and that prepare already spent the one-shot token so a re-prepare
    (next deploy cancel+retry) is required. Truncates any plan rtmr0 prefix.
    """

    plan_shape = (plan_vm_shape or plan_instance_type or "").strip() or plan_instance_type
    requested = (requested_instance_type or "").strip() or requested_instance_type
    prefix = short_measurement_hex_prefix(plan_rtmr0) if plan_rtmr0 else None
    rtmr_note = (
        f" Plan measurement rtmr0 prefix={prefix} (truncated; full value never logged)."
        if prefix
        else ""
    )
    return (
        "Eval deployment shape mismatch: validator-issued plan has "
        f"vm_shape/instance_type={plan_shape!r} but CLI --eval-instance-type="
        f"{requested!r}. A shape change requires a matching measurement pin "
        "(rtmr0) on the validator allowlist and a re-prepare; a stale rtmr0 pin "
        "surfaces only as a generic key-release denial deep in the TEE flow — "
        "not as a clear shape error. This abort is before Phala create (no spend). "
        "The one-time EVAL_RUN_TOKEN delivery was already consumed by prepare; "
        "re-run eval deploy (or eval cancel + retry) to re-prepare a fresh attempt."
        f"{rtmr_note}"
    )


def format_rtmr0_pin_mismatch_error(
    *,
    plan_rtmr0: str,
    expected_rtmr0: str,
) -> str:
    """Loud message when plan rtmr0 disagrees with --expected-measurement."""

    plan_prefix = short_measurement_hex_prefix(plan_rtmr0) or "?"
    expected_prefix = short_measurement_hex_prefix(expected_rtmr0) or "?"
    return (
        "Eval measurement pin mismatch on field rtmr0: "
        f"plan prefix={plan_prefix} vs --expected-measurement prefix={expected_prefix} "
        "(truncated; full digests never logged). A stale rtmr0 pin on the validator "
        "allowlist surfaces only as a generic key-release denial. Aborting before "
        "Phala create (no spend). Re-prepare after the allowlist pin matches the "
        "deployed shape."
    )


def load_expected_measurement_mapping(path: str | Path) -> dict[str, Any]:
    """Load a miner-supplied expected measurement JSON object from PATH."""

    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MeasurementError(f"expected measurement file could not be read: {path}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MeasurementError("expected measurement file is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise MeasurementError("expected measurement file must contain a JSON object")
    return parsed


def compare_plan_rtmr0_to_expected(
    plan_measurement: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> str | None:
    """Return a loud error string when plan rtmr0 disagrees with expected, else None.

    When ``expected`` has no ``rtmr0`` field the check is a no-op (optional pin).
    """

    expected_rtmr0 = expected.get("rtmr0")
    if expected_rtmr0 is None:
        return None
    if not isinstance(expected_rtmr0, str) or not expected_rtmr0.strip():
        raise MeasurementError("expected measurement rtmr0 must be a non-empty string")
    plan_rtmr0 = plan_measurement.get("rtmr0")
    if not isinstance(plan_rtmr0, str) or not plan_rtmr0.strip():
        raise MeasurementError("plan measurement is missing rtmr0 for pin comparison")
    plan_norm = plan_rtmr0.strip().lower()
    expected_norm = expected_rtmr0.strip().lower()
    if plan_norm.startswith("0x"):
        plan_norm = plan_norm[2:]
    if expected_norm.startswith("0x"):
        expected_norm = expected_norm[2:]
    if plan_norm == expected_norm:
        return None
    return format_rtmr0_pin_mismatch_error(
        plan_rtmr0=plan_rtmr0,
        expected_rtmr0=expected_rtmr0,
    )


def measurements_agree(
    miner_measurement: Mapping[str, Any],
    validator_entry: Mapping[str, Any],
) -> bool:
    """Whether the miner-reproduced record equals a validator allowlist entry.

    Compares the six canonical fields field-for-field (VAL-DEPLOY-004).
    """

    return canonical_measurement_subset(miner_measurement) == canonical_measurement_subset(
        validator_entry
    )


__all__ = [
    "AllowlistVerdict",
    "MeasurementError",
    "ProvisionOsIdentityError",
    "allowlist_verdict",
    "canonical_measurement_subset",
    "compare_plan_rtmr0_to_expected",
    "domain_allowlist_verdict",
    "format_eval_shape_mismatch_error",
    "format_rtmr0_pin_mismatch_error",
    "load_allowlist_entries",
    "load_expected_measurement_mapping",
    "measurement_uses_product_os_identity",
    "measurements_agree",
    "product_os_image_hash",
    "reproduce_measurement",
    "short_measurement_hex_prefix",
    "verify_provision_os_identity",
]
