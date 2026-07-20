"""Deterministic measurement tooling for the canonical Phala eval image.

The validator pins a canonical eval image by its *measurement*: the static,
allowlist-pinnable record ``{mrtd, rtmr0, rtmr1, rtmr2, compose_hash,
os_image_hash}`` (architecture.md sec 6/7). This module computes that record
deterministically so a validator and a miner independently agree on the same
value, and so any change to the image or its compose is detected as measurement
drift.

Two independent inputs feed the record:

* **compose_hash** -- the SHA-256 of the normalized ``app-compose.json`` (the
  same value dstack measures into RTMR3 and Phala returns from
  ``POST /cvms/provision``; see the Phala "verify your application" guide, where
  the compose-hash is ``sha256`` of the app-compose file bytes). Normalization
  (parse then re-serialize with sorted keys + compact separators) makes the hash
  stable and invariant to key ordering / insignificant whitespace, while any
  material change to the compose yields a new, reproducible hash. The bytes this
  module hashes are exactly the bytes a deployer should ship, so the offline hash
  matches the live CVM's ``compose_hash``.

* **mrtd / rtmr0-2 / os_image_hash** -- computed from the pinned dstack OS image
  by ``dstack-mr`` (``go install github.com/kvinwang/dstack-mr@latest``), which
  replays the TDX measurement of firmware/kernel/cmdline+initrd for a given
  vCPU/RAM shape. **Product ``os_image_hash`` is sealed as**
  ``sha256(MRTD || RTMR1 || RTMR2)`` (the same formula guest review, key-release,
  and quote bind recompute from attested registers). This is intentionally
  independent of the **dstack catalog / Phala provision** digest sometimes
  surfaced as ``mr_image`` or provision ``os_image_hash`` (e.g. offline
  ``digest.txt`` / teepod catalog ``bd369a…`` on residual dstack-0.5.9, which is
  **not** equal to the product formula ``5c6d…`` on the same registers). Catalog
  digests may be retained as optional ``dstack_mr_image`` for provision binding
  only; they must never overload the product field.

The MR registers are Intel TDX SHA-384 values (48 bytes -> 96 hex chars);
``os_image_hash`` (product formula) and the compose-hash are SHA-256 (64 hex).

The real ``dstack-mr`` needs the multi-hundred-MB dstack OS image files, so it is
invoked live at M6; here it is wrapped as a configurable subprocess so the
measurement record is produced the same way offline and on a live CVM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: TDX measurement registers (MRTD, RTMR0-2) are SHA-384 -> 48 bytes -> 96 hex.
REGISTER_HEX_WIDTH = 96
#: ``compose_hash`` and ``os_image_hash`` are SHA-256 -> 32 bytes -> 64 hex.
SHA256_HEX_WIDTH = 64

#: Binary used to compute image measurements; overridable via ``DSTACK_MR_BIN``.
DEFAULT_DSTACK_MR_BIN = "dstack-mr"

#: The static, allowlist-pinnable measurement fields (``rtmr3`` is runtime-only).
CANONICAL_MEASUREMENT_FIELDS = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "compose_hash",
    "os_image_hash",
)

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

# --------------------------------------------------------------------------- #
# Normalized compose-hash
# --------------------------------------------------------------------------- #


def normalize_app_compose(compose: Mapping[str, Any] | str) -> str:
    """Return the canonical serialization of an ``app-compose`` document.

    Accepts either a mapping or a JSON string and emits deterministic,
    sorted-key, compact JSON. Two semantically-equivalent inputs (reordered
    keys, differing insignificant whitespace) collapse to identical output, so
    hashing the result is normalization-invariant. These bytes are exactly what
    a deployer should write to ``app-compose.json`` so the offline hash matches
    the value dstack measures on the live CVM.
    """

    if isinstance(compose, str):
        document = json.loads(compose)
    elif isinstance(compose, Mapping):
        document = compose
    else:
        raise TypeError(
            f"app-compose must be a mapping or JSON string, not {type(compose).__name__}"
        )
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compose_hash(compose: Mapping[str, Any] | str) -> str:
    """SHA-256 (hex) of the normalized ``app-compose`` document."""

    normalized = normalize_app_compose(compose)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# dstack-mr image measurement wrapper
# --------------------------------------------------------------------------- #


def dstack_mr_binary(explicit: str | None = None) -> str:
    """Resolve the ``dstack-mr`` binary: explicit arg > ``DSTACK_MR_BIN`` > default."""

    if explicit:
        return explicit
    return os.environ.get("DSTACK_MR_BIN") or DEFAULT_DSTACK_MR_BIN


def dstack_mr_available(binary: str | None = None) -> bool:
    """Whether the resolved ``dstack-mr`` binary is present and executable."""

    resolved = dstack_mr_binary(binary)
    if os.path.sep in resolved:
        return os.path.isfile(resolved) and os.access(resolved, os.X_OK)
    return shutil.which(resolved) is not None


def _format_memory(memory: int | str) -> str:
    """Format a memory spec for ``dstack-mr -memory`` (int is interpreted as GiB)."""

    if isinstance(memory, bool):  # guard: bool is an int subclass
        raise TypeError("memory must be an int (GiB) or a size string like '4G'")
    if isinstance(memory, int):
        return f"{memory}G"
    return str(memory)


def _validate_hex(value: Any, *, width: int, field: str) -> str:
    """Validate ``value`` is a hex string of ``width`` chars; return it lowercased."""

    if not isinstance(value, str):
        raise ValueError(f"{field} is missing or not a string")
    candidate = value.strip()
    if len(candidate) != width or not _HEX_RE.match(candidate):
        raise ValueError(f"{field} is not a {width}-char hex digest (got {len(candidate)} chars)")
    return candidate.lower()


def product_os_image_hash(*, mrtd: str, rtmr1: str, rtmr2: str) -> str:
    """Product OS identity: ``sha256(MRTD || RTMR1 || RTMR2)`` (hex lowercase).

    Canonical single seal for assignment / allowlist / guest quote bind /
    key-release. Shared formula with
    :func:`agent_challenge.keyrelease.quote.os_image_hash_from_registers`.
    Do **not** equate this with Phala provision / teepod catalog digests that
    tools sometimes label ``mr_image``.
    """

    preimage = (
        bytes.fromhex(_validate_hex(mrtd, width=REGISTER_HEX_WIDTH, field="mrtd"))
        + bytes.fromhex(_validate_hex(rtmr1, width=REGISTER_HEX_WIDTH, field="rtmr1"))
        + bytes.fromhex(_validate_hex(rtmr2, width=REGISTER_HEX_WIDTH, field="rtmr2"))
    )
    return hashlib.sha256(preimage).hexdigest()


def measurement_uses_product_os_identity(measurement: Mapping[str, Any]) -> bool:
    """True when sealed ``os_image_hash`` equals product formula of its registers."""

    try:
        expected = product_os_image_hash(
            mrtd=str(measurement.get("mrtd") or ""),
            rtmr1=str(measurement.get("rtmr1") or ""),
            rtmr2=str(measurement.get("rtmr2") or ""),
        )
    except (TypeError, ValueError):
        return False
    sealed = measurement.get("os_image_hash")
    if not isinstance(sealed, str) or not sealed:
        return False
    return sealed.strip().lower() == expected


@dataclass(frozen=True)
class ImageMeasurement:
    """MRTD/RTMR0-2 + product OS identity for a pinned dstack image + VM shape.

    ``os_image_hash`` is always the product formula
    ``sha256(MRTD || RTMR1 || RTMR2)``. Optional ``dstack_mr_image`` retains a
    tool/catalog digest when present without overloading the product field.
    """

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    os_image_hash: str
    dstack_mr_image: str | None = None


def compute_image_measurement(
    metadata_path: Path | str,
    *,
    cpu: int,
    memory: int | str,
    dstack_mr_bin: str | None = None,
) -> ImageMeasurement:
    """Compute MRTD/RTMR0-2 and the **product** OS image hash for a pin + shape.

    Runs ``dstack-mr -cpu <cpu> -memory <mem> -json -metadata <path>`` and parses
    its JSON output. MR registers are validated as SHA-384-width hex.
    ``os_image_hash`` is **always** derived as ``sha256(MRTD || RTMR1 || RTMR2)``
    (guest/keyrelease-compatible). Tool ``mr_image`` (when present) is retained
    only as optional ``dstack_mr_image`` (catalog / provision binding) and never
    overloads the product field. Raises on tool failure or malformed registers so
    a bad measurement never yields a silently-wrong allowlist entry.
    """

    binary = dstack_mr_binary(dstack_mr_bin)
    cmd = [
        binary,
        "-cpu",
        str(cpu),
        "-memory",
        _format_memory(memory),
        "-json",
        "-metadata",
        str(metadata_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"dstack-mr failed (exit {proc.returncode}): {detail}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dstack-mr produced non-JSON output: {exc}") from exc

    mrtd = _validate_hex(data.get("mrtd"), width=REGISTER_HEX_WIDTH, field="mrtd")
    rtmr0 = _validate_hex(data.get("rtmr0"), width=REGISTER_HEX_WIDTH, field="rtmr0")
    rtmr1 = _validate_hex(data.get("rtmr1"), width=REGISTER_HEX_WIDTH, field="rtmr1")
    rtmr2 = _validate_hex(data.get("rtmr2"), width=REGISTER_HEX_WIDTH, field="rtmr2")
    product_os = product_os_image_hash(mrtd=mrtd, rtmr1=rtmr1, rtmr2=rtmr2)

    catalog: str | None = None
    raw_mr_image = data.get("mr_image")
    if raw_mr_image is not None and raw_mr_image != "":
        catalog = _validate_hex(raw_mr_image, width=SHA256_HEX_WIDTH, field="mr_image")

    return ImageMeasurement(
        mrtd=mrtd,
        rtmr0=rtmr0,
        rtmr1=rtmr1,
        rtmr2=rtmr2,
        os_image_hash=product_os,
        dstack_mr_image=catalog,
    )


# --------------------------------------------------------------------------- #
# Canonical, pinnable measurement record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CanonicalMeasurement:
    """The static, allowlist-pinnable measurement record (architecture sec 6/7)."""

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    compose_hash: str
    os_image_hash: str

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in CANONICAL_MEASUREMENT_FIELDS}

    def to_json(self) -> str:
        """Byte-stable serialization a validator copies verbatim into an allowlist."""

        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def build_canonical_measurement(
    *,
    metadata_path: Path | str,
    cpu: int,
    memory: int | str,
    compose: Mapping[str, Any] | str,
    dstack_mr_bin: str | None = None,
) -> CanonicalMeasurement:
    """Compute the full canonical measurement record for the pinned image+compose."""

    image = compute_image_measurement(
        metadata_path, cpu=cpu, memory=memory, dstack_mr_bin=dstack_mr_bin
    )
    return CanonicalMeasurement(
        mrtd=image.mrtd,
        rtmr0=image.rtmr0,
        rtmr1=image.rtmr1,
        rtmr2=image.rtmr2,
        compose_hash=compose_hash(compose),
        os_image_hash=image.os_image_hash,
    )


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(
        prog="agent-challenge-canonical-measure",
        description="Compute the canonical, pinnable measurement record for the eval image.",
    )
    parser.add_argument("--metadata", required=True, help="dstack image metadata.json path")
    parser.add_argument("--cpu", type=int, required=True, help="vCPU count of the pinned VM shape")
    parser.add_argument("--memory", required=True, help="memory of the pinned VM shape, e.g. 4G")
    parser.add_argument("--compose", required=True, help="path to the app-compose.json to pin")
    parser.add_argument("--dstack-mr", default=None, help="override the dstack-mr binary")
    args = parser.parse_args(argv)

    compose_doc = Path(args.compose).read_text()
    record = build_canonical_measurement(
        metadata_path=args.metadata,
        cpu=args.cpu,
        memory=args.memory,
        compose=compose_doc,
        dstack_mr_bin=args.dstack_mr,
    )
    print(record.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(_main())
