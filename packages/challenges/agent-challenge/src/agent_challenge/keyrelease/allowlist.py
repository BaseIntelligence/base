"""Validator-owned canonical-measurement allowlist for key release.

The allowlist is the authoritative source of *which* measurements a validator
trusts (architecture.md §3/§7): a quote releases the golden key only if its
measurement equals a pinned canonical entry across **every** register. This is
validator-owned configuration — there is deliberately NO API through which a
requester (miner) can add, alter, or select an entry (VAL-KEY-027); the endpoint
consults only its own configured allowlist and ignores any requester-supplied
"expected measurement".

An allowlist entry pins the full attested register set for a fixed VM shape:
``{mrtd, rtmr0, rtmr1, rtmr2, compose_hash, os_image_hash, key_provider}``. RTMR0
encodes the VM configuration (vCPU/RAM), so pinning it fixes (or enumerates) the
permitted shape (VAL-KEY-015); ``compose_hash`` / ``key_provider`` are the
RTMR3-replayed content the endpoint checks against these values (VAL-KEY-014).

An **empty** allowlist fails closed: nothing is canonical, so nothing releases
(never default-accept-any).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Registers compared for canonical membership. ``rtmr3`` is NOT pinned directly
#: (it is runtime); it is validated by event-log replay yielding ``compose_hash``
#: and ``key_provider``, which ARE pinned here.
ALLOWLIST_REGISTERS = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "compose_hash",
    "os_image_hash",
    "key_provider",
)

#: Env var naming a JSON file with the validator's canonical allowlist.
ALLOWLIST_FILE_ENV = "CHALLENGE_KEY_RELEASE_ALLOWLIST_FILE"


class AllowlistError(Exception):
    """The allowlist configuration is malformed (fail closed)."""


@dataclass(frozen=True)
class CanonicalEntry:
    """One pinned canonical measurement (full register set for a fixed shape)."""

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    compose_hash: str
    os_image_hash: str
    key_provider: str

    def as_dict(self) -> dict[str, str]:
        return {reg: getattr(self, reg) for reg in ALLOWLIST_REGISTERS}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> CanonicalEntry:
        missing = [reg for reg in ALLOWLIST_REGISTERS if reg not in data]
        if missing:
            raise AllowlistError(f"allowlist entry missing registers: {', '.join(missing)}")
        values: dict[str, str] = {}
        for reg in ALLOWLIST_REGISTERS:
            value = data[reg]
            if not isinstance(value, str) or not value:
                raise AllowlistError(f"allowlist entry register {reg!r} must be a non-empty string")
            values[reg] = value.strip().lower()
        return cls(**values)


@dataclass(frozen=True)
class MeasurementCandidate:
    """The register set derived from a presented quote, checked for membership."""

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    compose_hash: str
    os_image_hash: str
    key_provider: str

    def normalized(self) -> dict[str, str]:
        return {reg: str(getattr(self, reg)).strip().lower() for reg in ALLOWLIST_REGISTERS}


class MeasurementAllowlist:
    """A validator-owned set of canonical measurement entries (immutable at runtime).

    Membership requires an EXACT match on every register of some entry; a single
    differing register denies (VAL-KEY-009). An empty allowlist matches nothing.
    """

    def __init__(self, entries: Iterable[CanonicalEntry] = ()) -> None:
        self._entries: tuple[CanonicalEntry, ...] = tuple(entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[CanonicalEntry, ...]:
        return self._entries

    def is_empty(self) -> bool:
        return not self._entries

    def contains(self, candidate: MeasurementCandidate | Mapping[str, object]) -> bool:
        """Whether ``candidate`` exactly matches a canonical entry on all registers."""

        if isinstance(candidate, MeasurementCandidate):
            values = candidate.normalized()
        elif isinstance(candidate, Mapping):
            values = {}
            for reg in ALLOWLIST_REGISTERS:
                if reg not in candidate:
                    return False
                values[reg] = str(candidate[reg]).strip().lower()
        else:
            raise TypeError("candidate must be a MeasurementCandidate or mapping")
        return any(entry.as_dict() == values for entry in self._entries)

    @classmethod
    def from_entries(cls, raw: Iterable[Mapping[str, object]]) -> MeasurementAllowlist:
        return cls(CanonicalEntry.from_mapping(item) for item in raw)

    @classmethod
    def from_json(cls, text: str) -> MeasurementAllowlist:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AllowlistError(f"allowlist is not valid JSON: {exc}") from exc
        if isinstance(data, Mapping):
            data = data.get("entries", [])
        if not isinstance(data, list):
            raise AllowlistError("allowlist JSON must be a list of entries or {'entries': [...]}")
        return cls.from_entries(data)

    @classmethod
    def from_file(cls, path: str | Path) -> MeasurementAllowlist:
        file_path = Path(path)
        if not file_path.is_file():
            raise AllowlistError(f"allowlist file not found: {file_path}")
        return cls.from_json(file_path.read_text(encoding="utf-8"))


__all__ = [
    "ALLOWLIST_FILE_ENV",
    "ALLOWLIST_REGISTERS",
    "AllowlistError",
    "CanonicalEntry",
    "MeasurementAllowlist",
    "MeasurementCandidate",
]
