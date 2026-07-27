"""Load BASE-held attestation verify key from settings (fail-closed)."""

from __future__ import annotations

from typing import Protocol


class _HasConstationVerifyKey(Protocol):
    """Minimal settings surface for verify-key load (avoids full Settings import)."""

    @property
    def constation(self) -> _ConstationKeyHex: ...


class _ConstationKeyHex(Protocol):
    attestation_verify_key_hex: str | None


def load_attestation_verify_key(settings: _HasConstationVerifyKey) -> bytes | None:
    """Parse ``settings.constation.attestation_verify_key_hex`` to raw key bytes.

    Empty / missing / whitespace-only → ``None`` (router returns ``empty_key``).
    Non-empty hex is decoded with ``bytes.fromhex`` (invalid hex raises).
    """
    raw = settings.constation.attestation_verify_key_hex
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return bytes.fromhex(text)


__all__ = ["load_attestation_verify_key"]
