"""Authenticated encryption for the golden test material (encrypted at rest).

The golden expected-output (the frozen terminal-bench oracle) is a validator
secret: a miner who could read it would know the answer key. It is therefore
stored ONLY as ciphertext everywhere the miner can see (repo tree, canonical
image, deployable compose, mounted volume) and decrypted transiently inside the
enclave after the validator releases the key (architecture.md §4 C3).

This module provides the AEAD primitives (AES-256-GCM) used to package and
unseal that material. Decryption is authenticated: any single-byte mutation of
the ciphertext, nonce, header, or associated data fails closed
(:class:`GoldenDecryptionError`) and yields no plaintext (VAL-KEY-002).

The 32-byte key is never committed or baked into any miner-visible artifact
(VAL-KEY-003): it lives only server-side in the validator key-release endpoint
and, transiently, in enclave memory after release. :func:`load_golden_key` reads
it from a file (named directly or via the ``CHALLENGE_GOLDEN_KEY_FILE`` env var)
that resides OUTSIDE any repository.

``cryptography`` (a direct project dependency) is imported lazily inside the
crypto functions so this module's constants/paths are usable even where the
compiled backend is not installed.
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

#: AES-256 key length in bytes.
GOLDEN_KEY_BYTES = 32

#: Self-describing container header: format id + version. Authenticated as part
#: of the AEAD associated data, so mutating it fails decryption.
_MAGIC = b"ACGENC01"

#: AES-GCM nonce length (bytes). 96-bit nonces are the GCM-recommended size.
_NONCE_BYTES = 12

#: AES-GCM authentication tag length (bytes), appended to the ciphertext.
_TAG_BYTES = 16

#: Env var naming the file that holds the golden key (outside any repo).
KEY_FILE_ENV = "CHALLENGE_GOLDEN_KEY_FILE"


class GoldenCryptoError(Exception):
    """Base error for golden encryption/decryption/key handling."""


class GoldenKeyError(GoldenCryptoError):
    """The golden key is missing, malformed, or the wrong length (fail closed)."""


class GoldenDecryptionError(GoldenCryptoError):
    """Decryption failed integrity/authentication or the blob is malformed.

    Raised for every tamper case (ciphertext, nonce, header, or associated data)
    and for a wrong key; no plaintext is ever produced.
    """


def generate_golden_key() -> bytes:
    """Return a fresh cryptographically-random 256-bit golden key."""

    return os.urandom(GOLDEN_KEY_BYTES)


def _coerce_key(key: bytes | bytearray) -> bytes:
    if not isinstance(key, (bytes, bytearray)):
        raise GoldenKeyError("golden key must be bytes")
    if len(key) != GOLDEN_KEY_BYTES:
        raise GoldenKeyError(f"golden key must be {GOLDEN_KEY_BYTES} bytes, got {len(key)}")
    return bytes(key)


def encrypt_golden(
    plaintext: bytes,
    key: bytes | bytearray,
    *,
    associated_data: bytes = b"",
) -> bytes:
    """Encrypt ``plaintext`` under ``key`` and return the AEAD container bytes.

    Layout: ``_MAGIC ∥ nonce(12) ∥ ciphertext∥tag(16)``. The GCM associated data
    is ``_MAGIC ∥ associated_data`` so the header and any caller-supplied binding
    (e.g. the golden identity) are authenticated.
    """

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _coerce_key(key)
    nonce = os.urandom(_NONCE_BYTES)
    aad = _MAGIC + associated_data
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    return _MAGIC + nonce + ciphertext


def decrypt_golden(
    blob: bytes,
    key: bytes | bytearray,
    *,
    associated_data: bytes = b"",
) -> bytes:
    """Decrypt an AEAD container produced by :func:`encrypt_golden`; fail closed.

    Raises :class:`GoldenDecryptionError` for a malformed container, a bad
    header, or any authentication failure (tampered ciphertext/nonce/header/AAD
    or wrong key) — never returning partial or unauthenticated plaintext.
    """

    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _coerce_key(key)
    header = len(_MAGIC)
    minimum = header + _NONCE_BYTES + _TAG_BYTES
    if len(blob) < minimum:
        raise GoldenDecryptionError("golden ciphertext is too short to be valid")
    if blob[:header] != _MAGIC:
        raise GoldenDecryptionError("golden ciphertext has an unexpected header")
    nonce = blob[header : header + _NONCE_BYTES]
    ciphertext = blob[header + _NONCE_BYTES :]
    aad = _MAGIC + associated_data
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise GoldenDecryptionError("golden ciphertext failed authentication") from exc


def parse_key_material(raw: bytes | str) -> bytes:
    """Parse key material (hex, base64, or raw 32 bytes) into a 32-byte key."""

    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    text = raw.strip()
    # 64-char hex (the canonical on-disk form).
    if len(text) == GOLDEN_KEY_BYTES * 2:
        try:
            return _coerce_key(binascii.unhexlify(text))
        except (binascii.Error, ValueError):
            pass
    # Base64 (with or without padding/newlines).
    try:
        decoded = base64.b64decode(text, validate=True)
        if len(decoded) == GOLDEN_KEY_BYTES:
            return decoded
    except (binascii.Error, ValueError):
        pass
    # Raw 32 bytes.
    if len(raw) == GOLDEN_KEY_BYTES:
        return bytes(raw)
    raise GoldenKeyError("golden key material is not valid hex/base64/32-byte key")


def load_golden_key(source: bytes | bytearray | str | Path | None = None) -> bytes:
    """Load the golden key from ``source``, an env-named file, or fail closed.

    ``source`` may be raw key bytes, or a path (``str``/``Path``) to a key file.
    When ``source`` is ``None`` the file named by :data:`KEY_FILE_ENV` is read.
    The key must NEVER live inside a repository — this is why it is resolved from
    an external file/env rather than a packaged resource (VAL-KEY-003).
    """

    if isinstance(source, (bytes, bytearray)):
        return _coerce_key(bytes(source))

    path: Path | None
    if source is None:
        env_value = os.environ.get(KEY_FILE_ENV)
        path = Path(env_value) if env_value else None
    else:
        path = Path(source)

    if path is None:
        raise GoldenKeyError(f"no golden key configured (set {KEY_FILE_ENV} or pass a key/path)")
    if not path.is_file():
        raise GoldenKeyError(f"golden key file not found: {path}")
    return parse_key_material(path.read_bytes())


__all__ = [
    "GOLDEN_KEY_BYTES",
    "KEY_FILE_ENV",
    "GoldenCryptoError",
    "GoldenDecryptionError",
    "GoldenKeyError",
    "decrypt_golden",
    "encrypt_golden",
    "generate_golden_key",
    "load_golden_key",
    "parse_key_material",
]
