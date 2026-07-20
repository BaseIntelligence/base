"""Golden test material packaged encrypted at rest (validator secret).

The frozen terminal-bench oracle golden is the answer key: it is stored only as
ciphertext everywhere the miner can see (repo / canonical image / deployable
compose / mounted volume) and decrypted transiently inside the enclave after the
validator releases the key (architecture.md §4 C3).

* :mod:`agent_challenge.golden.crypto` — AES-256-GCM AEAD primitives + key
  loading (the key lives outside any repo; VAL-KEY-002/003).
* :mod:`agent_challenge.golden.package` — encrypt/decrypt the oracle golden and
  the on-disk artifact conventions (VAL-KEY-001).
"""

from __future__ import annotations

from agent_challenge.golden.crypto import (
    GOLDEN_KEY_BYTES,
    KEY_FILE_ENV,
    GoldenCryptoError,
    GoldenDecryptionError,
    GoldenKeyError,
    decrypt_golden,
    encrypt_golden,
    generate_golden_key,
    load_golden_key,
)

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
]
