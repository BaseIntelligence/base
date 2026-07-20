"""AEAD golden-crypto unit tests (feature ``golden-encrypted-at-rest``).

Fulfils the authenticated-encryption behaviour of VAL-KEY-002 (tamper is
detected, never silently accepted) and the key-handling primitives used to keep
the golden decryption key out of the repo/image/compose (VAL-KEY-003).

These are pure, offline unit tests: they exercise the crypto primitives with
ephemeral keys and sample data (no real golden key or committed artifact is
required), so they run in the milestone gate.
"""

from __future__ import annotations

import base64

import pytest

from agent_challenge.golden import crypto

# Assembled from fragments so this test source never itself carries the
# contiguous golden-plaintext marker (VAL-KEY-001 scans tracked files).
_MARKER = "harbor-independence/" + "oracle-golden"


def test_generate_key_is_256_bit_and_random():
    a = crypto.generate_golden_key()
    b = crypto.generate_golden_key()
    assert len(a) == crypto.GOLDEN_KEY_BYTES == 32
    assert isinstance(a, bytes)
    assert a != b  # overwhelmingly likely for a real CSPRNG


def test_roundtrip_recovers_exact_plaintext():
    key = crypto.generate_golden_key()
    plaintext = b'{"schema": "example", "results": {"t": 1}}\x00\x01\x02'
    blob = crypto.encrypt_golden(plaintext, key)
    assert crypto.decrypt_golden(blob, key) == plaintext


def test_ciphertext_is_not_plaintext_and_has_header():
    key = crypto.generate_golden_key()
    plaintext = (_MARKER + " marker payload").encode()
    blob = crypto.encrypt_golden(plaintext, key)
    # Ciphertext must not contain the plaintext (or its distinctive marker).
    assert plaintext not in blob
    assert _MARKER.encode() not in blob
    # Self-describing magic header + nonce so the format is unambiguous.
    assert blob.startswith(crypto._MAGIC)
    assert len(blob) >= len(crypto._MAGIC) + crypto._NONCE_BYTES + 16


def test_encryption_is_nonce_randomised():
    key = crypto.generate_golden_key()
    plaintext = b"same message"
    assert crypto.encrypt_golden(plaintext, key) != crypto.encrypt_golden(plaintext, key)


def test_tampered_ciphertext_body_fails_closed():
    key = crypto.generate_golden_key()
    blob = bytearray(crypto.encrypt_golden(b"secret golden bytes", key))
    # Flip a byte in the ciphertext/tag region (after magic+nonce).
    idx = len(crypto._MAGIC) + crypto._NONCE_BYTES + 1
    blob[idx] ^= 0x01
    with pytest.raises(crypto.GoldenDecryptionError):
        crypto.decrypt_golden(bytes(blob), key)


def test_tampered_nonce_fails_closed():
    key = crypto.generate_golden_key()
    blob = bytearray(crypto.encrypt_golden(b"secret golden bytes", key))
    blob[len(crypto._MAGIC)] ^= 0x01  # first nonce byte
    with pytest.raises(crypto.GoldenDecryptionError):
        crypto.decrypt_golden(bytes(blob), key)


def test_tampered_magic_header_fails_closed():
    key = crypto.generate_golden_key()
    blob = bytearray(crypto.encrypt_golden(b"secret golden bytes", key))
    blob[0] ^= 0x01
    with pytest.raises(crypto.GoldenDecryptionError):
        crypto.decrypt_golden(bytes(blob), key)


def test_every_single_byte_flip_is_detected():
    # Exhaustively confirm the AEAD detects a one-byte mutation anywhere.
    key = crypto.generate_golden_key()
    blob = crypto.encrypt_golden(b"golden expected-output", key)
    for i in range(len(blob)):
        mutated = bytearray(blob)
        mutated[i] ^= 0x01
        with pytest.raises(crypto.GoldenDecryptionError):
            crypto.decrypt_golden(bytes(mutated), key)


def test_wrong_key_fails_closed_with_no_plaintext():
    blob = crypto.encrypt_golden(b"secret golden bytes", crypto.generate_golden_key())
    with pytest.raises(crypto.GoldenDecryptionError):
        crypto.decrypt_golden(blob, crypto.generate_golden_key())


def test_associated_data_is_bound():
    key = crypto.generate_golden_key()
    blob = crypto.encrypt_golden(b"payload", key, associated_data=b"oracle")
    assert crypto.decrypt_golden(blob, key, associated_data=b"oracle") == b"payload"
    # A different AAD (e.g. a swapped golden identity) must not authenticate.
    with pytest.raises(crypto.GoldenDecryptionError):
        crypto.decrypt_golden(blob, key, associated_data=b"other")


def test_truncated_blob_fails_closed():
    key = crypto.generate_golden_key()
    blob = crypto.encrypt_golden(b"payload", key)
    for cut in (0, 4, len(crypto._MAGIC), len(crypto._MAGIC) + crypto._NONCE_BYTES):
        with pytest.raises(crypto.GoldenDecryptionError):
            crypto.decrypt_golden(blob[:cut], key)


@pytest.mark.parametrize("bad", [b"", b"\x00" * 16, b"\x00" * 31, b"\x00" * 33])
def test_wrong_length_key_is_rejected(bad):
    with pytest.raises(crypto.GoldenKeyError):
        crypto.encrypt_golden(b"x", bad)
    with pytest.raises(crypto.GoldenKeyError):
        crypto.decrypt_golden(b"\x00" * 40, bad)


def test_load_golden_key_from_hex_file(tmp_path):
    key = crypto.generate_golden_key()
    f = tmp_path / "golden.key"
    f.write_text(key.hex() + "\n")
    assert crypto.load_golden_key(f) == key


def test_load_golden_key_from_base64_file(tmp_path):
    key = crypto.generate_golden_key()
    f = tmp_path / "golden.b64"
    f.write_text(base64.b64encode(key).decode() + "\n")
    assert crypto.load_golden_key(f) == key


def test_load_golden_key_from_env(tmp_path, monkeypatch):
    key = crypto.generate_golden_key()
    f = tmp_path / "golden.key"
    f.write_text(key.hex())
    monkeypatch.setenv(crypto.KEY_FILE_ENV, str(f))
    assert crypto.load_golden_key() == key


def test_load_golden_key_missing_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv(crypto.KEY_FILE_ENV, raising=False)
    with pytest.raises(crypto.GoldenKeyError):
        crypto.load_golden_key()
    with pytest.raises(crypto.GoldenKeyError):
        crypto.load_golden_key(tmp_path / "does-not-exist.key")


def test_load_golden_key_malformed_fails_closed(tmp_path):
    f = tmp_path / "golden.key"
    f.write_text("not-a-valid-key")
    with pytest.raises(crypto.GoldenKeyError):
        crypto.load_golden_key(f)
