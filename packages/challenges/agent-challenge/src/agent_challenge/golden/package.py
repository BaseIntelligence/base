"""Package the golden test material encrypted at rest, and unseal it in-enclave.

The frozen terminal-bench oracle golden (``golden/tbench-2.1-oracle.json``) is
the answer key the miner must never see. This module encrypts it under the
validator golden key into ``golden/tbench-2.1-oracle.json.enc`` (the only form
that ships in the repo / canonical image / deployable compose) and decrypts it
transiently inside the enclave after the key is released
(architecture.md §4 C3, VAL-KEY-001).

Encryption binds the golden identity as AEAD associated data so a ciphertext
cannot be repurposed for a different golden artifact.

CLI::

    # Encrypt (validator-side packaging; key from a file OUTSIDE any repo).
    python -m agent_challenge.golden.package encrypt \
        --in golden/tbench-2.1-oracle.json \
        --out golden/tbench-2.1-oracle.json.enc \
        --key-file /path/outside/repo/golden.key

    # Decrypt (verification / in-enclave use).
    python -m agent_challenge.golden.package decrypt \
        --in golden/tbench-2.1-oracle.json.enc --key-file <file>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent_challenge.golden import crypto

#: Repository root (``.../agent-challenge``).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Directory holding the golden dataset artifacts (mounted read-only in the CVM).
GOLDEN_DIR = REPO_ROOT / "golden"

#: Plaintext oracle golden name — must NOT exist at rest (VAL-KEY-001).
ORACLE_PLAINTEXT_NAME = "tbench-2.1-oracle.json"

#: Encrypted-at-rest oracle golden name — the only form that ships.
ORACLE_CIPHERTEXT_NAME = ORACLE_PLAINTEXT_NAME + ".enc"

#: AEAD associated data binding the ciphertext to the oracle-golden identity.
ORACLE_AAD = ORACLE_PLAINTEXT_NAME.encode("utf-8")


def encrypted_oracle_path(golden_dir: Path | str = GOLDEN_DIR) -> Path:
    """Return the path to the encrypted oracle golden artifact."""

    return Path(golden_dir) / ORACLE_CIPHERTEXT_NAME


def encrypt_golden_bytes(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt oracle-golden ``plaintext`` (binds :data:`ORACLE_AAD`)."""

    return crypto.encrypt_golden(plaintext, key, associated_data=ORACLE_AAD)


def decrypt_golden_bytes(blob: bytes, key: bytes) -> bytes:
    """Decrypt an oracle-golden container; fail closed (binds :data:`ORACLE_AAD`)."""

    return crypto.decrypt_golden(blob, key, associated_data=ORACLE_AAD)


def encrypt_golden_file(src: Path | str, dst: Path | str, key: bytes) -> Path:
    """Encrypt the plaintext golden at ``src`` into the ciphertext file ``dst``."""

    dst = Path(dst)
    blob = encrypt_golden_bytes(Path(src).read_bytes(), key)
    dst.write_bytes(blob)
    return dst


def decrypt_golden_file(src: Path | str, key: bytes) -> bytes:
    """Return the decrypted plaintext bytes of the ciphertext file ``src``."""

    return decrypt_golden_bytes(Path(src).read_bytes(), key)


def load_encrypted_oracle_golden(
    key: bytes,
    *,
    golden_dir: Path | str = GOLDEN_DIR,
) -> dict[str, Any]:
    """Decrypt and parse the packaged oracle golden into a JSON document.

    Fails closed: a missing artifact, a decryption/authentication failure, or
    malformed JSON raises rather than yielding a partial/placeholder golden.
    """

    path = encrypted_oracle_path(golden_dir)
    if not path.is_file():
        raise crypto.GoldenDecryptionError(f"encrypted golden artifact missing: {path}")
    plaintext = decrypt_golden_bytes(path.read_bytes(), key)
    try:
        document = json.loads(plaintext)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise crypto.GoldenDecryptionError("decrypted golden is not valid JSON") from exc
    if not isinstance(document, dict):
        raise crypto.GoldenDecryptionError("decrypted golden is not a JSON object")
    return document


def _cmd_encrypt(args: argparse.Namespace) -> int:
    key = crypto.load_golden_key(args.key_file) if args.key_file else crypto.load_golden_key()
    out = encrypt_golden_file(args.in_path, args.out, key)
    print(f"encrypted {args.in_path} -> {out} ({out.stat().st_size} bytes)")
    return 0


def _cmd_decrypt(args: argparse.Namespace) -> int:
    key = crypto.load_golden_key(args.key_file) if args.key_file else crypto.load_golden_key()
    plaintext = decrypt_golden_file(args.in_path, key)
    if args.out:
        Path(args.out).write_bytes(plaintext)
        print(f"decrypted {args.in_path} -> {args.out} ({len(plaintext)} bytes)")
    else:
        sys.stdout.buffer.write(plaintext)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-challenge-golden-package", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encrypt", help="encrypt a plaintext golden into ciphertext")
    enc.add_argument("--in", dest="in_path", required=True, help="plaintext golden path")
    enc.add_argument("--out", required=True, help="output ciphertext path")
    enc.add_argument("--key-file", default=None, help=f"key file (else ${crypto.KEY_FILE_ENV})")
    enc.set_defaults(func=_cmd_encrypt)

    dec = sub.add_parser("decrypt", help="decrypt a ciphertext golden")
    dec.add_argument("--in", dest="in_path", required=True, help="ciphertext golden path")
    dec.add_argument("--out", default=None, help="output plaintext path (default: stdout)")
    dec.add_argument("--key-file", default=None, help=f"key file (else ${crypto.KEY_FILE_ENV})")
    dec.set_defaults(func=_cmd_decrypt)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())


__all__ = [
    "GOLDEN_DIR",
    "ORACLE_AAD",
    "ORACLE_CIPHERTEXT_NAME",
    "ORACLE_PLAINTEXT_NAME",
    "REPO_ROOT",
    "decrypt_golden_bytes",
    "decrypt_golden_file",
    "encrypt_golden_bytes",
    "encrypt_golden_file",
    "encrypted_oracle_path",
    "load_encrypted_oracle_golden",
    "main",
]
