"""Golden-encrypted-at-rest tests (feature ``golden-encrypted-at-rest``).

Fulfils:
  * VAL-KEY-001 — golden expected-output is ciphertext everywhere miner-visible
    (repo tree, generated compose); no cleartext golden marker at rest and the
    packaged blob is genuinely encrypted (not merely encoded).
  * VAL-KEY-002 — the packaged artifact is real AEAD ciphertext: it does not
    decrypt without the released key (tamper/wrong-key fails closed). The
    exhaustive AEAD behaviour lives in ``test_golden_crypto.py``.
  * VAL-KEY-003 — no golden decryption key (or key-derivation secret) is
    committed or baked into the repo / generated compose; the key lives only
    outside the repo (endpoint/enclave-resident).

The assertions that need the real released key are skipped when the key is not
available (so the milestone gate, which has no key, still runs), and exercised
during development by exporting ``CHALLENGE_GOLDEN_KEY_FILE``.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from agent_challenge.canonical import compose as ccompose
from agent_challenge.canonical import secrets_scan
from agent_challenge.golden import crypto, package

REPO_ROOT = package.REPO_ROOT
GOLDEN_DIR = package.GOLDEN_DIR
GOLDEN_MARKER = b"harbor-independence/" + b"oracle-golden"


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    return [REPO_ROOT / p.decode() for p in out.stdout.split(b"\x00") if p]


def _available_golden_key() -> bytes | None:
    try:
        return crypto.load_golden_key()
    except crypto.GoldenKeyError:
        return None


requires_key = pytest.mark.skipif(
    _available_golden_key() is None,
    reason="golden key not available (set CHALLENGE_GOLDEN_KEY_FILE)",
)


# --------------------------------------------------------------------------- #
# VAL-KEY-001 — golden expected-output is ciphertext at rest
# --------------------------------------------------------------------------- #


def test_plaintext_oracle_golden_absent_from_repo():
    plaintext = GOLDEN_DIR / package.ORACLE_PLAINTEXT_NAME
    assert not plaintext.exists(), f"plaintext golden must not be at rest: {plaintext}"
    # The historical parity copy under runs/ must not ship as plaintext either.
    assert not (REPO_ROOT / "runs" / "ours" / package.ORACLE_PLAINTEXT_NAME).exists()


def test_encrypted_oracle_artifact_is_present():
    enc = package.encrypted_oracle_path()
    assert enc.is_file(), f"encrypted golden artifact missing: {enc}"
    assert enc.name == package.ORACLE_CIPHERTEXT_NAME


def test_encrypted_artifact_is_ciphertext_not_encoding():
    blob = package.encrypted_oracle_path().read_bytes()
    # Self-describing AEAD container, not JSON / plaintext / base64 text.
    assert blob.startswith(crypto._MAGIC)
    assert GOLDEN_MARKER not in blob
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        json.loads(blob)
    # Encrypted body is high-entropy (near 8 bits/byte), i.e. not merely encoded.
    body = blob[len(crypto._MAGIC) + crypto._NONCE_BYTES :]
    assert _shannon_entropy(body) > 7.0


def test_golden_dir_has_no_plaintext_marker():
    hits = [h for h in secrets_scan.scan_path(GOLDEN_DIR) if h.pattern == "golden_oracle_plaintext"]
    assert hits == [], [h.member for h in hits]


def test_no_tracked_file_carries_golden_plaintext_marker():
    offenders: list[str] = []
    for path in _tracked_files():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if GOLDEN_MARKER in data:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"golden plaintext marker present in repo: {offenders}"


def test_generated_compose_has_no_golden_plaintext():
    compose = ccompose.generate_app_compose(
        orchestrator_image="ghcr.io/example/agent-challenge-canonical@sha256:" + "a" * 64,
        key_release_url="http://validator.example:8700",
    )
    blob = ccompose.render_app_compose_bytes(compose)
    assert GOLDEN_MARKER not in blob
    hits = [h for h in secrets_scan.scan_bytes(blob, member="app-compose.json")]
    assert [h for h in hits if h.pattern == "golden_oracle_plaintext"] == []


# --------------------------------------------------------------------------- #
# VAL-KEY-002 — the packaged artifact is genuine AEAD ciphertext
# --------------------------------------------------------------------------- #


def test_committed_artifact_not_decryptable_without_key():
    blob = package.encrypted_oracle_path().read_bytes()
    with pytest.raises(crypto.GoldenDecryptionError):
        package.decrypt_golden_bytes(blob, crypto.generate_golden_key())


@requires_key
def test_committed_artifact_decrypts_to_valid_golden():
    key = _available_golden_key()
    doc = package.load_encrypted_oracle_golden(key)
    assert doc["schema"].startswith(GOLDEN_MARKER.decode())
    assert isinstance(doc["results"], dict)
    assert doc["task_count"] == len(doc["results"]) == 89


@requires_key
def test_committed_artifact_tamper_fails_closed():
    key = _available_golden_key()
    blob = bytearray(package.encrypted_oracle_path().read_bytes())
    blob[-1] ^= 0x01
    with pytest.raises(crypto.GoldenDecryptionError):
        package.decrypt_golden_bytes(bytes(blob), key)


# --------------------------------------------------------------------------- #
# VAL-KEY-003 — no golden key committed / baked
# --------------------------------------------------------------------------- #


def test_no_key_files_tracked_in_repo():
    tracked = [p.relative_to(REPO_ROOT).as_posix() for p in _tracked_files()]
    assert [p for p in tracked if p.endswith(".key")] == []
    assert [p for p in tracked if p.startswith("secrets/")] == []


def test_gitignore_protects_key_and_secrets():
    gi = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "*.key" in gi
    assert "secrets/" in gi


def test_golden_source_has_no_hardcoded_key():
    # The crypto/packaging source must not embed a 32-byte key constant.
    import re

    for mod in (crypto.__file__, package.__file__):
        text = Path(mod).read_text()
        # No 64-hex or base64-32 literal that could be a baked key.
        assert not re.search(r"['\"][0-9a-fA-F]{64}['\"]", text), mod


@requires_key
def test_released_key_absent_from_repo_and_compose():
    key = _available_golden_key()
    import base64

    needles = [key.hex().encode(), base64.b64encode(key), key]
    # Not in the generated compose.
    compose = ccompose.generate_app_compose(
        orchestrator_image="ghcr.io/example/agent-challenge-canonical@sha256:" + "a" * 64,
        key_release_url="http://validator.example:8700",
    )
    blob = ccompose.render_app_compose_bytes(compose)
    for needle in needles:
        assert needle not in blob
    # Not in any tracked repo file.
    for path in _tracked_files():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for needle in needles:
            assert needle not in data, f"golden key material found in {path}"
