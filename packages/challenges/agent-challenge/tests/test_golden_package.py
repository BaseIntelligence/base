"""Packaging API + CLI tests for ``agent_challenge.golden.package``.

Covers the file-level encrypt/decrypt helpers, the fail-closed loader, and the
``encrypt``/``decrypt`` CLI (key from ``--key-file`` or the env var). These are
offline and self-contained (ephemeral key + tmp files), so they run in the gate.
"""

from __future__ import annotations

import json

import pytest

from agent_challenge.golden import crypto, package


@pytest.fixture
def key_file(tmp_path):
    key = crypto.generate_golden_key()
    path = tmp_path / "golden.key"
    path.write_text(key.hex())
    return path, key


def test_encrypt_then_decrypt_file_roundtrip(tmp_path, key_file):
    path, key = key_file
    src = tmp_path / "golden.json"
    payload = b'{"schema": "x", "results": {"t": 1}}'
    src.write_bytes(payload)
    enc = tmp_path / "golden.json.enc"

    package.encrypt_golden_file(src, enc, key)
    assert enc.read_bytes().startswith(crypto._MAGIC)
    assert package.decrypt_golden_file(enc, key) == payload


def test_load_encrypted_oracle_golden_happy(tmp_path, key_file):
    _, key = key_file
    doc = {"schema": "s", "results": {"a": 1}, "task_count": 1}
    enc = package.encrypted_oracle_path(tmp_path)
    enc.write_bytes(package.encrypt_golden_bytes(json.dumps(doc).encode(), key))
    assert package.load_encrypted_oracle_golden(key, golden_dir=tmp_path) == doc


def test_load_encrypted_oracle_golden_missing_fails_closed(tmp_path, key_file):
    _, key = key_file
    with pytest.raises(crypto.GoldenDecryptionError):
        package.load_encrypted_oracle_golden(key, golden_dir=tmp_path)


def test_load_encrypted_oracle_golden_bad_json_fails_closed(tmp_path, key_file):
    _, key = key_file
    package.encrypted_oracle_path(tmp_path).write_bytes(
        package.encrypt_golden_bytes(b"\xff\xfe not json", key)
    )
    with pytest.raises(crypto.GoldenDecryptionError):
        package.load_encrypted_oracle_golden(key, golden_dir=tmp_path)


def test_load_encrypted_oracle_golden_non_object_fails_closed(tmp_path, key_file):
    _, key = key_file
    package.encrypted_oracle_path(tmp_path).write_bytes(
        package.encrypt_golden_bytes(b"[1, 2, 3]", key)
    )
    with pytest.raises(crypto.GoldenDecryptionError):
        package.load_encrypted_oracle_golden(key, golden_dir=tmp_path)


def test_cli_encrypt_and_decrypt_with_key_file(tmp_path, key_file, capsys):
    path, key = key_file
    src = tmp_path / "in.json"
    src.write_bytes(b'{"golden": true}')
    enc = tmp_path / "out.enc"
    out = tmp_path / "back.json"

    rc = package.main(["encrypt", "--in", str(src), "--out", str(enc), "--key-file", str(path)])
    assert rc == 0
    assert enc.is_file()
    assert "encrypted" in capsys.readouterr().out

    rc = package.main(["decrypt", "--in", str(enc), "--out", str(out), "--key-file", str(path)])
    assert rc == 0
    assert out.read_bytes() == src.read_bytes()


def test_cli_uses_env_key_when_no_key_file(tmp_path, key_file, monkeypatch):
    path, key = key_file
    monkeypatch.setenv(crypto.KEY_FILE_ENV, str(path))
    src = tmp_path / "in.json"
    src.write_bytes(b'{"a": 1}')
    enc = tmp_path / "out.enc"

    assert package.main(["encrypt", "--in", str(src), "--out", str(enc)]) == 0
    assert package.decrypt_golden_file(enc, key) == src.read_bytes()


def test_cli_decrypt_to_stdout(tmp_path, key_file):
    path, key = key_file
    src = tmp_path / "in.json"
    src.write_bytes(b'{"stdout": 1}')
    enc = tmp_path / "out.enc"
    package.encrypt_golden_file(src, enc, key)

    import io
    import sys

    buffer = io.BytesIO()
    real_stdout = sys.stdout

    class _Shim:
        def __init__(self, buf):
            self.buffer = buf

    sys.stdout = _Shim(buffer)
    try:
        rc = package.main(["decrypt", "--in", str(enc), "--key-file", str(path)])
    finally:
        sys.stdout = real_stdout
    assert rc == 0
    assert buffer.getvalue() == src.read_bytes()
