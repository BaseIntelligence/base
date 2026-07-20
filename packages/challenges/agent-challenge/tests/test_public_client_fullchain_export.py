"""Public-only client fullchain export surface (product residual Mode B).

After GetTlsKey material_ready, live Path B could not harvest public client.crt
(SSH disabled, SCP fail, secret-free logs). These tests pin:

1. Export emits public fullchain (leaf + intermediates) for ----BEGIN CERT----.
2. Private key is never present in any export surface (log + path).
3. Fail closed when the chain is missing/empty.
4. Entrypoint bootstrap path invokes export after material_ready.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from agent_challenge.canonical import entrypoint
from agent_challenge.canonical import public_client_fullchain as pubfc

# --------------------------------------------------------------------------- #
# Crypto helpers (match bootstrap suite style; no invent roots for production)
# --------------------------------------------------------------------------- #


def _rsa_key(size: int = 2048):
    return rsa.generate_private_key(public_exponent=65537, key_size=size)


def _make_ca(*, cn: str = "export-test-ca") -> tuple[Any, x509.Certificate]:
    key = _rsa_key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _sign_cert(
    *,
    subject_cn: str,
    issuer_key,
    issuer_cert: x509.Certificate,
    client_auth: bool = True,
) -> tuple[Any, x509.Certificate]:
    key = _rsa_key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_cn)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
    )
    if client_auth:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
    cert = builder.sign(issuer_key, hashes.SHA256())
    return key, cert


def _pem_cert(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def _pem_key(key) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()


def _chain_files(tmp_path: Path) -> tuple[Path, Path, str, str]:
    """Write leaf+intermediate chain file + private key; return paths + PEMs."""

    ca_key, ca_cert = _make_ca(cn="dstack-inter")
    leaf_key, leaf = _sign_cert(
        subject_cn="guest-client",
        issuer_key=ca_key,
        issuer_cert=ca_cert,
        client_auth=True,
    )
    leaf_pem = _pem_cert(leaf)
    inter_pem = _pem_cert(ca_cert)
    chain = leaf_pem + inter_pem
    key_pem = _pem_key(leaf_key)
    cert_path = tmp_path / "client.crt"
    key_path = tmp_path / "client.key"
    cert_path.write_text(chain, encoding="utf-8")
    key_path.write_text(key_pem, encoding="utf-8")
    os.chmod(key_path, 0o600)
    return cert_path, key_path, chain, key_pem


# --------------------------------------------------------------------------- #
# Unit: extract + assert public-only + fail closed
# --------------------------------------------------------------------------- #


def test_extract_public_fullchain_leaf_and_intermediate():
    ca_key, ca_cert = _make_ca()
    _lk, leaf = _sign_cert(subject_cn="leaf", issuer_key=ca_key, issuer_cert=ca_cert)
    chain = _pem_cert(leaf) + _pem_cert(ca_cert)

    pem = pubfc.extract_public_fullchain_pem(chain)
    assert pem.count("BEGIN CERTIFICATE") == 2
    assert pem.count("END CERTIFICATE") == 2
    assert "PRIVATE KEY" not in pem.upper()
    # Order preserved: leaf first (PEM header is -----BEGIN CERTIFICATE-----).
    assert pem.startswith("-----BEGIN CERTIFICATE-----")


def test_extract_fails_closed_when_chain_empty():
    with pytest.raises(pubfc.PublicFullchainExportError, match="public_chain_missing"):
        pubfc.extract_public_fullchain_pem("")
    with pytest.raises(pubfc.PublicFullchainExportError, match="public_chain_missing"):
        pubfc.extract_public_fullchain_pem("   ")
    with pytest.raises(pubfc.PublicFullchainExportError, match="public_chain_missing"):
        pubfc.extract_public_fullchain_pem("not a pem at all")


def test_extract_fails_closed_when_private_key_in_input():
    ca_key, ca_cert = _make_ca()
    leaf_key, leaf = _sign_cert(subject_cn="leaf", issuer_key=ca_key, issuer_cert=ca_cert)
    blob = _pem_cert(leaf) + _pem_key(leaf_key)
    with pytest.raises(pubfc.PublicFullchainExportError, match="private_key_in_export"):
        pubfc.extract_public_fullchain_pem(blob)


def test_assert_public_only_rejects_private_key_variants():
    for header in (
        "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n",
        "-----BEGIN EC PRIVATE KEY-----\nabc\n-----END EC PRIVATE KEY-----\n",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----\n",
    ):
        with pytest.raises(pubfc.PublicFullchainExportError, match="private_key_in_export"):
            pubfc.assert_public_only(header)


def test_export_surface_contains_private_key_predicate():
    assert pubfc.export_surface_contains_private_key("-----BEGIN PRIVATE KEY-----\n") is True
    assert pubfc.export_surface_contains_private_key("-----BEGIN CERTIFICATE-----\n") is False
    assert pubfc.export_surface_contains_private_key("") is False


# --------------------------------------------------------------------------- #
# Unit: full export path + log surface
# --------------------------------------------------------------------------- #


def test_export_writes_public_file_and_emits_log(tmp_path, capsys):
    cert_path, key_path, chain, key_pem = _chain_files(tmp_path)
    export_path = tmp_path / "public" / "client-fullchain.pem"

    result = pubfc.export_public_client_fullchain(
        cert_path=cert_path,
        export_path=export_path,
        emit_log=True,
        write_file=True,
    )

    assert result.cert_count == 2
    assert result.pem_len > 0
    assert result.path_written is True
    assert export_path.is_file()
    written = export_path.read_text(encoding="utf-8")
    assert written == result.pem
    assert "BEGIN CERTIFICATE" in written
    assert "PRIVATE KEY" not in written.upper()
    # Private key file must never leak into the public path.
    assert key_pem[:40] not in written
    assert key_path.read_text(encoding="utf-8") != written

    expected_sha = hashlib.sha256(result.pem.encode("utf-8")).hexdigest()
    assert result.sha256_hex == expected_sha
    assert result.leaf_spki_sha256 is not None
    assert len(result.leaf_spki_sha256) == 64

    out = capsys.readouterr().out
    assert f"{pubfc.PUBLIC_FULLCHAIN_MARKER} stage=export_ready" in out
    assert "cert_count=2" in out
    assert f"sha256={expected_sha}" in out
    assert f"leaf_spki_sha256={result.leaf_spki_sha256}" in out
    assert "path_written=yes" in out
    assert "BEGIN CERTIFICATE" in out
    assert "END CERTIFICATE" in out
    assert f"{pubfc.PUBLIC_FULLCHAIN_MARKER} stage=export_end" in out
    assert "PRIVATE KEY" not in out.upper()
    assert key_pem[:40] not in out
    # Export surface object inspection
    assert pubfc.export_surface_contains_private_key(out) is False
    assert pubfc.export_surface_contains_private_key(written) is False


def test_export_fails_closed_when_cert_file_missing(tmp_path):
    missing = tmp_path / "nope.crt"
    with pytest.raises(pubfc.PublicFullchainExportError, match="public_chain_missing"):
        pubfc.export_public_client_fullchain(
            cert_path=missing,
            emit_log=False,
            write_file=False,
        )


def test_export_fails_closed_when_cert_file_empty(tmp_path):
    empty = tmp_path / "empty.crt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(pubfc.PublicFullchainExportError, match="public_chain_missing"):
        pubfc.export_public_client_fullchain(
            cert_path=empty,
            emit_log=False,
            write_file=False,
        )


def test_export_log_only_when_path_not_writable(tmp_path, capsys, monkeypatch):
    """Path write may fail; log surface still exposes public PEM for phala logs."""

    cert_path, _key_path, _chain, key_pem = _chain_files(tmp_path)

    def _fail_write(pem: str, path: Path) -> bool:  # noqa: ARG001
        return False

    monkeypatch.setattr(pubfc, "write_public_fullchain_file", _fail_write)
    result = pubfc.export_public_client_fullchain(
        cert_path=cert_path,
        export_path=tmp_path / "blocked" / "client-fullchain.pem",
        emit_log=True,
        write_file=True,
    )
    assert result.path_written is False
    out = capsys.readouterr().out
    assert "path_written=no" in out
    assert "BEGIN CERTIFICATE" in out
    assert "PRIVATE KEY" not in out.upper()
    assert key_pem[:40] not in out


def test_write_public_fullchain_file_refuses_private_key(tmp_path):
    bad = "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"
    target = tmp_path / "should-not-write.pem"
    with pytest.raises(pubfc.PublicFullchainExportError, match="private_key_in_export"):
        pubfc.write_public_fullchain_file(bad, target)
    assert not target.exists()


def test_resolve_public_export_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom-export.pem"
    monkeypatch.setenv(pubfc.PUBLIC_EXPORT_PATH_ENV, str(custom))
    assert pubfc.resolve_public_export_path() == custom
    monkeypatch.delenv(pubfc.PUBLIC_EXPORT_PATH_ENV, raising=False)
    assert pubfc.resolve_public_export_path() == pubfc.DEFAULT_PUBLIC_EXPORT_PATH


# --------------------------------------------------------------------------- #
# Entrypoint integration: export runs after material_ready; private key absent
# --------------------------------------------------------------------------- #


def _patch_tcp_ok(monkeypatch) -> None:
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def close(self) -> None:
            return None

    def _fake_create_connection(address, timeout=None, **_kwargs):  # noqa: ANN001, ARG001
        return _Conn()

    monkeypatch.setattr(entrypoint.socket, "create_connection", _fake_create_connection)


def test_entrypoint_exports_public_fullchain_after_material_ready(monkeypatch, tmp_path, capsys):
    cert_path = tmp_path / "client.crt"
    key_path = tmp_path / "client.key"
    ca_path = tmp_path / "ca.crt"
    export_path = tmp_path / "varlog" / "client-fullchain.pem"

    ca_key, ca_cert = _make_ca(cn="server-trust-ca")
    leaf_key, leaf = _sign_cert(
        subject_cn="guest-client",
        issuer_key=ca_key,
        issuer_cert=ca_cert,
        client_auth=True,
    )
    # Leaf first, then intermediate (issuer) for fullchain length == 2.
    leaf_pem = _pem_cert(leaf)
    inter_pem = _pem_cert(ca_cert)
    key_pem = _pem_key(leaf_key)
    server_ca_pem = _pem_cert(ca_cert)

    class _Resp:
        key = key_pem
        certificate_chain = [leaf_pem, inter_pem]

    class _FakeDstack:
        def __init__(self, timeout: int = 90) -> None:
            assert timeout >= 60

        def get_tls_key(self, **kwargs: Any):
            assert kwargs.get("usage_ra_tls") is True
            return _Resp()

    import dstack_sdk

    monkeypatch.setattr(dstack_sdk, "DstackClient", _FakeDstack, raising=False)
    monkeypatch.setenv("KEY_RELEASE_RA_TLS_HOST", "84.32.70.61")
    monkeypatch.setenv("KEY_RELEASE_RA_TLS_PORT", "8701")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", str(cert_path))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_KEY_FILE", str(key_path))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CA_FILE", str(ca_path))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM", server_ca_pem)
    monkeypatch.setenv(pubfc.PUBLIC_EXPORT_PATH_ENV, str(export_path))
    _patch_tcp_ok(monkeypatch)

    def fake_backend(args: list[str]) -> int:  # noqa: ARG001
        return 0

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.main",
        fake_backend,
        raising=True,
    )

    rc = entrypoint.main(["run", "--job-dir", "/tmp/job"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "ra_tls_bootstrap stage=material_ready" in out
    assert f"{pubfc.PUBLIC_FULLCHAIN_MARKER} stage=export_ready" in out
    assert "BEGIN CERTIFICATE" in out
    assert "END CERTIFICATE" in out
    assert f"{pubfc.PUBLIC_FULLCHAIN_MARKER} stage=export_end" in out
    assert "ra_tls_bootstrap stage=tcp_connect_ok" in out

    # Private key never on export surfaces (stdout + public path).
    assert "PRIVATE KEY" not in out.upper()
    assert key_pem[:48] not in out
    assert export_path.is_file()
    surface = export_path.read_text(encoding="utf-8")
    assert "BEGIN CERTIFICATE" in surface
    assert "PRIVATE KEY" not in surface.upper()
    assert key_pem[:48] not in surface
    # Secret paths still hold the key for mTLS dial only.
    assert key_path.is_file()
    assert "PRIVATE KEY" in key_path.read_text(encoding="utf-8").upper()
    # Public export path is distinct from secret client.crt (same certs, no key).
    assert cert_path.read_text(encoding="utf-8").strip()
    assert "PRIVATE KEY" not in cert_path.read_text(encoding="utf-8").upper()


def test_entrypoint_fails_closed_when_public_chain_export_impossible(monkeypatch, tmp_path, capsys):
    """If cert file becomes empty after write checks, export fails closed."""

    cert_path = tmp_path / "client.crt"
    key_path = tmp_path / "client.key"
    ca_path = tmp_path / "ca.crt"
    ca_key, ca_cert = _make_ca()
    leaf_key, leaf = _sign_cert(
        subject_cn="prebaked",
        issuer_key=ca_key,
        issuer_cert=ca_cert,
        client_auth=True,
    )
    # Pre-write empty cert so material check for is_file() passes... actually
    # empty fails later at extract. Use pe-filled then overwrite via monkeypatch.
    cert_path.write_text(_pem_cert(leaf), encoding="utf-8")
    key_path.write_text(_pem_key(leaf_key), encoding="utf-8")
    monkeypatch.setenv("KEY_RELEASE_RA_TLS_HOST", "127.0.0.1")
    monkeypatch.setenv("KEY_RELEASE_RA_TLS_PORT", "8701")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", str(cert_path))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_KEY_FILE", str(key_path))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CA_FILE", str(ca_path))
    monkeypatch.setenv(
        "CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM",
        _pem_cert(ca_cert),
    )

    def _export_raises(**_kwargs):
        raise pubfc.PublicFullchainExportError("public_chain_missing: forced")

    monkeypatch.setattr(entrypoint, "export_public_client_fullchain", _export_raises)

    def _must_not_run(*_a, **_k):  # pragma: no cover
        raise AssertionError("backend must not run when public export fails closed")

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.main",
        _must_not_run,
        raising=True,
    )

    rc = entrypoint.main(["run", "--job-dir", "/tmp/job"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ra_tls_bootstrap stage=fail" in out
    assert "reason=public_chain_missing" in out
