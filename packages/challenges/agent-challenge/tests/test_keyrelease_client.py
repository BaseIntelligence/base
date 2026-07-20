"""Unit tests for the in-CVM golden-test key-release client (fail-closed).

Exercises the client's transport/protocol error mapping directly (with a fake
``urlopen``) so each key-unavailable case maps to the right typed
:class:`KeyReleaseError` subclass, all of which carry the fail-closed reason code.
The orchestrator-level fail-closed wiring (VAL-ORCH-035) is covered in
``tests/test_own_runner_backend_keyrelease.py``.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import socket
import urllib.error
from typing import Any

import pytest

from agent_challenge.keyrelease import client as kc
from agent_challenge.keyrelease.client import (
    KEY_RELEASE_FAILED_REASON,
    KEY_RELEASE_TAG,
    GoldenKeyReleaseClient,
    KeyReleaseDenied,
    KeyReleaseError,
    KeyReleaseMidExchangeError,
    KeyReleaseProtocolError,
    KeyReleaseUnreachable,
    key_release_report_data,
)

GOLDEN_KEY = b"golden-decryption-key-0123456789"
FAKE_QUOTE = "ab" * 64


class _FakeQuote:
    def __init__(self, quote: str = FAKE_QUOTE) -> None:
        self.quote = quote


class _StaticQuoteProvider:
    def __init__(self, quote: str = FAKE_QUOTE) -> None:
        self._quote = quote
        self.report_data_seen: bytes | None = None

    def get_quote(self, report_data: bytes):
        self.report_data_seen = report_data
        return _FakeQuote(self._quote)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_urlopen(handlers):
    """Build a fake urlopen dispatching on the request URL suffix.

    ``handlers`` maps ``"nonce"``/``"release"`` to either a bytes body (returned)
    or a callable/exception (raised).
    """

    def _urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        key = "nonce" if url.endswith("/nonce") else "release"
        handler = handlers[key]
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            return handler(request)
        return _FakeResponse(handler)

    return _urlopen


def _client(handlers, **kwargs) -> GoldenKeyReleaseClient:
    return GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=_StaticQuoteProvider(),
        urlopen=_make_urlopen(handlers),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# report_data binding
# --------------------------------------------------------------------------- #
def test_key_release_report_data_is_deterministic_and_tagged():
    a = key_release_report_data("nonce-1", b"pub")
    b = key_release_report_data("nonce-1", b"pub")
    assert a == b
    assert len(a) == 32  # SHA-256 digest


def test_key_release_report_data_changes_with_nonce_and_pubkey():
    base_rd = key_release_report_data("nonce-1", b"pub")
    assert key_release_report_data("nonce-2", b"pub") != base_rd
    assert key_release_report_data("nonce-1", b"other-pub") != base_rd


def test_key_release_tag_distinct_from_result_tag():
    # Cross-protocol separation: the key-release tag must differ from the
    # result-attestation tag so a result quote cannot release the golden key.
    from agent_challenge.canonical.report_data import PHALA_REPORT_DATA_TAG

    assert KEY_RELEASE_TAG.decode() != PHALA_REPORT_DATA_TAG


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_acquire_golden_key_success_returns_key_bytes():
    handlers = {
        "nonce": json.dumps({"nonce": "fresh-nonce"}).encode(),
        "release": json.dumps({"key": base64.b64encode(GOLDEN_KEY).decode()}).encode(),
    }
    provider = _StaticQuoteProvider()
    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=provider,
        ra_tls_pubkey=b"enclave-pub",
        urlopen=_make_urlopen(handlers),
    )
    key = client.acquire_golden_key()
    assert key == GOLDEN_KEY
    # The quote bound the issued nonce via the key-release report_data.
    assert provider.report_data_seen == key_release_report_data("fresh-nonce", b"enclave-pub")


def test_v2_acquire_binds_the_immutable_run_and_issued_key_nonce(monkeypatch) -> None:
    # Isolate from suite residual cookies written by entrypoint provision tests.
    monkeypatch.delenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", raising=False)
    monkeypatch.delenv("CHALLENGE_PHALA_RA_TLS_SPKI_SHA256", raising=False)
    monkeypatch.delenv("CHALLENGE_PHALA_RA_TLS_KEY_FILE", raising=False)
    monkeypatch.delenv("CHALLENGE_PHALA_RA_TLS_CA_FILE", raising=False)
    seen: dict[str, Any] = {}

    def release(request):
        seen["body"] = json.loads(request.data)
        return _FakeResponse(json.dumps({"key": base64.b64encode(GOLDEN_KEY).decode()}).encode())

    handlers = {
        "nonce": AssertionError("v2 must not request a replacement legacy nonce"),
        "release": release,
    }
    provider = _StaticQuoteProvider()
    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=provider,
        ra_tls_pubkey=b"enclave-pub",
        urlopen=_make_urlopen(handlers),
    )
    spki_digest = hashlib.sha256(b"enclave-pub").hexdigest()
    assert (
        client.acquire_golden_key(
            eval_run_id="eval-run-001",
            key_release_nonce="key-nonce-001",
            ra_tls_spki_digest=spki_digest,
        )
        == GOLDEN_KEY
    )
    assert seen["body"]["nonce"] == "key-nonce-001"
    assert seen["body"]["eval_run_id"] == "eval-run-001"
    assert provider.report_data_seen == key_release_report_data(
        "",
        b"enclave-pub",
        eval_run_id="eval-run-001",
        key_release_nonce="key-nonce-001",
        ra_tls_spki_digest=spki_digest,
    )


def test_v2_acquire_with_cert_file_does_not_raise_on_matching_spki(tmp_path, monkeypatch) -> None:
    """Empty pubkey + cert file: client SPKI resolves from leaf, frame may proceed.

    Reproduces the live fail-closed mode where sha256(empty) was forced as the
    caller-supplied digest and mismatched the cert-derived actual digest.
    """
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ra-tls-client")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "client.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    monkeypatch.setenv(kc.KEY_RELEASE_TLS_CERT_ENV, str(cert_path))

    spki = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    cert_spki = hashlib.sha256(spki).hexdigest()
    empty_spki = hashlib.sha256(b"").hexdigest()
    assert cert_spki != empty_spki

    seen: dict[str, Any] = {}

    def release(request):
        seen["body"] = json.loads(request.data)
        return _FakeResponse(json.dumps({"key": base64.b64encode(GOLDEN_KEY).decode()}).encode())

    handlers = {
        "nonce": AssertionError("v2 must not request a replacement legacy nonce"),
        "release": release,
    }
    provider = _StaticQuoteProvider()
    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=provider,
        ra_tls_pubkey=b"",  # live often has no PUBKEY env
        urlopen=_make_urlopen(handlers),
    )
    # Caller supplies the cert-derived digest (own_runner post-fix path).
    assert (
        client.acquire_golden_key(
            eval_run_id="eval-run-cert",
            key_release_nonce="key-nonce-cert",
            ra_tls_spki_digest=cert_spki,
        )
        == GOLDEN_KEY
    )
    assert seen["body"]["eval_run_id"] == "eval-run-cert"


def test_v2_empty_spki_binding_rejection_is_differentiated(tmp_path, monkeypatch) -> None:
    """Optional: empty-SPKI force-bind gets a distinct ProtocolError message."""
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ra-tls-client")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "client.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    monkeypatch.setenv(kc.KEY_RELEASE_TLS_CERT_ENV, str(cert_path))

    empty_spki = hashlib.sha256(b"").hexdigest()
    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=_StaticQuoteProvider(),
        ra_tls_pubkey=b"",
        urlopen=_make_urlopen(
            {
                "nonce": AssertionError("should not run"),
                "release": AssertionError("should not run"),
            }
        ),
    )
    with pytest.raises(KeyReleaseProtocolError, match="empty-pubkey binding"):
        client.acquire_golden_key(
            eval_run_id="eval-run-empty",
            key_release_nonce="key-nonce-empty",
            ra_tls_spki_digest=empty_spki,
        )


def test_acquire_golden_key_rejects_malformed_key_encoding():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": json.dumps({"key": "!!!not-base64!!!"}).encode(),
    }
    with pytest.raises(KeyReleaseProtocolError):
        _client(handlers).acquire_golden_key()


# --------------------------------------------------------------------------- #
# Deny paths
# --------------------------------------------------------------------------- #
def test_release_http_error_maps_to_denied():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": urllib.error.HTTPError(
            "https://validator.test:8700/release",
            403,
            "Forbidden",
            {},
            io.BytesIO(json.dumps({"reason": "measurement not allowlisted"}).encode()),
        ),
    }
    with pytest.raises(KeyReleaseDenied):
        _client(handlers).acquire_golden_key()


def test_release_released_false_maps_to_denied():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": json.dumps({"released": False, "reason": "stale nonce"}).encode(),
    }
    with pytest.raises(KeyReleaseDenied):
        _client(handlers).acquire_golden_key()


def test_release_missing_key_maps_to_denied():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": json.dumps({"status": "ok"}).encode(),
    }
    with pytest.raises(KeyReleaseDenied):
        _client(handlers).acquire_golden_key()


# --------------------------------------------------------------------------- #
# Unreachable paths (connection refused / DNS failure)
# --------------------------------------------------------------------------- #
def test_nonce_connection_refused_maps_to_unreachable():
    handlers = {
        "nonce": urllib.error.URLError(ConnectionRefusedError("refused")),
        "release": b"",
    }
    with pytest.raises(KeyReleaseUnreachable):
        _client(handlers).acquire_golden_key()


def test_nonce_dns_failure_maps_to_unreachable():
    handlers = {
        "nonce": urllib.error.URLError(socket.gaierror("Name or service not known")),
        "release": b"",
    }
    with pytest.raises(KeyReleaseUnreachable):
        _client(handlers).acquire_golden_key()


def test_direct_connection_refused_maps_to_unreachable():
    handlers = {"nonce": ConnectionRefusedError("refused"), "release": b""}
    with pytest.raises(KeyReleaseUnreachable):
        _client(handlers).acquire_golden_key()


# --------------------------------------------------------------------------- #
# Mid-exchange drop (after nonce issued)
# --------------------------------------------------------------------------- #
def test_release_incomplete_read_maps_to_mid_exchange():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": http.client.IncompleteRead(b"partial"),
    }
    with pytest.raises(KeyReleaseMidExchangeError):
        _client(handlers).acquire_golden_key()


def test_release_connection_reset_maps_to_mid_exchange():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": ConnectionResetError("peer reset"),
    }
    with pytest.raises(KeyReleaseMidExchangeError):
        _client(handlers).acquire_golden_key()


def test_release_empty_body_maps_to_mid_exchange():
    handlers = {"nonce": json.dumps({"nonce": "n"}).encode(), "release": b""}
    with pytest.raises(KeyReleaseMidExchangeError):
        _client(handlers).acquire_golden_key()


# --------------------------------------------------------------------------- #
# Protocol errors
# --------------------------------------------------------------------------- #
def test_nonce_missing_field_maps_to_protocol_error():
    handlers = {"nonce": json.dumps({"not_nonce": "x"}).encode(), "release": b""}
    with pytest.raises(KeyReleaseProtocolError):
        _client(handlers).acquire_golden_key()


def test_release_non_json_maps_to_protocol_error():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": b"this is not json",
    }
    with pytest.raises(KeyReleaseProtocolError):
        _client(handlers).acquire_golden_key()


def test_no_quote_provider_fails_closed():
    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=None,
        urlopen=_make_urlopen({"nonce": b"", "release": b""}),
    )
    with pytest.raises(KeyReleaseError):
        client.acquire_golden_key()


def test_generic_urlerror_during_release_maps_to_mid_exchange():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": urllib.error.URLError("some transport failure"),
    }
    with pytest.raises(KeyReleaseMidExchangeError):
        _client(handlers).acquire_golden_key()


def test_generic_urlerror_during_nonce_maps_to_unreachable():
    handlers = {"nonce": urllib.error.URLError("some transport failure"), "release": b""}
    with pytest.raises(KeyReleaseUnreachable):
        _client(handlers).acquire_golden_key()


def test_timeout_during_nonce_maps_to_unreachable():
    handlers = {"nonce": TimeoutError("timed out"), "release": b""}
    with pytest.raises(KeyReleaseUnreachable):
        _client(handlers).acquire_golden_key()


def test_timeout_during_release_maps_to_mid_exchange():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": TimeoutError("timed out"),
    }
    with pytest.raises(KeyReleaseMidExchangeError):
        _client(handlers).acquire_golden_key()


def test_oserror_during_nonce_maps_to_unreachable():
    handlers = {"nonce": OSError("host down"), "release": b""}
    with pytest.raises(KeyReleaseUnreachable):
        _client(handlers).acquire_golden_key()


def test_oserror_during_release_maps_to_mid_exchange():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": OSError("host down"),
    }
    with pytest.raises(KeyReleaseMidExchangeError):
        _client(handlers).acquire_golden_key()


def test_non_dict_json_response_maps_to_protocol_error():
    handlers = {"nonce": json.dumps(["not", "a", "dict"]).encode(), "release": b""}
    with pytest.raises(KeyReleaseProtocolError):
        _client(handlers).acquire_golden_key()


def test_quote_provider_failure_after_nonce_fails_closed():
    class _BoomProvider:
        def get_quote(self, report_data: bytes):
            raise RuntimeError("dstack socket unavailable")

    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=_BoomProvider(),
        urlopen=_make_urlopen({"nonce": json.dumps({"nonce": "n"}).encode(), "release": b""}),
    )
    with pytest.raises(KeyReleaseError):
        client.acquire_golden_key()


def test_denied_http_error_surfaces_json_reason():
    handlers = {
        "nonce": json.dumps({"nonce": "n"}).encode(),
        "release": urllib.error.HTTPError(
            "https://validator.test:8700/release",
            403,
            "Forbidden",
            {},
            io.BytesIO(json.dumps({"reason": "nonce consumed"}).encode()),
        ),
    }
    with pytest.raises(KeyReleaseDenied, match="nonce consumed"):
        _client(handlers).acquire_golden_key()


# --------------------------------------------------------------------------- #
# Every failure carries the fail-closed reason code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc_cls",
    [
        KeyReleaseError,
        KeyReleaseUnreachable,
        KeyReleaseDenied,
        KeyReleaseMidExchangeError,
        KeyReleaseProtocolError,
    ],
)
def test_all_key_release_errors_carry_fail_closed_reason_code(exc_cls):
    assert exc_cls("x").reason_code == KEY_RELEASE_FAILED_REASON


def test_reason_code_is_a_known_own_runner_reason_code():
    from agent_challenge.evaluation.own_runner.reason_codes import is_known_reason_code

    assert is_known_reason_code(kc.KEY_RELEASE_FAILED_REASON)
