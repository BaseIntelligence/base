"""Raw RA-TLS key-release transport contract tests."""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import UTC, datetime, timedelta

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    QuoteStructureError,
    build_rtmr3_event_log,
    build_tdx_quote,
)
from agent_challenge.keyrelease.server import (
    FRAME_MAX_BYTES,
    REASON_FRAME_TOO_LARGE,
    REASON_VERIFIER_UNAVAILABLE,
    build_frame,
    parse_frame,
    spki_sha256_from_certificate,
    validate_ratls_certificate,
)

MRTD = "11" * 48
RTMR0 = "22" * 48
RTMR1 = "33" * 48
RTMR2 = "44" * 48
COMPOSE_HASH = "ab" * 32
KEY_PROVIDER_PAYLOAD = b'{"name":"kms","id":"kms-1"}'


def _event_log() -> tuple[list[dict[str, object]], str]:
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex(COMPOSE_HASH)),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
        ]
    )


def _cert() -> bytes:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ratls-client")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(minutes=5))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)


def _ratls_cert(*, issuer_key=None, issuer_cert=None, client_auth: bool = True) -> bytes:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=hashlib.sha512(b"ratls-cert:" + public_key).digest(),
    )
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ratls-client")])
    issuer = issuer_cert.subject if issuer_cert is not None else name
    signer = issuer_key or key

    def octet_string(value: bytes) -> bytes:
        length = len(value)
        if length < 128:
            return b"\x04" + bytes([length]) + value
        width = (length.bit_length() + 7) // 8
        return b"\x04" + bytes([0x80 | width]) + length.to_bytes(width, "big") + value

    event_bytes = json.dumps(event_log, separators=(",", ":")).encode()
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(minutes=5))
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
                    if client_auth
                    else x509.oid.ExtendedKeyUsageOID.SERVER_AUTH
                ]
            ),
            critical=True,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier("1.3.6.1.4.1.62397.1.1"),
                octet_string(bytes.fromhex(quote)),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier("1.3.6.1.4.1.62397.1.2"),
                octet_string(event_bytes),
            ),
            critical=False,
        )
    )
    return builder.sign(signer, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


def test_raw_frame_is_big_endian_and_canonical_json() -> None:
    payload = {"schema_version": 1, "eval_run_id": "eval-1", "nonce": "nonce-1"}
    frame = build_frame(payload)
    assert frame[:4] == struct.pack(">I", len(frame) - 4)
    assert parse_frame(frame[4:]) == payload


def test_raw_frame_rejects_noncanonical_and_oversized_payload() -> None:
    with pytest.raises(ValueError):
        parse_frame(b'{"nonce":"nonce-1","eval_run_id":"eval-1","schema_version":1}')
    with pytest.raises(ValueError, match=REASON_FRAME_TOO_LARGE):
        build_frame({"value": "x" * FRAME_MAX_BYTES})


def test_spki_digest_is_derived_from_certificate_der() -> None:
    certificate = _cert()
    parsed = x509.load_der_x509_certificate(certificate)
    expected = parsed.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert spki_sha256_from_certificate(certificate) == hashlib.sha256(expected).hexdigest()


def test_ratls_certificate_extracts_quote_and_event_log_and_binds_spki() -> None:
    certificate = _ratls_cert()
    digest, quote, event_log = validate_ratls_certificate(certificate)
    assert digest == spki_sha256_from_certificate(certificate)
    assert quote
    assert event_log == _event_log()[0]


def test_ratls_certificate_without_client_auth_is_rejected() -> None:
    with pytest.raises(ValueError, match="client authentication"):
        validate_ratls_certificate(_ratls_cert(client_auth=False))


def test_service_strict_quote_path_rejects_v3_layout() -> None:
    from agent_challenge.keyrelease.quote import parse_tdx_quote_v4

    quote = bytes(632).hex()
    with pytest.raises(QuoteStructureError):
        parse_tdx_quote_v4(quote)


def test_reason_code_for_verifier_outage_is_retryable() -> None:
    assert REASON_VERIFIER_UNAVAILABLE == "verifier_unavailable"


def test_key_release_request_has_no_peer_identity_field() -> None:
    from agent_challenge.keyrelease.server import validate_framed_request

    event_log, _ = _event_log()
    request = {
        "schema_version": 1,
        "eval_run_id": "eval-1",
        "nonce": "nonce-1",
        "quote_hex": "aa" * 1200,
        "event_log": event_log,
    }
    from agent_challenge.keyrelease.server import build_frame

    encoded = build_frame(request)[4:]
    assert set(validate_framed_request(encoded)) == set(request)
    request["ra_tls_pubkey"] = "00"
    with pytest.raises(ValueError):
        validate_framed_request(json.dumps(request, separators=(",", ":")).encode())


def test_key_release_deny_log_is_durable_and_secret_free(capsys) -> None:
    """Host framed denials must leave a scrapable stderr trail without secrets."""

    from agent_challenge.keyrelease.server import (
        REASON_MALFORMED_REQUEST,
        _log_key_release_deny,
    )

    sentinel = "super-secret-golden-key-xyz"
    _log_key_release_deny(reason=REASON_MALFORMED_REQUEST, eval_run_id="eval-run-42")
    _log_key_release_deny(
        reason=f"measurement_not_allowlisted {sentinel}",
        eval_run_id=f"eval-run-99/{sentinel}",
    )
    err = capsys.readouterr().err
    assert "key_release_deny reason=malformed_request eval_run_id=eval-run-42" in err
    # Sanitization keeps only the first token and strips path freeloaders.
    assert sentinel not in err
    assert "key_release_deny reason=measurement_not_allowlisted" in err
    assert "eval_run_id=eval-run-99" in err
