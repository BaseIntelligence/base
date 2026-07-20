"""Host KR key-provider decode + allowlist-miss social codes + invalid_quote detail.

Live residual (post GetQuote normalize image@sha256:6e163501 / compose 2f76f9fd):
host durable ``key_release_deny reason=invalid_quote``, ledger denied / nonce
consumed. Ranked root: KR dual-verify CERT path builds allowlist candidate with
raw RTMR3 key-provider hex/JSON while the live pin is ``key_provider=phala``.
Review already maps the kms/phala JSON family → ``phala``; KR did not, so a
true allowlist miss was raised as ``QuoteVerificationError`` → wrong social
code ``invalid_quote``.

Discriminators (offline only, no Phala create):

1. Live-shaped JSON/hex key-provider with allowlist pin ``phala`` grants (leaves
   invalid_quote).
2. Pure allowlist miss from dual-verify / authorize path is durable
   ``measurement_not_allowlisted`` / wire ``measurement_rejected``, not
   ``invalid_quote``.
3. ``invalid_quote`` deny logs can include secret-free
   ``detail=cert_dcap|cert_structure|cert_rtmr3|cert_allowlist|frame_structure|frame_dcap``.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from agent_challenge.keyrelease.allowlist import CanonicalEntry, MeasurementAllowlist
from agent_challenge.keyrelease.client import key_release_report_data
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    QuoteVerdict,
    QuoteVerificationError,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    decode_key_provider,
    os_image_hash_from_registers,
    replay_rtmr3,
)
from agent_challenge.keyrelease.server import (
    INVALID_QUOTE_DETAIL_TOKENS,
    REASON_INVALID_QUOTE,
    REASON_MEASUREMENT_NOT_ALLOWLISTED,
    REASON_MEASUREMENT_REJECTED,
    KeyReleaseService,
    _log_key_release_deny,
    _protocol_reason,
)

MRTD = "11" * 48
RTMR0 = "22" * 48
RTMR1 = "33" * 48
RTMR2 = "44" * 48
COMPOSE_HASH = "ab" * 32
# Live dstack KMS family payload (RTMR3 event hex/JSON).
LIVE_KEY_PROVIDER_JSON = b'{"name":"kms","id":"kms-live-1"}'
# Alternative live-shaped name that must also collapse onto the pin.
LIVE_KEY_PROVIDER_PHALA_JSON = b'{"name":"phala","id":"app-1"}'
GOLDEN_KEY = bytes(range(32))
ENCLAVE_PUBKEY = b"enclave-ra-tls-pubkey-0123456789"
SENTINEL = "super-secret-golden-key-SENTINEL-xyz"


def _event_log(provider_payload: bytes = LIVE_KEY_PROVIDER_JSON):
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex(COMPOSE_HASH)),
            (KEY_PROVIDER_EVENT, provider_payload),
            ("instance-id", b"instance-xyz"),
        ]
    )


def _entry_pin_phala() -> CanonicalEntry:
    return CanonicalEntry(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        compose_hash=COMPOSE_HASH,
        os_image_hash=os_image_hash_from_registers(MRTD, RTMR1, RTMR2),
        key_provider="phala",
    )


def _make_service(**kwargs) -> KeyReleaseService:
    params = {
        "allowlist": MeasurementAllowlist([_entry_pin_phala()]),
        "verifier": StaticQuoteVerifier(tcb_status="UpToDate"),
        "golden_key_loader": lambda: GOLDEN_KEY,
    }
    params.update(kwargs)
    return KeyReleaseService(**params)


def test_decode_key_provider_maps_kms_json_family_to_phala() -> None:
    assert decode_key_provider(LIVE_KEY_PROVIDER_JSON.hex()) == "phala"
    assert decode_key_provider(LIVE_KEY_PROVIDER_PHALA_JSON.hex()) == "phala"
    assert decode_key_provider(b'{"name":"phala-kms"}'.hex()) == "phala"
    # Plain identifier still accepted (offline fixtures).
    assert decode_key_provider(b"phala".hex()) == "phala"
    assert decode_key_provider(b"validator-kms".hex()) == "validator-kms"


def test_decode_key_provider_rejects_invalid_payload() -> None:
    with pytest.raises(QuoteVerificationError):
        decode_key_provider("zznothex")
    with pytest.raises(QuoteVerificationError):
        decode_key_provider(b'{"name":""}'.hex())
    with pytest.raises(QuoteVerificationError):
        decode_key_provider(None)  # type: ignore[arg-type]


def test_live_shaped_json_key_provider_with_pin_phala_grants() -> None:
    """Unit: live JSON key-provider + pin phala leaves invalid_quote and grants."""

    service = _make_service()
    event_log, rtmr3 = _event_log(LIVE_KEY_PROVIDER_JSON)
    # Replay still surfaces raw hex; candidate decode happens inside KR.
    replay = replay_rtmr3(event_log)
    assert replay.key_provider == LIVE_KEY_PROVIDER_JSON.hex()
    nonce = service.issue_nonce()
    report_data = key_release_report_data(nonce, ENCLAVE_PUBKEY)
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )
    out = service.authorize_release(
        nonce=nonce,
        quote_hex=quote,
        ra_tls_pubkey_hex=ENCLAVE_PUBKEY.hex(),
        event_log=event_log,
        session_peer_pubkey=ENCLAVE_PUBKEY,
    )
    assert out.released is True
    assert out.key == GOLDEN_KEY
    assert out.reason is None


def test_raw_hex_pin_no_longer_required_for_json_payload() -> None:
    """Discriminator: pinning raw JSON hex (old misfold) fails; pin 'phala' grants."""

    raw_pin = CanonicalEntry(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        compose_hash=COMPOSE_HASH,
        os_image_hash=os_image_hash_from_registers(MRTD, RTMR1, RTMR2),
        key_provider=LIVE_KEY_PROVIDER_JSON.hex(),
    )
    service = _make_service(allowlist=MeasurementAllowlist([raw_pin]))
    event_log, rtmr3 = _event_log(LIVE_KEY_PROVIDER_JSON)
    nonce = service.issue_nonce()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    out = service.authorize_release(
        nonce=nonce,
        quote_hex=quote,
        ra_tls_pubkey_hex=ENCLAVE_PUBKEY.hex(),
        event_log=event_log,
        session_peer_pubkey=ENCLAVE_PUBKEY,
    )
    # Decoded candidate is 'phala', so raw-hex pin misses. Social code must be
    # measurement_*, not invalid_quote (old misfold path).
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED
    assert out.reason != REASON_INVALID_QUOTE


def test_allowlist_miss_is_measurement_not_invalid_quote() -> None:
    service = _make_service()
    # Wrong MRTD → pure measurement miss after valid structure/verify.
    event_log, rtmr3 = _event_log(LIVE_KEY_PROVIDER_JSON)
    nonce = service.issue_nonce()
    quote = build_tdx_quote(
        mrtd="ff" * 48,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    out = service.authorize_release(
        nonce=nonce,
        quote_hex=quote,
        ra_tls_pubkey_hex=ENCLAVE_PUBKEY.hex(),
        event_log=event_log,
        session_peer_pubkey=ENCLAVE_PUBKEY,
    )
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED
    assert _protocol_reason(out.reason) == REASON_MEASUREMENT_REJECTED


def test_invalid_quote_detail_tokens_are_closed_set() -> None:
    assert INVALID_QUOTE_DETAIL_TOKENS == frozenset(
        {
            "cert_dcap",
            "cert_structure",
            "cert_rtmr3",
            "cert_allowlist",
            "frame_structure",
            "frame_dcap",
        }
    )


@pytest.mark.parametrize(
    "token",
    sorted(
        {
            "cert_dcap",
            "cert_structure",
            "cert_rtmr3",
            "cert_allowlist",
            "frame_structure",
            "frame_dcap",
        }
    ),
)
def test_invalid_quote_deny_log_includes_detail_token_secret_free(token, capsys) -> None:
    _log_key_release_deny(
        reason=REASON_INVALID_QUOTE,
        eval_run_id="eval-iq-1",
        detail=token,
    )
    # Sentinel freeloader must never appear even if jammed into detail.
    _log_key_release_deny(
        reason=REASON_INVALID_QUOTE,
        eval_run_id=f"eval-iq-1/{SENTINEL}",
        detail=f"{token} {SENTINEL}",
    )
    err = capsys.readouterr().err
    assert f"key_release_deny reason=invalid_quote eval_run_id=eval-iq-1 detail={token}" in err
    assert SENTINEL not in err
    # Measurement denials still do not invent invalid_quote detail.
    _log_key_release_deny(
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
        eval_run_id="eval-ok",
        detail=token,
    )
    err2 = capsys.readouterr().err
    assert "detail=" not in err2
    assert "measurement_not_allowlisted" in err2


def test_authorize_release_structure_fail_carries_frame_structure_detail() -> None:
    service = _make_service()
    nonce = service.issue_nonce()
    out = service.authorize_release(
        nonce=nonce,
        quote_hex="aabb",  # too short / not a TDX quote
        ra_tls_pubkey_hex=ENCLAVE_PUBKEY.hex(),
        event_log=_event_log()[0],
        session_peer_pubkey=ENCLAVE_PUBKEY,
    )
    assert out.released is False
    assert out.reason == REASON_INVALID_QUOTE
    assert getattr(out, "detail", None) == "frame_structure"


def test_authorize_release_verifier_fail_carries_frame_dcap_detail() -> None:
    class _BadVerifier:
        def verify(self, quote_hex: str) -> QuoteVerdict:
            raise QuoteVerificationError("signature broken")

    service = _make_service(verifier=_BadVerifier())
    event_log, rtmr3 = _event_log()
    nonce = service.issue_nonce()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    out = service.authorize_release(
        nonce=nonce,
        quote_hex=quote,
        ra_tls_pubkey_hex=ENCLAVE_PUBKEY.hex(),
        event_log=event_log,
        session_peer_pubkey=ENCLAVE_PUBKEY,
    )
    assert out.released is False
    assert out.reason == REASON_INVALID_QUOTE
    assert getattr(out, "detail", None) == "frame_dcap"


def _ratls_cert_with_provider(
    provider_payload: bytes = LIVE_KEY_PROVIDER_JSON,
    *,
    mrtd: str = MRTD,
) -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    event_log, rtmr3 = _event_log(provider_payload)
    quote = build_tdx_quote(
        mrtd=mrtd,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=hashlib.sha512(b"ratls-cert:" + public_key).digest(),
    )

    def octet_string(value: bytes) -> bytes:
        length = len(value)
        if length < 128:
            return b"\x04" + bytes([length]) + value
        width = (length.bit_length() + 7) // 8
        return b"\x04" + bytes([0x80 | width]) + length.to_bytes(width, "big") + value

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ratls-client")])
    event_bytes = __import__("json").dumps(event_log, separators=(",", ":")).encode()
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(minutes=5))
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
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
    return builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


async def test_cert_dual_verify_allowlist_miss_is_measurement_not_invalid_quote(
    database_session,
    monkeypatch,
) -> None:
    """Framed cert dual-verify allowlist miss must not social-code as invalid_quote."""

    # Reuse durable ledger helpers (same tests/ dir is on sys.path under pytest).
    import test_keyrelease_durable_state as durable

    from agent_challenge.keyrelease.server import build_frame, spki_sha256_from_certificate

    eval_run_id, plan, _ = await durable._create_run(database_session, monkeypatch)
    # Certificate measurement uses wrong MRTD → cert dual-verify allowlist miss.
    certificate = _ratls_cert_with_provider(mrtd="ff" * 48)
    spki = spki_sha256_from_certificate(certificate)
    event_log, rtmr3 = _event_log(LIVE_KEY_PROVIDER_JSON)
    report_data = key_release_report_data(
        "",
        b"",
        eval_run_id=eval_run_id,
        key_release_nonce=plan["key_release_nonce"],
        ra_tls_spki_digest=spki,
    )
    # Body quote may be canonical - dual-verify on CERT must still deny first.
    body_quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )
    payload = build_frame(
        {
            "schema_version": 1,
            "eval_run_id": eval_run_id,
            "nonce": plan["key_release_nonce"],
            "quote_hex": body_quote,
            "event_log": event_log,
        }
    )[4:]
    service = KeyReleaseService(
        allowlist=MeasurementAllowlist([_entry_pin_phala()]),
        verifier=StaticQuoteVerifier(tcb_status="UpToDate"),
        golden_key_loader=lambda: GOLDEN_KEY,
        session_context_factory=database_session,
    )
    key, reason, detail = await service.authorize_framed_request(
        payload,
        peer_certificate_der=certificate,
    )
    assert key is None
    assert reason == REASON_MEASUREMENT_REJECTED
    assert reason != REASON_INVALID_QUOTE
    # Measurement paths do not attach invalid_quote detail tokens.
    assert detail is None


def test_build_candidate_decodes_key_provider() -> None:
    service = _make_service()
    event_log, rtmr3 = _event_log(LIVE_KEY_PROVIDER_JSON)
    replay = replay_rtmr3(event_log)
    report = type(
        "R",
        (),
        {
            "mrtd": MRTD,
            "rtmr0": RTMR0,
            "rtmr1": RTMR1,
            "rtmr2": RTMR2,
            "rtmr3": rtmr3,
        },
    )()
    candidate = service._build_candidate(report, replay)
    assert candidate.key_provider == "phala"
