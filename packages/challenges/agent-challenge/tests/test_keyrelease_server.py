"""Validator golden key-release endpoint tests (M3, VAL-KEY-004..015).

Exercises the validator-operated key-release SERVER (nonce issuance/freshness/
single-use; the happy path verify quote -> measurement == canonical allowlist ->
nonce+RA-TLS binding in report_data -> golden key over RA-TLS; and the
TCB-downgrade / off-shape / content-based RTMR3-replay rejections) through its
public surfaces: the :class:`KeyReleaseService` decision core (with an in-process
RA-TLS session) and the real HTTP endpoint.

Quotes are assembled deterministically with :func:`build_tdx_quote` /
:func:`build_rtmr3_event_log` and the cryptographic verifier is stubbed
(:class:`StaticQuoteVerifier`); a live dstack quote is exercised at M6.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from agent_challenge.golden.crypto import decrypt_golden, encrypt_golden
from agent_challenge.keyrelease.allowlist import (
    AllowlistError,
    CanonicalEntry,
    MeasurementAllowlist,
    MeasurementCandidate,
)
from agent_challenge.keyrelease.client import (
    KEY_RELEASE_TAG,
    GoldenKeyReleaseClient,
    key_release_report_data,
)
from agent_challenge.keyrelease.nonce import DEFAULT_NONCE_TTL_SECONDS, NonceState, NonceStore
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    QuoteVerificationError,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
    parse_td_report,
    replay_rtmr3,
    runtime_event_digest,
)
from agent_challenge.keyrelease.server import (
    REASON_CONSUMED_NONCE,
    REASON_EVENT_LOG_REQUIRED,
    REASON_INVALID_QUOTE,
    REASON_MEASUREMENT_NOT_ALLOWLISTED,
    REASON_RA_TLS_PEER_MISMATCH,
    REASON_RA_TLS_REQUIRED,
    REASON_REPORT_DATA_MISMATCH,
    REASON_RTMR3_MISMATCH,
    REASON_STALE_NONCE,
    REASON_TCB_UNACCEPTABLE,
    REASON_UNKNOWN_NONCE,
    EvalRunKeyReleaseBinding,
    KeyReleaseService,
    ReleaseOutcome,
    make_server,
)

GOLDEN_KEY = bytes(range(32))
ENCLAVE_PUBKEY = b"enclave-ra-tls-pubkey-0123456789"

MRTD = "11" * 48
RTMR0 = "22" * 48
RTMR1 = "33" * 48
RTMR2 = "44" * 48
COMPOSE_HASH = "ab" * 32
KEY_PROVIDER_PAYLOAD = b'{"name":"kms","id":"kms-1"}'


def _event_log():
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex(COMPOSE_HASH)),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
            ("instance-id", b"instance-xyz"),
        ]
    )


def _canonical_entry() -> CanonicalEntry:
    return CanonicalEntry(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        compose_hash=COMPOSE_HASH,
        os_image_hash=os_image_hash_from_registers(MRTD, RTMR1, RTMR2),
        # Live/KMS JSON payloads decode to the stable pin "phala".
        key_provider="phala",
    )


def _make_service(**kwargs) -> KeyReleaseService:
    params = {
        "allowlist": MeasurementAllowlist([_canonical_entry()]),
        "verifier": StaticQuoteVerifier(tcb_status="UpToDate"),
        "golden_key_loader": lambda: GOLDEN_KEY,
    }
    params.update(kwargs)
    return KeyReleaseService(**params)


def _canonical_request(
    service: KeyReleaseService,
    *,
    nonce: str | None = None,
    ra_tls_pubkey: bytes = ENCLAVE_PUBKEY,
    rtmr3: str | None = None,
    report_data_nonce: str | None = None,
    report_data_pubkey: bytes | None = None,
    tag: bytes = KEY_RELEASE_TAG,
    event_log=None,
    os_image_hash: str | None = None,
):
    """Build a fully-canonical release request (kwargs for authorize_release)."""

    if nonce is None:
        nonce = service.issue_nonce()
    if event_log is None:
        event_log, replay_rtmr3_hex = _event_log()
    else:
        _, replay_rtmr3_hex = _event_log()
    quote_rtmr3 = rtmr3 if rtmr3 is not None else replay_rtmr3_hex

    rd_nonce = report_data_nonce if report_data_nonce is not None else nonce
    rd_pubkey = report_data_pubkey if report_data_pubkey is not None else ra_tls_pubkey
    if tag == KEY_RELEASE_TAG:
        report_data = key_release_report_data(rd_nonce, rd_pubkey)
    else:
        import hashlib

        report_data = hashlib.sha256(tag + rd_nonce.encode() + rd_pubkey).digest()

    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=quote_rtmr3,
        report_data=report_data,
    )
    vm_config = {
        "os_image_hash": os_image_hash
        if os_image_hash is not None
        else os_image_hash_from_registers(MRTD, RTMR1, RTMR2)
    }
    return {
        "nonce": nonce,
        "quote_hex": quote,
        "ra_tls_pubkey_hex": ra_tls_pubkey.hex(),
        "event_log": event_log,
        "vm_config": vm_config,
        "session_peer_pubkey": ra_tls_pubkey,
    }


# =========================================================================== #
# VAL-KEY-004 / 005 / 006 -- nonce issuance, freshness, single-use, expiry
# =========================================================================== #
def test_nonce_is_fresh_high_entropy_and_tracked():
    store = NonceStore()
    a = store.issue()
    b = store.issue()
    assert a != b
    # token_urlsafe(32) -> >= 43 chars of URL-safe base64 (256-bit entropy).
    assert len(a) >= 43 and len(b) >= 43
    assert store.is_outstanding(a)
    assert store.is_outstanding(b)
    assert not store.is_outstanding("never-issued")


def test_service_issue_nonce_distinct():
    service = _make_service()
    nonces = {service.issue_nonce() for _ in range(50)}
    assert len(nonces) == 50


def test_nonce_single_use_second_use_denied():
    store = NonceStore()
    nonce = store.issue()
    assert store.consume(nonce) is NonceState.OK
    assert store.consume(nonce) is NonceState.CONSUMED


def test_nonce_single_use_across_release_attempts_no_key_on_reuse():
    service = _make_service()
    req = _canonical_request(service)
    first = service.authorize_release(**req)
    assert first.released is True
    # Same nonce presented again (even with a fresh valid quote binding) is denied.
    second = service.authorize_release(**req)
    assert second.released is False
    assert second.key is None
    assert second.reason == REASON_CONSUMED_NONCE


def test_nonce_consumed_even_when_first_attempt_denied():
    # Pin an allowlist whose os_image_hash the canonical quote cannot match so the
    # first attempt is denied for a non-nonce reason (bad measurement).
    entry = _canonical_entry().as_dict()
    entry["os_image_hash"] = "ff" * 32
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    nonce = service.issue_nonce()
    denied = service.authorize_release(**_canonical_request(service, nonce=nonce))
    assert denied.released is False
    assert denied.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED
    # ...but the nonce is now consumed, so a subsequent attempt is denied.
    reused = service.authorize_release(**_canonical_request(service, nonce=nonce))
    assert reused.released is False
    assert reused.reason == REASON_CONSUMED_NONCE


def test_unknown_nonce_denied():
    service = _make_service()
    req = _canonical_request(service, nonce="never-issued-by-us")
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_UNKNOWN_NONCE
    assert out.key is None


def test_expired_nonce_denied():
    clock = {"t": 1000.0}
    store = NonceStore(ttl_seconds=30.0, clock=lambda: clock["t"])
    service = _make_service(nonce_store=store)
    nonce = service.issue_nonce()
    clock["t"] += 31.0  # advance past TTL
    out = service.authorize_release(**_canonical_request(service, nonce=nonce))
    assert out.released is False
    assert out.reason == REASON_STALE_NONCE
    assert out.key is None


def test_nonce_store_expiry_state():
    clock = {"t": 0.0}
    store = NonceStore(ttl_seconds=10.0, clock=lambda: clock["t"])
    nonce = store.issue()
    clock["t"] = 10.0
    assert store.is_outstanding(nonce)  # boundary: exactly TTL still valid
    clock["t"] = 10.001
    assert not store.is_outstanding(nonce)
    assert store.consume(nonce) is NonceState.EXPIRED


def test_default_nonce_ttl_is_bounded():
    assert 0 < DEFAULT_NONCE_TTL_SECONDS <= 600


# =========================================================================== #
# VAL-KEY-007 -- happy path: canonical quote + fresh nonce -> key over RA-TLS
# =========================================================================== #
def test_happy_path_releases_golden_key():
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is True
    assert out.key == GOLDEN_KEY
    assert out.reason is None


def test_released_key_decrypts_the_golden_ciphertext():
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is True
    # The released key is the real golden key: it decrypts golden ciphertext.
    ciphertext = encrypt_golden(b"the-golden-oracle", GOLDEN_KEY, associated_data=b"oracle")
    assert decrypt_golden(ciphertext, out.key, associated_data=b"oracle") == b"the-golden-oracle"


def test_release_requires_ra_tls_session():
    service = _make_service()
    req = _canonical_request(service)
    req["session_peer_pubkey"] = None  # plain (non-RA-TLS) request
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_RA_TLS_REQUIRED
    assert out.key is None


def test_plain_http_attempt_does_not_consume_the_nonce():
    service = _make_service()
    nonce = service.issue_nonce()
    plain = _canonical_request(service, nonce=nonce)
    plain["session_peer_pubkey"] = None
    assert service.authorize_release(**plain).released is False
    # The nonce was never a real (RA-TLS) attempt, so it is still usable.
    assert service.authorize_release(**_canonical_request(service, nonce=nonce)).released is True


# =========================================================================== #
# VAL-KEY-008 / 013 -- signature + TCB verification precede release
# =========================================================================== #
def test_invalid_signature_quote_denied():
    service = _make_service(verifier=StaticQuoteVerifier(valid=False))
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_INVALID_QUOTE
    assert out.key is None


def test_verifier_error_fails_closed():
    class _BoomVerifier:
        def verify(self, quote_hex):
            raise RuntimeError("collateral fetch failed")

    service = _make_service(verifier=_BoomVerifier())
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    # Non-definitive verifier exceptions stay retryable (VAL-KEY-005/028).
    from agent_challenge.keyrelease.server import REASON_VERIFIER_UNAVAILABLE

    assert out.reason == REASON_VERIFIER_UNAVAILABLE


def test_malformed_quote_denied():
    service = _make_service()
    req = _canonical_request(service)
    req["quote_hex"] = "abcd"  # far too short to hold a TD report
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_INVALID_QUOTE


@pytest.mark.parametrize(
    "tcb", ["OutOfDate", "ConfigurationNeeded", "Revoked", "SWHardeningNeeded"]
)
def test_tcb_downgrade_quotes_rejected(tcb):
    service = _make_service(verifier=StaticQuoteVerifier(tcb_status=tcb))
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_TCB_UNACCEPTABLE
    assert out.key is None


def test_uptodate_tcb_accepted_others_not():
    service = _make_service(verifier=StaticQuoteVerifier(tcb_status="UpToDate"))
    assert service.authorize_release(**_canonical_request(service)).released is True


def test_endpoint_and_dcap_qvl_agree_on_accept_and_deny():
    # The endpoint's accept/deny signature verdict tracks the verifier's verdict
    # (dcap-qvl in production): a verifier that accepts -> released; one that
    # rejects -> denied. (Same quote bytes, discriminating verifier.)
    accept = _make_service(verifier=StaticQuoteVerifier(valid=True))
    reject = _make_service(verifier=StaticQuoteVerifier(valid=False))
    assert accept.authorize_release(**_canonical_request(accept)).released is True
    assert reject.authorize_release(**_canonical_request(reject)).released is False


# =========================================================================== #
# VAL-KEY-009 / 015 -- measurement == canonical allowlist across ALL registers
# =========================================================================== #
def test_release_requires_exact_measurement_match():
    service = _make_service()
    assert service.authorize_release(**_canonical_request(service)).released is True


def test_empty_allowlist_fails_closed():
    service = _make_service(allowlist=MeasurementAllowlist())
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


@pytest.mark.parametrize("register", ["mrtd", "rtmr0", "rtmr1", "rtmr2"])
def test_single_register_mismatch_denied(register):
    # An allowlist whose one register differs from the quote -> denied.
    entry = _canonical_entry().as_dict()
    entry[register] = "ee" * 48
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_compose_hash_mismatch_denied():
    entry = _canonical_entry().as_dict()
    entry["compose_hash"] = "cd" * 32
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_os_image_hash_mismatch_denied():
    # os_image_hash is derived from the attested registers, so a non-canonical
    # os_image is expressed via an allowlist whose pinned os_image_hash the
    # derived value cannot match (a requester-supplied value is ignored).
    entry = _canonical_entry().as_dict()
    entry["os_image_hash"] = "ff" * 32
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_key_provider_mismatch_denied():
    entry = _canonical_entry().as_dict()
    # Candidate still decodes JSON kms family → "phala"; pin a different id.
    entry["key_provider"] = "evil-kms"
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_off_shape_rtmr0_quote_denied():
    # RTMR0 encodes the VM shape (vCPU/RAM). A quote whose RTMR0 is not the
    # allowlisted shape is denied even though every other register matches.
    service = _make_service()
    off_shape = _canonical_request(service)
    # Re-mint the quote with a different RTMR0 but the same report_data binding.
    nonce = off_shape["nonce"]
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0="99" * 48,  # off-shape RTMR0
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    off_shape["quote_hex"] = quote
    out = service.authorize_release(**off_shape)
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_enumerated_allowed_shape_releases():
    # An allowlist may enumerate more than one permitted shape.
    other_shape = _canonical_entry().as_dict()
    other_shape["rtmr0"] = "99" * 48
    allowlist = MeasurementAllowlist([_canonical_entry(), CanonicalEntry.from_mapping(other_shape)])
    service = _make_service(allowlist=allowlist)
    assert service.authorize_release(**_canonical_request(service)).released is True


# =========================================================================== #
# VAL-KEY-010 / 011 -- nonce in report_data; content-type tag prevents reuse
# =========================================================================== #
def test_report_data_without_issued_nonce_denied():
    service = _make_service()
    nonce = service.issue_nonce()
    # report_data binds a DIFFERENT nonce than the one presented.
    req = _canonical_request(service, nonce=nonce, report_data_nonce="some-other-nonce")
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_REPORT_DATA_MISMATCH
    assert out.key is None


def test_result_tag_quote_denied_at_key_release():
    # A quote whose report_data uses the RESULT-attestation tag (not the
    # key-release tag) is denied even though nonce + measurement are canonical.
    from agent_challenge.canonical.report_data import PHALA_REPORT_DATA_TAG

    service = _make_service()
    req = _canonical_request(service, tag=PHALA_REPORT_DATA_TAG.encode())
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_REPORT_DATA_MISMATCH


def test_key_release_tag_quote_released():
    service = _make_service()
    assert service.authorize_release(**_canonical_request(service, tag=KEY_RELEASE_TAG)).released


def test_v2_eval_run_key_release_binds_run_nonce_and_ra_tls_spki() -> None:
    run_id = "eval-run-001"
    key_nonce = "key-nonce-001"
    service = _make_service(
        eval_run_bindings=[
            EvalRunKeyReleaseBinding(
                eval_run_id=run_id,
                key_release_nonce=key_nonce,
                expires_at_ms=(time.time_ns() // 1_000_000) + 60_000,
            )
        ]
    )
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(
            "",
            ENCLAVE_PUBKEY,
            eval_run_id=run_id,
            key_release_nonce=key_nonce,
            ra_tls_spki_digest=hashlib.sha256(ENCLAVE_PUBKEY).hexdigest(),
        ),
    )
    request = {
        "nonce": key_nonce,
        "quote_hex": quote,
        "ra_tls_pubkey_hex": ENCLAVE_PUBKEY.hex(),
        "event_log": event_log,
        "vm_config": {},
        "session_peer_pubkey": ENCLAVE_PUBKEY,
        "eval_run_id": run_id,
    }
    assert service.authorize_release(**request).released
    assert service.authorize_release(**request).reason == REASON_UNKNOWN_NONCE


def test_v2_client_and_validator_service_agree_on_release_wire() -> None:
    run_id = "eval-run-001"
    key_nonce = "key-nonce-001"
    service = _make_service(
        eval_run_bindings=[
            EvalRunKeyReleaseBinding(
                eval_run_id=run_id,
                key_release_nonce=key_nonce,
                expires_at_ms=(time.time_ns() // 1_000_000) + 60_000,
            )
        ]
    )
    event_log, rtmr3 = _event_log()

    class _Provider:
        def get_quote(self, report_data: bytes):
            return type(
                "_Response",
                (),
                {
                    "quote": build_tdx_quote(
                        mrtd=MRTD,
                        rtmr0=RTMR0,
                        rtmr1=RTMR1,
                        rtmr2=RTMR2,
                        rtmr3=rtmr3,
                        report_data=report_data,
                    ),
                    "event_log": event_log,
                    "vm_config": {},
                },
            )()

    class _Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(request, timeout=None):
        assert request.full_url.endswith("/release")
        payload = json.loads(request.data)
        outcome = service.authorize_release(
            nonce=payload["nonce"],
            quote_hex=payload["quote"],
            ra_tls_pubkey_hex=payload["ra_tls_pubkey"],
            event_log=payload["event_log"],
            vm_config=payload["vm_config"],
            session_peer_pubkey=ENCLAVE_PUBKEY,
            eval_run_id=payload["eval_run_id"],
        )
        assert outcome.released
        return _Response(json.dumps({"key": base64.b64encode(outcome.key).decode()}).encode())

    client = GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=_Provider(),
        ra_tls_pubkey=ENCLAVE_PUBKEY,
        urlopen=_urlopen,
    )
    assert (
        client.acquire_golden_key(
            eval_run_id=run_id,
            key_release_nonce=key_nonce,
            ra_tls_spki_digest=hashlib.sha256(ENCLAVE_PUBKEY).hexdigest(),
        )
        == GOLDEN_KEY
    )


# =========================================================================== #
# VAL-KEY-012 -- key bound to attesting enclave via RA-TLS pubkey (anti-relay)
# =========================================================================== #
def test_matching_ra_tls_peer_key_releases():
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service, ra_tls_pubkey=ENCLAVE_PUBKEY))
    assert out.released is True


def test_relayed_quote_with_different_peer_key_denied():
    # Attacker replays a genuine canonical quote (report_data binds the enclave's
    # key) over a session whose peer key differs -> denied (anti-relay).
    service = _make_service()
    req = _canonical_request(service, ra_tls_pubkey=ENCLAVE_PUBKEY)
    req["session_peer_pubkey"] = b"attacker-relay-session-pubkey-01"
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_RA_TLS_PEER_MISMATCH
    assert out.key is None


def test_attacker_swaps_bound_key_breaks_report_data():
    # If the attacker instead sets ra_tls_pubkey to their own key (to match the
    # session), report_data (which binds the enclave key) no longer matches.
    service = _make_service()
    nonce = service.issue_nonce()
    attacker_key = b"attacker-relay-session-pubkey-01"
    req = _canonical_request(
        service,
        nonce=nonce,
        ra_tls_pubkey=attacker_key,  # bound in request + session
        report_data_pubkey=ENCLAVE_PUBKEY,  # but the quote binds the enclave key
    )
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_REPORT_DATA_MISMATCH


# =========================================================================== #
# VAL-KEY-014 -- RTMR3 validated by replayed content (compose-hash), not value
# =========================================================================== #
def test_rtmr3_replay_matches_releases():
    service = _make_service()
    # Sanity: the canonical event log replays to the quote's RTMR3.
    event_log, rtmr3 = _event_log()
    req = _canonical_request(service, event_log=event_log, rtmr3=rtmr3)
    assert service.authorize_release(**req).released is True


def test_arbitrary_rtmr3_value_without_matching_replay_denied():
    service = _make_service()
    req = _canonical_request(service, rtmr3="de" * 48)  # attacker-set RTMR3
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_RTMR3_MISMATCH


def test_event_log_required_for_release():
    service = _make_service()
    req = _canonical_request(service)
    req["event_log"] = []
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_EVENT_LOG_REQUIRED


def test_inconsistent_event_digest_denied():
    # An event whose logged digest does not match its payload is rejected: the
    # payload cannot be forged while keeping a matching RTMR3.
    service = _make_service()
    event_log, rtmr3 = _event_log()
    # Tamper the compose-hash event payload without fixing its digest.
    for entry in event_log:
        if entry["event"] == COMPOSE_HASH_EVENT:
            entry["event_payload"] = "cd" * 32
    req = _canonical_request(service, event_log=event_log, rtmr3=rtmr3)
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_RTMR3_MISMATCH


def test_noncanonical_compose_hash_denied_via_consistent_replay():
    # A self-consistent event log (replays to its RTMR3) whose compose-hash is
    # simply not the allowlisted value -> measurement mismatch (not RTMR3).
    service = _make_service()
    bad_compose = bytes.fromhex("cd" * 32)
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bad_compose),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
        ]
    )
    req = _canonical_request(service, event_log=event_log, rtmr3=rtmr3)
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_replay_rtmr3_extracts_compose_and_key_provider():
    event_log, rtmr3 = _event_log()
    replay = replay_rtmr3(event_log)
    assert replay.rtmr3 == rtmr3
    assert replay.compose_hash == COMPOSE_HASH
    assert replay.key_provider == KEY_PROVIDER_PAYLOAD.hex()


def test_replay_rtmr3_rejects_forged_digest():
    event_log, _ = _event_log()
    event_log[1]["digest"] = "00" * 48  # wrong digest for the payload
    with pytest.raises(QuoteVerificationError):
        replay_rtmr3(event_log)


def test_runtime_event_digest_matches_dstack_formula():
    import hashlib

    name = "compose-hash"
    payload = bytes.fromhex(COMPOSE_HASH)
    expected = hashlib.sha384(
        (0x08000001).to_bytes(4, "little") + b":" + name.encode() + b":" + payload
    ).digest()
    assert runtime_event_digest(name, payload) == expected


# =========================================================================== #
# Quote structural parse round-trip
# =========================================================================== #
def test_build_and_parse_td_report_round_trip():
    rd = b"\xaa" * 32
    quote = build_tdx_quote(
        mrtd=MRTD, rtmr0=RTMR0, rtmr1=RTMR1, rtmr2=RTMR2, rtmr3="55" * 48, report_data=rd
    )
    report = parse_td_report(quote)
    assert report.mrtd == MRTD
    assert report.rtmr0 == RTMR0
    assert report.rtmr1 == RTMR1
    assert report.rtmr2 == RTMR2
    assert report.rtmr3 == "55" * 48
    assert report.report_data == rd.ljust(64, b"\x00")


# =========================================================================== #
# ReleaseOutcome invariants
# =========================================================================== #
def test_deny_outcome_never_carries_key():
    out = ReleaseOutcome.deny(REASON_UNKNOWN_NONCE)
    assert out.released is False
    assert out.key is None
    assert out.reason == REASON_UNKNOWN_NONCE


# =========================================================================== #
# HTTP endpoint integration (wire contract matches the in-CVM client)
# =========================================================================== #
@pytest.fixture
def running_server():
    service = _make_service()
    server = make_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    base = f"http://{host}:{port}"
    try:
        yield service, base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_json(url, *, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def test_health_endpoint(running_server):
    _service, base = running_server
    status, body = _http_json(f"{base}/health")
    assert status == 200
    assert body == {"status": "ok"}


def test_nonce_endpoint_returns_distinct_high_entropy_nonces(running_server):
    service, base = running_server
    _, first = _http_json(f"{base}/nonce")
    _, second = _http_json(f"{base}/nonce")
    assert first["nonce"] != second["nonce"]
    assert len(first["nonce"]) >= 43
    # Each issued nonce is tracked by the endpoint as outstanding.
    assert service.nonce_store.is_outstanding(first["nonce"])
    assert service.nonce_store.is_outstanding(second["nonce"])


def test_release_over_http_with_ra_tls_header_returns_key(running_server):
    service, base = running_server
    _, nonce_body = _http_json(f"{base}/nonce")
    nonce = nonce_body["nonce"]
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    status, body = _http_json(
        f"{base}/release",
        method="POST",
        body={
            "nonce": nonce,
            "quote": quote,
            "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
            "event_log": event_log,
            "vm_config": {"os_image_hash": os_image_hash_from_registers(MRTD, RTMR1, RTMR2)},
        },
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    assert status == 200
    assert body["released"] is True
    assert base64.b64decode(body["key"], validate=True) == GOLDEN_KEY


def test_release_over_http_without_ra_tls_header_denied(running_server):
    service, base = running_server
    _, nonce_body = _http_json(f"{base}/nonce")
    nonce = nonce_body["nonce"]
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    status, body = _http_json(
        f"{base}/release",
        method="POST",
        body={
            "nonce": nonce,
            "quote": quote,
            "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
            "event_log": event_log,
        },
    )
    assert status == 200
    assert body["released"] is False
    assert "key" not in body
    assert body["reason"] == REASON_RA_TLS_REQUIRED


def test_release_malformed_json_returns_400(running_server):
    _service, base = running_server
    req = urllib.request.Request(
        f"{base}/release",
        data=b"{not json",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status, body = resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        status, body = exc.code, json.loads(exc.read())
    assert status == 400
    assert body["released"] is False


def test_client_can_fetch_nonce_from_real_server(running_server):
    # The landed in-CVM client's nonce fetch works against the real endpoint.
    from agent_challenge.keyrelease.client import GoldenKeyReleaseClient

    _service, base = running_server
    client = GoldenKeyReleaseClient(base, quote_provider=None)
    nonce = client.request_nonce()
    assert isinstance(nonce, str) and nonce


def test_unknown_route_returns_404(running_server):
    _service, base = running_server
    req = urllib.request.Request(f"{base}/nope", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 404


def test_nonce_via_post(running_server):
    _service, base = running_server
    status, body = _http_json(f"{base}/nonce", method="POST", body={})
    assert status == 200
    assert body["nonce"]


# =========================================================================== #
# Allowlist loading (validator-owned config)
# =========================================================================== #
def test_allowlist_from_json_list_and_entries_wrapper():
    entry = _canonical_entry().as_dict()
    from_list = MeasurementAllowlist.from_json(json.dumps([entry]))
    from_wrapped = MeasurementAllowlist.from_json(json.dumps({"entries": [entry]}))
    assert len(from_list) == 1
    assert len(from_wrapped) == 1
    assert from_list.contains(MeasurementCandidate(**entry))


def test_allowlist_from_file(tmp_path):
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps([_canonical_entry().as_dict()]))
    allowlist = MeasurementAllowlist.from_file(path)
    assert len(allowlist) == 1


def test_allowlist_missing_register_rejected():
    entry = _canonical_entry().as_dict()
    del entry["compose_hash"]
    with pytest.raises(AllowlistError):
        MeasurementAllowlist.from_entries([entry])


def test_allowlist_malformed_json_rejected():
    with pytest.raises(AllowlistError):
        MeasurementAllowlist.from_json("{not json")


def test_allowlist_non_list_json_rejected():
    with pytest.raises(AllowlistError):
        MeasurementAllowlist.from_json(json.dumps(123))


def test_allowlist_missing_file_rejected(tmp_path):
    with pytest.raises(AllowlistError):
        MeasurementAllowlist.from_file(tmp_path / "does-not-exist.json")


def test_allowlist_empty_string_register_rejected():
    entry = _canonical_entry().as_dict()
    entry["mrtd"] = ""
    with pytest.raises(AllowlistError):
        MeasurementAllowlist.from_entries([entry])


def test_allowlist_contains_dict_missing_key_is_false():
    allowlist = MeasurementAllowlist([_canonical_entry()])
    partial = _canonical_entry().as_dict()
    del partial["rtmr0"]
    assert allowlist.contains(partial) is False


def test_allowlist_is_empty():
    assert MeasurementAllowlist().is_empty()
    assert not MeasurementAllowlist([_canonical_entry()]).is_empty()


# =========================================================================== #
# DcapQvlVerifier parsing (trustless quote verification adapter)
# =========================================================================== #
def _completed(returncode, stdout="", stderr=""):
    import subprocess

    return subprocess.CompletedProcess(["dcap-qvl"], returncode, stdout=stdout, stderr=stderr)


def test_dcap_qvl_verifier_accepts_uptodate():
    from agent_challenge.keyrelease.quote import DcapQvlVerifier

    seen: list[list[str]] = []

    def runner(args):
        seen.append(list(args))
        # dcap-qvl expects a file path, not the hex body, as the last arg.
        assert len(args) == 4
        assert args[0] == "dcap-qvl"
        assert args[1] == "verify"
        assert args[2] == "--hex"
        path = Path(args[3])
        assert path.is_file()
        assert path.read_text(encoding="ascii") == "00" * 700
        assert len(args[3]) < 400  # never argv the 10k-char quote body
        return _completed(0, stdout=json.dumps({"status": "UpToDate", "advisory_ids": ["INTEL-1"]}))

    verdict = DcapQvlVerifier(runner=runner).verify("00" * 700)
    assert verdict.tcb_status == "UpToDate"
    assert verdict.advisory_ids == ("INTEL-1",)
    assert seen and not Path(seen[0][3]).exists()  # temp file cleaned up


def test_dcap_qvl_verifier_nonzero_exit_rejects():
    from agent_challenge.keyrelease.quote import DcapQvlVerifier

    verifier = DcapQvlVerifier(runner=lambda args: _completed(1, stderr="bad signature"))
    with pytest.raises(QuoteVerificationError):
        verifier.verify("00" * 700)


def test_dcap_qvl_verifier_non_json_rejects():
    from agent_challenge.keyrelease.quote import DcapQvlVerifier

    verifier = DcapQvlVerifier(runner=lambda args: _completed(0, stdout="not-json"))
    with pytest.raises(QuoteVerificationError):
        verifier.verify("00" * 700)


def test_dcap_qvl_verifier_missing_status_rejects():
    from agent_challenge.keyrelease.quote import DcapQvlVerifier

    verifier = DcapQvlVerifier(runner=lambda args: _completed(0, stdout=json.dumps({})))
    with pytest.raises(QuoteVerificationError):
        verifier.verify("00" * 700)


def test_dcap_qvl_verifier_out_of_date_status_surfaced():
    from agent_challenge.keyrelease.quote import DcapQvlVerifier

    verifier = DcapQvlVerifier(
        runner=lambda args: _completed(0, stdout=json.dumps({"tcbStatus": "OutOfDate"}))
    )
    assert verifier.verify("00" * 700).tcb_status == "OutOfDate"


# =========================================================================== #
# Quote parse errors + os_image_hash + non-runtime RTMR3 event
# =========================================================================== #
def test_parse_td_report_too_short_rejected():
    from agent_challenge.keyrelease.quote import QuoteStructureError, parse_td_report

    with pytest.raises(QuoteStructureError):
        parse_td_report("00" * 10)


def test_parse_quote_hex_malformed_rejected():
    from agent_challenge.keyrelease.quote import QuoteStructureError, parse_quote_hex

    with pytest.raises(QuoteStructureError):
        parse_quote_hex("zzzz")


def test_parse_quote_hex_accepts_0x_prefix():
    from agent_challenge.keyrelease.quote import parse_quote_hex

    assert parse_quote_hex("0xdeadbeef") == bytes.fromhex("deadbeef")


def test_os_image_hash_matches_measurement_definition():
    import hashlib

    expected = hashlib.sha256(
        bytes.fromhex(MRTD) + bytes.fromhex(RTMR1) + bytes.fromhex(RTMR2)
    ).hexdigest()
    assert os_image_hash_from_registers(MRTD, RTMR1, RTMR2) == expected


def test_replay_rtmr3_non_runtime_event_uses_logged_digest():
    # A non-runtime RTMR3 event contributes its logged 48-byte digest to the fold.
    from agent_challenge.keyrelease.quote import APP_IMR, _rtmr_extend

    digest = "7a" * 48
    log = [{"imr": APP_IMR, "event_type": 1, "digest": digest, "event": "x", "event_payload": ""}]
    replay = replay_rtmr3(log)
    assert replay.rtmr3 == _rtmr_extend(bytes(48), bytes.fromhex(digest)).hex()


def test_replay_rtmr3_non_runtime_event_missing_digest_rejected():
    from agent_challenge.keyrelease.quote import APP_IMR

    log = [{"imr": APP_IMR, "event_type": 1, "event": "x", "event_payload": ""}]
    with pytest.raises(QuoteVerificationError):
        replay_rtmr3(log)


def test_replay_rtmr3_ignores_non_app_imr_events():
    log, rtmr3 = _event_log()
    log.insert(
        0,
        {"imr": 0, "event_type": 1, "digest": "aa" * 48, "event": "acpi", "event_payload": ""},
    )
    assert replay_rtmr3(log).rtmr3 == rtmr3


# =========================================================================== #
# Service construction from environment (fail closed)
# =========================================================================== #
def test_from_env_loads_allowlist_and_tcb(tmp_path, monkeypatch):
    from agent_challenge.keyrelease.allowlist import ALLOWLIST_FILE_ENV
    from agent_challenge.keyrelease.server import (
        ACCEPTABLE_TCB_ENV,
        NONCE_TTL_ENV,
        KeyReleaseService,
    )

    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps([_canonical_entry().as_dict()]))
    monkeypatch.setenv(ALLOWLIST_FILE_ENV, str(path))
    monkeypatch.setenv(ACCEPTABLE_TCB_ENV, "UpToDate, SWHardeningNeeded")
    monkeypatch.setenv(NONCE_TTL_ENV, "45")

    service = KeyReleaseService.from_env(
        verifier=StaticQuoteVerifier(), golden_key_loader=lambda: GOLDEN_KEY
    )
    assert len(service.allowlist) == 1
    assert service.authorize_release(**_canonical_request(service)).released is True


def test_from_env_empty_allowlist_when_unconfigured(monkeypatch):
    from agent_challenge.keyrelease.allowlist import ALLOWLIST_FILE_ENV
    from agent_challenge.keyrelease.server import KeyReleaseService

    monkeypatch.delenv(ALLOWLIST_FILE_ENV, raising=False)
    service = KeyReleaseService.from_env(verifier=StaticQuoteVerifier())
    assert service.allowlist.is_empty()


def test_golden_key_loader_failure_fails_closed():
    def _boom():
        from agent_challenge.golden.crypto import GoldenKeyError

        raise GoldenKeyError("no key configured")

    service = _make_service(golden_key_loader=_boom)
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is False
    assert out.key is None
    assert out.reason == "golden_key_unavailable"


# =========================================================================== #
# VAL-KEY-016 -- key transported ONLY over RA-TLS, never over a plaintext channel
# =========================================================================== #
SENTINEL_KEY = b"SENTINEL-GOLDEN-KEY-DO-NOT-LEAK1"  # 32 bytes


def _sentinel_service(**kwargs) -> KeyReleaseService:
    return _make_service(golden_key_loader=lambda: SENTINEL_KEY, **kwargs)


def test_val_key_016_no_ra_tls_session_denies_with_no_key():
    # A release attempt without an established RA-TLS session (no attested peer
    # key) never yields key material, regardless of an otherwise valid request.
    service = _sentinel_service()
    req = _canonical_request(service)
    req["session_peer_pubkey"] = None
    out = service.authorize_release(**req)
    assert out.released is False
    assert out.key is None
    assert out.reason == REASON_RA_TLS_REQUIRED


def test_val_key_016_key_only_over_ra_tls_http(running_server):
    # Over the real HTTP endpoint: a request WITHOUT the attested RA-TLS peer
    # header gets no key; the SAME request WITH the header releases the key.
    service, base = running_server

    def _release(nonce, headers):
        event_log, rtmr3 = _event_log()
        quote = build_tdx_quote(
            mrtd=MRTD,
            rtmr0=RTMR0,
            rtmr1=RTMR1,
            rtmr2=RTMR2,
            rtmr3=rtmr3,
            report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
        )
        return _http_json(
            f"{base}/release",
            method="POST",
            body={
                "nonce": nonce,
                "quote": quote,
                "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
                "event_log": event_log,
                "vm_config": {"os_image_hash": os_image_hash_from_registers(MRTD, RTMR1, RTMR2)},
            },
            headers=headers,
        )

    _, nonce_body = _http_json(f"{base}/nonce")
    _, plain = _release(nonce_body["nonce"], {})
    assert plain["released"] is False
    assert "key" not in plain

    _, nonce_body2 = _http_json(f"{base}/nonce")
    _, ra_tls = _release(nonce_body2["nonce"], {"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()})
    assert ra_tls["released"] is True
    assert "key" in ra_tls


# =========================================================================== #
# VAL-KEY-018 -- the key is never echoed in responses, logs, or errors
# =========================================================================== #
def _key_sentinels() -> list[str]:
    return [
        base64.b64encode(SENTINEL_KEY).decode(),
        SENTINEL_KEY.decode(),
        SENTINEL_KEY.hex(),
    ]


def test_val_key_018_key_absent_from_every_deny_response():
    service = _sentinel_service()
    b64 = base64.b64encode(SENTINEL_KEY).decode()

    bad_event_log, bad_rtmr3 = build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex("cd" * 32)),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
        ]
    )
    deny_reqs = {
        "unknown_nonce": _canonical_request(service, nonce="never-issued"),
        "bad_measurement": _canonical_request(service, event_log=bad_event_log, rtmr3=bad_rtmr3),
        "wrong_report_data": _canonical_request(service, report_data_nonce="mismatch"),
    }
    for name, req in deny_reqs.items():
        out = service.authorize_release(**req)
        assert out.released is False, name
        assert out.key is None, name
        # The reason string never carries the key sentinel (raw or base64).
        assert b64 not in (out.reason or ""), name
        assert SENTINEL_KEY.decode() not in (out.reason or ""), name

    # And a no-RA-TLS deny likewise carries no key.
    no_ra = _canonical_request(service)
    no_ra["session_peer_pubkey"] = None
    out = service.authorize_release(**no_ra)
    assert out.key is None
    assert b64 not in (out.reason or "")


def test_val_key_018_key_only_in_success_body_not_in_logs(running_server, capsys):
    service, base = running_server
    # Point the running server's loader at the sentinel key for this check.
    service._golden_key_loader = lambda: SENTINEL_KEY  # noqa: SLF001 - test-only override
    sentinels = _key_sentinels()

    # A deny path (no RA-TLS header) then the success path (RA-TLS header).
    _, nonce_body = _http_json(f"{base}/nonce")
    event_log, rtmr3 = _event_log()

    def _quote(nonce):
        return build_tdx_quote(
            mrtd=MRTD,
            rtmr0=RTMR0,
            rtmr1=RTMR1,
            rtmr2=RTMR2,
            rtmr3=rtmr3,
            report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
        )

    _, denied = _http_json(
        f"{base}/release",
        method="POST",
        body={
            "nonce": nonce_body["nonce"],
            "quote": _quote(nonce_body["nonce"]),
            "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
            "event_log": event_log,
        },
    )
    assert denied["released"] is False
    for s in sentinels:
        assert s not in json.dumps(denied)

    _, nonce_body2 = _http_json(f"{base}/nonce")
    status, released = _http_json(
        f"{base}/release",
        method="POST",
        body={
            "nonce": nonce_body2["nonce"],
            "quote": _quote(nonce_body2["nonce"]),
            "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
            "event_log": event_log,
            "vm_config": {"os_image_hash": os_image_hash_from_registers(MRTD, RTMR1, RTMR2)},
        },
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    assert status == 200
    assert released["released"] is True
    # The key sentinel appears ONLY as the base64 key in the success body.
    assert released["key"] == base64.b64encode(SENTINEL_KEY).decode()

    # The endpoint logs nothing (handler log_message is silenced) -> the key
    # never reaches captured stdout/stderr.
    captured = capsys.readouterr()
    for s in sentinels:
        assert s not in captured.out
        assert s not in captured.err


def test_val_key_018_handler_log_message_is_silenced():
    from agent_challenge.keyrelease.server import make_handler

    handler_cls = make_handler(_sentinel_service())
    # The overridden log_message is a no-op (never writes request/response data).
    assert handler_cls.log_message is not BaseHTTPRequestHandler.log_message


def test_framed_deny_helper_emits_secret_free_host_trail(capsys):
    """Every raw framed deny leaves a durable host stderr line (ledger may be null)."""

    from agent_challenge.keyrelease.server import (
        REASON_MEASUREMENT_NOT_ALLOWLISTED,
        _log_key_release_deny,
    )

    _log_key_release_deny(
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
        eval_run_id="eval-deny-trail-1",
    )
    err = capsys.readouterr().err
    assert "key_release_deny reason=measurement_not_allowlisted" in err
    assert "eval_run_id=eval-deny-trail-1" in err
    # HTTP handler silence (VAL-KEY-018) is separate: deny trail is intentional,
    # secret-free, and never carries the golden key bytes.
    assert base64.b64encode(GOLDEN_KEY).decode() not in err
    assert GOLDEN_KEY.decode("latin-1", errors="ignore") not in err
