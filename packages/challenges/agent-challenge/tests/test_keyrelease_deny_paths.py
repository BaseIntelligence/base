"""Validator key-release DENY matrix (M3, VAL-KEY-019..025 + VAL-KEY-030).

The happy path (``key-release-endpoint-happy-path``) landed the conjunctive,
fail-closed :meth:`KeyReleaseService.authorize_release` decision core. This module
is its adversarial complement: it pins the *full deny matrix* as a behavioral
contract, driving each independent block through the public surface (the decision
core with an in-process RA-TLS session, plus the real HTTP endpoint) and asserting
the hard invariant on every path -- **the golden key is never released and zero
key bytes ever leave the endpoint**:

* VAL-KEY-019  non-canonical / tampered image measurement -> denied
* VAL-KEY-020  invalid / forged / malformed quote (and verifier error) -> denied
* VAL-KEY-021  stale / expired nonce -> denied
* VAL-KEY-022  reused (already-consumed) nonce -> denied
* VAL-KEY-023  unknown / never-issued nonce -> denied
* VAL-KEY-024  missing nonce binding in report_data -> denied
* VAL-KEY-025  a captured valid quote cannot be replayed against a fresh nonce
* VAL-KEY-030  empty / unconfigured / unparseable allowlist -> releases nothing

A distinct sentinel golden key is used throughout so a leak (in a response body,
reason string, or over the wire) would be unmistakable.
"""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from agent_challenge.canonical.report_data import PHALA_REPORT_DATA_TAG
from agent_challenge.keyrelease.allowlist import (
    ALLOWLIST_FILE_ENV,
    AllowlistError,
    CanonicalEntry,
    MeasurementAllowlist,
)
from agent_challenge.keyrelease.client import KEY_RELEASE_TAG, key_release_report_data
from agent_challenge.keyrelease.nonce import NonceStore
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
)
from agent_challenge.keyrelease.server import (
    REASON_CONSUMED_NONCE,
    REASON_INVALID_QUOTE,
    REASON_MEASUREMENT_NOT_ALLOWLISTED,
    REASON_REPORT_DATA_MISMATCH,
    REASON_STALE_NONCE,
    REASON_UNKNOWN_NONCE,
    KeyReleaseService,
    make_server,
)

# A deliberately unmistakable golden key: any leak of these bytes (raw / base64 /
# hex) on ANY deny path is a hard failure.
SENTINEL_KEY = b"SENTINEL-DENY-MATRIX-GOLDEN-KEY!"  # 31 bytes -> pad below
SENTINEL_KEY = SENTINEL_KEY.ljust(32, b"!")  # exactly 32 bytes
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
        "golden_key_loader": lambda: SENTINEL_KEY,
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
    report_data_override: bytes | None = None,
    tag: bytes = KEY_RELEASE_TAG,
    event_log=None,
    os_image_hash: str | None = None,
    mrtd: str = MRTD,
    rtmr0: str = RTMR0,
    rtmr1: str = RTMR1,
    rtmr2: str = RTMR2,
):
    """Build a fully-canonical release request (kwargs for authorize_release).

    Individual fields can be perturbed to drive a single deny branch while every
    other check stays valid (so the assertion under test is the sole cause).
    """

    if nonce is None:
        nonce = service.issue_nonce()
    if event_log is None:
        event_log, replay_rtmr3_hex = _event_log()
    else:
        _, replay_rtmr3_hex = _event_log()
    quote_rtmr3 = rtmr3 if rtmr3 is not None else replay_rtmr3_hex

    if report_data_override is not None:
        report_data = report_data_override
    else:
        rd_nonce = report_data_nonce if report_data_nonce is not None else nonce
        rd_pubkey = report_data_pubkey if report_data_pubkey is not None else ra_tls_pubkey
        if tag == KEY_RELEASE_TAG:
            report_data = key_release_report_data(rd_nonce, rd_pubkey)
        else:
            import hashlib

            report_data = hashlib.sha256(tag + rd_nonce.encode() + rd_pubkey).digest()

    quote = build_tdx_quote(
        mrtd=mrtd,
        rtmr0=rtmr0,
        rtmr1=rtmr1,
        rtmr2=rtmr2,
        rtmr3=quote_rtmr3,
        report_data=report_data,
    )
    vm_config = {
        "os_image_hash": os_image_hash
        if os_image_hash is not None
        else os_image_hash_from_registers(mrtd, rtmr1, rtmr2)
    }
    return {
        "nonce": nonce,
        "quote_hex": quote,
        "ra_tls_pubkey_hex": ra_tls_pubkey.hex(),
        "event_log": event_log,
        "vm_config": vm_config,
        "session_peer_pubkey": ra_tls_pubkey,
    }


def _assert_no_key(outcome, *, reason: str | None = None):
    """The universal deny invariant: not released, no key bytes, a deny reason."""

    assert outcome.released is False
    assert outcome.key is None
    assert outcome.reason is not None
    # The sentinel key must never appear anywhere in the reason string.
    assert SENTINEL_KEY.hex() not in outcome.reason
    assert SENTINEL_KEY.decode("latin-1") not in outcome.reason
    if reason is not None:
        assert outcome.reason == reason


# =========================================================================== #
# Sanity: the canonical request DOES release (so every deny below is a real
# discriminator, not a request that would fail anyway).
# =========================================================================== #
def test_canonical_request_releases_sentinel_key():
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is True
    assert out.key == SENTINEL_KEY


# =========================================================================== #
# VAL-KEY-019 -- non-canonical / tampered image measurement -> denied
# =========================================================================== #
@pytest.mark.parametrize("register", ["mrtd", "rtmr0", "rtmr1", "rtmr2"])
def test_val_key_019_tampered_register_denied(register):
    # The quote reflects a modified image: one measurement register differs from
    # every allowlisted entry. No key -> the modified image cannot decrypt golden.
    service = _make_service()
    req = _canonical_request(service, **{register: "ee" * 48})
    _assert_no_key(service.authorize_release(**req), reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


def test_val_key_019_noncanonical_compose_hash_denied():
    # A self-consistent event log (replays to its own RTMR3) whose compose-hash is
    # simply not the allowlisted value: measurement mismatch, no key.
    service = _make_service()
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex("cd" * 32)),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
        ]
    )
    req = _canonical_request(service, event_log=event_log, rtmr3=rtmr3)
    _assert_no_key(service.authorize_release(**req), reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


def test_val_key_019_noncanonical_os_image_hash_denied():
    # os_image_hash is derived from the attested registers (a requester-supplied
    # value is ignored), so a non-canonical os_image is expressed via an allowlist
    # whose pinned os_image_hash the derived value cannot match.
    entry = _canonical_entry().as_dict()
    entry["os_image_hash"] = "ff" * 32
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


def test_val_key_019_noncanonical_key_provider_denied():
    entry = _canonical_entry().as_dict()
    # Candidate from kms JSON decodes to "phala"; pin a different id → miss.
    entry["key_provider"] = "attacker-kms"
    service = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(entry)]))
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


def test_val_key_019_tampered_image_never_decrypts_golden():
    # End-to-end phrasing of the anti-cheat guarantee: a tampered image is denied
    # the key, and the key it would need is never produced for it.
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service, mrtd="ee" * 48))
    _assert_no_key(out, reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


# =========================================================================== #
# VAL-KEY-020 -- invalid / forged / malformed quote -> denied
# =========================================================================== #
def test_val_key_020_forged_signature_denied():
    service = _make_service(verifier=StaticQuoteVerifier(valid=False))
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)), reason=REASON_INVALID_QUOTE
    )


def test_val_key_020_malformed_quote_too_short_denied():
    service = _make_service()
    req = _canonical_request(service)
    req["quote_hex"] = "abcd"  # far too short to hold a TD report
    _assert_no_key(service.authorize_release(**req), reason=REASON_INVALID_QUOTE)


def test_val_key_020_nonhex_quote_denied():
    service = _make_service()
    req = _canonical_request(service)
    req["quote_hex"] = "zz" * 400  # not valid hex
    _assert_no_key(service.authorize_release(**req), reason=REASON_INVALID_QUOTE)


def test_val_key_020_verifier_error_fails_closed():
    class _BoomVerifier:
        def verify(self, quote_hex):
            raise RuntimeError("collateral fetch failed / verifier unreachable")

    from agent_challenge.keyrelease.server import REASON_VERIFIER_UNAVAILABLE

    service = _make_service(verifier=_BoomVerifier())
    # Fail closed with zero key; the disposition is retryable, not definitive-invalid.
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_VERIFIER_UNAVAILABLE,
    )


# =========================================================================== #
# VAL-KEY-021 -- stale / expired nonce -> denied (measurement/quote notwithstanding)
# =========================================================================== #
def test_val_key_021_expired_nonce_denied():
    clock = {"t": 1000.0}
    store = NonceStore(ttl_seconds=30.0, clock=lambda: clock["t"])
    service = _make_service(nonce_store=store)
    nonce = service.issue_nonce()
    clock["t"] += 31.0  # advance strictly past the TTL
    # The quote is otherwise perfectly valid + canonical; only the nonce is stale.
    _assert_no_key(
        service.authorize_release(**_canonical_request(service, nonce=nonce)),
        reason=REASON_STALE_NONCE,
    )


# =========================================================================== #
# VAL-KEY-022 -- reused (already-consumed) nonce -> denied
# =========================================================================== #
def test_val_key_022_reused_nonce_after_success_denied():
    service = _make_service()
    req = _canonical_request(service)
    first = service.authorize_release(**req)
    assert first.released is True and first.key == SENTINEL_KEY
    # The identical (already-consumed) nonce cannot obtain the key a second time.
    _assert_no_key(service.authorize_release(**req), reason=REASON_CONSUMED_NONCE)


def test_val_key_022_nonce_consumed_even_after_denied_attempt():
    # A nonce is burned on ANY completed RA-TLS release attempt, even a denied one,
    # so it cannot be retried on a later valid quote.
    service = _make_service()
    nonce = service.issue_nonce()
    denied = service.authorize_release(**_canonical_request(service, nonce=nonce, mrtd="ee" * 48))
    _assert_no_key(denied, reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)
    reused = service.authorize_release(**_canonical_request(service, nonce=nonce))
    _assert_no_key(reused, reason=REASON_CONSUMED_NONCE)


# =========================================================================== #
# VAL-KEY-023 -- unknown / never-issued nonce -> denied
# =========================================================================== #
def test_val_key_023_unknown_nonce_denied():
    service = _make_service()
    # An attacker-chosen value the endpoint never issued.
    req = _canonical_request(service, nonce="attacker-forged-nonce-never-issued")
    _assert_no_key(service.authorize_release(**req), reason=REASON_UNKNOWN_NONCE)


# =========================================================================== #
# VAL-KEY-024 -- missing nonce binding in report_data -> denied
# =========================================================================== #
def test_val_key_024_report_data_empty_nonce_binding_denied():
    # A valid, canonical quote whose report_data binds an EMPTY nonce (no nonce
    # encoded) is denied even though a real issued nonce is presented in-band.
    service = _make_service()
    nonce = service.issue_nonce()
    req = _canonical_request(service, nonce=nonce, report_data_nonce="")
    _assert_no_key(service.authorize_release(**req), reason=REASON_REPORT_DATA_MISMATCH)


def test_val_key_024_report_data_all_zero_denied():
    # report_data carries no binding at all (all-zero 64-byte field): denied.
    service = _make_service()
    nonce = service.issue_nonce()
    req = _canonical_request(service, nonce=nonce, report_data_override=bytes(64))
    _assert_no_key(service.authorize_release(**req), reason=REASON_REPORT_DATA_MISMATCH)


def test_val_key_024_wrong_nonce_in_report_data_denied():
    service = _make_service()
    nonce = service.issue_nonce()
    req = _canonical_request(service, nonce=nonce, report_data_nonce="a-different-nonce")
    _assert_no_key(service.authorize_release(**req), reason=REASON_REPORT_DATA_MISMATCH)


def test_val_key_024_result_tag_quote_denied_at_key_release():
    # A quote minted for the RESULT-attestation purpose (result domain tag) is not
    # a valid key-release binding: its report_data does not encode the key-release
    # nonce binding, so it is denied (cross-protocol reuse blocked).
    service = _make_service()
    req = _canonical_request(service, tag=PHALA_REPORT_DATA_TAG.encode())
    _assert_no_key(service.authorize_release(**req), reason=REASON_REPORT_DATA_MISMATCH)


# =========================================================================== #
# VAL-KEY-025 -- a captured valid quote cannot be replayed against a fresh nonce
# =========================================================================== #
def test_val_key_025_captured_quote_replayed_against_new_nonce_denied():
    service = _make_service()
    # 1) A legitimate release for nonce N1 (this quote is now "captured" by an
    #    on-path observer of the encrypted transcript metadata / a malicious host).
    n1 = service.issue_nonce()
    req1 = _canonical_request(service, nonce=n1)
    released = service.authorize_release(**req1)
    assert released.released is True and released.key == SENTINEL_KEY
    captured_quote = req1["quote_hex"]

    # 2) A fresh, legitimate nonce N2 is issued. Replaying the CAPTURED quote (whose
    #    report_data binds N1) against N2 is denied: the binding does not match N2.
    n2 = service.issue_nonce()
    replay = _canonical_request(service, nonce=n2)
    replay["quote_hex"] = captured_quote
    out = service.authorize_release(**replay)
    _assert_no_key(out, reason=REASON_REPORT_DATA_MISMATCH)


def test_val_key_025_captured_quote_denied_for_every_fresh_nonce():
    service = _make_service()
    n1 = service.issue_nonce()
    req1 = _canonical_request(service, nonce=n1)
    assert service.authorize_release(**req1).released is True
    captured_quote = req1["quote_hex"]

    # The captured quote never releases the key for any subsequent fresh nonce.
    for _ in range(5):
        fresh = service.issue_nonce()
        attempt = _canonical_request(service, nonce=fresh)
        attempt["quote_hex"] = captured_quote
        _assert_no_key(service.authorize_release(**attempt), reason=REASON_REPORT_DATA_MISMATCH)


def test_val_key_025_captured_quote_reused_with_original_nonce_denied():
    # Replaying the captured quote with its ORIGINAL nonce N1 (already consumed by
    # the first release) is likewise denied -- single-use closes this door too.
    service = _make_service()
    n1 = service.issue_nonce()
    req1 = _canonical_request(service, nonce=n1)
    assert service.authorize_release(**req1).released is True
    _assert_no_key(service.authorize_release(**req1), reason=REASON_CONSUMED_NONCE)


# =========================================================================== #
# VAL-KEY-030 -- empty / unconfigured / unparseable allowlist -> releases nothing
# =========================================================================== #
def test_val_key_030_empty_allowlist_denies_canonical_request():
    # An empty allowlist has no canonical measurement to match: even a genuine,
    # canonical, fresh-nonce, correct-tag, RA-TLS-bound quote is denied.
    service = _make_service(allowlist=MeasurementAllowlist())
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


def test_val_key_030_empty_allowlist_is_not_match_anything():
    # Sweeping several distinct (well-formed) measurements, the empty allowlist
    # matches NONE of them (never "accept-any").
    service = _make_service(allowlist=MeasurementAllowlist())
    for reg in ("mrtd", "rtmr0", "rtmr1", "rtmr2"):
        out = service.authorize_release(**_canonical_request(service, **{reg: "ee" * 48}))
        _assert_no_key(out, reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)
    # And the otherwise-canonical measurement is likewise unmatched.
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


def test_val_key_030_from_env_unconfigured_allowlist_denies(monkeypatch):
    monkeypatch.delenv(ALLOWLIST_FILE_ENV, raising=False)
    service = KeyReleaseService.from_env(
        verifier=StaticQuoteVerifier(), golden_key_loader=lambda: SENTINEL_KEY
    )
    assert service.allowlist.is_empty()
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


def test_val_key_030_from_env_missing_allowlist_file_fails_closed(tmp_path, monkeypatch):
    # A configured-but-missing allowlist file fails closed (raises) rather than
    # silently building an accept-any / releasing service.
    monkeypatch.setenv(ALLOWLIST_FILE_ENV, str(tmp_path / "does-not-exist.json"))
    with pytest.raises(AllowlistError):
        KeyReleaseService.from_env(
            verifier=StaticQuoteVerifier(), golden_key_loader=lambda: SENTINEL_KEY
        )


def test_val_key_030_from_env_unparseable_allowlist_fails_closed(tmp_path, monkeypatch):
    bad = tmp_path / "allowlist.json"
    bad.write_text("{ this is not valid json ")
    monkeypatch.setenv(ALLOWLIST_FILE_ENV, str(bad))
    with pytest.raises(AllowlistError):
        KeyReleaseService.from_env(
            verifier=StaticQuoteVerifier(), golden_key_loader=lambda: SENTINEL_KEY
        )


def test_val_key_030_from_env_empty_entries_file_denies(tmp_path, monkeypatch):
    # A well-formed but empty allowlist file ({"entries": []}) parses to an empty
    # allowlist -> still denies all (fail-closed on misconfig).
    empty = tmp_path / "allowlist.json"
    empty.write_text(json.dumps({"entries": []}))
    monkeypatch.setenv(ALLOWLIST_FILE_ENV, str(empty))
    service = KeyReleaseService.from_env(
        verifier=StaticQuoteVerifier(), golden_key_loader=lambda: SENTINEL_KEY
    )
    assert service.allowlist.is_empty()
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


# =========================================================================== #
# Cross-cutting: EVERY deny branch returns zero key bytes (single matrix)
# =========================================================================== #
def test_full_deny_matrix_never_returns_key_bytes():
    service = _make_service()

    def _fresh(**kw):
        return _canonical_request(service, nonce=service.issue_nonce(), **kw)

    cases: dict[str, dict] = {
        "tampered_mrtd": _fresh(mrtd="ee" * 48),
        "unknown_nonce": _canonical_request(service, nonce="never-issued-value"),
        "missing_nonce_binding": _fresh(report_data_nonce=""),
        "zero_report_data": _fresh(report_data_override=bytes(64)),
        "wrong_tag": _fresh(tag=PHALA_REPORT_DATA_TAG.encode()),
    }
    for name, req in cases.items():
        out = service.authorize_release(**req)
        assert out.released is False, name
        assert out.key is None, name
        assert SENTINEL_KEY.hex() not in (out.reason or ""), name

    # os_image_hash is derived from the attested registers; a non-canonical
    # os_image is expressed via an allowlist whose pinned value cannot be matched.
    oimg_entry = _canonical_entry().as_dict()
    oimg_entry["os_image_hash"] = "ff" * 32
    oimg = _make_service(allowlist=MeasurementAllowlist([CanonicalEntry.from_mapping(oimg_entry)]))
    _assert_no_key(oimg.authorize_release(**_canonical_request(oimg)))

    # Forged signature + verifier error branches (separate verifiers).
    forged = _make_service(verifier=StaticQuoteVerifier(valid=False))
    _assert_no_key(forged.authorize_release(**_canonical_request(forged)))

    # Empty allowlist denies the canonical request too.
    empty = _make_service(allowlist=MeasurementAllowlist())
    _assert_no_key(empty.authorize_release(**_canonical_request(empty)))


# =========================================================================== #
# HTTP wire contract: deny responses carry NO key field / NO key bytes
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


def _release_body(nonce, *, report_data):
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )
    return {
        "nonce": nonce,
        "quote": quote,
        "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
        "event_log": event_log,
        "vm_config": {"os_image_hash": os_image_hash_from_registers(MRTD, RTMR1, RTMR2)},
    }


def test_http_unknown_nonce_deny_carries_no_key(running_server):
    _service, base = running_server
    fake = "never-issued-over-http"
    body = _release_body(fake, report_data=key_release_report_data(fake, ENCLAVE_PUBKEY))
    status, resp = _http_json(
        f"{base}/release",
        method="POST",
        body=body,
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    assert status == 200
    assert resp["released"] is False
    assert "key" not in resp
    assert resp["reason"] == REASON_UNKNOWN_NONCE
    for token in (SENTINEL_KEY.hex(), SENTINEL_KEY.decode("latin-1")):
        assert token not in json.dumps(resp)


def test_http_replayed_quote_against_new_nonce_denied(running_server):
    service, base = running_server
    # Release for N1 over the wire (captured quote).
    _, n1_body = _http_json(f"{base}/nonce")
    n1 = n1_body["nonce"]
    body1 = _release_body(n1, report_data=key_release_report_data(n1, ENCLAVE_PUBKEY))
    status, ok = _http_json(
        f"{base}/release",
        method="POST",
        body=body1,
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    assert status == 200 and ok["released"] is True

    # Replay the captured quote for a fresh N2 -> denied, no key over the wire.
    _, n2_body = _http_json(f"{base}/nonce")
    n2 = n2_body["nonce"]
    replay_body = dict(body1)
    replay_body["nonce"] = n2  # captured quote still binds N1
    status, resp = _http_json(
        f"{base}/release",
        method="POST",
        body=replay_body,
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    assert status == 200
    assert resp["released"] is False
    assert "key" not in resp
    assert resp["reason"] == REASON_REPORT_DATA_MISMATCH


def test_http_empty_allowlist_denies_over_wire():
    service = _make_service(allowlist=MeasurementAllowlist())
    server = make_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    base = f"http://{host}:{port}"
    try:
        _, nonce_body = _http_json(f"{base}/nonce")
        nonce = nonce_body["nonce"]
        body = _release_body(nonce, report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY))
        status, resp = _http_json(
            f"{base}/release",
            method="POST",
            body=body,
            headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
        )
        assert status == 200
        assert resp["released"] is False
        assert "key" not in resp
        assert resp["reason"] == REASON_MEASUREMENT_NOT_ALLOWLISTED
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
