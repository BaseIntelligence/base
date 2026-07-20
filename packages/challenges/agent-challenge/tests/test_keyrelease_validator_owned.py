"""Validator-owned allowlist authority + fail-closed conjunction + backward-compat.

Feature ``key-release-validator-owned-allowlist`` (M3), assertions VAL-KEY-026..029.
This module is the behavioral contract that the golden key-release authority is
**validator/subnet-owned (never the miner)**, that release is a **fail-closed
conjunction of every check** (no single passing check short-circuits, and any
verifier error/timeout denies), and that with the Phala path OFF the eval keeps
its **legacy golden handling** with no key-release dependency at all.

It builds on the happy-path (``key-release-endpoint-happy-path``) and deny-matrix
(``key-release-deny-paths``) features and drives the same public surfaces: the
:class:`KeyReleaseService` decision core (with an in-process RA-TLS session), the
real HTTP endpoint (the ``curl`` surface for VAL-KEY-027), and the in-CVM eval
backend (``own_runner_backend``) for the flag-off invariant.

* VAL-KEY-026  a genuine quote from a miner-chosen (non-canonical) compose is denied
* VAL-KEY-027  the allowlist is validator-owned config with NO miner-facing mutation surface
* VAL-KEY-028  ALL checks are conjunctive and the endpoint fails closed
* VAL-KEY-029  flag OFF ⇒ legacy golden handling unchanged, no key-release call

A distinct sentinel golden key is used throughout so any leak (in a body, reason,
or over the wire) on a deny path would be unmistakable.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest

from agent_challenge.canonical.report_data import PHALA_REPORT_DATA_TAG
from agent_challenge.evaluation import own_runner_backend as backend
from agent_challenge.evaluation.own_runner.orchestrator import JobResult
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
)
from agent_challenge.keyrelease.allowlist import (
    CanonicalEntry,
    MeasurementAllowlist,
)
from agent_challenge.keyrelease.client import (
    KEY_RELEASE_TAG,
    KEY_RELEASE_URL_ENV,
    key_release_report_data,
)
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
    KeyReleaseService,
    make_server,
)

# An unmistakable sentinel golden key: any leak of these bytes (raw / base64 /
# hex) on ANY deny path is a hard failure.
SENTINEL_KEY = b"VALIDATOR-OWNED-SENTINEL-KEY!!!!"  # exactly 32 bytes
ENCLAVE_PUBKEY = b"enclave-ra-tls-pubkey-0123456789"

MRTD = "11" * 48
RTMR0 = "22" * 48
RTMR1 = "33" * 48
RTMR2 = "44" * 48
COMPOSE_HASH = "ab" * 32
KEY_PROVIDER_PAYLOAD = b'{"name":"kms","id":"kms-1"}'

# Sentinel distinguishing "caller did not pass session_peer_pubkey" (default to
# the bound RA-TLS key -> a releasing request) from an explicit ``None`` (models
# a plain, non-RA-TLS request).
_DEFAULT_SESSION = object()


def _event_log(compose_hash: str = COMPOSE_HASH):
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, bytes.fromhex(compose_hash)),
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
    session_peer_pubkey=_DEFAULT_SESSION,
    mrtd: str = MRTD,
    rtmr0: str = RTMR0,
    rtmr1: str = RTMR1,
    rtmr2: str = RTMR2,
):
    """Build a fully-canonical (releasing) request; perturb one field per call."""

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
    session = ra_tls_pubkey if session_peer_pubkey is _DEFAULT_SESSION else session_peer_pubkey
    return {
        "nonce": nonce,
        "quote_hex": quote,
        "ra_tls_pubkey_hex": ra_tls_pubkey.hex(),
        "event_log": event_log,
        "vm_config": vm_config,
        "session_peer_pubkey": session,
    }


def _assert_no_key(outcome, *, reason: str | None = None):
    """The universal deny invariant: not released, no key bytes, a deny reason."""

    assert outcome.released is False
    assert outcome.key is None
    assert outcome.reason is not None
    assert SENTINEL_KEY.hex() not in outcome.reason
    assert SENTINEL_KEY.decode("latin-1") not in outcome.reason
    if reason is not None:
        assert outcome.reason == reason


# =========================================================================== #
# VAL-KEY-026 -- a miner-chosen (non-canonical) compose cannot obtain the key
# =========================================================================== #
def test_val_key_026_genuine_quote_miner_compose_denied():
    # A GENUINE quote (valid signature, acceptable TCB, fresh nonce, correct
    # key-release tag, RA-TLS peer bound) from a CVM running a miner-authored
    # compose: the event log is self-consistent (replays to its own RTMR3) but the
    # compose-hash is the miner's, not the validator's canonical value. Quote
    # validity does NOT grant release -- allowlist membership does.
    service = _make_service()
    miner_compose = "de" * 32  # miner-authored compose (not allowlisted)
    event_log, rtmr3 = _event_log(compose_hash=miner_compose)
    req = _canonical_request(service, event_log=event_log, rtmr3=rtmr3)
    _assert_no_key(service.authorize_release(**req), reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


def test_val_key_026_canonical_compose_releases_contrast():
    # The exact same genuine-quote machinery DOES release for the canonical
    # compose, proving the denial above is caused solely by allowlist membership.
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is True
    assert out.key == SENTINEL_KEY


def test_val_key_026_membership_governs_not_quote_validity():
    # Sweep several distinct miner-chosen composes -- each is a perfectly valid,
    # self-consistent, genuinely-signed quote, yet none obtains the key because
    # none is the validator's canonical compose.
    service = _make_service()
    for miner_compose in ("00" * 32, "cd" * 32, "ef" * 32, "12" * 32):
        event_log, rtmr3 = _event_log(compose_hash=miner_compose)
        req = _canonical_request(service, event_log=event_log, rtmr3=rtmr3)
        out = service.authorize_release(**req)
        _assert_no_key(out, reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


def test_val_key_026_miner_compose_never_yields_golden_key_bytes():
    # End-to-end phrasing of the anti-cheat guarantee: the key a miner-chosen
    # compose would need to decrypt golden is never produced for it.
    service = _make_service()
    event_log, rtmr3 = _event_log(compose_hash="99" * 32)
    out = service.authorize_release(**_canonical_request(service, event_log=event_log, rtmr3=rtmr3))
    _assert_no_key(out, reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


# =========================================================================== #
# VAL-KEY-027 -- allowlist is validator-owned; NO miner-facing mutation surface
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


def _http(url, *, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _release_body(nonce, *, compose_hash=COMPOSE_HASH, extra=None):
    event_log, rtmr3 = _event_log(compose_hash=compose_hash)
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    body = {
        "nonce": nonce,
        "quote": quote,
        "ra_tls_pubkey": ENCLAVE_PUBKEY.hex(),
        "event_log": event_log,
        "vm_config": {"os_image_hash": os_image_hash_from_registers(MRTD, RTMR1, RTMR2)},
    }
    if extra:
        body.update(extra)
    return body


@pytest.mark.parametrize(
    "path",
    ["/allowlist", "/allowlist/add", "/allowlist/set", "/measurement", "/config", "/trust"],
)
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
def test_val_key_027_no_allowlist_mutation_endpoint(running_server, path, method):
    # Probe the service surface for ANY allowlist-mutation affordance: none exists
    # (unknown routes / unsupported methods never 2xx), and the validator's own
    # allowlist is left intact after every probe.
    service, base = running_server
    body = None if method in ("GET", "DELETE") else {"entries": [_canonical_entry().as_dict()]}
    status, _ = _http(f"{base}{path}", method=method, body=body)
    assert status >= 400  # never a successful mutation
    # The validator-owned allowlist is unchanged by the probe.
    assert len(service.allowlist) == 1
    assert service.allowlist.contains(_canonical_entry().as_dict())


def test_val_key_027_only_known_routes_exist(running_server):
    # The entire miner-reachable surface is health / nonce / release. There is no
    # route to read, add, alter, or select the trusted measurement.
    _service, base = running_server
    assert _http(f"{base}/health")[0] == 200
    assert _http(f"{base}/nonce")[0] == 200
    # A route that would expose/mutate the allowlist simply does not exist.
    assert _http(f"{base}/allowlist")[0] == 404


def test_val_key_027_requester_supplied_expected_measurement_ignored(running_server):
    # A miner presents a genuine quote for a NON-canonical compose and tries to
    # get the server to trust it by supplying its own "expected measurement"
    # (several field spellings). The server ignores all of them and consults only
    # its own allowlist -> denied, no key.
    service, base = running_server
    _, nonce_raw = _http(f"{base}/nonce")
    nonce = json.loads(nonce_raw)["nonce"]
    miner_compose = "de" * 32
    miner_entry = _canonical_entry().as_dict()
    miner_entry["compose_hash"] = miner_compose
    extra = {
        "expected_measurement": miner_entry,
        "measurement": miner_entry,
        "allowlist": [miner_entry],
        "canonical": miner_entry,
    }
    body = _release_body(nonce, compose_hash=miner_compose, extra=extra)
    status, raw = _http(
        f"{base}/release",
        method="POST",
        body=body,
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    resp = json.loads(raw)
    assert status == 200
    assert resp["released"] is False
    assert "key" not in resp
    assert resp["reason"] == REASON_MEASUREMENT_NOT_ALLOWLISTED


def test_val_key_027_bogus_expected_measurement_does_not_block_canonical(running_server):
    # The converse: a CANONICAL quote carrying a bogus requester-supplied
    # "expected measurement" pointing at garbage STILL releases -- proving the
    # field has no effect; the server's own allowlist governs either way.
    service, base = running_server
    _, nonce_raw = _http(f"{base}/nonce")
    nonce = json.loads(nonce_raw)["nonce"]
    body = _release_body(
        nonce,
        extra={"expected_measurement": {"mrtd": "00" * 48}, "allowlist": []},
    )
    status, raw = _http(
        f"{base}/release",
        method="POST",
        body=body,
        headers={"X-RA-TLS-Peer-Key": ENCLAVE_PUBKEY.hex()},
    )
    resp = json.loads(raw)
    assert status == 200
    assert resp["released"] is True
    assert "key" in resp


def test_val_key_027_requester_supplied_os_image_hash_cannot_widen_allowlist():
    # Even the one attested field a request carries (vm_config.os_image_hash) is
    # only ever CHECKED against the fixed allowlist, never used to widen it: a
    # tampered-register quote that supplies the CANONICAL os_image_hash is still
    # denied (its other registers are not allowlisted).
    service = _make_service()
    canonical_os = os_image_hash_from_registers(MRTD, RTMR1, RTMR2)
    req = _canonical_request(service, mrtd="ee" * 48, os_image_hash=canonical_os)
    _assert_no_key(service.authorize_release(**req), reason=REASON_MEASUREMENT_NOT_ALLOWLISTED)


def test_val_key_027_requester_supplied_os_image_hash_is_ignored_registers_win():
    # Defense in depth (M3 scrutiny): the one measurement field a request can
    # carry (vm_config.os_image_hash) is NEVER trusted. The os_image_hash checked
    # against the allowlist is ALWAYS derived from the attested quote registers
    # (sha256(MRTD‖RTMR1‖RTMR2)), so a canonical quote whose request supplies a
    # bogus os_image_hash STILL releases -- the attested registers decide, not the
    # request. (Discriminator: the pre-hardening code trusted the request value and
    # would DENY this otherwise-canonical quote.)
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service, os_image_hash="ff" * 32))
    assert out.released is True
    assert out.key == SENTINEL_KEY


def test_val_key_027_allowlist_authority_is_construction_time_only():
    # The allowlist is fixed at service construction (validator-side config). The
    # request-handling API surface (authorize_release) exposes no parameter that
    # adds/selects an entry: the only measurement inputs are the attested quote +
    # its vm_config, both checked against the server-owned allowlist.
    service = _make_service(allowlist=MeasurementAllowlist())  # empty => nothing canonical
    # No request can populate the allowlist; an empty one releases nothing.
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )
    assert service.allowlist.is_empty()


# =========================================================================== #
# VAL-KEY-028 -- ALL checks are conjunctive and the endpoint fails closed
# =========================================================================== #
def test_val_key_028_baseline_all_checks_pass_releases():
    # Sanity anchor: when EVERY check passes, the key is released. Every deny
    # below flips exactly one check, so each is a real discriminator.
    service = _make_service()
    out = service.authorize_release(**_canonical_request(service))
    assert out.released is True
    assert out.key == SENTINEL_KEY


def test_val_key_028_single_check_failure_denies_invalid_signature():
    service = _make_service(verifier=StaticQuoteVerifier(valid=False))
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)), reason=REASON_INVALID_QUOTE
    )


def test_val_key_028_single_check_failure_denies_noncanonical_measurement():
    service = _make_service()
    _assert_no_key(
        service.authorize_release(**_canonical_request(service, mrtd="ee" * 48)),
        reason=REASON_MEASUREMENT_NOT_ALLOWLISTED,
    )


@pytest.mark.parametrize("tcb", ["OutOfDate", "ConfigurationNeeded", "Revoked"])
def test_val_key_028_single_check_failure_denies_unacceptable_tcb(tcb):
    service = _make_service(verifier=StaticQuoteVerifier(tcb_status=tcb))
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)), reason=REASON_TCB_UNACCEPTABLE
    )


def test_val_key_028_single_check_failure_denies_wrong_content_type_tag():
    service = _make_service()
    req = _canonical_request(service, tag=PHALA_REPORT_DATA_TAG.encode())
    _assert_no_key(service.authorize_release(**req), reason=REASON_REPORT_DATA_MISMATCH)


def test_val_key_028_single_check_failure_denies_nonce_not_bound_in_report_data():
    service = _make_service()
    nonce = service.issue_nonce()
    _assert_no_key(
        service.authorize_release(**_canonical_request(service, nonce=nonce, report_data_nonce="")),
        reason=REASON_REPORT_DATA_MISMATCH,
    )


def test_val_key_028_single_check_failure_denies_unknown_nonce():
    service = _make_service()
    _assert_no_key(
        service.authorize_release(**_canonical_request(service, nonce="never-issued-value")),
        reason=REASON_UNKNOWN_NONCE,
    )


def test_val_key_028_single_check_failure_denies_stale_nonce():
    clock = {"t": 1000.0}
    store = NonceStore(ttl_seconds=30.0, clock=lambda: clock["t"])
    service = _make_service(nonce_store=store)
    nonce = service.issue_nonce()
    clock["t"] += 31.0
    _assert_no_key(
        service.authorize_release(**_canonical_request(service, nonce=nonce)),
        reason=REASON_STALE_NONCE,
    )


def test_val_key_028_single_check_failure_denies_consumed_nonce():
    service = _make_service()
    req = _canonical_request(service)
    assert service.authorize_release(**req).released is True
    _assert_no_key(service.authorize_release(**req), reason=REASON_CONSUMED_NONCE)


def test_val_key_028_single_check_failure_denies_ra_tls_peer_mismatch():
    service = _make_service()
    req = _canonical_request(service, session_peer_pubkey=b"attacker-relay-session-pubkey-01")
    _assert_no_key(service.authorize_release(**req), reason=REASON_RA_TLS_PEER_MISMATCH)


def test_val_key_028_single_check_failure_denies_no_ra_tls_session():
    service = _make_service()
    req = _canonical_request(service, session_peer_pubkey=None)
    _assert_no_key(service.authorize_release(**req), reason=REASON_RA_TLS_REQUIRED)


def test_val_key_028_single_check_failure_denies_missing_event_log():
    service = _make_service()
    req = _canonical_request(service)
    req["event_log"] = []
    _assert_no_key(service.authorize_release(**req), reason=REASON_EVENT_LOG_REQUIRED)


def test_val_key_028_single_check_failure_denies_rtmr3_replay_mismatch():
    service = _make_service()
    _assert_no_key(
        service.authorize_release(**_canonical_request(service, rtmr3="de" * 48)),
        reason=REASON_RTMR3_MISMATCH,
    )


def test_val_key_028_verifier_error_fails_closed():
    class _BoomVerifier:
        def verify(self, quote_hex):
            raise RuntimeError("collateral fetch failed / verifier unreachable")

    from agent_challenge.keyrelease.server import REASON_VERIFIER_UNAVAILABLE

    service = _make_service(verifier=_BoomVerifier())
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_VERIFIER_UNAVAILABLE,
    )


def test_val_key_028_verifier_timeout_fails_closed():
    class _TimeoutVerifier:
        def verify(self, quote_hex):
            raise TimeoutError("quote verification collateral fetch timed out")

    from agent_challenge.keyrelease.server import REASON_VERIFIER_UNAVAILABLE

    service = _make_service(verifier=_TimeoutVerifier())
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)),
        reason=REASON_VERIFIER_UNAVAILABLE,
    )


def test_val_key_028_verifier_indeterminate_result_fails_closed():
    # A verifier that returns an unknown/indeterminate TCB posture (not on the
    # acceptable list) is treated as a failure -> denied, never released.
    service = _make_service(verifier=StaticQuoteVerifier(tcb_status="Unknown"))
    _assert_no_key(
        service.authorize_release(**_canonical_request(service)), reason=REASON_TCB_UNACCEPTABLE
    )


def test_val_key_028_no_single_passing_check_short_circuits_release():
    # The conjunction invariant in one sweep: every request below satisfies ALL
    # but one check; NONE releases and NONE returns key bytes. A single passing
    # check can never short-circuit to a release.
    def _fresh_service():
        return _make_service()

    # Each case builds its own fresh service so a fresh nonce is available.
    cases = []

    s = _fresh_service()
    cases.append((s, _canonical_request(s, mrtd="ee" * 48)))  # measurement

    s = _fresh_service()
    cases.append((s, _canonical_request(s, tag=PHALA_REPORT_DATA_TAG.encode())))  # tag

    s = _fresh_service()
    cases.append((s, _canonical_request(s, report_data_nonce="")))  # nonce binding

    s = _fresh_service()
    cases.append((s, _canonical_request(s, nonce="never-issued")))  # unknown nonce

    s = _fresh_service()
    cases.append(
        (s, _canonical_request(s, session_peer_pubkey=b"relay-peer-key-0123456789012"))
    )  # anti-relay

    s = _fresh_service()
    cases.append((s, _canonical_request(s, rtmr3="de" * 48)))  # rtmr3 replay

    for service, req in cases:
        _assert_no_key(service.authorize_release(**req))

    # signature + TCB failures (separate verifiers)
    _assert_no_key(
        _make_service(verifier=StaticQuoteVerifier(valid=False)).authorize_release(
            **_canonical_request(_make_service())
        )
    )
    tcb_service = _make_service(verifier=StaticQuoteVerifier(tcb_status="OutOfDate"))
    _assert_no_key(tcb_service.authorize_release(**_canonical_request(tcb_service)))


# =========================================================================== #
# VAL-KEY-029 -- flag OFF ⇒ legacy golden handling unchanged (no key-release dep)
# =========================================================================== #
def _legacy_job_result() -> JobResult:
    return JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )


class _RunRecorder:
    def __init__(self) -> None:
        self.called = False

    async def __call__(self, **kwargs: Any) -> JobResult:
        self.called = True
        return _legacy_job_result()


def _result_lines(out: str) -> list[dict]:
    return [
        json.loads(ln[len(RESULT_LINE_PREFIX) :])
        for ln in out.splitlines()
        if ln.startswith(RESULT_LINE_PREFIX)
    ]


def test_val_key_029_flag_off_acquire_helper_returns_none(monkeypatch):
    # With the Phala key-release endpoint unset (flag off) the eval has NO
    # key-release dependency: the helper short-circuits to None.
    monkeypatch.delenv(KEY_RELEASE_URL_ENV, raising=False)
    assert backend._acquire_golden_key_if_required() is None


def test_val_key_029_flag_off_runs_legacy_path_with_no_key_release_call(
    monkeypatch, tmp_path, capsys
):
    # Flag off: the legacy own_runner path runs unchanged; the key-release client
    # is NEVER constructed and the encrypted-at-rest in-enclave decrypt is NEVER
    # invoked -- golden handling is exactly as today.
    monkeypatch.delenv(KEY_RELEASE_URL_ENV, raising=False)
    monkeypatch.delenv(backend.PHALA_ATTESTATION_ENABLED_ENV, raising=False)

    def _forbidden_client(*args: Any, **kwargs: Any):
        raise AssertionError("key-release client constructed while the Phala flag is OFF")

    def _forbidden_decrypt(*args: Any, **kwargs: Any):
        raise AssertionError("in-enclave golden decrypt invoked while the Phala flag is OFF")

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _forbidden_client)
    monkeypatch.setattr(backend, "_decrypt_golden_in_enclave", _forbidden_decrypt)
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    out = capsys.readouterr().out
    lines = _result_lines(out)
    assert rc == 0
    assert recorder.called is True  # legacy eval path ran
    assert len(lines) == 1
    assert lines[0]["status"] == "completed"


def test_val_key_029_flag_off_default_config_has_phala_disabled():
    # The validator-config default keeps the Phala attestation path OFF, so an
    # unconfigured deployment retains legacy golden handling with no key-release.
    from agent_challenge.sdk.config import ChallengeSettings

    settings = ChallengeSettings()
    assert settings.phala_attestation_enabled is False
    assert settings.terminal_bench_execution_backend == "own_runner"


def test_val_key_029_flag_off_result_line_carries_no_attestation():
    # A flag-off run's result line is the legacy benchmark result -- no attestation
    # envelope / Phala-tier fields are attached.
    from agent_challenge.canonical import attested_result as ar

    payload = build_benchmark_result(
        status="completed", score=1.0, resolved=1, total=1, reason_code=None
    )
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload
