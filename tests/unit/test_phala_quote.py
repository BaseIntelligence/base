"""Unit tests for the base-side TDX quote primitives (M4 verifier support).

Pins the structural parser, the dstack RTMR3 event-log replay, the OS-image
identity, and the ``dcap-qvl`` adapter's accept / reject / park mapping used by
the Phala-tier verifier.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from base.worker.phala_quote import (
    DcapQvlVerifier,
    QuoteStructureError,
    QuoteVerificationError,
    StaticQuoteVerifier,
    VerifierUnavailableError,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
    parse_td_report,
    replay_rtmr3,
    runtime_event_digest,
)

MRTD = "a1" * 48
RTMR0 = "b0" * 48
RTMR1 = "b1" * 48
RTMR2 = "b2" * 48


def _quote(report_data: str = "ab" * 64) -> str:
    _log, rtmr3 = build_rtmr3_event_log([("compose-hash", bytes.fromhex("c3" * 32))])
    return build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data,
    )


def test_parse_td_report_round_trips_registers() -> None:
    report = parse_td_report(_quote(report_data="ff" * 32))
    assert report.mrtd == MRTD
    assert report.rtmr0 == RTMR0
    assert report.rtmr1 == RTMR1
    assert report.rtmr2 == RTMR2
    assert report.report_data == bytes.fromhex("ff" * 32).ljust(64, b"\x00")


def test_parse_td_report_accepts_0x_prefix() -> None:
    report = parse_td_report("0x" + _quote())
    assert report.mrtd == MRTD


@pytest.mark.parametrize("bad", ["", "zz", "ab" * 4])
def test_parse_td_report_rejects_malformed_or_short(bad: str) -> None:
    with pytest.raises(QuoteStructureError):
        parse_td_report(bad)


def test_os_image_hash_matches_sha256_of_registers() -> None:
    expected = hashlib.sha256(
        bytes.fromhex(MRTD) + bytes.fromhex(RTMR1) + bytes.fromhex(RTMR2)
    ).hexdigest()
    assert os_image_hash_from_registers(MRTD, RTMR1, RTMR2) == expected


def test_replay_rtmr3_surfaces_compose_and_key_provider() -> None:
    compose = bytes.fromhex("c3" * 32)
    provider = b"kms-root"
    log, rtmr3 = build_rtmr3_event_log(
        [("compose-hash", compose), ("key-provider", provider)]
    )
    replay = replay_rtmr3(log)
    assert replay.rtmr3 == rtmr3
    assert replay.compose_hash == compose.hex()
    assert replay.key_provider == provider.hex()


def test_replay_rtmr3_ignores_non_app_imr_entries() -> None:
    log, rtmr3 = build_rtmr3_event_log([("compose-hash", bytes.fromhex("c3" * 32))])
    noise = [{"imr": 0, "event": "boot", "digest": "aa" * 48}, *log]
    assert replay_rtmr3(noise).rtmr3 == rtmr3


def test_replay_rtmr3_rejects_inconsistent_digest() -> None:
    log, _rtmr3 = build_rtmr3_event_log([("compose-hash", bytes.fromhex("c3" * 32))])
    log[0]["event_payload"] = "ee" * 32  # digest no longer matches payload
    with pytest.raises(QuoteVerificationError):
        replay_rtmr3(log)


def test_replay_rtmr3_rejects_non_mapping_entry() -> None:
    with pytest.raises(QuoteVerificationError):
        replay_rtmr3([["not", "a", "mapping"]])  # type: ignore[list-item]


def test_replay_rtmr3_rejects_bad_payload_hex() -> None:
    from base.worker.phala_quote import APP_IMR, DSTACK_RUNTIME_EVENT_TYPE

    bad = [
        {
            "imr": APP_IMR,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "event": "compose-hash",
            "event_payload": "zz",
        }
    ]
    with pytest.raises(QuoteVerificationError):
        replay_rtmr3(bad)


def test_replay_rtmr3_accepts_non_runtime_event_with_raw_digest() -> None:
    digest = runtime_event_digest("compose-hash", bytes.fromhex("c3" * 32))
    entry = {
        "imr": 3,
        "event_type": 1,
        "event": "some-tcg-event",
        "digest": digest.hex(),
    }
    replay = replay_rtmr3([entry])
    assert len(bytes.fromhex(replay.rtmr3)) == 48


def test_build_tdx_quote_honors_header_and_tail() -> None:
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3="d3" * 48,
        report_data="ab" * 32,
        header=b"\x01" * 48,
        tail=b"\x09\x09",
    )
    raw = bytes.fromhex(quote)
    assert raw[:48] == b"\x01" * 48
    assert raw.endswith(b"\x09\x09")


def test_static_verifier_default_is_uptodate() -> None:
    assert StaticQuoteVerifier().verify("00" * 8).tcb_status == "UpToDate"


def test_dcap_qvl_parses_advisories_and_alt_status_keys() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        body = json.dumps({"tcbStatus": "UpToDate", "advisoryIDs": ["INTEL-SA-1"]})
        return subprocess.CompletedProcess(args, returncode=0, stdout=body, stderr="")

    verdict = DcapQvlVerifier(runner=runner).verify("00" * 8)
    assert verdict.tcb_status == "UpToDate"
    assert verdict.advisory_ids == ("INTEL-SA-1",)


def test_dcap_qvl_unparseable_json_is_reject() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, returncode=0, stdout="not json", stderr=""
        )

    with pytest.raises(QuoteVerificationError):
        DcapQvlVerifier(runner=runner).verify("00" * 8)


def test_dcap_qvl_non_object_json_is_reject() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, returncode=0, stdout="[1,2,3]", stderr=""
        )

    with pytest.raises(QuoteVerificationError):
        DcapQvlVerifier(runner=runner).verify("00" * 8)


def test_dcap_qvl_missing_status_is_reject() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode=0, stdout="{}", stderr="")

    with pytest.raises(QuoteVerificationError):
        DcapQvlVerifier(runner=runner).verify("00" * 8)


def test_dcap_qvl_generic_subprocess_error_is_park() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.SubprocessError("boom")

    with pytest.raises(VerifierUnavailableError):
        DcapQvlVerifier(runner=runner).verify("00" * 8)
