"""Score-path event_log[].event projection: live dstack → 1-128 visible ASCII.

Live residual (image@sha256:ffbb60a9 / compose bb68f5e1): after KR grant,
decrypt_ok, job_done trials=3, emit_start → AttestationEmissionError
'schema-v2 Eval emission is invalid: event_log[].event must be a 1-128 character
visible ASCII id' → phala_attestation_failed.

Root cause: schema-v2 wire (eval_wire._id) requires event_log[].event to be a
1–128 character visible ASCII id. Live dstack GetQuote often yields IMR0–2 (and
occasionally incomplete) entries with empty/missing/non-string ``event`` after
KR coerce + empty-IMR3 fill. Identity IMR3 events already have proper names;
early boot entries may not. Port a closed projection after coerce+IMR3 fill:
strip control bytes, derive visible id from event_type when empty and RTMR3 is
unaffected (IMR!=3 or non-runtime), else fail closed with a typed emit detail.

Discriminators (would fail a wrong implementation):
  * empty / missing / non-string ``event`` on IMR0–2 entries pass wire after
    obtain_quote / schema-v2 emit projection
  * IMR3 runtime empty event stays fail-closed (RTMR3-bound; no invent)
  * identity event names preserved; RTMR3 self-check still green
No Phala create; targeted offline only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.keyrelease.quote import (
    APP_IMR,
    COMPOSE_HASH_EVENT,
    DSTACK_RUNTIME_EVENT_TYPE,
    KEY_PROVIDER_EVENT,
    build_rtmr3_event_log,
    build_tdx_quote,
    replay_rtmr3,
)

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}


def _identity_event_log() -> tuple[list[dict[str, Any]], str]:
    compose_payload = bytes.fromhex(MEASUREMENT["compose_hash"])
    provider_payload = b'{"name":"phala"}'
    bootstrap = bytes.fromhex("11" * 8)
    filled, rtmr3 = build_rtmr3_event_log(
        [
            ("instance-id", bootstrap),
            (COMPOSE_HASH_EVENT, compose_payload),
            ("boot-mr-done", bootstrap),
            (KEY_PROVIDER_EVENT, provider_payload),
        ]
    )
    return filled, rtmr3


def _live_shaped_event_log_with_empty_imr_events() -> tuple[list[dict[str, Any]], str]:
    """Live residual: IMR0–2 empty/missing event + filled IMR3 identities.

    KR-style closed keys with empty ``digest`` on IMR3 (fill path) not needed
    here; events already have digests from ``build_rtmr3_event_log``. The score
    residual is the empty early-boot ``event`` field the wire rejects.
    """

    filled, rtmr3 = _identity_event_log()
    # Prepend live-shaped IMR0/1 entries (empty / missing / non-string event).
    early: list[dict[str, Any]] = [
        {
            "imr": 0,
            "event_type": 1,
            "digest": "aa" * 48,
            "event": "",  # live residual: empty string
            "event_payload": "",
        },
        {
            "imr": 1,
            "event_type": 2,
            "digest": "bb" * 48,
            # Missing event key — event_type-only residual surface
            "event_payload": "cc" * 4,
            "digest_extra_drop": True,  # coerce path also drops extras
        },
        {
            "imr": 2,
            "event_type": 3,
            "digest": "dd" * 48,
            "event": None,  # non-string / null residual
            "event_payload": "",
        },
        {
            "imr": 0,
            "event_type": 4,
            "digest": "ee" * 48,
            "event": "acpi\x00\x01",  # control bytes in non-IMR3 name
            "event_payload": "",
        },
    ]
    # Keep identities exactly so RTMR3 is unchanged
    return early + [dict(e) for e in filled], rtmr3


class _LiveEventLogQuoteResponse:
    def __init__(
        self,
        quote_hex: str,
        event_log: list[dict[str, Any]],
        vm_config: dict[str, Any] | None = None,
    ) -> None:
        self.quote = quote_hex
        self.event_log = json.dumps(event_log)
        self.vm_config = json.dumps(
            vm_config
            or {
                "vcpu": 1,
                "memory_mb": 2048,
                "os_image_hash": MEASUREMENT["os_image_hash"],
            }
        )
        self.report_data = ""


class _Provider:
    def __init__(self, response: _LiveEventLogQuoteResponse) -> None:
        self._response = response
        self.calls: list[bytes] = []

    def get_quote(self, report_data: bytes) -> _LiveEventLogQuoteResponse:
        self.calls.append(report_data)
        return self._response


def test_unprojected_empty_event_fails_wire() -> None:
    """Discriminator: raw empty/missing event fails schema-v2 wire before projection."""

    filled, rtmr3 = _identity_event_log()
    # Explicit closed 5-key entry with empty event string (the live residual
    # after KR coerce keeps the key; value is "") — wire rejects via _id.
    events = [
        {
            "imr": 0,
            "event_type": 1,
            "digest": "aa" * 48,
            "event": "",
            "event_payload": "",
        },
        *[dict(e) for e in filled],
    ]
    with pytest.raises(ew.EvalWireError, match=r"event_log\[\]\.event"):
        ew.validate_eval_phala_attestation(
            {
                "tdx_quote": "ab" * 600,
                "event_log": events,
                "report_data": "00" * 64,
                "measurement": {**MEASUREMENT, "rtmr3": rtmr3},
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            }
        )


def test_project_empty_missing_nonstring_event_to_visible_ascii() -> None:
    """Empty/missing/null/control-stripped event on non-IMR3 → 1–128 visible ASCII."""

    raw, _ = _live_shaped_event_log_with_empty_imr_events()
    projected = ar._project_eval_event_log(raw)
    assert projected, "projected log must be non-empty"
    for entry in projected:
        event = entry["event"]
        assert isinstance(event, str)
        assert 1 <= len(event) <= 128
        assert all("!" <= ch <= "~" for ch in event)
        assert set(entry) == {
            "imr",
            "event_type",
            "digest",
            "event",
            "event_payload",
        }

    # Control bytes stripped from non-IMR3 name
    control_entries = [e for e in projected if e["event"].startswith("acpi")]
    assert control_entries
    assert control_entries[0]["event"] == "acpi"

    # Identity names preserved (RTMR3uring names)
    names = {e["event"] for e in projected if e["imr"] == APP_IMR}
    assert COMPOSE_HASH_EVENT in names
    assert KEY_PROVIDER_EVENT in names


def test_project_imr3_runtime_empty_event_fails_closed() -> None:
    """IMR3 runtime empty event is RTMR3-bound: do not invent a name."""

    events, _ = _identity_event_log()
    broken = [dict(e) for e in events]
    # Break the compose-hash name (runtime IMR3) to empty — fail closed.
    for entry in broken:
        if entry["event"] == COMPOSE_HASH_EVENT:
            entry["event"] = ""
            break
    with pytest.raises(ar.AttestationEmissionError, match=r"(?i)event|RTMR3|imr3"):
        ar._project_eval_event_log(broken)


def test_obtain_quote_projects_event_ids_and_passes_wire() -> None:
    """obtain_quote applies event projection after coerce; wire accepts the log."""

    raw_log, expected_rtmr3 = _live_shaped_event_log_with_empty_imr_events()
    quote_hex = build_tdx_quote(
        mrtd=MEASUREMENT["mrtd"],
        rtmr0=MEASUREMENT["rtmr0"],
        rtmr1=MEASUREMENT["rtmr1"],
        rtmr2=MEASUREMENT["rtmr2"],
        rtmr3=expected_rtmr3,
        report_data=b"z" * 32,
    )
    provider = _Provider(_LiveEventLogQuoteResponse(quote_hex, raw_log))
    result = ar.obtain_quote(provider, b"z" * 32)

    assert result.event_log
    for entry in result.event_log:
        assert ew._id(entry["event"], "event_log[].event") == entry["event"]

    # RTMR3 still matches (only non-IMRT3 names projected)
    assert replay_rtmr3(result.event_log).rtmr3 == expected_rtmr3

    # Full attestation shape with projected log is wire-valid
    att = ew.validate_eval_phala_attestation(
        {
            "tdx_quote": result.quote,
            "event_log": result.event_log,
            "report_data": "00" * 64,
            "measurement": {**MEASUREMENT, "rtmr3": expected_rtmr3},
            "vm_config": result.vm_config,
        }
    )
    assert att["event_log"]


def test_schema_v2_emit_with_live_shaped_empty_event_log(monkeypatch) -> None:
    """schema-v2 emit green with live-shaped empty/missing event residuals."""

    del monkeypatch
    raw_log, expected_rtmr3 = _live_shaped_event_log_with_empty_imr_events()

    class _DstackProvider:
        def get_quote(self, report_data: bytes) -> _LiveEventLogQuoteResponse:
            quote_hex = build_tdx_quote(
                mrtd=MEASUREMENT["mrtd"],
                rtmr0=MEASUREMENT["rtmr0"],
                rtmr1=MEASUREMENT["rtmr1"],
                rtmr2=MEASUREMENT["rtmr2"],
                rtmr3=expected_rtmr3,
                report_data=report_data,
            )
            return _LiveEventLogQuoteResponse(quote_hex, raw_log)

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    task_ids = ["task-a", "task-b"]
    record = ew.build_canonical_score_record(
        eval_run_id="eval-event-id-project",
        policy=policy,
        trial_scores_by_task={"task-a": [1.0], "task-b": [0.0]},
    )
    line = ar.emit_attested_benchmark_result(
        benchmark_result={
            "status": "completed",
            "score": 0.5,
            "resolved": 1,
            "total": 2,
            "reason_code": None,
        },
        canonical_measurement=dict(MEASUREMENT),
        rtmr3="0" * 96,
        agent_hash="f" * 64,
        task_ids=task_ids,
        scores={},
        quote_provider=_DstackProvider(),
        manifest_sha256="1" * 64,
        eval_run_id="eval-event-id-project",
        submission_id="submission-event-id-project",
        score_nonce="score-nonce-event-id",
        score_record=record,
        image_digest="registry.example/eval@sha256:" + "d" * 64,
        vm_config=None,
    )
    payload = json.loads(line.split("=", 1)[1])
    assert ew.validate_eval_result_request(payload) == payload
    for entry in payload["execution_proof"]["attestation"]["event_log"]:
        event = entry["event"]
        assert isinstance(event, str)
        assert 1 <= len(event) <= 128
        assert all("!" <= ch <= "~" for ch in event)


def test_schema_v2_emit_fails_closed_imr3_runtime_empty_event() -> None:
    """Emit path fails closed with typed attach when RTMR3-bound name is empty."""

    events, expected_rtmr3 = _identity_event_log()
    broken = [dict(e) for e in events]
    for entry in broken:
        if entry["event"] == COMPOSE_HASH_EVENT:
            entry["event"] = ""
            break

    class _BrokenProvider:
        def get_quote(self, report_data: bytes) -> _LiveEventLogQuoteResponse:
            quote_hex = build_tdx_quote(
                mrtd=MEASUREMENT["mrtd"],
                rtmr0=MEASUREMENT["rtmr0"],
                rtmr1=MEASUREMENT["rtmr1"],
                rtmr2=MEASUREMENT["rtmr2"],
                rtmr3=expected_rtmr3,
                report_data=report_data,
            )
            return _LiveEventLogQuoteResponse(quote_hex, broken)

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    record = ew.build_canonical_score_record(
        eval_run_id="eval-event-id-imr3-empty",
        policy=policy,
        trial_scores_by_task={"task-a": [1.0]},
    )
    with pytest.raises(ar.AttestationEmissionError, match=r"(?i)event|RTMR3|imr3|schema-v2"):
        ar.emit_attested_benchmark_result(
            benchmark_result={
                "status": "completed",
                "score": 1.0,
                "resolved": 1,
                "total": 1,
                "reason_code": None,
            },
            canonical_measurement=dict(MEASUREMENT),
            rtmr3="0" * 96,
            agent_hash="f" * 64,
            task_ids=["task-a"],
            scores={},
            quote_provider=_BrokenProvider(),
            manifest_sha256="1" * 64,
            eval_run_id="eval-event-id-imr3-empty",
            submission_id="submission-event-id-imr3-empty",
            score_nonce="score-nonce-imr3-empty",
            score_record=record,
            image_digest="registry.example/eval@sha256:" + "d" * 64,
            vm_config=None,
        )


def test_project_preserves_rtmr3_for_identity_only_log() -> None:
    """Identity-only log projection is a no-op that keeps RTMR3 green."""

    events, expected = _identity_event_log()
    projected = ar._project_eval_event_log(events)
    assert replay_rtmr3(projected).rtmr3 == expected
    names = [e["event"] for e in projected]
    assert COMPOSE_HASH_EVENT in names
    assert KEY_PROVIDER_EVENT in names
    # DSTACK runtime type marker still present for identities
    for entry in projected:
        if entry["event"] in {COMPOSE_HASH_EVENT, KEY_PROVIDER_EVENT}:
            assert entry["imr"] == APP_IMR
            assert entry["event_type"] == DSTACK_RUNTIME_EVENT_TYPE
