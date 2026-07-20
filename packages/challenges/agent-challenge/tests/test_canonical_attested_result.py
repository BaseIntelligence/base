"""Behavioral tests for in-image attested-result emission (M1).

Fulfils the offline slice of the ``attestation-quote-emission-selfverify`` feature:
  * VAL-IMG-025 emitted envelope conforms to the BASE ExecutionProof Phala tier schema
  * VAL-IMG-026 an envelope missing a required field fails conformance validation
  * VAL-IMG-027 the BASE_BENCHMARK_RESULT= line stays parseable AND carries the envelope
  * VAL-IMG-028 the envelope/report_data agent_hash matches the submitted agent
  * VAL-IMG-034 get_quote failure fails closed (no fabricated/attested-looking result)

The live self-verify assertions this feature also fulfils (VAL-IMG-020/021/022/024)
are exercised against a real Phala CVM in M6; here the dstack quote provider is
faked so the emission path, envelope conformance, and fail-closed behavior are
verified offline. The envelope shape is pinned to base's real
``ExecutionProof``/``PhalaAttestation`` model in
``base/tests/unit/test_worker_proof_phala.py`` (cross-repo conformance).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import report_data as rd
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    validate_benchmark_result,
)

# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #
MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}
RTMR3 = "d" * 96
AGENT_HASH = "f" * 64
TASK_IDS = ["task-b", "task-a", "task-c"]
SCORES = {"task-a": 1.0, "task-b": 0.0, "task-c": 0.5}
NONCE = "nonce-123"
MANIFEST = "1" * 64
UNIT_ID = "submission-phala-1"
FAKE_QUOTE = "ab" * 600  # plausible non-empty hex TDX quote
FAKE_EVENT_LOG = [{"imr": 3, "event": "compose-hash", "digest": "c" * 64}]
FAKE_VM_CONFIG = {"vcpu": 1, "memory_mb": 2048}


class FakeQuoteResponse:
    """Mimics dstack ``GetQuoteResponse`` (event_log/vm_config are JSON strings)."""

    def __init__(self, quote, *, event_log=FAKE_EVENT_LOG, vm_config=FAKE_VM_CONFIG):
        self.quote = quote
        self.event_log = json.dumps(event_log) if not isinstance(event_log, str) else event_log
        self.vm_config = json.dumps(vm_config) if not isinstance(vm_config, str) else vm_config
        self.report_data = ""


class FakeQuoteProvider:
    """Records the report_data handed to get_quote and returns a canned response."""

    def __init__(self, response=None, *, raises=None):
        self._response = response if response is not None else FakeQuoteResponse(FAKE_QUOTE)
        self._raises = raises
        self.calls: list[bytes] = []

    def get_quote(self, report_data: bytes):
        self.calls.append(report_data)
        if self._raises is not None:
            raise self._raises
        return self._response


def _benchmark_result(status="completed", score=0.5, resolved=1, total=2):
    return {
        "status": status,
        "score": score,
        "resolved": resolved,
        "total": total,
        "reason_code": None,
    }


def _emit_kwargs(**overrides):
    base = dict(
        benchmark_result=_benchmark_result(),
        canonical_measurement=dict(MEASUREMENT),
        rtmr3=RTMR3,
        agent_hash=AGENT_HASH,
        task_ids=list(TASK_IDS),
        scores=dict(SCORES),
        validator_nonce=NONCE,
        quote_provider=FakeQuoteProvider(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        vm_config=dict(FAKE_VM_CONFIG),
    )
    base.update(overrides)
    return base


def _parse_result_line(line: str) -> dict:
    assert line.startswith(RESULT_LINE_PREFIX)
    return json.loads(line[len(RESULT_LINE_PREFIX) :])


# --------------------------------------------------------------------------- #
# Envelope construction + conformance (VAL-IMG-025)
# --------------------------------------------------------------------------- #
def test_phala_tier_constant_matches_base() -> None:
    assert ar.PHALA_TDX_TIER == "phala-tdx"


def test_build_attestation_has_all_named_fields() -> None:
    att = ar.build_phala_attestation(
        tdx_quote=FAKE_QUOTE,
        event_log=FAKE_EVENT_LOG,
        report_data_hex="ab" * 64,
        measurement=ar.build_measurement(MEASUREMENT, rtmr3=RTMR3),
        vm_config=FAKE_VM_CONFIG,
    )
    assert set(att) >= {"tdx_quote", "event_log", "report_data", "measurement", "vm_config"}
    assert set(att["measurement"]) == {
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "rtmr3",
        "compose_hash",
        "os_image_hash",
    }
    # conformance validator accepts it
    ar.validate_phala_attestation(att)


def test_execution_proof_envelope_conforms() -> None:
    line = ar.emit_attested_benchmark_result(**_emit_kwargs())
    payload = _parse_result_line(line)
    envelope = payload[ar.EXECUTION_PROOF_RESULT_KEY]
    # validates against the self-contained ExecutionProof-Phala-tier conformance check
    ar.validate_execution_proof_envelope(envelope)
    assert envelope["tier"] == ar.PHALA_TDX_TIER
    assert envelope["version"] == ar.EXECUTION_PROOF_VERSION
    assert isinstance(envelope["manifest_sha256"], str)
    assert set(envelope["worker_signature"]) == {"worker_pubkey", "sig"}
    att = envelope["attestation"]
    assert att["tdx_quote"] == FAKE_QUOTE
    assert att["measurement"]["mrtd"] == MEASUREMENT["mrtd"]
    assert att["measurement"]["rtmr3"] == RTMR3
    assert att["vm_config"] == FAKE_VM_CONFIG


def test_report_data_hex_is_64_byte_field_bound_to_run() -> None:
    line = ar.emit_attested_benchmark_result(**_emit_kwargs())
    envelope = _parse_result_line(line)[ar.EXECUTION_PROOF_RESULT_KEY]
    report_data_hex = envelope["attestation"]["report_data"]
    expected = rd.report_data_hex(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=rd.scores_digest(SCORES),
        validator_nonce=NONCE,
    )
    assert report_data_hex == expected
    assert len(report_data_hex) == 128


def test_value_handed_to_get_quote_is_the_32_byte_digest() -> None:
    provider = FakeQuoteProvider()
    ar.emit_attested_benchmark_result(**_emit_kwargs(quote_provider=provider))
    assert len(provider.calls) == 1
    handed = provider.calls[0]
    assert isinstance(handed, bytes)
    assert len(handed) <= 64  # VAL-IMG-016: never >64 bytes to get_quote
    expected_digest = rd.report_data(
        canonical_measurement=MEASUREMENT,
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=rd.scores_digest(SCORES),
        validator_nonce=NONCE,
    )
    assert handed == expected_digest
    assert len(handed) == 32


# --------------------------------------------------------------------------- #
# Missing required field fails conformance (VAL-IMG-026)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field", ["version", "tier", "manifest_sha256", "worker_signature", "attestation"]
)
def test_execution_proof_missing_field_rejected(field: str) -> None:
    line = ar.emit_attested_benchmark_result(**_emit_kwargs())
    envelope = _parse_result_line(line)[ar.EXECUTION_PROOF_RESULT_KEY]
    del envelope[field]
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


@pytest.mark.parametrize(
    "field", ["tdx_quote", "event_log", "report_data", "measurement", "vm_config"]
)
def test_attestation_missing_field_rejected(field: str) -> None:
    att = ar.build_phala_attestation(
        tdx_quote=FAKE_QUOTE,
        event_log=FAKE_EVENT_LOG,
        report_data_hex="ab" * 64,
        measurement=ar.build_measurement(MEASUREMENT, rtmr3=RTMR3),
        vm_config=FAKE_VM_CONFIG,
    )
    del att[field]
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


@pytest.mark.parametrize(
    "field", ["mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3", "compose_hash", "os_image_hash"]
)
def test_attestation_measurement_missing_register_rejected(field: str) -> None:
    att = ar.build_phala_attestation(
        tdx_quote=FAKE_QUOTE,
        event_log=FAKE_EVENT_LOG,
        report_data_hex="ab" * 64,
        measurement=ar.build_measurement(MEASUREMENT, rtmr3=RTMR3),
        vm_config=FAKE_VM_CONFIG,
    )
    del att["measurement"][field]
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_wrong_typed_field_rejected() -> None:
    line = ar.emit_attested_benchmark_result(**_emit_kwargs())
    envelope = _parse_result_line(line)[ar.EXECUTION_PROOF_RESULT_KEY]
    envelope["worker_signature"] = "not-an-object"
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


# --------------------------------------------------------------------------- #
# BASE_BENCHMARK_RESULT line stays parseable and carries the envelope (VAL-IMG-027)
# --------------------------------------------------------------------------- #
def test_extended_line_preserves_five_core_fields_and_parses() -> None:
    line = ar.emit_attested_benchmark_result(**_emit_kwargs())
    payload = _parse_result_line(line)
    # Legacy five-field contract intact + still schema-valid (additive-only).
    for key in ("status", "score", "resolved", "total", "reason_code"):
        assert key in payload
    assert payload["status"] == "completed"
    assert payload["score"] == 0.5
    validate_benchmark_result(payload)
    # Attestation envelope attached and independently parseable from the SAME line.
    assert ar.EXECUTION_PROOF_RESULT_KEY in payload
    assert payload[ar.EXECUTION_PROOF_RESULT_KEY]["tier"] == ar.PHALA_TDX_TIER


def test_exactly_one_result_line_emitted(capsys) -> None:
    ar.emit_attested_benchmark_result(**_emit_kwargs())
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)]
    assert len(lines) == 1


# --------------------------------------------------------------------------- #
# agent_hash binding (VAL-IMG-028)
# --------------------------------------------------------------------------- #
def test_envelope_reports_agent_hash_matching_submission() -> None:
    submitted_agent = b"def agent(): return 42\n"
    agent_hash = hashlib.sha256(submitted_agent).hexdigest()
    line = ar.emit_attested_benchmark_result(**_emit_kwargs(agent_hash=agent_hash))
    payload = _parse_result_line(line)
    binding = payload[ar.ATTESTATION_BINDING_RESULT_KEY]
    assert binding["agent_hash"] == agent_hash
    # and report_data is derived with that exact agent_hash (recompute & compare)
    report_data_hex = payload[ar.EXECUTION_PROOF_RESULT_KEY]["attestation"]["report_data"]
    expected = rd.report_data_hex(
        canonical_measurement=MEASUREMENT,
        agent_hash=agent_hash,
        task_ids=TASK_IDS,
        scores_digest=rd.scores_digest(SCORES),
        validator_nonce=NONCE,
    )
    assert report_data_hex == expected


def test_changing_agent_hash_changes_report_data() -> None:
    line_a = ar.emit_attested_benchmark_result(**_emit_kwargs(agent_hash="1" * 64))
    line_b = ar.emit_attested_benchmark_result(**_emit_kwargs(agent_hash="2" * 64))
    rd_a = _parse_result_line(line_a)[ar.EXECUTION_PROOF_RESULT_KEY]["attestation"]["report_data"]
    rd_b = _parse_result_line(line_b)[ar.EXECUTION_PROOF_RESULT_KEY]["attestation"]["report_data"]
    assert rd_a != rd_b


def test_binding_block_is_self_consistent_with_report_data() -> None:
    line = ar.emit_attested_benchmark_result(**_emit_kwargs())
    payload = _parse_result_line(line)
    binding = payload[ar.ATTESTATION_BINDING_RESULT_KEY]
    # A verifier can recompute report_data purely from the reported binding block.
    recomputed = rd.report_data_hex(
        canonical_measurement=binding["canonical_measurement"],
        agent_hash=binding["agent_hash"],
        task_ids=binding["task_ids"],
        scores_digest=rd.scores_digest(binding["scores"]),
        validator_nonce=binding["validator_nonce"],
    )
    assert recomputed == payload[ar.EXECUTION_PROOF_RESULT_KEY]["attestation"]["report_data"]
    assert binding["scores_digest"] == rd.scores_digest(SCORES)


# --------------------------------------------------------------------------- #
# Fail-closed on get_quote failure (VAL-IMG-034)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "provider",
    [
        FakeQuoteProvider(raises=RuntimeError("dstack socket unavailable")),
        FakeQuoteProvider(raises=TimeoutError("get_quote timed out")),
        FakeQuoteProvider(response=FakeQuoteResponse("")),  # empty quote
        FakeQuoteProvider(response=FakeQuoteResponse("   ")),  # whitespace quote
        FakeQuoteProvider(response=FakeQuoteResponse(None)),  # malformed quote
    ],
)
def test_get_quote_failure_fails_closed(provider) -> None:
    line, attested = ar.emit_attested_or_failclosed(**_emit_kwargs(quote_provider=provider))
    assert attested is False
    payload = _parse_result_line(line)
    # No fabricated / attested-looking output.
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload
    assert ar.ATTESTATION_BINDING_RESULT_KEY not in payload
    assert "attestation" not in json.dumps(payload).lower() or payload.get("status") == "failed"
    # Not a silent passing result.
    assert payload["status"] == "failed"
    assert payload["score"] == 0.0
    assert payload["reason_code"] == ar.PHALA_ATTESTATION_FAILED_REASON
    # And the line still validates as a well-formed benchmark result.
    validate_benchmark_result(payload)


def test_emit_attested_raises_on_quote_failure() -> None:
    provider = FakeQuoteProvider(raises=RuntimeError("boom"))
    with pytest.raises(ar.AttestationEmissionError):
        ar.emit_attested_benchmark_result(**_emit_kwargs(quote_provider=provider))


def test_failclosed_never_emits_a_fabricated_quote(capsys) -> None:
    provider = FakeQuoteProvider(response=FakeQuoteResponse(""))
    ar.emit_attested_or_failclosed(**_emit_kwargs(quote_provider=provider))
    out = capsys.readouterr().out
    assert FAKE_QUOTE not in out
    assert "tdx_quote" not in out


def test_happy_path_reports_attested_true() -> None:
    line, attested = ar.emit_attested_or_failclosed(**_emit_kwargs())
    assert attested is True
    assert ar.EXECUTION_PROOF_RESULT_KEY in _parse_result_line(line)


# --------------------------------------------------------------------------- #
# Oversized preimage guard (VAL-IMG-016 continuity)
# --------------------------------------------------------------------------- #
def test_obtain_quote_rejects_oversized_report_data() -> None:
    provider = FakeQuoteProvider()
    with pytest.raises(ar.AttestationEmissionError):
        ar.obtain_quote(provider, b"x" * 65)


# --------------------------------------------------------------------------- #
# Validator robustness (type / shape rejections) — hardens VAL-IMG-026
# --------------------------------------------------------------------------- #
def _valid_attestation() -> dict:
    return ar.build_phala_attestation(
        tdx_quote=FAKE_QUOTE,
        event_log=FAKE_EVENT_LOG,
        report_data_hex="ab" * 64,
        measurement=ar.build_measurement(MEASUREMENT, rtmr3=RTMR3),
        vm_config=FAKE_VM_CONFIG,
    )


def test_validate_attestation_rejects_non_mapping() -> None:
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(["not", "a", "mapping"])


def test_validate_attestation_rejects_empty_quote() -> None:
    att = _valid_attestation()
    att["tdx_quote"] = ""
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_attestation_rejects_empty_report_data() -> None:
    att = _valid_attestation()
    att["report_data"] = ""
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_attestation_rejects_non_list_event_log() -> None:
    att = _valid_attestation()
    att["event_log"] = {"not": "a list"}
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_attestation_rejects_event_log_non_object_entries() -> None:
    att = _valid_attestation()
    att["event_log"] = ["not-an-object"]
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_attestation_rejects_non_object_vm_config() -> None:
    att = _valid_attestation()
    att["vm_config"] = "not-an-object"
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_attestation_rejects_non_object_measurement() -> None:
    att = _valid_attestation()
    att["measurement"] = "not-an-object"
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_attestation_rejects_empty_register() -> None:
    att = _valid_attestation()
    att["measurement"]["rtmr0"] = ""
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_phala_attestation(att)


def test_validate_envelope_rejects_non_mapping() -> None:
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope("nope")


def test_validate_envelope_rejects_non_int_version() -> None:
    envelope = ar.build_execution_proof_envelope(
        manifest_sha256=MANIFEST, attestation=_valid_attestation()
    )
    envelope["version"] = "1"
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


def test_validate_envelope_rejects_bool_tier() -> None:
    envelope = ar.build_execution_proof_envelope(
        manifest_sha256=MANIFEST, attestation=_valid_attestation()
    )
    envelope["tier"] = True
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


def test_validate_envelope_rejects_empty_manifest() -> None:
    envelope = ar.build_execution_proof_envelope(
        manifest_sha256=MANIFEST, attestation=_valid_attestation()
    )
    envelope["manifest_sha256"] = ""
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


def test_validate_envelope_rejects_signature_missing_field() -> None:
    envelope = ar.build_execution_proof_envelope(
        manifest_sha256=MANIFEST, attestation=_valid_attestation()
    )
    envelope["worker_signature"] = {"worker_pubkey": "pk"}
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


def test_validate_envelope_rejects_signature_non_str_field() -> None:
    envelope = ar.build_execution_proof_envelope(
        manifest_sha256=MANIFEST, attestation=_valid_attestation()
    )
    envelope["worker_signature"] = {"worker_pubkey": "pk", "sig": 123}
    with pytest.raises(ar.EnvelopeSchemaError):
        ar.validate_execution_proof_envelope(envelope)


def test_envelope_accepts_real_worker_signature_and_optional_fields() -> None:
    envelope = ar.build_execution_proof_envelope(
        manifest_sha256=MANIFEST,
        attestation=_valid_attestation(),
        worker_signature={"worker_pubkey": "0xpub", "sig": "0xsig"},
        image_digest="sha256:" + "a" * 64,
        provider={"name": "phala"},
    )
    ar.validate_execution_proof_envelope(envelope)
    assert envelope["worker_signature"] == {"worker_pubkey": "0xpub", "sig": "0xsig"}
    assert envelope["image_digest"].startswith("sha256:")
    assert envelope["provider"]["name"] == "phala"


def test_placeholder_worker_signature_is_schema_valid() -> None:
    sig = ar.placeholder_worker_signature()
    assert sig == {"worker_pubkey": "", "sig": ""}


# --------------------------------------------------------------------------- #
# Quote coercion + provider
# --------------------------------------------------------------------------- #
def test_obtain_quote_parses_json_string_event_log_and_vm_config() -> None:
    resp = FakeQuoteResponse(FAKE_QUOTE)
    resp.event_log = json.dumps([{"imr": 3, "digest": "c" * 64}])
    # Partial {vcpu} alone cannot project onto schema-v2 (needs memory); full
    # schema-shaped surface projects to the exact three keys.
    resp.vm_config = json.dumps({"vcpu": 2, "memory_mb": 2048, "os_image_hash": "e" * 64})
    provider = FakeQuoteProvider(response=resp)
    quote = ar.obtain_quote(provider, b"x" * 32)
    assert quote.event_log == [{"imr": 3, "digest": "c" * 64}]
    assert quote.vm_config == {
        "vcpu": 2,
        "memory_mb": 2048,
        "os_image_hash": "e" * 64,
    }


def test_obtain_quote_partial_vm_config_fails_closed() -> None:
    """Non-empty vm_config missing required fields fails at obtain_quote."""

    resp = FakeQuoteResponse(FAKE_QUOTE)
    resp.vm_config = json.dumps({"vcpu": 2})
    with pytest.raises(ar.AttestationEmissionError, match="memory"):
        ar.obtain_quote(FakeQuoteProvider(response=resp), b"x" * 32)


def test_obtain_quote_rejects_non_bytes_report_data() -> None:
    with pytest.raises(ar.AttestationEmissionError):
        ar.obtain_quote(FakeQuoteProvider(), "not-bytes")  # type: ignore[arg-type]


def test_obtain_quote_rejects_malformed_event_log_json() -> None:
    resp = FakeQuoteResponse(FAKE_QUOTE)
    resp.event_log = "{not-json"
    with pytest.raises(ar.AttestationEmissionError):
        ar.obtain_quote(FakeQuoteProvider(response=resp), b"x" * 32)


def test_obtain_quote_rejects_malformed_vm_config_json() -> None:
    resp = FakeQuoteResponse(FAKE_QUOTE)
    resp.vm_config = "{not-json"
    with pytest.raises(ar.AttestationEmissionError):
        ar.obtain_quote(FakeQuoteProvider(response=resp), b"x" * 32)


def test_obtain_quote_empty_event_log_and_vm_config_default() -> None:
    resp = FakeQuoteResponse(FAKE_QUOTE)
    resp.event_log = ""
    resp.vm_config = ""
    quote = ar.obtain_quote(FakeQuoteProvider(response=resp), b"x" * 32)
    assert quote.event_log == []
    assert quote.vm_config == {}


def test_dstack_quote_provider_uses_dstack_client(monkeypatch) -> None:
    import dstack_sdk

    created: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            created["args"] = args
            created["kwargs"] = kwargs

        def get_quote(self, report_data):
            created["report_data"] = report_data
            return FakeQuoteResponse(FAKE_QUOTE)

    monkeypatch.setattr(dstack_sdk, "DstackClient", FakeClient)

    with_endpoint = ar.DstackQuoteProvider("http://localhost:8090")
    resp = with_endpoint.get_quote(b"z" * 32)
    assert created["args"] == ("http://localhost:8090",)
    assert created["kwargs"].get("timeout") == ar.DSTACK_QUOTE_TIMEOUT_SECONDS
    assert resp.quote == FAKE_QUOTE

    created.clear()
    default = ar.DstackQuoteProvider()
    default.get_quote(b"z" * 32)
    assert created["args"] == ()
    assert created["kwargs"].get("timeout") == ar.DSTACK_QUOTE_TIMEOUT_SECONDS


def test_dstack_quote_provider_wallclocks_get_quote(monkeypatch) -> None:
    """Stuck get_quote must fail closed without re-joining an indefinite hang."""

    import time

    import dstack_sdk

    class HangClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_quote(self, report_data):  # noqa: ARG002
            # Indefinite hang discriminates ThreadPoolExecutor re-join hangs
            # from a daemon-thread wallclock that returns at the deadline.
            while True:
                time.sleep(1.0)

    monkeypatch.setattr(dstack_sdk, "DstackClient", HangClient)
    provider = ar.DstackQuoteProvider(timeout_seconds=0.05)
    t0 = time.monotonic()
    with pytest.raises(ar.AttestationEmissionError, match="wallclock|timed out|exceeded"):
        provider.get_quote(b"z" * 32)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"get_quote wallclock re-joined hung RPC: elapsed={elapsed:.3f}s"


# --------------------------------------------------------------------------- #
# build helpers
# --------------------------------------------------------------------------- #
def test_build_measurement_accepts_canonical_dataclass() -> None:
    from agent_challenge.canonical.measurement import CanonicalMeasurement

    cm = CanonicalMeasurement(**MEASUREMENT)
    measurement = ar.build_measurement(cm, rtmr3=RTMR3)
    assert measurement["mrtd"] == MEASUREMENT["mrtd"]
    assert measurement["rtmr3"] == RTMR3


def test_build_measurement_rejects_bad_type() -> None:
    with pytest.raises(TypeError):
        ar.build_measurement(12345, rtmr3=RTMR3)  # type: ignore[arg-type]


def test_build_attestation_binding_accepts_dataclass() -> None:
    from agent_challenge.canonical.measurement import CanonicalMeasurement

    cm = CanonicalMeasurement(**MEASUREMENT)
    binding = ar.build_attestation_binding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores=SCORES,
        scores_digest=rd.scores_digest(SCORES),
        validator_nonce=NONCE,
        canonical_measurement=cm,
    )
    assert binding["canonical_measurement"]["mrtd"] == MEASUREMENT["mrtd"]
    assert binding["task_ids"] == sorted(TASK_IDS)


def test_emit_failclosed_result_direct(capsys) -> None:
    line = ar.emit_failclosed_result(total=5)
    payload = _parse_result_line(line)
    assert payload["status"] == "failed"
    assert payload["score"] == 0.0
    assert payload["total"] == 5
    assert payload["reason_code"] == ar.PHALA_ATTESTATION_FAILED_REASON
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload


def test_failclosed_total_falls_back_to_task_count() -> None:
    provider = FakeQuoteProvider(raises=RuntimeError("boom"))
    # benchmark_result without a usable total -> total derived from task_ids.
    line, attested = ar.emit_attested_or_failclosed(
        **_emit_kwargs(
            quote_provider=provider,
            benchmark_result={
                "status": "completed",
                "score": 0.5,
                "resolved": 1,
                "total": True,  # bool is rejected as a real total
                "reason_code": None,
            },
        )
    )
    assert attested is False
    assert _parse_result_line(line)["total"] == len(TASK_IDS)
