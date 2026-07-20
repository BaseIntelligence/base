"""Score-path vm_config projection: dstack → exact schema-v2 keys (offline).

Live residual (image@sha256:a27bdadd / compose 09a44f24): after KR grant,
decrypt_ok, job_done trials=3, emit_start → AttestationEmissionError
'schema-v2 Eval emission is invalid: vm_config has invalid fields' →
phala_attestation_failed.

Root cause: schema-v2 wire (eval_wire.py:734) requires exact set
{vcpu, memory_mb, os_image_hash}. Score obtain_quote/_coerce_vm_config
forwarded raw dstack vm_config (cpu_count/memory_size/extras). Review already
projects via _normalize_vm_config (cpu_count→vcpu, memory_size bytes→memory_mb,
drop extras). Port that projection into the score emit path.

Discriminators (would fail a wrong implementation):
  * dstack-shaped {cpu_count, memory_size, extras} must pass
    validate_eval_phala_attestation after obtain_quote / schema-v2 emit project
  * extras filtered so set(vm_config) == {vcpu, memory_mb, os_image_hash}
  * memory_size (bytes) maps to memory_mb via integer // 1MiB
  * missing vcpu/cpu_count or memory_mb/memory_size fails closed
No Phala create; targeted offline only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    build_rtmr3_event_log,
    build_tdx_quote,
)

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}

# Live-shaped dstack fixture (extras + native field names).
DSTACK_VM_CONFIG_RAW: dict[str, Any] = {
    "cpu_count": 2,
    "memory_size": 4 * 1024 * 1024 * 1024,  # 4 GiB in bytes
    "os_image_hash": MEASUREMENT["os_image_hash"],
    "qemu_single_pass_add_pages": True,
    "num_gpus": 0,
    "hugepages": False,
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


class _DstackVmConfigQuoteResponse:
    """GetQuote response carrying dstack-native vm_config (JSON string)."""

    def __init__(
        self,
        quote_hex: str,
        event_log: list[dict[str, Any]],
        vm_config: dict[str, Any],
    ) -> None:
        self.quote = quote_hex
        self.event_log = json.dumps(event_log)
        self.vm_config = json.dumps(vm_config)
        self.report_data = ""


class _Provider:
    def __init__(self, response: _DstackVmConfigQuoteResponse) -> None:
        self._response = response
        self.calls: list[bytes] = []

    def get_quote(self, report_data: bytes) -> _DstackVmConfigQuoteResponse:
        self.calls.append(report_data)
        return self._response


def test_project_dstack_vm_config_maps_cpu_memory_drops_extras() -> None:
    """cpu_count→vcpu, memory_size bytes→memory_mb, extras dropped, int coerce."""

    projected = ar._project_eval_vm_config(
        DSTACK_VM_CONFIG_RAW,
        os_image_hash=MEASUREMENT["os_image_hash"],
    )
    assert set(projected) == {"vcpu", "memory_mb", "os_image_hash"}
    assert projected["vcpu"] == 2
    assert projected["memory_mb"] == 4096
    assert projected["os_image_hash"] == MEASUREMENT["os_image_hash"]
    assert isinstance(projected["vcpu"], int)
    assert isinstance(projected["memory_mb"], int)


def test_project_dstack_vm_config_accepts_schema_keys_unchanged() -> None:
    """Already-schema keys stay as-is (int-coerced) with extras dropped."""

    projected = ar._project_eval_vm_config(
        {
            "vcpu": "1",
            "memory_mb": "2048",
            "os_image_hash": MEASUREMENT["os_image_hash"],
            "stray": True,
        },
        os_image_hash=MEASUREMENT["os_image_hash"],
    )
    assert projected == {
        "vcpu": 1,
        "memory_mb": 2048,
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }


def test_project_fills_os_image_hash_from_measurement_when_raw_omits() -> None:
    """os_image_hash may come from measurement when absente from dstack surface."""

    projected = ar._project_eval_vm_config(
        {"cpu_count": 1, "memory_size": 2 * 1024 * 1024 * 1024},
        os_image_hash=MEASUREMENT["os_image_hash"],
    )
    assert projected["os_image_hash"] == MEASUREMENT["os_image_hash"]
    assert projected["vcpu"] == 1
    assert projected["memory_mb"] == 2048


def test_project_missing_vcpu_fails_closed() -> None:
    """Neither vcpu nor cpu_count → fail closed (no silent default)."""

    with pytest.raises(ar.AttestationEmissionError, match="vcpu"):
        ar._project_eval_vm_config(
            {
                "memory_size": 2 * 1024 * 1024 * 1024,
                "os_image_hash": MEASUREMENT["os_image_hash"],
            },
            os_image_hash=MEASUREMENT["os_image_hash"],
        )


def test_project_missing_memory_fails_closed() -> None:
    """Neither memory_mb nor memory_size → fail closed (no silent default)."""

    with pytest.raises(ar.AttestationEmissionError, match="memory"):
        ar._project_eval_vm_config(
            {"cpu_count": 1, "os_image_hash": MEASUREMENT["os_image_hash"]},
            os_image_hash=MEASUREMENT["os_image_hash"],
        )


def test_project_invalid_memory_size_fails_closed() -> None:
    with pytest.raises(ar.AttestationEmissionError, match="memory"):
        ar._project_eval_vm_config(
            {
                "cpu_count": 1,
                "memory_size": 0,
                "os_image_hash": MEASUREMENT["os_image_hash"],
            },
            os_image_hash=MEASUREMENT["os_image_hash"],
        )


def test_project_non_object_fails_closed() -> None:
    with pytest.raises(ar.AttestationEmissionError, match="object"):
        ar._project_eval_vm_config(
            ["not", "an", "object"],
            os_image_hash=MEASUREMENT["os_image_hash"],
        )


def test_raw_dstack_vm_config_fails_wire_without_projection() -> None:
    """Discriminator: unprojected dstack shape is rejected by schema-v2 wire."""

    events, rtmr3 = _identity_event_log()
    with pytest.raises(ew.EvalWireError, match="vm_config has invalid fields"):
        ew.validate_eval_phala_attestation(
            {
                "tdx_quote": "ab" * 600,
                "event_log": events,
                "report_data": "00" * 64,
                "measurement": {**MEASUREMENT, "rtmr3": rtmr3},
                "vm_config": dict(DSTACK_VM_CONFIG_RAW),
            }
        )


def test_obtain_quote_projects_dstack_vm_config() -> None:
    """Score-path obtain_quote projects dstack vm_config to the three schema keys."""

    events, rtmr3 = _identity_event_log()
    quote_hex = build_tdx_quote(
        mrtd=MEASUREMENT["mrtd"],
        rtmr0=MEASUREMENT["rtmr0"],
        rtmr1=MEASUREMENT["rtmr1"],
        rtmr2=MEASUREMENT["rtmr2"],
        rtmr3=rtmr3,
        report_data=b"z" * 32,
    )
    provider = _Provider(_DstackVmConfigQuoteResponse(quote_hex, events, DSTACK_VM_CONFIG_RAW))
    result = ar.obtain_quote(provider, b"z" * 32)
    assert set(result.vm_config) == {"vcpu", "memory_mb", "os_image_hash"}
    assert result.vm_config["vcpu"] == 2
    assert result.vm_config["memory_mb"] == 4096
    assert result.vm_config["os_image_hash"] == MEASUREMENT["os_image_hash"]


def test_schema_v2_emit_with_dstack_shaped_quote_vm_config(monkeypatch) -> None:
    """schema-v2 emit projects quote.vm_config so wire validate accepts emission."""

    del monkeypatch  # kept for future env-based residual harness
    events, expected_rtmr3 = _identity_event_log()

    class _DstackProvider:
        def get_quote(self, report_data: bytes) -> _DstackVmConfigQuoteResponse:
            quote_hex = build_tdx_quote(
                mrtd=MEASUREMENT["mrtd"],
                rtmr0=MEASUREMENT["rtmr0"],
                rtmr1=MEASUREMENT["rtmr1"],
                rtmr2=MEASUREMENT["rtmr2"],
                rtmr3=expected_rtmr3,
                report_data=report_data,
            )
            # Live residual: raw dstack fields + extras, no env override.
            return _DstackVmConfigQuoteResponse(quote_hex, events, dict(DSTACK_VM_CONFIG_RAW))

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    task_ids = ["task-a", "task-b"]
    record = ew.build_canonical_score_record(
        eval_run_id="eval-vm-config-project",
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
        eval_run_id="eval-vm-config-project",
        submission_id="submission-vm-config-project",
        score_nonce="score-nonce-vm-config",
        score_record=record,
        image_digest="registry.example/eval@sha256:" + "d" * 64,
        # Critical: no schema-shaped env override — force quote.vm_config path.
        vm_config=None,
    )
    payload = json.loads(line.split("=", 1)[1])
    assert ew.validate_eval_result_request(payload) == payload
    vm = payload["execution_proof"]["attestation"]["vm_config"]
    assert set(vm) == {"vcpu", "memory_mb", "os_image_hash"}
    assert vm["vcpu"] == 2
    assert vm["memory_mb"] == 4096
    assert vm["os_image_hash"] == MEASUREMENT["os_image_hash"]


def test_schema_v2_emit_fails_closed_when_vm_config_missing_cpu() -> None:
    """Emit fails closed (typed AttestationEmissionError) when projection cannot derive vcpu."""

    events, expected_rtmr3 = _identity_event_log()

    class _ProviderMissingCpu:
        def get_quote(self, report_data: bytes) -> _DstackVmConfigQuoteResponse:
            quote_hex = build_tdx_quote(
                mrtd=MEASUREMENT["mrtd"],
                rtmr0=MEASUREMENT["rtmr0"],
                rtmr1=MEASUREMENT["rtmr1"],
                rtmr2=MEASUREMENT["rtmr2"],
                rtmr3=expected_rtmr3,
                report_data=report_data,
            )
            return _DstackVmConfigQuoteResponse(
                quote_hex,
                events,
                {
                    "memory_size": 2 * 1024 * 1024 * 1024,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            )

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    record = ew.build_canonical_score_record(
        eval_run_id="eval-vm-config-missing-cpu",
        policy=policy,
        trial_scores_by_task={"task-a": [1.0]},
    )
    with pytest.raises(ar.AttestationEmissionError, match="(?i)vm_config|vcpu"):
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
            quote_provider=_ProviderMissingCpu(),
            manifest_sha256="1" * 64,
            eval_run_id="eval-vm-config-missing-cpu",
            submission_id="submission-vm-config-missing-cpu",
            score_nonce="score-nonce-missing-cpu",
            score_record=record,
            image_digest="registry.example/eval@sha256:" + "d" * 64,
            vm_config=None,
        )
