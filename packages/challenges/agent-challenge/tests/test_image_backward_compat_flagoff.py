"""Backward-compat + config-layer flag-off tests for the canonical image (M1).

Fulfils the ``image-backward-compat-flagoff`` feature:
  * VAL-IMG-029 flag off -> legacy own_runner path, NO attestation envelope emitted
  * VAL-IMG-030 flag off -> zero dstack socket / ``get_quote`` interactions
  * VAL-IMG-031 flag on (k=1, variance off) scoring is byte-identical to flag off
  * VAL-IMG-032 default configuration has the Phala path disabled (safe default)
  * VAL-IMG-033 canonical_measurement bound in report_data == the envelope
                measurement block == the pinnable allowlist record

The in-image emission gate is the ``CHALLENGE_PHALA_ATTESTATION_ENABLED`` env var
read by :func:`own_runner_backend._phala_attestation_enabled`; the
validator-config layer exposes the SAME switch as the default-off
:class:`ChallengeSettings` field ``phala_attestation_enabled`` (whose env var,
via the ``CHALLENGE_`` prefix, is exactly that gate). These tests pin both
layers and prove that config-off == gate-off == byte-identical legacy behavior.
The dstack quote provider is faked so the flag-on path is exercised offline (the
live self-verify assertions run against a real CVM in M6).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.canonical import report_data as rd
from agent_challenge.canonical.measurement import (
    CANONICAL_MEASUREMENT_FIELDS,
    CanonicalMeasurement,
)
from agent_challenge.evaluation.own_runner.orchestrator import JobResult, TrialOutcome
from agent_challenge.evaluation.own_runner.result_schema import (
    REQUIRED_FIELDS,
    RESULT_LINE_PREFIX,
    build_benchmark_result,
    format_benchmark_result_line,
)
from agent_challenge.evaluation.own_runner_backend import (
    PHALA_ATTESTATION_ENABLED_ENV,
    PHALA_EVAL_PLAN_ENV,
    PHALA_RTMR3_ENV,
    _phala_attestation_enabled,
    main,
)
from agent_challenge.sdk.config import ChallengeSettings

ATTESTED_REVIEW_ENABLED_ENV = "CHALLENGE_ATTESTED_REVIEW_ENABLED"

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
NONCE = "nonce-xyz"
FAKE_QUOTE = "ab" * 600
CONFIG_EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.yaml"


class _FakeQuoteResponse:
    def __init__(self, quote: str = FAKE_QUOTE) -> None:
        self.quote = quote
        self.event_log = json.dumps(
            [
                {
                    "imr": 3,
                    "event_type": 134217729,
                    "digest": "c" * 96,
                    "event": "compose-hash",
                    "event_payload": MEASUREMENT["compose_hash"],
                }
            ]
        )
        self.vm_config = json.dumps(
            {"vcpu": 1, "memory_mb": 2048, "os_image_hash": MEASUREMENT["os_image_hash"]}
        )
        self.report_data = ""


def _make_spy_provider() -> tuple[type, dict[str, int]]:
    """A monitored dstack provider that counts construction + get_quote calls."""

    events = {"instances": 0, "quote_calls": 0}

    class _Spy:
        def __init__(self, endpoint: str | None = None) -> None:
            events["instances"] += 1

        def get_quote(self, report_data: bytes) -> Any:
            events["quote_calls"] += 1
            return _FakeQuoteResponse()

    return _Spy, events


def _canned_result() -> JobResult:
    return JobResult(
        status="completed",
        score=0.5,
        resolved=1,
        total=2,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[
            TrialOutcome(
                task_name="hello-world",
                trial_name="hello-world__attempt-0",
                status="completed",
                rewards={"reward": 0.5},
            )
        ],
        benchmark_result=build_benchmark_result(
            status="completed", score=0.5, resolved=1, total=2, reason_code=None
        ),
    )


def _patch_run(monkeypatch) -> None:
    async def _fake_run(**kwargs: Any) -> JobResult:
        return _canned_result()

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.run_own_runner_job", _fake_run
    )


def _set_eval_plan_env(monkeypatch) -> None:
    monkeypatch.setenv(PHALA_ATTESTATION_ENABLED_ENV, "1")
    monkeypatch.setenv(ATTESTED_REVIEW_ENABLED_ENV, "1")
    monkeypatch.setenv(PHALA_RTMR3_ENV, RTMR3)
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "submission_id": "submission-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": AGENT_HASH,
        "selected_tasks": [
            {
                "task_id": "hello-world",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "d" * 64,
            "compose_hash": MEASUREMENT["compose_hash"],
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "3" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("3" * 64)).hexdigest(),
            "measurement": {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "os_image_hash": MEASUREMENT["os_image_hash"],
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-run-001/result",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": (time.time_ns() // 1_000_000) - 1_000,
        "expires_at_ms": (time.time_ns() // 1_000_000) + 60_000,
    }
    monkeypatch.setenv(PHALA_EVAL_PLAN_ENV, json.dumps(plan))
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._acquire_golden_key_if_required",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.assert_agent_artifact_matches_plan",
        lambda **_: AGENT_HASH,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._preflight_eval_plan_tasks",
        lambda **_: {},
    )


def _result_line(out: str) -> dict:
    lines = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)]
    assert len(lines) == 1, f"expected exactly one result line, got {len(lines)}"
    return json.loads(lines[0][len(RESULT_LINE_PREFIX) :])


# --------------------------------------------------------------------------- #
# VAL-IMG-032: default configuration has the Phala path disabled
# --------------------------------------------------------------------------- #
def test_default_config_has_phala_attestation_disabled(monkeypatch) -> None:
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    settings = ChallengeSettings()
    assert settings.phala_attestation_enabled is False


def test_config_flag_env_var_is_the_in_image_gate(monkeypatch) -> None:
    # The ChallengeSettings field maps (via the CHALLENGE_ prefix) to exactly the
    # in-image gate env var, so a config-off deployment gates the image off.
    assert PHALA_ATTESTATION_ENABLED_ENV == "CHALLENGE_PHALA_ATTESTATION_ENABLED"
    monkeypatch.setenv(PHALA_ATTESTATION_ENABLED_ENV, "1")
    monkeypatch.setenv(ATTESTED_REVIEW_ENABLED_ENV, "1")
    assert ChallengeSettings().phala_attestation_enabled is True
    monkeypatch.setenv(PHALA_ATTESTATION_ENABLED_ENV, "0")
    monkeypatch.setenv(ATTESTED_REVIEW_ENABLED_ENV, "0")
    assert ChallengeSettings().phala_attestation_enabled is False


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("false", False)],
)
def test_config_flag_and_in_image_gate_agree(monkeypatch, value: str, expected: bool) -> None:
    monkeypatch.setenv(PHALA_ATTESTATION_ENABLED_ENV, value)
    monkeypatch.setenv(ATTESTED_REVIEW_ENABLED_ENV, value)
    assert _phala_attestation_enabled() is expected
    assert ChallengeSettings().phala_attestation_enabled is expected


def test_config_example_documents_phala_disabled_default() -> None:
    text = CONFIG_EXAMPLE.read_text()
    assert "phala_attestation_enabled: false" in text


# --------------------------------------------------------------------------- #
# VAL-IMG-029: flag off -> legacy own_runner path, no attestation emitted
# --------------------------------------------------------------------------- #
def test_flag_off_emits_legacy_line_without_attestation(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    _patch_run(monkeypatch)
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    out = capsys.readouterr().out
    payload = _result_line(out)
    # Only the legacy five-field result -- no attestation blocks anywhere.
    assert set(payload) == set(REQUIRED_FIELDS)
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload
    assert ar.ATTESTATION_BINDING_RESULT_KEY not in payload
    assert "tdx_quote" not in out
    assert "attestation" not in out
    assert payload["status"] == "completed"


# --------------------------------------------------------------------------- #
# VAL-IMG-030: flag off -> zero dstack socket / get_quote interactions
# --------------------------------------------------------------------------- #
def test_flag_off_makes_zero_dstack_interactions(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    _patch_run(monkeypatch)
    spy, events = _make_spy_provider()
    monkeypatch.setattr(ar, "DstackQuoteProvider", spy)
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    assert events == {"instances": 0, "quote_calls": 0}


def test_flag_off_never_constructs_quote_provider(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    _patch_run(monkeypatch)

    class _Boom:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("dstack provider must not be constructed when flag off")

        def get_quote(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("get_quote must not be called when flag off")

    monkeypatch.setattr(ar, "DstackQuoteProvider", _Boom)
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0


def test_flag_on_does_interact_with_dstack(monkeypatch, tmp_path, capsys) -> None:
    # Positive control: the monitored provider IS a real discriminator -- when the
    # flag is on the image constructs it and calls get_quote exactly once.
    _patch_run(monkeypatch)
    _set_eval_plan_env(monkeypatch)
    spy, events = _make_spy_provider()
    monkeypatch.setattr(ar, "DstackQuoteProvider", spy)
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    assert events == {"instances": 1, "quote_calls": 1}


# --------------------------------------------------------------------------- #
# VAL-IMG-031: flag on (k=1) scoring is byte-identical to flag off
# --------------------------------------------------------------------------- #
def test_flag_on_k1_scoring_byte_identical_to_flag_off(monkeypatch, tmp_path, capsys) -> None:
    _patch_run(monkeypatch)

    # Flag OFF: legacy line.
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    rc_off = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "off")])
    off_out = capsys.readouterr().out
    off_line = next(ln for ln in off_out.splitlines() if ln.startswith(RESULT_LINE_PREFIX))
    off_payload = _result_line(off_out)

    # Flag ON (k=1, variance off): strict Eval request with the same score.
    _set_eval_plan_env(monkeypatch)
    monkeypatch.setattr(ar, "DstackQuoteProvider", _make_spy_provider()[0])
    rc_on = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "on")])
    on_payload = _result_line(capsys.readouterr().out)

    assert rc_off == 0 and rc_on == 0
    # Flag-off carries exactly the five scoring fields.
    assert set(off_payload) == set(REQUIRED_FIELDS)
    assert ew.validate_eval_result_request(on_payload) == on_payload
    on_score = ew.decode_score_f64be(on_payload["score_record"]["final"]["job_score_f64be"])
    assert on_score == off_payload["score"]
    # The off-path serialization remains untouched and reconstructs byte-exactly.
    assert format_benchmark_result_line(off_payload) == off_line


# --------------------------------------------------------------------------- #
# VAL-IMG-033: canonical_measurement is self-consistent across binding /
# envelope / pinnable allowlist record
# --------------------------------------------------------------------------- #
def _emit_attested_line() -> str:
    spy, _events = _make_spy_provider()
    return ar.emit_attested_benchmark_result(
        benchmark_result=build_benchmark_result(
            status="completed", score=0.5, resolved=1, total=2, reason_code=None
        ),
        canonical_measurement=dict(MEASUREMENT),
        rtmr3=RTMR3,
        agent_hash=AGENT_HASH,
        task_ids=["task-b", "task-a"],
        scores={"task-a": 1.0, "task-b": 0.0},
        validator_nonce=NONCE,
        quote_provider=spy(),
        manifest_sha256="1" * 64,
        vm_config={"vcpu": 1},
    )


def test_canonical_measurement_consistent_across_binding_envelope_and_record() -> None:
    line = _emit_attested_line()
    payload = json.loads(line[len(RESULT_LINE_PREFIX) :])

    record = CanonicalMeasurement(**MEASUREMENT).as_dict()
    binding_meas = payload[ar.ATTESTATION_BINDING_RESULT_KEY]["canonical_measurement"]
    envelope_meas = payload[ar.EXECUTION_PROOF_RESULT_KEY]["attestation"]["measurement"]

    # The binding's canonical_measurement is exactly the pinnable allowlist record.
    assert binding_meas == record
    # The envelope's measurement block carries the same static registers (+ runtime rtmr3).
    assert {field: envelope_meas[field] for field in CANONICAL_MEASUREMENT_FIELDS} == record
    assert envelope_meas["rtmr3"] == RTMR3

    # report_data was derived from exactly that measurement (self-consistent binding).
    report_data_hex = payload[ar.EXECUTION_PROOF_RESULT_KEY]["attestation"]["report_data"]
    expected = rd.report_data_hex(
        canonical_measurement=record,
        agent_hash=AGENT_HASH,
        task_ids=["task-b", "task-a"],
        scores_digest=rd.scores_digest({"task-a": 1.0, "task-b": 0.0}),
        validator_nonce=NONCE,
    )
    assert report_data_hex == expected


def test_report_data_changes_when_bound_canonical_measurement_changes() -> None:
    common = dict(
        agent_hash=AGENT_HASH,
        task_ids=["task-a"],
        scores_digest=rd.scores_digest({"task-a": 1.0}),
        validator_nonce=NONCE,
    )
    baseline = rd.report_data_hex(canonical_measurement=dict(MEASUREMENT), **common)
    drifted_measurement = dict(MEASUREMENT)
    drifted_measurement["compose_hash"] = "9" * 64
    drifted = rd.report_data_hex(canonical_measurement=drifted_measurement, **common)
    assert baseline != drifted
