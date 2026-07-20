"""Post-grant stage markers + score-path GetQuote KR normalize (offline).

Live residual (image@sha256:6e163501 + host KR 32ed505b): key_release_state=granted,
guest frame_send, then BASE_BENCHMARK_RESULT reason_code=phala_attestation_failed with
NO trial/post-grant stage markers. Ranked cause: score-path obtain_quote/emit does not
reuse KR GetQuote normalizers (quote_hex lower/0x strip, event_log coerce + empty IMR3
runtime_event_digest fill), so RTMR3 self-check / validate_eval_phala_attestation fails;
also no decrypt_ok/job_*/emit_* markers.

Discriminators (would fail a wrong implementation):
  * score obtain_quote must inherit KR normalize so empty-IMR3 dstack GetQuote passes
    RTMR3 self-check used by schema-v2 emit;
  * happy-path fakes must surface guest_eval stages decrypt_ok → job_start →
    job_done → emit_start → score_quote_ok in order;
  * emit failures surface guest_eval_fail stage=emit class/detail without secrets.
No Phala create; targeted offline only.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.own_runner.orchestrator import JobResult, TrialOutcome
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
)
from agent_challenge.evaluation.own_runner_backend import (
    PHALA_ATTESTATION_ENABLED_ENV,
    PHALA_EVAL_PLAN_ENV,
    PHALA_RTMR3_ENV,
    main,
)
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    build_rtmr3_event_log,
    build_tdx_quote,
    parse_tdx_quote_v4,
    replay_rtmr3,
    runtime_event_digest,
)

ATTESTED_REVIEW_ENABLED_ENV = "CHALLENGE_ATTESTED_REVIEW_ENABLED"

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}


def _identity_events_empty_digests() -> tuple[list[dict[str, Any]], str]:
    """dstack-shaped IMR3 log with empty digests (live GetQuote residual)."""

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
    empty: list[dict[str, Any]] = []
    for entry in filled:
        raw = dict(entry)
        raw["digest"] = ""  # live residual: IMR3 digests blank until fill
        raw["event_payload"] = "0x" + str(raw["event_payload"]).upper()
        raw["extra_dstack_field"] = "drop-me"
        empty.append(raw)
    return empty, rtmr3


class _DstackShapedScoreQuoteResponse:
    """Mimics live dstack GetQuote: 0x mixed-case quote + empty IMR3 digests."""

    def __init__(self, quote_hex: str, event_log: list[dict[str, Any]]) -> None:
        if quote_hex.startswith(("0x", "0X")):
            self.quote = quote_hex
        else:
            self.quote = "0x" + quote_hex.upper()
        self.event_log = json.dumps(event_log)
        self.vm_config = json.dumps(
            {"vcpu": 1, "memory_mb": 2048, "os_image_hash": MEASUREMENT["os_image_hash"]}
        )
        self.report_data = ""


class _DstackShapedScoreProvider:
    def __init__(self, response: _DstackShapedScoreQuoteResponse) -> None:
        self._response = response
        self.calls: list[bytes] = []

    def get_quote(self, report_data: bytes) -> _DstackShapedScoreQuoteResponse:
        self.calls.append(report_data)
        return self._response


def test_score_path_obtain_quote_normalizes_empty_imr3_and_passes_rtmr3_self_check() -> None:
    """Score-path GetQuote reuses KR normalize so empty IMR3 digests self-check."""

    empty_log, expected_rtmr3 = _identity_events_empty_digests()
    # Discriminator vs shallow coerce: empty digests fail closed register schema
    # (non-empty 96-char hex required by validate_eval_phala_attestation).
    for entry in empty_log:
        assert entry["digest"] == ""
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(
            {
                "tdx_quote": "ab" * 600,
                "event_log": [
                    {
                        "imr": e["imr"],
                        "event_type": e["event_type"],
                        "digest": e["digest"],
                        "event": e["event"],
                        "event_payload": str(e["event_payload"]).removeprefix("0x").lower()
                        if isinstance(e["event_payload"], str)
                        else e["event_payload"],
                    }
                    for e in empty_log
                ],
                "report_data": "00" * 64,
                "measurement": {
                    **MEASUREMENT,
                    "rtmr3": expected_rtmr3,
                },
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                },
            }
        )

    quote_hex = build_tdx_quote(
        mrtd=MEASUREMENT["mrtd"],
        rtmr0=MEASUREMENT["rtmr0"],
        rtmr1=MEASUREMENT["rtmr1"],
        rtmr2=MEASUREMENT["rtmr2"],
        rtmr3=expected_rtmr3,
        report_data=b"z" * 32,
    )
    provider = _DstackShapedScoreProvider(_DstackShapedScoreQuoteResponse(quote_hex, empty_log))
    result = ar.obtain_quote(provider, b"z" * 32)

    # quote_hex lower / 0x stripped
    assert result.quote == quote_hex.lower().removeprefix("0x")
    assert not result.quote.startswith(("0x", "0X"))
    assert result.quote == result.quote.lower()

    # Closed 5-key projection + filled IMR3 digests match runtime_event_digest.
    closed = {"imr", "event_type", "digest", "event", "event_payload"}
    assert all(set(entry) == closed for entry in result.event_log)
    for entry in result.event_log:
        expected = runtime_event_digest(
            str(entry["event"]),
            bytes.fromhex(str(entry["event_payload"])),
        ).hex()
        assert entry["digest"] == expected
        assert entry["event_payload"] == str(entry["event_payload"]).lower()
        assert not str(entry["event_payload"]).startswith("0x")

    # RTMR3 self-check used by schema-v2 emit must pass after normalize.
    parsed_rtmr3 = parse_tdx_quote_v4(result.quote).rtmr3
    replayed = replay_rtmr3(result.event_log).rtmr3
    assert parsed_rtmr3 == replayed == expected_rtmr3


def test_score_path_schema_v2_emit_with_empty_imr3_event_log(monkeypatch) -> None:
    """schema-v2 emit obtains a quote, self-checks RTMR3, validates wire after normalize."""

    empty_log, expected_rtmr3 = _identity_events_empty_digests()
    report_data_holder: dict[str, bytes] = {}

    class _Provider:
        def get_quote(self, report_data: bytes) -> _DstackShapedScoreQuoteResponse:
            report_data_holder["rd"] = report_data
            # Pin quote report_data to the call so parse path is consistent.
            quote_hex = build_tdx_quote(
                mrtd=MEASUREMENT["mrtd"],
                rtmr0=MEASUREMENT["rtmr0"],
                rtmr1=MEASUREMENT["rtmr1"],
                rtmr2=MEASUREMENT["rtmr2"],
                rtmr3=expected_rtmr3,
                report_data=report_data,
            )
            return _DstackShapedScoreQuoteResponse(quote_hex, empty_log)

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    task_ids = ["task-a", "task-b"]
    record = ew.build_canonical_score_record(
        eval_run_id="eval-score-normalize",
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
        rtmr3="0" * 96,  # supplier is overridden by quote parse after self-check
        agent_hash="f" * 64,
        task_ids=task_ids,
        scores={},
        quote_provider=_Provider(),
        manifest_sha256="1" * 64,
        eval_run_id="eval-score-normalize",
        submission_id="submission-score-normalize",
        score_nonce="score-nonce-normalize",
        score_record=record,
        image_digest="registry.example/eval@sha256:" + "d" * 64,
        vm_config={
            "vcpu": 1,
            "memory_mb": 2048,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
    )
    payload = json.loads(line.split("=", 1)[1])
    assert ew.validate_eval_result_request(payload) == payload
    attestation = payload["execution_proof"]["attestation"]
    assert attestation["measurement"]["rtmr3"] == expected_rtmr3
    # Digests were filled and project closed keys.
    for entry in attestation["event_log"]:
        assert set(entry) == {"imr", "event_type", "digest", "event", "event_payload"}
        assert entry["digest"] and len(entry["digest"]) == 96


def _canned_result(*, trials: int = 1) -> JobResult:
    outcomes = [
        TrialOutcome(
            task_name="hello-world",
            trial_name=f"hello-world__attempt-{i}",
            status="completed",
            rewards={"reward": 1.0},
        )
        for i in range(trials)
    ]
    return JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=trials,
        n_completed_trials=trials,
        n_errored_trials=0,
        trial_outcomes=outcomes,
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )


def _fake_provider_factory(quote: str = "ab" * 600, *, raises: Exception | None = None):
    class _FakeProvider:
        def __init__(self, endpoint: str | None = None) -> None:
            self.endpoint = endpoint

        def get_quote(self, report_data: bytes) -> Any:
            if raises is not None:
                raise raises

            class _Resp:
                def __init__(self) -> None:
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
                        {
                            "vcpu": 1,
                            "memory_mb": 2048,
                            "os_image_hash": MEASUREMENT["os_image_hash"],
                        }
                    )
                    self.report_data = ""

            return _Resp()

    return _FakeProvider


def _set_happy_phala_env(monkeypatch, *, trials: int = 1) -> None:
    monkeypatch.setenv(PHALA_ATTESTATION_ENABLED_ENV, "1")
    monkeypatch.setenv(ATTESTED_REVIEW_ENABLED_ENV, "1")
    monkeypatch.setenv(PHALA_RTMR3_ENV, "d" * 96)
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval-run-markers",
        "submission_id": "submission-markers",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": "f" * 64,
        "selected_tasks": [
            {
                "task_id": "hello-world",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
            }
        ],
        "k": trials,
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
        "result_endpoint": "/evaluation/v1/runs/eval-run-markers/result",
        "key_release_nonce": "key-nonce-markers",
        "score_nonce": "score-nonce-markers",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": (time.time_ns() // 1_000_000) - 1_000,
        "expires_at_ms": (time.time_ns() // 1_000_000) + 60_000,
    }
    monkeypatch.setenv(PHALA_EVAL_PLAN_ENV, json.dumps(plan))
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.assert_agent_artifact_matches_plan",
        lambda **_: "f" * 64,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._preflight_eval_plan_tasks",
        lambda **_: {},
    )


def test_post_grant_stage_markers_order_on_happy_path(monkeypatch, tmp_path, capsys) -> None:
    """decrypt_ok → job_start → job_done → emit_start → score_quote_ok on fakes."""

    trials = 2
    _set_happy_phala_env(monkeypatch, trials=trials)
    sentinel_key = b"super-secret-golden-key-SENTINEL-markers-xyz"

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._acquire_golden_key_if_required",
        lambda **_: sentinel_key,
    )

    def _decrypt_ok(key: bytes) -> dict[str, Any]:
        if key != sentinel_key:
            raise RuntimeError("bad key")
        return {"tasks": {}}

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._decrypt_golden_in_enclave",
        _decrypt_ok,
    )

    async def _fake_run(**kwargs: Any) -> JobResult:
        return _canned_result(trials=trials)

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.run_own_runner_job",
        _fake_run,
    )
    monkeypatch.setattr(ar, "DstackQuoteProvider", _fake_provider_factory())

    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    out = capsys.readouterr().out
    assert rc == 0

    stages = []
    for line in out.splitlines():
        if line.startswith("guest_eval stage="):
            # "guest_eval stage=NAME ..."
            token = line.split()[1]
            stages.append(token.split("=", 1)[1])

    required = ["decrypt_ok", "job_start", "job_done", "emit_start", "score_quote_ok"]
    positions = [stages.index(name) for name in required]
    assert positions == sorted(positions), f"stage order wrong: {stages}"

    job_done_line = next(ln for ln in out.splitlines() if "guest_eval stage=job_done" in ln)
    assert f"trials={trials}" in job_done_line
    # Secret-free: sentinel key must never appear.
    assert "SENTINEL" not in out
    assert sentinel_key.decode() not in out

    result_lines = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)]
    assert len(result_lines) == 1
    payload = json.loads(result_lines[0][len(RESULT_LINE_PREFIX) :])
    assert "execution_proof" in payload


def test_emit_failure_surfaces_guest_eval_fail_stage_emit(monkeypatch, tmp_path, capsys) -> None:
    """Emit failures print guest_eval_fail stage=emit class/detail without secrets."""

    _set_happy_phala_env(monkeypatch, trials=1)
    sentinel = "leak-me-SENTINEL-api-key-should-not-appear"

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._acquire_golden_key_if_required",
        lambda **_: b"granted-key",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._decrypt_golden_in_enclave",
        lambda _key: {"tasks": {}},
    )

    async def _fake_run(**kwargs: Any) -> JobResult:
        return _canned_result(trials=1)

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.run_own_runner_job",
        _fake_run,
    )
    monkeypatch.setattr(
        ar,
        "DstackQuoteProvider",
        _fake_provider_factory(raises=RuntimeError(f"dstack boom {sentinel}")),
    )

    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    out = capsys.readouterr().out
    assert rc != 0

    assert "guest_eval stage=decrypt_ok" in out
    assert "guest_eval stage=job_start" in out
    assert "guest_eval stage=job_done" in out
    assert "guest_eval stage=emit_start" in out
    fail_line = next(ln for ln in out.splitlines() if ln.startswith("guest_eval_fail "))
    assert "stage=emit" in fail_line
    assert "class=" in fail_line
    assert "detail=" in fail_line
    # Secret-free: raw api-key-ish detail is redacted or class-only; sentinel not dumped.
    assert sentinel not in out
    assert "SENTINEL" not in out

    result_lines = [
        ln[len(RESULT_LINE_PREFIX) :]
        for ln in out.splitlines()
        if ln.startswith(RESULT_LINE_PREFIX)
    ]
    assert len(result_lines) == 1
    payload = json.loads(result_lines[0])
    assert payload["status"] == "failed"
    assert payload["reason_code"] == ar.PHALA_ATTESTATION_FAILED_REASON
    assert "execution_proof" not in payload
