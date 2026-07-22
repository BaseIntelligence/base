"""Backend wiring tests for Phala attested-result emission (M1).

Verifies ``own_runner_backend.main`` dispatch:
  * gate OFF  -> legacy plain BASE_BENCHMARK_RESULT line, no attestation, rc 0
  * gate ON   -> attested line carrying the ExecutionProof envelope, rc 0
  * gate ON + get_quote failure     -> fail-closed failed line, rc != 0 (VAL-IMG-034)
  * gate ON + missing binding env    -> fail-closed failed line, rc != 0 (VAL-IMG-034)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation import own_runner_backend as backend
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

ATTESTED_REVIEW_ENABLED_ENV = "CHALLENGE_ATTESTED_REVIEW_ENABLED"

MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b0" * 48,
    "rtmr1": "b1" * 48,
    "rtmr2": "b2" * 48,
    "compose_hash": "c" * 64,
    "os_image_hash": "e" * 64,
}
FAKE_QUOTE = "ab" * 600


class _FakeQuoteResponse:
    def __init__(self, quote: str) -> None:
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


def _fake_provider_factory(quote: str = FAKE_QUOTE, *, raises: Exception | None = None):
    class _FakeProvider:
        def __init__(self, endpoint: str | None = None) -> None:
            self.endpoint = endpoint

        def get_quote(self, report_data: bytes) -> Any:
            if raises is not None:
                raise raises
            return _FakeQuoteResponse(quote)

    return _FakeProvider


def _canned_result() -> JobResult:
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
        trial_outcomes=[
            TrialOutcome(
                task_name="hello-world",
                trial_name="hello-world__attempt-0",
                status="completed",
                rewards={"reward": 1.0},
            )
        ],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
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
        "eval_run_id": "eval-run-001",
        "submission_id": "submission-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": "f" * 64,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
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
        lambda **_: "f" * 64,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.assert_package_tree_matches_plan",
        lambda **_: "b" * 64,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend._preflight_eval_plan_tasks",
        lambda **_: {},
    )


def _result_line(out: str) -> dict:
    lines = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)]
    assert len(lines) == 1
    return json.loads(lines[0][len(RESULT_LINE_PREFIX) :])


def test_gate_off_emits_legacy_line_without_attestation(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    _patch_run(monkeypatch)
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    payload = _result_line(capsys.readouterr().out)
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload
    assert payload["status"] == "completed"


def test_gate_on_emits_strict_eval_result_request(monkeypatch, tmp_path, capsys) -> None:
    _patch_run(monkeypatch)
    _set_eval_plan_env(monkeypatch)
    monkeypatch.setattr(ar, "DstackQuoteProvider", _fake_provider_factory())
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    payload = _result_line(capsys.readouterr().out)
    assert ew.validate_eval_result_request(payload) == payload
    envelope = payload["execution_proof"]
    assert envelope["tier"] == ar.PHALA_TDX_TIER
    assert envelope["attestation"]["tdx_quote"] == FAKE_QUOTE
    assert payload["agent_hash"] == "f" * 64


def test_gate_on_quote_failure_fails_closed(monkeypatch, tmp_path, capsys) -> None:
    _patch_run(monkeypatch)
    _set_eval_plan_env(monkeypatch)
    monkeypatch.setattr(
        ar, "DstackQuoteProvider", _fake_provider_factory(raises=RuntimeError("no socket"))
    )
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc != 0
    out = capsys.readouterr().out
    payload = _result_line(out)
    assert "execution_proof" not in payload
    assert payload["status"] == "failed"
    assert payload["score"] == 0.0
    assert payload["reason_code"] == ar.PHALA_ATTESTATION_FAILED_REASON
    assert FAKE_QUOTE not in out
    assert "tdx_quote" not in out


def test_gate_on_missing_binding_env_fails_closed(monkeypatch, tmp_path, capsys) -> None:
    _patch_run(monkeypatch)
    monkeypatch.setenv(PHALA_ATTESTATION_ENABLED_ENV, "1")
    monkeypatch.setenv(ATTESTED_REVIEW_ENABLED_ENV, "1")
    # No immutable plan is provided.
    monkeypatch.delenv(PHALA_EVAL_PLAN_ENV, raising=False)
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc != 0
    payload = _result_line(capsys.readouterr().out)
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload
    assert payload["status"] == "failed"
    assert payload["reason_code"] == ar.PHALA_ATTESTATION_FAILED_REASON


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--task", "unexpected-task"],
        ["--n-attempts", "2"],
    ],
)
def test_gate_on_rejects_cli_values_that_cross_the_immutable_plan(
    monkeypatch, tmp_path, capsys, extra_args
) -> None:
    _set_eval_plan_env(monkeypatch)
    calls = {"key_release": 0, "runner": 0}

    def _no_key_release(**_: Any) -> None:
        calls["key_release"] += 1
        pytest.fail("mismatched CLI values must fail before key release")

    async def _no_runner(**_: Any) -> JobResult:
        calls["runner"] += 1
        pytest.fail("mismatched CLI values must fail before evaluation")

    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", _no_key_release)
    monkeypatch.setattr(backend, "run_own_runner_job", _no_runner)
    rc = main(["run", "--task", "hello-world", *extra_args, "--job-dir", str(tmp_path / "job")])
    assert rc != 0
    assert calls == {"key_release": 0, "runner": 0}
    assert _result_line(capsys.readouterr().out)["reason_code"] == "terminal_bench_failed"


def test_gate_on_uses_plan_for_key_release_images_and_attempts(
    monkeypatch, tmp_path, capsys
) -> None:
    _set_eval_plan_env(monkeypatch)
    seen: dict[str, Any] = {}

    def _key_release(*, eval_plan: dict[str, Any] | None = None) -> None:
        seen["key_release_plan"] = eval_plan
        return None

    async def _run(**kwargs: Any) -> JobResult:
        seen["run_kwargs"] = kwargs
        return _canned_result()

    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", _key_release)
    monkeypatch.setattr(backend, "run_own_runner_job", _run)
    monkeypatch.setattr(ar, "DstackQuoteProvider", _fake_provider_factory())
    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    plan = seen["key_release_plan"]
    assert plan is not None
    assert seen["run_kwargs"]["n_attempts"] == plan["k"] == 1
    assert seen["run_kwargs"]["live_registry_refs"] == {
        "hello-world": "registry.example/task@sha256:" + "d" * 64
    }
    assert seen["run_kwargs"]["eval_plan"] == plan
    assert ew.validate_eval_result_request(_result_line(capsys.readouterr().out))


def test_expired_plan_blocks_key_release_and_evaluation(monkeypatch, tmp_path, capsys) -> None:
    _set_eval_plan_env(monkeypatch)
    plan = json.loads(os.environ[PHALA_EVAL_PLAN_ENV])
    plan["issued_at_ms"] = 1
    plan["expires_at_ms"] = 2
    monkeypatch.setenv(PHALA_EVAL_PLAN_ENV, json.dumps(plan))

    def _unexpected(**_: Any) -> None:
        pytest.fail("expired plans must fail before work or key release")

    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", _unexpected)
    monkeypatch.setattr(backend, "run_own_runner_job", _unexpected)
    assert main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")]) != 0
    assert _result_line(capsys.readouterr().out)["reason_code"] == "terminal_bench_failed"


def test_task_config_mismatch_blocks_key_release_and_container_work(
    monkeypatch, tmp_path, capsys
) -> None:
    _set_eval_plan_env(monkeypatch)

    def _mismatch(**_: Any) -> dict[str, Any]:
        raise ValueError("task content digest does not match Eval plan")

    def _unexpected(**_: Any) -> None:
        pytest.fail("task mismatch must fail before key release or execution")

    monkeypatch.setattr(backend, "_preflight_eval_plan_tasks", _mismatch)
    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", _unexpected)
    monkeypatch.setattr(backend, "run_own_runner_job", _unexpected)
    assert main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")]) != 0
    assert _result_line(capsys.readouterr().out)["reason_code"] == "terminal_bench_failed"


def test_agent_source_hash_mismatch_blocks_key_release_and_execution(
    monkeypatch, tmp_path, capsys
) -> None:
    _set_eval_plan_env(monkeypatch)

    def _unexpected(**_: Any) -> None:
        pytest.fail("agent mismatch must fail before key release or execution")

    def _mismatch(**_: Any) -> str:
        raise ValueError("agent artifact does not match immutable Eval plan agent_hash")

    monkeypatch.setattr(backend, "assert_agent_artifact_matches_plan", _mismatch)
    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", _unexpected)
    monkeypatch.setattr(backend, "run_own_runner_job", _unexpected)
    assert main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")]) != 0
    assert _result_line(capsys.readouterr().out)["reason_code"] == "terminal_bench_failed"


def test_plan_agent_hash_matches_zip_artifact_domain(tmp_path) -> None:
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"submitted-bytes")
    digest = hashlib.sha256(b"submitted-bytes").hexdigest()
    assert backend.agent_artifact_sha256(zip_path) == digest
    assert (
        backend.assert_agent_artifact_matches_plan(
            artifact_path=zip_path,
            plan_agent_hash=digest,
        )
        == digest
    )
