from __future__ import annotations

import pytest
from base.challenge_sdk.executor import DockerRunResult

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import (
    ContainerEvaluationError,
    PrismContainerEvaluator,
    _classify_failure,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.modes import execution_mode_from_value

CONTRACT_CODE = "def build_model(ctx): pass\ndef get_recipe(ctx): return {}"


def _settings(**overrides) -> PrismSettings:
    base = {
        "shared_token": "secret",
        "docker_backend": "broker",
        "docker_broker_url": "http://broker",
        "docker_broker_token": "token",
    }
    base.update(overrides)
    return PrismSettings(**base)


def _evaluator(**overrides) -> PrismContainerEvaluator:
    return PrismContainerEvaluator(
        settings=_settings(**overrides), ctx=PrismContext(sequence_length=16)
    )


def test_hard_timeout_property_sits_above_budget_plus_grace_and_timeout():
    settings = _settings(
        base_eval_budget_seconds=1200,
        base_eval_watchdog_grace_seconds=120,
        base_eval_timeout_seconds=1800,
    )
    # The outer docker cap must give the runner's graceful budget + hard watchdog time to fire.
    assert settings.base_eval_hard_timeout_seconds >= 1200 + 120
    assert settings.base_eval_hard_timeout_seconds >= settings.base_eval_timeout_seconds

    tight = _settings(
        base_eval_budget_seconds=600,
        base_eval_watchdog_grace_seconds=60,
        base_eval_timeout_seconds=300,
    )
    # Even with a small configured outer timeout, the floor keeps it above budget + grace.
    assert tight.base_eval_hard_timeout_seconds == 600 + 60 + 60


def test_payload_context_carries_budget_grace_and_quota():
    settings = _settings(
        base_eval_budget_seconds=900,
        base_eval_watchdog_grace_seconds=90,
        base_eval_artifacts_quota_bytes=123456,
    )
    evaluator = PrismContainerEvaluator(settings=settings, ctx=PrismContext(sequence_length=16))
    payload = evaluator._payload(
        submission_id="sub",
        code_hash="ch",
        arch_hash="ah",
        files=(),
        architecture_entrypoint="architecture.py",
        training_entrypoint="training.py",
        build_model_symbol="build_model",
        train_symbol="train",
        gpu_allocation={"actual_gpu_count": 1},
        execution_mode=execution_mode_from_value("gpu_proxy_eval"),
    )
    context = payload["context"]
    assert context["budget_seconds"] == 900
    assert context["watchdog_grace_seconds"] == 90
    assert context["artifacts_quota_bytes"] == 123456


def test_classify_failure_budget_marker():
    rule_id, explanation = _classify_failure("PRISM_RUNNER_BUDGET_EXCEEDED: cap hit", 7)
    assert rule_id == "prism:budget-exceeded"
    assert "budget" in explanation


def test_classify_failure_artifacts_quota_marker():
    rule_id, _ = _classify_failure("PRISM_RUNNER_ARTIFACTS_QUOTA: over quota", 8)
    assert rule_id == "prism:artifacts-quota"


def test_classify_failure_oom_by_exit_code_and_marker():
    by_code, _ = _classify_failure("worker killed", 137)
    assert by_code == "prism:resource-oom"
    by_marker, _ = _classify_failure("RuntimeError: CUDA out of memory", 1)
    assert by_marker == "prism:resource-oom"


def test_classify_failure_generic_runtime_error():
    rule_id, _ = _classify_failure("Traceback (most recent call last): ValueError: boom", 1)
    assert rule_id == "prism:runtime-error"


def test_classify_failure_marker_takes_precedence_over_oom_exit_code():
    # A budget watchdog exit must not be misclassified as OOM even on a SIGKILL-style code.
    rule_id, _ = _classify_failure("PRISM_RUNNER_BUDGET_EXCEEDED: cap", 137)
    assert rule_id == "prism:budget-exceeded"


def _raise_via_fake_run(monkeypatch, result: DockerRunResult, **settings_overrides):
    def fake_run(self, spec, timeout_seconds):
        return result

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    evaluator = _evaluator(**settings_overrides)
    with pytest.raises(ContainerEvaluationError) as raised:
        evaluator.evaluate(
            submission_id="sub-fail",
            code=CONTRACT_CODE,
            code_hash="code",
            arch_hash="arch",
            backend="base_gpu",
        )
    return raised.value


def test_oom_returncode_lands_failed_with_oom_reason(monkeypatch):
    error = _raise_via_fake_run(
        monkeypatch,
        DockerRunResult("prism-eval", "", "RuntimeError: CUDA out of memory", 137),
    )
    assert error.evidence[0].rule_id == "prism:resource-oom"


def test_budget_marker_exit_lands_failed_with_budget_reason(monkeypatch):
    error = _raise_via_fake_run(
        monkeypatch,
        DockerRunResult("prism-eval", "", "PRISM_RUNNER_BUDGET_EXCEEDED: cap hit", 7),
    )
    assert error.evidence[0].rule_id == "prism:budget-exceeded"


def test_artifacts_quota_marker_exit_lands_failed_with_quota_reason(monkeypatch):
    error = _raise_via_fake_run(
        monkeypatch,
        DockerRunResult("prism-eval", "", "PRISM_RUNNER_ARTIFACTS_QUOTA: over quota", 8),
    )
    assert error.evidence[0].rule_id == "prism:artifacts-quota"


def test_runtime_crash_exit_lands_failed_with_runtime_reason(monkeypatch):
    error = _raise_via_fake_run(
        monkeypatch,
        DockerRunResult("prism-eval", "", "Traceback ...\nRuntimeError: miner exploded", 1),
    )
    assert error.evidence[0].rule_id == "prism:runtime-error"


def test_docker_timeout_lands_failed_with_budget_reason(monkeypatch):
    error = _raise_via_fake_run(
        monkeypatch,
        DockerRunResult("prism-eval", "", "", 124, timed_out=True),
        base_eval_budget_seconds=2,
        base_eval_watchdog_grace_seconds=1,
        base_eval_timeout_seconds=5,
    )
    assert error.evidence[0].rule_id == "prism:budget-exceeded"
    assert "safety cap" in str(error)


def test_reap_job_invokes_cleanup_and_is_best_effort(monkeypatch):
    calls: list[str] = []

    def fake_cleanup(self, job_id):
        calls.append(job_id)

    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.cleanup_job", fake_cleanup
    )
    _evaluator().reap_job("sub-reap")
    assert calls == ["sub-reap"]


def test_reap_job_swallows_cleanup_errors(monkeypatch):
    def boom_cleanup(self, job_id):
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.cleanup_job", boom_cleanup
    )
    # Reaping must never raise into the terminal-run finally that calls it.
    _evaluator().reap_job("sub-reap")
