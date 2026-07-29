"""Trial failure diagnostics on the BASE_BENCHMARK_RESULT line + stderr.

Production Terminal-Bench failures currently collapse to the opaque
``harbor_trial_failed`` reason code with no persisted detail. These tests pin
the additive ``trial_diagnostics`` payload field and the matching one-line
stderr breadcrumbs so the real per-trial crash cause is visible without
breaking the five-field harbor contract.
"""

from __future__ import annotations

import json

import pytest

from agent_challenge.evaluation.own_runner.orchestrator import JobResult, TrialOutcome
from agent_challenge.evaluation.own_runner.redaction import (
    REDACTED_GATEWAY_TOKEN,
    LogRedactor,
)
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
    build_trial_diagnostics,
    format_benchmark_result_line,
    validate_benchmark_result,
)
from agent_challenge.evaluation.own_runner_backend import _emit_job_result


def _failed(
    task: str,
    attempt: int = 0,
    *,
    reason: str | None = "harbor_trial_failed",
    error: str | None = "trial crashed: RuntimeError: boom",
    errored: bool = True,
    status: str = "failed",
) -> TrialOutcome:
    return TrialOutcome(
        task_name=task,
        trial_name=f"{task}__attempt-{attempt}",
        status=status,
        rewards=None,
        reason_code=reason,
        errored=errored,
        error_text=error,
    )


def _ok(task: str = "ok-task") -> TrialOutcome:
    return TrialOutcome(
        task_name=task,
        trial_name=f"{task}__attempt-0",
        status="completed",
        rewards={"reward": 1.0},
        errored=False,
    )


def _job(
    outcomes: list[TrialOutcome],
    *,
    status: str = "failed",
    reason: str | None = None,
    score: float = 0.0,
    resolved: int | None = None,
) -> JobResult:
    total = len(outcomes)
    errored = sum(1 for o in outcomes if o.errored)
    resolved_n = resolved if resolved is not None else 0
    return JobResult(
        status=status,
        score=score,
        resolved=resolved_n,
        total=total,
        reason_code=reason,
        pass_at_k={},
        n_total_trials=total,
        n_completed_trials=total - errored,
        n_errored_trials=errored,
        trial_outcomes=outcomes,
        benchmark_result=build_benchmark_result(
            status=status,
            score=score,
            resolved=resolved_n,
            total=total,
            reason_code=reason,
        ),
    )


# --------------------------------------------------------------------------- #
# S1 — build_trial_diagnostics filters + shape
# --------------------------------------------------------------------------- #


def test_build_includes_only_failed_or_errored_preserving_order() -> None:
    outcomes = [
        _ok("a"),
        _failed("b", error="err-b"),
        _ok("c"),
        _failed("d", error="err-d", errored=False, status="failed"),  # failed, not errored
        TrialOutcome(  # errored but status completed (defensive)
            task_name="e",
            trial_name="e__attempt-0",
            status="completed",
            errored=True,
            reason_code="harbor_trial_failed",
            error_text="err-e",
        ),
    ]
    diags = build_trial_diagnostics(outcomes)
    assert [d["task_name"] for d in diags] == ["b", "d", "e"]
    for entry in diags:
        assert set(entry) >= {
            "task_name",
            "trial_name",
            "status",
            "errored",
            "reason_code",
            "error_text",
        }
    assert diags[0]["error_text"] == "err-b"
    assert diags[0]["reason_code"] == "harbor_trial_failed"
    assert diags[0]["errored"] is True
    assert diags[1]["errored"] is False
    assert diags[1]["status"] == "failed"


def test_build_returns_empty_for_all_successful() -> None:
    assert build_trial_diagnostics([_ok("x"), _ok("y")]) == []


def test_build_truncates_error_text_with_ellipsis_marker() -> None:
    long = "X" * 5000
    diags = build_trial_diagnostics([_failed("t", error=long)], max_error_chars=100)
    assert len(diags) == 1
    text = diags[0]["error_text"]
    assert text is not None
    assert len(text) < 5000
    assert text.startswith("X" * 100)
    assert text.endswith("…[truncated]")
    assert "X" * 101 not in text.replace("…[truncated]", "")


def test_build_caps_at_limit_and_appends_truncation_marker() -> None:
    outcomes = [_failed(f"t{i}", error=f"e{i}") for i in range(5)]
    diags = build_trial_diagnostics(outcomes, limit=3)
    # 3 real entries + 1 marker documenting the remainder.
    assert len(diags) == 4
    assert [d["task_name"] for d in diags[:3]] == ["t0", "t1", "t2"]
    marker = diags[-1]
    assert marker["reason_code"] == "diagnostics_truncated"
    assert marker["errored"] is True
    assert marker["status"] == "failed"
    assert "2" in (marker["error_text"] or "")  # omitted count


# --------------------------------------------------------------------------- #
# S3 — redaction is mandatory on error_text
# --------------------------------------------------------------------------- #


def test_build_redacts_secret_in_error_text() -> None:
    secret = "scoped-gw-token-9f2a-SECRET"
    redactor = LogRedactor(gateway_token=secret)
    diags = build_trial_diagnostics(
        [_failed("leak", error=f"trial crashed: RuntimeError: used {secret}")],
        redactor=redactor,
    )
    assert len(diags) == 1
    assert secret not in (diags[0]["error_text"] or "")
    assert REDACTED_GATEWAY_TOKEN in (diags[0]["error_text"] or "")


def test_emitted_payload_redacts_secret_in_trial_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: secret in outcome.error_text must not appear on the result line."""

    monkeypatch.delenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", raising=False)
    monkeypatch.delenv("PHALA_ATTESTATION_ENABLED", raising=False)
    secret = "or-live-secret-token-abc123"
    # Emit-time redactor sources secrets from process env (defense in depth).
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setenv("BASE_GATEWAY_TOKEN", secret)

    result = _job(
        [_failed("crash-task", error=f"trial crashed: ValueError: bearer {secret}")]
    )
    rc = _emit_job_result(result, ["crash-task"])
    assert rc == 0
    captured = capsys.readouterr()
    out_line = [ln for ln in captured.out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)][-1]
    assert secret not in out_line
    payload = json.loads(out_line[len(RESULT_LINE_PREFIX) :])
    assert "trial_diagnostics" in payload
    assert secret not in json.dumps(payload["trial_diagnostics"])
    # stderr diagnostic also redacted
    assert secret not in captured.err
    assert any("agent_challenge_trial_diagnostic" in ln for ln in captured.err.splitlines())


# --------------------------------------------------------------------------- #
# S4 — schema accepts additive trial_diagnostics
# --------------------------------------------------------------------------- #


def test_payload_with_trial_diagnostics_validates() -> None:
    payload = build_benchmark_result(
        status="failed", score=0.0, resolved=0, total=1, reason_code="harbor_trial_failed"
    )
    payload["trial_diagnostics"] = build_trial_diagnostics(
        [_failed("t", error="trial crashed: RuntimeError: boom")]
    )
    validate_benchmark_result(payload)  # must not raise
    line = format_benchmark_result_line(payload)
    assert "trial_diagnostics" in line
    parsed = json.loads(line[len(RESULT_LINE_PREFIX) :])
    assert parsed["status"] == "failed"
    assert parsed["reason_code"] == "harbor_trial_failed"
    assert isinstance(parsed["trial_diagnostics"], list)
    assert parsed["trial_diagnostics"][0]["task_name"] == "t"


# --------------------------------------------------------------------------- #
# S5 — _emit_job_result legacy wiring
# --------------------------------------------------------------------------- #


def test_emit_adds_trial_diagnostics_only_when_nonempty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", raising=False)
    monkeypatch.delenv("PHALA_ATTESTATION_ENABLED", raising=False)

    # Successful job: no trial_diagnostics key (byte-compatible additive contract).
    ok = _job([_ok("hello-world")], status="completed", reason=None, score=1.0, resolved=1)
    assert _emit_job_result(ok, ["hello-world"]) == 0
    out_ok = capsys.readouterr().out
    line_ok = [ln for ln in out_ok.splitlines() if ln.startswith(RESULT_LINE_PREFIX)][-1]
    payload_ok = json.loads(line_ok[len(RESULT_LINE_PREFIX) :])
    assert "trial_diagnostics" not in payload_ok

    # Failed job: key present with the crashed trial.
    failed = _job(
        [_failed("boom", error="trial crashed: OSError: no space left on device")],
        status="failed",
        reason="harbor_trial_failed",
    )
    assert _emit_job_result(failed, ["boom"]) == 0
    captured = capsys.readouterr()
    line_fail = [ln for ln in captured.out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)][-1]
    payload_fail = json.loads(line_fail[len(RESULT_LINE_PREFIX) :])
    assert "trial_diagnostics" in payload_fail
    assert payload_fail["trial_diagnostics"][0]["task_name"] == "boom"
    assert "no space left" in payload_fail["trial_diagnostics"][0]["error_text"]

    # stderr one-liner, single line, newlines collapsed.
    diag_lines = [
        ln for ln in captured.err.splitlines() if ln.startswith("agent_challenge_trial_diagnostic")
    ]
    assert len(diag_lines) == 1
    diag = diag_lines[0]
    assert "task=boom" in diag
    assert "trial=boom__attempt-0" in diag
    assert "reason=harbor_trial_failed" in diag
    assert "error=" in diag
    assert "\n" not in diag


def test_emit_stderr_collapses_newlines_in_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", raising=False)
    monkeypatch.delenv("PHALA_ATTESTATION_ENABLED", raising=False)
    result = _job(
        [_failed("nl", error="line1\nline2\r\nline3")],
        status="failed",
        reason="harbor_trial_failed",
    )
    assert _emit_job_result(result, ["nl"]) == 0
    err = capsys.readouterr().err
    prefix = "agent_challenge_trial_diagnostic"
    diag_lines = [ln for ln in err.splitlines() if ln.startswith(prefix)]
    assert len(diag_lines) == 1
    assert "line1" in diag_lines[0] and "line2" in diag_lines[0] and "line3" in diag_lines[0]
    # The diagnostic itself is exactly one physical line.
    assert diag_lines[0].count("\n") == 0
