from __future__ import annotations

import pytest

from agent_challenge.evaluation.runner import (
    _normalize_terminal_bench_result,
    _parse_terminal_bench_summary,
    _terminal_bench_stderr,
)
from agent_challenge.sdk.executors import DockerRunResult


def _run_result(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    timed_out: bool = False,
) -> DockerRunResult:
    return DockerRunResult(
        container_name="terminal-bench",
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
    )


@pytest.mark.parametrize(
    ("stdout", "expected_status", "expected_score", "expected_reason"),
    (
        (
            'harbor done\nBASE_BENCHMARK_RESULT={"score": 0.75, "status": "completed"}',
            "completed",
            0.75,
            None,
        ),
        ("harbor done without result line", "failed", 0.0, "harbor_result_missing"),
        ("BASE_BENCHMARK_RESULT={not-json", "failed", 0.0, "harbor_result_malformed"),
        ("BASE_BENCHMARK_RESULT={}", "failed", 0.0, "harbor_result_partial"),
        (
            'BASE_BENCHMARK_RESULT={"score": 0.0, "status": "failed", '
            '"reason_code": "harbor_result_missing"}',
            "failed",
            0.0,
            "harbor_result_missing",
        ),
        (
            'BASE_BENCHMARK_RESULT={"score": 1.0, "status": "failed"}',
            "failed",
            0.0,
            "harbor_result_invalid",
        ),
        (
            'BASE_BENCHMARK_RESULT={"score": 1.5, "status": "completed"}',
            "failed",
            0.0,
            "harbor_result_invalid",
        ),
        (
            'BASE_BENCHMARK_RESULT={"score": -0.1, "status": "completed"}',
            "failed",
            0.0,
            "harbor_result_invalid",
        ),
        (
            'BASE_BENCHMARK_RESULT={"score": 0.5, "status": "weird"}',
            "failed",
            0.0,
            "harbor_result_invalid",
        ),
        (
            'BASE_BENCHMARK_RESULT={"score": true, "status": "completed"}',
            "failed",
            0.0,
            "harbor_result_invalid",
        ),
    ),
)
def test_normalize_terminal_bench_result_payload_shapes(
    stdout: str, expected_status: str, expected_score: float, expected_reason: str | None
) -> None:
    normalized = _normalize_terminal_bench_result(_run_result(stdout=stdout))

    assert normalized.status == expected_status
    assert normalized.score == expected_score
    assert normalized.reason_code == expected_reason


def test_normalize_terminal_bench_timeout_overrides_valid_payload() -> None:
    normalized = _normalize_terminal_bench_result(
        _run_result(
            stdout='BASE_BENCHMARK_RESULT={"score": 1.0, "status": "completed"}',
            returncode=124,
            timed_out=True,
        )
    )

    assert normalized.status == "timed_out"
    assert normalized.score == 0.0
    assert normalized.reason_code == "timed_out"


def test_normalize_terminal_bench_nonzero_exit_discards_valid_payload_score() -> None:
    normalized = _normalize_terminal_bench_result(
        _run_result(
            stdout='BASE_BENCHMARK_RESULT={"score": 1.0, "status": "completed"}',
            stderr="harbor failed",
            returncode=2,
        )
    )

    assert normalized.status == "failed"
    assert normalized.score == 0.0
    assert normalized.reason_code == "harbor_nonzero_exit"
    assert normalized.payload == {"score": 1.0, "status": "completed"}


def test_parse_terminal_bench_summary_keeps_stdout_diagnostic_only() -> None:
    assert _parse_terminal_bench_summary("BASE_BENCHMARK_RESULT={not-json") == {}
    assert _parse_terminal_bench_summary("no result line") == {}


def test_terminal_bench_reason_code_appends_to_benchmark_stderr_only() -> None:
    stderr = _terminal_bench_stderr("harbor stderr\n", "harbor_nonzero_exit")

    assert stderr == "harbor stderr\nagent_challenge_reason_code=harbor_nonzero_exit"
