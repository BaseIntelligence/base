"""Contract tests for the own-runner BASE_BENCHMARK_RESULT emitter + schema.

These tests pin the emitter to the EXACT line/shape the current runner parses.
They import and call the *real* existing parser
(`agent_challenge.evaluation.runner._parse_terminal_bench_summary_with_reason`
and `_normalize_terminal_bench_result`) so any drift in the emitted line is
caught here rather than at result-ingestion time.
"""

from __future__ import annotations

import json

import pytest

from agent_challenge.evaluation.own_runner.result_schema import (
    BENCHMARK_RESULT_SCHEMA,
    REQUIRED_FIELDS,
    RESULT_LINE_PREFIX,
    ResultSchemaError,
    build_benchmark_result,
    derive_benchmark_result_from_stats,
    format_benchmark_result_line,
    validate_benchmark_result,
)

# Real, existing parse/normalize routines = the contract authority. Importing
# (not copying) them guarantees drift is detected.
from agent_challenge.evaluation.runner import (
    _normalize_terminal_bench_result,
    _parse_terminal_bench_summary_with_reason,
)
from agent_challenge.sdk.executors import DockerRunResult


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _harbor_line(summary: dict) -> str:
    """Reproduce the EXACT line the current harbor embedded script emits.

    Mirrors runner.py:1437 verbatim: prefix + json.dumps(summary, sort_keys=True).
    """
    return "BASE_BENCHMARK_RESULT=" + json.dumps(summary, sort_keys=True)


def _run_result(stdout: str, *, returncode: int = 0, timed_out: bool = False) -> DockerRunResult:
    return DockerRunResult(
        container_name="own-runner",
        stdout=stdout,
        stderr="",
        returncode=returncode,
        timed_out=timed_out,
    )


# --------------------------------------------------------------------------- #
# S1 — Golden byte-compatibility against a harbor-produced line               #
# --------------------------------------------------------------------------- #
def test_prefix_matches_runner_contract() -> None:
    assert RESULT_LINE_PREFIX == "BASE_BENCHMARK_RESULT="


def test_required_fields_are_exactly_the_harbor_five() -> None:
    assert set(REQUIRED_FIELDS) == {"status", "score", "resolved", "total", "reason_code"}


@pytest.mark.parametrize(
    "summary",
    (
        {"status": "completed", "score": 0.5, "resolved": 2, "total": 4, "reason_code": None},
        {"status": "completed", "score": 1.0, "resolved": 1, "total": 1, "reason_code": None},
        {"status": "completed", "score": 0.0, "resolved": 0, "total": 3, "reason_code": None},
        {
            "status": "failed",
            "score": 0.0,
            "resolved": 0,
            "total": 0,
            "reason_code": "harbor_result_missing",
        },
        # Non-terminating binary float (1/3) — proves byte-identical float repr.
        {
            "status": "completed",
            "score": 1.0 / 3.0,
            "resolved": 1,
            "total": 3,
            "reason_code": None,
        },
    ),
)
def test_emitted_line_is_byte_identical_to_harbor(summary: dict) -> None:
    """The emitter must produce the SAME bytes the current harbor script prints."""
    expected = _harbor_line(summary)
    actual = format_benchmark_result_line(
        build_benchmark_result(
            status=summary["status"],
            score=summary["score"],
            resolved=summary["resolved"],
            total=summary["total"],
            reason_code=summary["reason_code"],
        )
    )
    assert actual == expected


def test_emitted_line_parses_via_real_parser_with_identical_keys_and_types() -> None:
    """Feed the emitted line into the REAL parser; assert same keys AND types."""
    summary = {"status": "completed", "score": 0.5, "resolved": 2, "total": 4, "reason_code": None}
    harbor_line = _harbor_line(summary)
    my_line = format_benchmark_result_line(build_benchmark_result(**summary))

    harbor_parsed, harbor_reason = _parse_terminal_bench_summary_with_reason(harbor_line)
    my_parsed, my_reason = _parse_terminal_bench_summary_with_reason(my_line)

    assert harbor_reason is None
    assert my_reason is None
    # Same keys.
    assert set(my_parsed) == set(harbor_parsed)
    # Same values AND same python types per key (e.g. resolved/total int, score float).
    for key in harbor_parsed:
        assert my_parsed[key] == harbor_parsed[key]
        assert type(my_parsed[key]) is type(harbor_parsed[key])


def test_emitted_line_normalizes_to_intended_outcome_via_real_runner() -> None:
    """Drive the emitted line through the real downstream normalizer."""
    line = format_benchmark_result_line(
        build_benchmark_result(
            status="completed", score=0.75, resolved=3, total=4, reason_code=None
        )
    )
    normalized = _normalize_terminal_bench_result(_run_result(line))
    assert normalized.status == "completed"
    assert normalized.score == 0.75
    assert normalized.reason_code is None


def test_failure_line_normalizes_to_failed() -> None:
    line = format_benchmark_result_line(
        build_benchmark_result(
            status="failed",
            score=0.0,
            resolved=0,
            total=0,
            reason_code="harbor_result_missing",
        )
    )
    normalized = _normalize_terminal_bench_result(_run_result(line))
    assert normalized.status == "failed"
    assert normalized.score == 0.0
    assert normalized.reason_code == "harbor_result_missing"


# --------------------------------------------------------------------------- #
# S2 — Schema rejects malformed / missing-required-field output BEFORE emit    #
# --------------------------------------------------------------------------- #
def test_validate_accepts_well_formed_payload() -> None:
    validate_benchmark_result(
        {"status": "completed", "score": 0.5, "resolved": 2, "total": 4, "reason_code": None}
    )  # must not raise


@pytest.mark.parametrize(
    "bad",
    (
        # Missing each required field in turn.
        {"score": 0.5, "resolved": 2, "total": 4, "reason_code": None},
        {"status": "completed", "resolved": 2, "total": 4, "reason_code": None},
        {"status": "completed", "score": 0.5, "total": 4, "reason_code": None},
        {"status": "completed", "score": 0.5, "resolved": 2, "reason_code": None},
        {"status": "completed", "score": 0.5, "resolved": 2, "total": 4},
    ),
)
def test_validate_rejects_missing_required_field(bad: dict) -> None:
    with pytest.raises(ResultSchemaError):
        validate_benchmark_result(bad)


@pytest.mark.parametrize(
    "bad",
    (
        # status not in enum.
        {"status": "errored", "score": 0.5, "resolved": 2, "total": 4, "reason_code": None},
        # status wrong type.
        {"status": 1, "score": 0.5, "resolved": 2, "total": 4, "reason_code": None},
        # score wrong type (string).
        {"status": "completed", "score": "0.5", "resolved": 2, "total": 4, "reason_code": None},
        # score is bool (must be rejected as a number).
        {"status": "completed", "score": True, "resolved": 2, "total": 4, "reason_code": None},
        # score out of range.
        {"status": "completed", "score": 1.5, "resolved": 2, "total": 4, "reason_code": None},
        {"status": "completed", "score": -0.1, "resolved": 2, "total": 4, "reason_code": None},
        # resolved not integer.
        {"status": "completed", "score": 0.5, "resolved": 2.5, "total": 4, "reason_code": None},
        # resolved bool.
        {"status": "completed", "score": 0.5, "resolved": True, "total": 4, "reason_code": None},
        # resolved negative.
        {"status": "completed", "score": 0.5, "resolved": -1, "total": 4, "reason_code": None},
        # total not integer.
        {"status": "completed", "score": 0.5, "resolved": 2, "total": "4", "reason_code": None},
        # reason_code wrong type (int).
        {"status": "completed", "score": 0.5, "resolved": 2, "total": 4, "reason_code": 7},
        # not an object at all.
        ["not", "an", "object"],
    ),
)
def test_validate_rejects_malformed_payload(bad: object) -> None:
    with pytest.raises(ResultSchemaError):
        validate_benchmark_result(bad)  # type: ignore[arg-type]


def test_format_refuses_to_emit_invalid_payload() -> None:
    with pytest.raises(ResultSchemaError):
        format_benchmark_result_line({"status": "completed"})  # missing fields


def test_build_rejects_invalid_inputs() -> None:
    with pytest.raises(ResultSchemaError):
        build_benchmark_result(status="completed", score=2.0, resolved=0, total=1, reason_code=None)


# --------------------------------------------------------------------------- #
# S3 — Additive-only: schema permits extra fields; core stays harbor-exact     #
# --------------------------------------------------------------------------- #
def test_schema_permits_additive_fields() -> None:
    # Forward-compat: extra observability fields must NOT be rejected.
    payload = {
        "status": "completed",
        "score": 0.5,
        "resolved": 2,
        "total": 4,
        "reason_code": None,
        "pass_at_k": {"pass@2": 0.5},  # additive
        "trial_count": 4,  # additive
    }
    validate_benchmark_result(payload)  # must not raise


def test_canonical_builder_emits_exactly_the_five_keys() -> None:
    result = build_benchmark_result(
        status="completed", score=0.5, resolved=2, total=4, reason_code=None
    )
    assert set(result) == {"status", "score", "resolved", "total", "reason_code"}


def test_schema_is_a_mapping_with_required_block() -> None:
    assert isinstance(BENCHMARK_RESULT_SCHEMA, dict)
    assert set(BENCHMARK_RESULT_SCHEMA["required"]) == {
        "status",
        "score",
        "resolved",
        "total",
        "reason_code",
    }


# --------------------------------------------------------------------------- #
# S4 — Harbor derivation parity (reproduce runner.py:1413-1434 exactly)        #
# --------------------------------------------------------------------------- #
def test_derive_from_stats_matches_harbor_algorithm_single_metric() -> None:
    data = {
        "n_total_trials": 4,
        "stats": {
            "n_completed_trials": 4,
            "n_errored_trials": 0,
            "evals": {"agent__suite": {"metrics": [{"mean": 0.5}]}},
        },
    }
    result = derive_benchmark_result_from_stats(data)
    assert result == {
        "status": "completed",
        "score": 0.5,
        "resolved": 2,  # round(0.5 * 4) banker's rounding
        "total": 4,
        "reason_code": None,
    }
    validate_benchmark_result(result)


def test_derive_from_stats_multi_metric_flattens_values() -> None:
    # No "mean" key -> ALL dict values become separate samples (runner.py:1424-1425).
    data = {
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "evals": {"e": {"metrics": [{"correctness": 0.5, "speed": 0.75}]}},
        },
    }
    result = derive_benchmark_result_from_stats(data)
    assert result["score"] == pytest.approx(0.625)
    assert result["status"] == "completed"


def test_derive_from_stats_errored_is_failed() -> None:
    data = {
        "n_total_trials": 2,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 1,
            "evals": {"e": {"metrics": [{"mean": 0.0}]}},
        },
    }
    result = derive_benchmark_result_from_stats(data)
    assert result["status"] == "failed"
    assert result["total"] == 2


def test_derive_banker_rounding() -> None:
    # round(0.5)=0, round(2.5)=2 — banker's rounding (G2).
    data = {
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "evals": {"e": {"metrics": [{"mean": 0.5}]}},
        },
    }
    assert derive_benchmark_result_from_stats(data)["resolved"] == 0  # round(0.5*1)=0


def test_derive_then_emit_roundtrips_through_real_parser() -> None:
    data = {
        "n_total_trials": 4,
        "stats": {
            "n_completed_trials": 4,
            "n_errored_trials": 0,
            "evals": {"agent__suite": {"metrics": [{"mean": 0.75}]}},
        },
    }
    line = format_benchmark_result_line(derive_benchmark_result_from_stats(data))
    parsed, reason = _parse_terminal_bench_summary_with_reason(line)
    assert reason is None
    assert parsed["status"] == "completed"
    assert parsed["score"] == 0.75
    assert parsed["resolved"] == 3
    assert parsed["total"] == 4
    assert parsed["reason_code"] is None
