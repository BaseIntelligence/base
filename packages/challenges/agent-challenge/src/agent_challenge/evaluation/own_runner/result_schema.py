"""BASE_BENCHMARK_RESULT emitter + result schema for the own-runner backend.

This module emits the EXACT ``BASE_BENCHMARK_RESULT=<json>`` line that the
*current* runner already parses, so downstream result ingestion stays
byte-for-byte UNCHANGED. The authoritative contract lives in
:mod:`agent_challenge.evaluation.runner`:

* the producer  -- the embedded ``python - <<'PY'`` script at
  ``runner.py:1399-1438`` which prints
  ``"BASE_BENCHMARK_RESULT=" + json.dumps(summary, sort_keys=True)``;
* the consumers -- ``_parse_terminal_bench_summary_with_reason``
  (``runner.py:1498-1509``) and ``_normalize_terminal_bench_result``
  (``runner.py:1443-1483``).

Contract (BINDING -- additive only, never rename/remove/retype a field)
----------------------------------------------------------------------
A harbor-produced summary is a JSON object with EXACTLY these five keys::

    status      str   -- "completed" | "failed"
    score       float -- 0.0 <= score <= 1.0 (flat mean of metric values)
    resolved    int   -- round(score * total)  (banker's rounding, gate G2)
    total       int   -- n_total_trials or completed + errored
    reason_code str | None

The line is ``BASE_BENCHMARK_RESULT=`` followed by
``json.dumps(summary, sort_keys=True)`` (default separators), so the keys are
serialized in sorted order: ``reason_code, resolved, score, status, total``.

The schema below validates *structure and types only*. It deliberately does NOT
encode the cross-field business rule (e.g. ``failed`` with ``score > 0``): a
faithful harbor run can emit that shape, and the downstream normalizer is the
component that maps it to ``harbor_result_invalid``. Encoding that rule here
would diverge from harbor's emitter. Extra (additive) fields are permitted so
future observability data can ride along without breaking the parser.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import IO, Any

# The prefix the runner scans for (runner.py:1499). Must be byte-identical.
RESULT_LINE_PREFIX = "BASE_BENCHMARK_RESULT="

# The five core fields a harbor-produced summary always carries. Order here is
# documentation only; the wire order is enforced by json.dumps(sort_keys=True).
REQUIRED_FIELDS: tuple[str, ...] = ("status", "score", "resolved", "total", "reason_code")

# Allowed status values (runner.py:1462 -> {"completed", "failed"}).
STATUS_VALUES: tuple[str, ...] = ("completed", "failed")

# JSON-Schema (Draft-07 style) description of a benchmark-result object. Used as
# the single source of truth for validation. ``additionalProperties: true`` keeps
# the contract additive-only: extra fields are permitted, the five core fields
# are required with fixed types.
BENCHMARK_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BaseBenchmarkResult",
    "type": "object",
    "additionalProperties": True,
    "required": list(REQUIRED_FIELDS),
    "properties": {
        "status": {"type": "string", "enum": list(STATUS_VALUES)},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "resolved": {"type": "integer", "minimum": 0},
        "total": {"type": "integer", "minimum": 0},
        "reason_code": {"type": ["string", "null"]},
    },
}


class ResultSchemaError(ValueError):
    """Raised when a benchmark-result payload violates BENCHMARK_RESULT_SCHEMA."""


# --------------------------------------------------------------------------- #
# Zero-dependency JSON-Schema validation                                       #
# --------------------------------------------------------------------------- #
# A self-contained validator for the small JSON-Schema subset this module uses
# (type/enum/minimum/maximum/required/properties/additionalProperties). This
# avoids adding ``jsonschema`` to the package dependency set (pyproject is owned
# by the wiring task) while still validating against a real schema document.


def _matches_type(value: Any, json_type: str) -> bool:
    if json_type == "object":
        return isinstance(value, Mapping)
    if json_type == "array":
        return isinstance(value, (list, tuple))
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        # JSON integers exclude bool (bool is an int subclass in Python).
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "null":
        return value is None
    raise ValueError(f"unsupported schema type: {json_type!r}")


def _check_type(value: Any, type_spec: Any, path: str) -> None:
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    if not any(_matches_type(value, t) for t in types):
        raise ResultSchemaError(
            f"{path or '<root>'}: expected type {type_spec!r}, got {type(value).__name__}"
        )


def _validate_against(instance: Any, schema: Mapping[str, Any], path: str) -> None:
    if "type" in schema:
        _check_type(instance, schema["type"], path)

    if "enum" in schema and instance not in schema["enum"]:
        raise ResultSchemaError(f"{path or '<root>'}: {instance!r} not in enum {schema['enum']!r}")

    # Numeric bounds only apply to actual numbers (skip non-numbers; their type
    # error, if any, is raised above).
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ResultSchemaError(f"{path or '<root>'}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ResultSchemaError(f"{path or '<root>'}: {instance} > maximum {schema['maximum']}")

    if schema.get("type") == "object" or isinstance(instance, Mapping):
        if isinstance(instance, Mapping):
            for required in schema.get("required", []):
                if required not in instance:
                    raise ResultSchemaError(
                        f"{path or '<root>'}: missing required field {required!r}"
                    )
            properties = schema.get("properties", {})
            for key, subschema in properties.items():
                if key in instance:
                    child = f"{path}.{key}" if path else key
                    _validate_against(instance[key], subschema, child)


def validate_benchmark_result(payload: Any) -> None:
    """Validate ``payload`` against :data:`BENCHMARK_RESULT_SCHEMA`.

    Raises :class:`ResultSchemaError` on the first violation. A valid payload
    returns ``None``.
    """
    if not isinstance(payload, Mapping):
        raise ResultSchemaError(f"<root>: expected object, got {type(payload).__name__}")
    _validate_against(payload, BENCHMARK_RESULT_SCHEMA, "")


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #
def build_benchmark_result(
    *,
    status: str,
    score: float,
    resolved: int,
    total: int,
    reason_code: str | None,
) -> dict[str, Any]:
    """Build a validated benchmark-result dict with EXACTLY the five core fields.

    Types are normalized to match a harbor-produced summary byte-for-byte:
    ``score`` -> float, ``resolved``/``total`` -> int. The result is validated
    before being returned so malformed values are rejected at construction time.
    """
    result: dict[str, Any] = {
        "status": status,
        # Preserve harbor's float typing (json renders 0.0 -> "0.0"). bool is
        # rejected by validation below rather than silently coerced.
        "score": float(score)
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else score,
        "resolved": resolved if isinstance(resolved, bool) else _as_int(resolved),
        "total": total if isinstance(total, bool) else _as_int(total),
        "reason_code": reason_code,
    }
    validate_benchmark_result(result)
    return result


def _as_int(value: Any) -> Any:
    """Coerce exact-integer floats to int; leave everything else untouched.

    Non-integral or non-numeric values pass through so schema validation reports
    the precise violation instead of this helper masking it.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Harbor outcome derivation (reproduces runner.py:1410-1437 byte-for-byte)     #
# --------------------------------------------------------------------------- #
def derive_benchmark_result_from_stats(data: Mapping[str, Any]) -> dict[str, Any]:
    """Derive ``{status, score, resolved, total, reason_code}`` from a result dict.

    This reproduces the success-path arithmetic of the current embedded harbor
    script (runner.py:1413-1434) EXACTLY -- same flat-mean score, same
    ``round(score * total)`` banker's rounding, same
    ``total or completed + errored`` fallback, same ``status`` derivation from
    the errored-trial count. CPython ``sum``/``/`` are used verbatim to preserve
    ε=0 floating-point parity (gate G2).

    ``data`` is the parsed harbor result JSON (``n_total_trials`` at top level,
    ``stats`` holding ``n_completed_trials``/``n_errored_trials``/``evals``).
    """
    stats = data.get("stats", {}) if isinstance(data, Mapping) else {}
    total = int(data.get("n_total_trials") or 0) if isinstance(data, Mapping) else 0
    completed = int(stats.get("n_completed_trials") or 0)
    errored = int(stats.get("n_errored_trials") or 0)

    score = 0.0
    evals = stats.get("evals", {})
    metric_values: list[float] = []
    for eval_stats in evals.values():
        for metric in eval_stats.get("metrics", []):
            if "mean" in metric:
                metric_values.append(float(metric["mean"]))
            else:
                metric_values.extend(float(value) for value in metric.values())
    if metric_values:
        score = sum(metric_values) / len(metric_values)

    return build_benchmark_result(
        status="completed" if errored == 0 else "failed",
        score=score,
        resolved=round(score * total),
        total=total or completed + errored,
        reason_code=None,
    )


# --------------------------------------------------------------------------- #
# Emission                                                                     #
# --------------------------------------------------------------------------- #
def format_benchmark_result_line(payload: Mapping[str, Any]) -> str:
    """Validate ``payload`` then format the exact ``BASE_BENCHMARK_RESULT=`` line.

    Output is byte-identical to the current harbor producer:
    ``RESULT_LINE_PREFIX + json.dumps(payload, sort_keys=True)``. Raises
    :class:`ResultSchemaError` before producing any output if ``payload`` is
    invalid, so malformed results never reach the wire.
    """
    validate_benchmark_result(payload)
    return RESULT_LINE_PREFIX + json.dumps(payload, sort_keys=True)


def emit_benchmark_result_line(payload: Mapping[str, Any], *, stream: IO[str] | None = None) -> str:
    """Validate, format, and print the result line (mirrors harbor's ``print``).

    Returns the emitted line. Writes to ``stream`` (default: ``sys.stdout``)
    followed by a newline, matching ``print(...)`` in the embedded harbor script.
    """
    line = format_benchmark_result_line(payload)
    target = stream if stream is not None else sys.stdout
    target.write(line + "\n")
    return line


__all__ = [
    "BENCHMARK_RESULT_SCHEMA",
    "REQUIRED_FIELDS",
    "RESULT_LINE_PREFIX",
    "STATUS_VALUES",
    "ResultSchemaError",
    "build_benchmark_result",
    "derive_benchmark_result_from_stats",
    "emit_benchmark_result_line",
    "format_benchmark_result_line",
    "validate_benchmark_result",
]
