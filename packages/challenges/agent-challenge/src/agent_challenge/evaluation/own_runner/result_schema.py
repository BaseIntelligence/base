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
from collections.abc import Mapping, Sequence
from typing import IO, Any, Protocol

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
# Trial failure diagnostics (additive observability on the result line)        #
# --------------------------------------------------------------------------- #


class _TrialDiagnosticSource(Protocol):
    """Minimal trial-outcome surface consumed by :func:`build_trial_diagnostics`."""

    task_name: str
    trial_name: str
    status: str
    errored: bool
    reason_code: str | None
    error_text: str | None


class _TextRedactor(Protocol):
    """Narrow redaction seam (satisfied by :class:`~.redaction.LogRedactor`)."""

    def redact(self, text: str | None) -> str | None: ...


#: Marker ``reason_code`` on the final entry when failed trials exceed ``limit``.
DIAGNOSTICS_TRUNCATED_REASON = "diagnostics_truncated"

#: Explicit suffix appended when an individual ``error_text`` is length-capped.
ERROR_TEXT_TRUNCATION_MARKER = "\u2026[truncated]"


def _truncate_error_text(text: str | None, max_error_chars: int) -> str | None:
    """Cap ``text`` at ``max_error_chars`` and append an explicit ellipsis marker."""

    if text is None:
        return None
    if max_error_chars < 0:
        raise ValueError("max_error_chars must be >= 0")
    if len(text) <= max_error_chars:
        return text
    return text[:max_error_chars] + ERROR_TEXT_TRUNCATION_MARKER


def build_trial_diagnostics(
    outcomes: Sequence[Any],
    *,
    limit: int = 20,
    max_error_chars: int = 2000,
    redactor: _TextRedactor | None = None,
) -> list[dict[str, Any]]:
    """Project failed/errored trial outcomes into additive diagnostics dicts.

    Production Terminal-Bench crashes currently surface only the opaque
    ``harbor_trial_failed`` reason code. This helper lifts the per-trial
    ``error_text`` (and identity fields) into a compact list that can ride on
    the ``BASE_BENCHMARK_RESULT`` line under the additive ``trial_diagnostics``
    key without touching the five core contract fields.

    Inclusion rule
    --------------
    A trial is included iff ``errored is True`` OR ``status == "failed"``.
    Successful trials are omitted so a fully green job stays free of noise.

    Ordering + cap
    --------------
    Input order is preserved (deterministic). At most ``limit`` real entries are
    kept. When more failed trials exist, a final marker entry is appended with
    ``reason_code="diagnostics_truncated"`` and an ``error_text`` that records how
    many additional failures were omitted (so the list length is ``limit + 1``
    when truncated).

    Redaction
    ---------
    ``error_text`` may carry gateway tokens or miner env values. When a
    ``redactor`` is supplied (typically :class:`LogRedactor`), every emitted
    ``error_text`` is passed through ``redactor.redact`` before truncation.
    Callers MUST supply an active redactor whenever secrets may still be present.

    Parameters
    ----------
    outcomes :
        Trial outcomes (duck-typed; :class:`TrialOutcome` satisfies the protocol).
    limit :
        Max real (non-marker) diagnostic entries. Must be >= 0.
    max_error_chars :
        Per-entry cap on ``error_text`` length before the ellipsis marker.
    redactor :
        Optional text redactor (``LogRedactor`` or any object with ``redact``).

    Returns
    -------
    list[dict[str, Any]]
        Each entry has keys ``task_name``, ``trial_name``, ``status``,
        ``errored``, ``reason_code``, ``error_text``. Empty when no
        failed/errored trials.
    """

    if limit < 0:
        raise ValueError("limit must be >= 0")

    selected: list[Any] = [
        outcome
        for outcome in outcomes
        if bool(getattr(outcome, "errored", False))
        or getattr(outcome, "status", None) == "failed"
    ]
    if not selected:
        return []

    omitted = max(0, len(selected) - limit)
    kept = selected[:limit]
    entries: list[dict[str, Any]] = []
    for outcome in kept:
        raw_error = getattr(outcome, "error_text", None)
        if redactor is not None:
            raw_error = redactor.redact(raw_error)
        entries.append(
            {
                "task_name": outcome.task_name,
                "trial_name": outcome.trial_name,
                "status": outcome.status,
                "errored": bool(outcome.errored),
                "reason_code": outcome.reason_code,
                "error_text": _truncate_error_text(raw_error, max_error_chars),
            }
        )

    if omitted:
        entries.append(
            {
                "task_name": "__truncated__",
                "trial_name": "__truncated__",
                "status": "failed",
                "errored": True,
                "reason_code": DIAGNOSTICS_TRUNCATED_REASON,
                "error_text": f"+{omitted} more failed trial(s) omitted",
            }
        )
    return entries


def collapse_error_text_to_single_line(text: str | None) -> str:
    """Collapse newlines/CRs in ``text`` so a stderr diagnostic stays one line."""

    if not text:
        return ""
    return " ".join(text.replace("\r", "\n").splitlines())


def format_trial_diagnostic_stderr_line(entry: Mapping[str, Any]) -> str:
    """Format one flat ``key=value`` stderr diagnostic for a failed trial.

    Shape mirrors ``agent_challenge_reason_code=...`` breadcrumbs elsewhere::

        agent_challenge_trial_diagnostic task=<t> trial=<n> reason=<c> error=<e>

    Newlines inside ``error`` are collapsed so the whole diagnostic is one line.
    """

    task = entry.get("task_name") or ""
    trial = entry.get("trial_name") or ""
    reason = entry.get("reason_code")
    reason_s = "" if reason is None else str(reason)
    raw_error = entry.get("error_text")
    error = collapse_error_text_to_single_line(
        raw_error if isinstance(raw_error, str) else None
    )
    return (
        f"agent_challenge_trial_diagnostic task={task} trial={trial} "
        f"reason={reason_s} error={error}"
    )


def emit_trial_diagnostic_stderr_lines(
    entries: Sequence[Mapping[str, Any]],
    *,
    stream: IO[str] | None = None,
) -> list[str]:
    """Print one stderr diagnostic per real (non-truncation-marker) entry.

    Truncation marker entries (``reason_code == diagnostics_truncated``) are
    skipped so operators only see concrete per-trial failures on stderr. Returns
    the lines that were written.
    """

    target = stream if stream is not None else sys.stderr
    written: list[str] = []
    for entry in entries:
        if entry.get("reason_code") == DIAGNOSTICS_TRUNCATED_REASON:
            continue
        line = format_trial_diagnostic_stderr_line(entry)
        target.write(line + "\n")
        written.append(line)
    try:
        target.flush()
    except Exception:  # noqa: BLE001 - best-effort flush only
        pass
    return written


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
    "DIAGNOSTICS_TRUNCATED_REASON",
    "ERROR_TEXT_TRUNCATION_MARKER",
    "REQUIRED_FIELDS",
    "RESULT_LINE_PREFIX",
    "STATUS_VALUES",
    "ResultSchemaError",
    "build_benchmark_result",
    "build_trial_diagnostics",
    "collapse_error_text_to_single_line",
    "derive_benchmark_result_from_stats",
    "emit_benchmark_result_line",
    "emit_trial_diagnostic_stderr_lines",
    "format_benchmark_result_line",
    "format_trial_diagnostic_stderr_line",
    "validate_benchmark_result",
]
