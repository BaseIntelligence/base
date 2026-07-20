"""Seed-scale long-context benchmark helper for scorecard multimetric.v1.1.

Restored/reimplemented after 0920289 deleted source modules left only pyc stubs.
Primary Official Comparison consumers should prefer
:mod:`prism_challenge.evaluator.scorecard_suite` (fixture + GPU-ready hooks).
This module keeps a low-dependency seed-scale accuracy helper and re-exports the
scorecard suite aggregators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prism_challenge.evaluator.scorecard_suite import (
    LONG_CTX_CHANCE,
    LONG_CTX_RELATIVE_FLOOR,
    LONG_CTX_SUITE_TASKS,
    LongCtxSuiteResult,
    LongCtxTaskScore,
    aggregate_long_ctx_suite,
    documented_floors_relative_to_chance,
    relative_to_chance,
    run_long_ctx_fixture_suite,
    score_closed_choice_accuracy,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class LongContextResult:
    """Legacy-shaped result used by older suite pyc; score ∈ [0, 1]."""

    score: float
    collapse_penalty: float
    accuracy_by_length: dict[int, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "collapse_penalty": self.collapse_penalty,
            "accuracy_by_length": dict(self.accuracy_by_length),
        }


def long_context_from_length_accuracies(
    accuracy_by_length: dict[int, float],
) -> LongContextResult:
    """Aggregate per-length accuracies with a simple collapse penalty (short→long)."""
    if not accuracy_by_length:
        return LongContextResult(score=0.0, collapse_penalty=0.0, accuracy_by_length={})
    ordered = sorted((int(k), float(v)) for k, v in accuracy_by_length.items())
    values = [clamp(v) for _, v in ordered]
    collapse = max(0.0, values[0] - values[-1]) if values else 0.0
    mean_acc = sum(values) / len(values)
    score = clamp(mean_acc * (1.0 - collapse))
    return LongContextResult(
        score=score,
        collapse_penalty=collapse,
        accuracy_by_length={k: clamp(v) for k, v in ordered},
    )


__all__ = [
    "LONG_CTX_CHANCE",
    "LONG_CTX_RELATIVE_FLOOR",
    "LONG_CTX_SUITE_TASKS",
    "LongContextResult",
    "LongCtxSuiteResult",
    "LongCtxTaskScore",
    "aggregate_long_ctx_suite",
    "clamp",
    "documented_floors_relative_to_chance",
    "long_context_from_length_accuracies",
    "relative_to_chance",
    "run_long_ctx_fixture_suite",
    "score_closed_choice_accuracy",
]
