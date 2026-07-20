from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prism_challenge.runtime_config import BenchmarkWeights, ScoreWeights

OFFICIAL_LM_EVAL_TASK_IDS: dict[str, str] = {
    "gsm8k": "gsm8k",
    "math": "hendrycks_math",
    "arc_challenge": "arc_challenge",
    "humaneval": "humaneval",
    "mmlu": "mmlu",
    "ifeval": "ifeval",
    "truthfulqa": "truthfulqa_mc2",
}
OFFICIAL_NEEDLE_TASK_ID: Literal["prism_needle"] = "prism_needle"
OFFICIAL_BENCHMARK_SCORE_KEYS: tuple[str, ...] = (
    "gsm8k",
    "math",
    "arc_challenge",
    "humaneval",
    "mmlu",
    "ifeval",
    "truthfulqa",
    "needle",
)
LM_EVAL_OUTPUT_PATH = "artifacts/benchmarks/lm_eval_results.json"
NEEDLE_OUTPUT_PATH = "artifacts/benchmarks/needle_results.json"

_PREFERRED_METRICS: dict[str, tuple[str, ...]] = {
    "gsm8k": ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match", "acc"),
    "math": ("exact_match", "exact_match,none", "acc"),
    "arc_challenge": ("acc_norm", "acc_norm,none", "acc", "acc,none"),
    "humaneval": ("pass@1", "pass_at_1", "exact_match"),
    "mmlu": ("acc", "acc,none"),
    "ifeval": (
        "prompt_level_strict_acc",
        "instruction_level_strict_acc",
        "prompt_level_loose_acc",
    ),
    "truthfulqa": ("mc2", "mc2,none", "acc", "acc,none"),
}


class LmEvalHarnessSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_ids: list[str] = Field(min_length=len(OFFICIAL_LM_EVAL_TASK_IDS))
    output_path: str = Field(default=LM_EVAL_OUTPUT_PATH, min_length=1)
    model: str = Field(default="hf", min_length=1)
    model_args: str | None = Field(default=None, min_length=1)
    batch_size: str = Field(default="auto", min_length=1)
    limit: int | None = Field(default=None, gt=0)

    def command(self) -> list[str]:
        command = [
            "lm_eval",
            "--model",
            self.model,
            "--tasks",
            ",".join(self.task_ids),
            "--output_path",
            self.output_path,
            "--batch_size",
            self.batch_size,
        ]
        if self.model_args:
            command.extend(["--model_args", self.model_args])
        if self.limit is not None:
            command.extend(["--limit", str(self.limit)])
        return command

    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump())


class NeedleScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_match_weight: float = Field(default=0.70, ge=0, le=1)
    contains_answer_weight: float = Field(default=0.20, ge=0, le=1)
    normalized_position_weight: float = Field(default=0.10, ge=0, le=1)
    primary_metric: Literal["weighted_retrieval"] = "weighted_retrieval"

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> NeedleScoringConfig:
        total = (
            self.exact_match_weight + self.contains_answer_weight + self.normalized_position_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"needle scoring weights must sum to 1.0, got {total:.6f}")
        return self


class NeedleBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Literal["prism_needle"] = OFFICIAL_NEEDLE_TASK_ID
    context_lengths: list[int] = Field(default_factory=lambda: [4096, 8192, 16384], min_length=1)
    positions: list[float] = Field(default_factory=lambda: [0.10, 0.50, 0.90], min_length=1)
    trials: int = Field(default=3, ge=1)
    output_path: str = Field(default=NEEDLE_OUTPUT_PATH, min_length=1)
    scoring: NeedleScoringConfig = Field(default_factory=NeedleScoringConfig)

    @model_validator(mode="after")
    def validate_ranges(self) -> NeedleBenchmarkConfig:
        if any(length <= 0 for length in self.context_lengths):
            raise ValueError("needle context lengths must be positive")
        if any(position < 0.0 or position > 1.0 for position in self.positions):
            raise ValueError("needle positions must be normalized between 0.0 and 1.0")
        return self

    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump())


@dataclass(frozen=True)
class ParsedBenchmarkScore:
    benchmark_key: str
    task_id: str
    score: float
    metric: str
    result_keys: tuple[str, ...]
    stderr: float | None = None


@dataclass(frozen=True)
class BenchmarkValidationMetadata:
    official_scoring_ready: bool
    missing_benchmark_keys: tuple[str, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkParseResult:
    benchmark_scores: dict[str, float]
    benchmark_fingerprints: dict[str, str]
    capability_metadata: dict[str, Any]
    noise_metadata: dict[str, Any]
    contamination_metadata: dict[str, Any]
    validation: BenchmarkValidationMetadata
    parsed_scores: tuple[ParsedBenchmarkScore, ...] = ()

    def manifest_metric_fields(self) -> dict[str, Any]:
        return {
            "benchmark_scores": dict(sorted(self.benchmark_scores.items())),
            "benchmark_capability_metadata": self.capability_metadata,
            "benchmark_noise_metadata": self.noise_metadata,
            "benchmark_contamination_metadata": self.contamination_metadata,
        }


@dataclass(frozen=True)
class BenchmarkSanityComponent:
    raw_score: float
    formula_cap: float
    capped_contribution: float
    benchmark_weights: dict[str, float] = field(default_factory=dict)


def official_lm_eval_spec(
    *,
    output_path: str = LM_EVAL_OUTPUT_PATH,
    model: str = "hf",
    model_args: str | None = None,
    batch_size: str = "auto",
    limit: int | None = None,
) -> LmEvalHarnessSpec:
    return LmEvalHarnessSpec(
        task_ids=list(OFFICIAL_LM_EVAL_TASK_IDS.values()),
        output_path=output_path,
        model=model,
        model_args=model_args,
        batch_size=batch_size,
        limit=limit,
    )


def official_needle_config(
    *,
    context_lengths: list[int] | None = None,
    positions: list[float] | None = None,
    trials: int = 3,
    output_path: str = NEEDLE_OUTPUT_PATH,
    scoring: NeedleScoringConfig | None = None,
) -> NeedleBenchmarkConfig:
    payload: dict[str, Any] = {"trials": trials, "output_path": output_path}
    if context_lengths is not None:
        payload["context_lengths"] = context_lengths
    if positions is not None:
        payload["positions"] = positions
    if scoring is not None:
        payload["scoring"] = scoring
    return NeedleBenchmarkConfig.model_validate(payload)


def parse_official_benchmark_outputs(
    lm_eval_output: str | Path | dict[str, Any],
    needle_output: str | Path | dict[str, Any],
    *,
    lm_eval_spec: LmEvalHarnessSpec | None = None,
    needle_config: NeedleBenchmarkConfig | None = None,
) -> BenchmarkParseResult:
    lm_spec = lm_eval_spec or official_lm_eval_spec()
    needle_spec = needle_config or official_needle_config()
    lm_payload = _load_json_payload(lm_eval_output)
    needle_payload = _load_json_payload(needle_output)
    parsed_scores = list(parse_lm_eval_output(lm_payload))
    parsed_scores.append(parse_needle_output(needle_payload, needle_spec))
    benchmark_scores = {score.benchmark_key: score.score for score in parsed_scores}
    missing = tuple(key for key in OFFICIAL_BENCHMARK_SCORE_KEYS if key not in benchmark_scores)
    errors = tuple(f"missing benchmark result: {key}" for key in missing)
    return BenchmarkParseResult(
        benchmark_scores=benchmark_scores,
        benchmark_fingerprints={
            "lm_eval": lm_spec.fingerprint(),
            "needle": needle_spec.fingerprint(),
        },
        capability_metadata=_capability_metadata(parsed_scores, lm_spec, needle_spec),
        noise_metadata=_noise_metadata(parsed_scores, lm_payload, needle_payload),
        contamination_metadata=_contamination_metadata(lm_payload, needle_payload),
        validation=BenchmarkValidationMetadata(
            official_scoring_ready=not missing,
            missing_benchmark_keys=missing,
            errors=errors,
        ),
        parsed_scores=tuple(parsed_scores),
    )


def parse_lm_eval_output(payload: dict[str, Any]) -> tuple[ParsedBenchmarkScore, ...]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("LM Eval Harness output requires a results object")
    parsed: list[ParsedBenchmarkScore] = []
    for benchmark_key, task_id in OFFICIAL_LM_EVAL_TASK_IDS.items():
        task_results = _matching_task_results(results, task_id)
        if not task_results:
            continue
        parsed.append(_parse_task_group(benchmark_key, task_id, task_results))
    return tuple(parsed)


def parse_needle_output(
    payload: dict[str, Any], config: NeedleBenchmarkConfig | None = None
) -> ParsedBenchmarkScore:
    needle_config = config or official_needle_config()
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Needle output requires a non-empty results list")
    row_scores = [_score_needle_row(row, needle_config.scoring) for row in rows]
    score = _clamp(sum(row_scores) / len(row_scores))
    return ParsedBenchmarkScore(
        benchmark_key="needle",
        task_id=needle_config.task_id,
        score=score,
        metric=needle_config.scoring.primary_metric,
        result_keys=(needle_config.task_id,),
    )


def benchmark_sanity_component(
    benchmark_scores: dict[str, float],
    benchmark_weights: BenchmarkWeights,
    score_weights: ScoreWeights,
    *,
    track: Literal["architecture", "training"],
) -> BenchmarkSanityComponent:
    weights = benchmark_weights.model_dump()
    raw = _weighted_score(benchmark_scores, weights)
    formula = (
        score_weights.architecture_formula
        if track == "architecture"
        else score_weights.training_formula
    )
    cap = float(formula.get("benchmark_sanity", 0.0))
    return BenchmarkSanityComponent(
        raw_score=raw,
        formula_cap=cap,
        capped_contribution=min(raw * cap, cap),
        benchmark_weights=weights,
    )


def _parse_task_group(
    benchmark_key: str, task_id: str, task_results: dict[str, dict[str, Any]]
) -> ParsedBenchmarkScore:
    preferred_metrics = _PREFERRED_METRICS[benchmark_key]
    metric = _first_present_metric(task_results, preferred_metrics)
    if metric is None:
        raise ValueError(f"{task_id} result missing supported metrics {preferred_metrics}")
    values = [
        _require_float(result[metric], f"{task_id}.{metric}") for result in task_results.values()
    ]
    stderr = _average_stderr(task_results, metric)
    return ParsedBenchmarkScore(
        benchmark_key=benchmark_key,
        task_id=task_id,
        score=_clamp(sum(values) / len(values)),
        metric=metric,
        result_keys=tuple(sorted(task_results)),
        stderr=stderr,
    )


def _matching_task_results(results: dict[str, Any], task_id: str) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for key, value in results.items():
        if key == task_id or key.startswith(f"{task_id}_"):
            if not isinstance(value, dict):
                raise ValueError(f"{key} result must be an object")
            matches[str(key)] = value
    return matches


def _first_present_metric(
    task_results: dict[str, dict[str, Any]], metrics: tuple[str, ...]
) -> str | None:
    for metric in metrics:
        if all(metric in result for result in task_results.values()):
            return metric
    return None


def _average_stderr(task_results: dict[str, dict[str, Any]], metric: str) -> float | None:
    stderr_key = f"{metric}_stderr"
    values = [
        _require_float(result[stderr_key], stderr_key)
        for result in task_results.values()
        if stderr_key in result
    ]
    if not values:
        return None
    return max(0.0, sum(values) / len(values))


def _score_needle_row(row: Any, scoring: NeedleScoringConfig) -> float:
    if not isinstance(row, dict):
        raise ValueError("Needle result rows must be objects")
    exact_match = _require_float(row.get("exact_match", 0.0), "needle.exact_match")
    contains_answer = _require_float(row.get("contains_answer", 0.0), "needle.contains_answer")
    position_score = (
        1.0 - abs(_require_float(row.get("position", 0.5), "needle.position") - 0.5) * 2.0
    )
    return _clamp(
        scoring.exact_match_weight * exact_match
        + scoring.contains_answer_weight * contains_answer
        + scoring.normalized_position_weight * _clamp(position_score)
    )


def _capability_metadata(
    scores: list[ParsedBenchmarkScore],
    lm_spec: LmEvalHarnessSpec,
    needle_config: NeedleBenchmarkConfig,
) -> dict[str, Any]:
    return {
        "lm_eval_task_ids": list(lm_spec.task_ids),
        "needle_task_id": needle_config.task_id,
        "score_keys": list(OFFICIAL_BENCHMARK_SCORE_KEYS),
        "parsed_metrics": {
            score.benchmark_key: {
                "task_id": score.task_id,
                "metric": score.metric,
                "result_keys": list(score.result_keys),
            }
            for score in scores
        },
    }


def _noise_metadata(
    scores: list[ParsedBenchmarkScore],
    lm_payload: dict[str, Any],
    needle_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stderr_by_benchmark": {
            score.benchmark_key: score.stderr for score in scores if score.stderr is not None
        },
        "lm_eval_versions": lm_payload.get("versions", {}),
        "needle_trial_count": len(needle_payload.get("results", [])),
    }


def _contamination_metadata(
    lm_payload: dict[str, Any], needle_payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "lm_eval": lm_payload.get(
            "contamination",
            {"provided": False, "validation_failure": "missing lm_eval contamination metadata"},
        ),
        "needle": needle_payload.get(
            "contamination",
            {"provided": False, "validation_failure": "missing needle contamination metadata"},
        ),
    }


def _weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return _clamp(
        sum(_clamp(float(scores.get(key, 0.0))) * float(weight) for key, weight in weights.items())
    )


def _load_json_payload(payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    path = Path(payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"benchmark output must be a JSON object: {path}")
    return loaded


def _require_float(value: Any, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
