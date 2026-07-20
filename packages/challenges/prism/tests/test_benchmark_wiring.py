from __future__ import annotations

import pytest
from pydantic import ValidationError

from prism_challenge.evaluator.benchmarks.official import (
    OFFICIAL_BENCHMARK_SCORE_KEYS,
    OFFICIAL_LM_EVAL_TASK_IDS,
    benchmark_sanity_component,
    official_lm_eval_spec,
    official_needle_config,
    parse_official_benchmark_outputs,
)
from prism_challenge.runtime_config import BenchmarkWeights, ScoreWeights


def lm_eval_fixture() -> dict:
    return {
        "results": {
            "gsm8k": {"exact_match,strict-match": 0.50, "exact_match,strict-match_stderr": 0.02},
            "hendrycks_math_algebra": {"exact_match": 0.25, "exact_match_stderr": 0.01},
            "hendrycks_math_counting_and_probability": {"exact_match": 0.75},
            "arc_challenge": {"acc_norm": 0.60},
            "humaneval": {"pass@1": 0.40},
            "mmlu": {"acc": 0.70},
            "ifeval": {"prompt_level_strict_acc": 0.80},
            "truthfulqa_mc2": {"mc2": 0.30},
        },
        "versions": {"gsm8k": 3},
        "contamination": {"provided": True, "report_path": "artifacts/contamination.json"},
    }


def needle_fixture() -> dict:
    return {
        "results": [
            {"context_length": 4096, "position": 0.1, "trial": 1, "exact_match": 1.0},
            {
                "context_length": 8192,
                "position": 0.5,
                "trial": 1,
                "exact_match": 0.0,
                "contains_answer": 1.0,
            },
        ],
        "contamination": {"provided": True, "report_path": "artifacts/needle-contamination.json"},
    }


def test_generated_lm_eval_spec_pins_executable_task_ids() -> None:
    spec = official_lm_eval_spec(output_path="artifacts/benchmarks/lm_eval.json")

    assert spec.task_ids == list(OFFICIAL_LM_EVAL_TASK_IDS.values())
    assert {
        "gsm8k",
        "hendrycks_math",
        "arc_challenge",
        "humaneval",
        "mmlu",
        "ifeval",
        "truthfulqa_mc2",
    }.issubset(set(spec.task_ids))
    assert "artifacts/benchmarks/lm_eval.json" in spec.command()
    assert "GSM8K" not in spec.command()
    assert "ARC-Challenge" not in spec.command()


def test_needle_config_declares_context_positions_trials_output_and_scoring() -> None:
    config = official_needle_config(
        context_lengths=[2048, 4096], positions=[0.0, 0.5, 1.0], trials=2
    )

    assert config.task_id == "prism_needle"
    assert config.context_lengths == [2048, 4096]
    assert config.positions == [0.0, 0.5, 1.0]
    assert config.trials == 2
    assert config.output_path == "artifacts/benchmarks/needle_results.json"
    assert config.scoring.primary_metric == "weighted_retrieval"


def test_parse_benchmark_outputs_returns_manifest_ready_scores_and_metadata() -> None:
    result = parse_official_benchmark_outputs(lm_eval_fixture(), needle_fixture())

    assert set(result.benchmark_scores) == set(OFFICIAL_BENCHMARK_SCORE_KEYS)
    assert result.benchmark_scores["math"] == pytest.approx(0.50)
    assert 0.0 <= result.benchmark_scores["needle"] <= 1.0
    assert result.validation.official_scoring_ready is True
    assert result.benchmark_fingerprints["lm_eval"]
    assert result.capability_metadata["parsed_metrics"]["truthfulqa"]["task_id"] == "truthfulqa_mc2"
    assert result.noise_metadata["stderr_by_benchmark"]["gsm8k"] == pytest.approx(0.02)
    assert result.contamination_metadata["lm_eval"]["provided"] is True
    assert result.manifest_metric_fields()["benchmark_scores"] == result.benchmark_scores


def test_missing_benchmark_result_creates_official_validation_failure_metadata() -> None:
    payload = lm_eval_fixture()
    payload["results"].pop("humaneval")

    result = parse_official_benchmark_outputs(payload, needle_fixture())

    assert result.validation.official_scoring_ready is False
    assert result.validation.missing_benchmark_keys == ("humaneval",)
    assert "missing benchmark result: humaneval" in result.validation.errors


def test_benchmark_sanity_component_is_capped_by_score_formula_share() -> None:
    score_weights = ScoreWeights(
        final_architecture_weight=0.7,
        final_recipe_weight=0.3,
        architecture_formula={
            "learning_scaling_dynamics": 0.35,
            "standardized_lm_quality": 0.20,
            "compute_efficiency": 0.15,
            "parameter_efficiency": 0.10,
            "diagnostics_health": 0.10,
            "robustness_stability": 0.05,
            "benchmark_sanity": 0.05,
        },
        training_formula={
            "architecture_normalized_heldout_improvement": 0.30,
            "learning_stability_dynamics": 0.25,
            "benchmark_sanity": 0.15,
            "compute_efficiency": 0.10,
            "reproducibility_stability": 0.10,
            "robustness_failure_behavior": 0.05,
            "artifact_completeness": 0.05,
        },
    )
    component = benchmark_sanity_component(
        {key: 1.0 for key in OFFICIAL_BENCHMARK_SCORE_KEYS},
        BenchmarkWeights(),
        score_weights,
        track="architecture",
    )

    assert component.raw_score == pytest.approx(1.0)
    assert component.formula_cap == pytest.approx(0.05)
    assert component.capped_contribution == pytest.approx(0.05)

    with pytest.raises(ValidationError, match="benchmark_sanity cannot exceed"):
        ScoreWeights(
            final_architecture_weight=0.7,
            final_recipe_weight=0.3,
            architecture_formula={"benchmark_sanity": 0.6, "learning_scaling_dynamics": 0.4},
            training_formula={
                "architecture_normalized_heldout_improvement": 0.30,
                "learning_stability_dynamics": 0.25,
                "benchmark_sanity": 0.15,
                "compute_efficiency": 0.10,
                "reproducibility_stability": 0.10,
                "robustness_failure_behavior": 0.05,
                "artifact_completeness": 0.05,
            },
        )
