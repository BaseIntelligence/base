"""Official Comparison Complete View v1.3 (machine JSON identity + multi-axis rules).

Complete View expands multimetric.v1.1 into a MAX A→Z architecture comparison
dashboard and, as of v1.3, adds a dedicated reasoning/logic panel. Product identity
is additive:

* ``scorecard_id = multimetric.complete.v1.3``
* top-level machine document schema ``complete_view.v1.3``
* protocol pin remains ``prism_official_compare.v1``
* historical Complete View identity ``multimetric.complete.v1.2`` preserved
* multimetric.v1.1 scorecard annex remains valid further up the chain

This module freezes:

1. Metric matrix catalogue (must-have + nice-to-have) mapped to VAL-COMPLETE /
   VAL-REASON panels (P0–P9 complete view residual + **P10_reasoning_logic**).
2. Multi-axis comparison object (per-axis leads including ``reasoning``,
   disagreement matrix, expanded TIE_POLAR honesty; never an opaque sole weighted
   crown).
3. Empty/partial dashboard builders that publish honest null + reason instead of
   inventing suite results (including seed-scale logic probe shells).

Production leaderboard emission remains orthogonal (held-out primary path). Prism
lab reports use PROVIDER_TRUST / LAB-GPU / IMAGE_PIN labels — not a Prism TEE
product goal. Seed-scale logic is lab/architecture comparison only — not human
AGI reasoning and not GSM8K/MMLU primary for ~7M from-scratch seeds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .official_comparison import (
    OFFICIAL_EPS_BPB,
    OFFICIAL_EPS_HELDOUT,
    OFFICIAL_EPS_LONG_CTX,
    OFFICIAL_LONG_CTX_FLOOR,
    OFFICIAL_MIN_PUBLIC_SEEDS,
    PROTOCOL_ID,
    AxisLead,
    OfficialScoreRecord,
    PolarConflictResult,
    compare_official_scorecard,
    detect_polar_conflict,
)
from .official_comparison import (
    SCORECARD_ID as MULTIMETRIC_V1_1_SCORECARD_ID,
)

# --- Complete View machine identity (VAL-COMPLETE / VAL-REASON panel locks) --------

# Active Complete View identity (VAL-REASON-001).
COMPLETE_VIEW_SCORECARD_ID = "multimetric.complete.v1.3"
COMPLETE_VIEW_SCHEMA = "complete_view.v1.3"
COMPLETE_VIEW_DASHBOARD_ID = "scorecard_complete_view.v1.3"
COMPLETE_VIEW_PROTOCOL_ID = PROTOCOL_ID

# Immediate predecessor Complete View (preserved; not rewritten by v1.3).
COMPLETE_VIEW_V1_2_SCORECARD_ID = "multimetric.complete.v1.2"
COMPLETE_VIEW_V1_2_SCHEMA = "complete_view.v1.2"
COMPLETE_VIEW_V1_2_DASHBOARD_ID = "scorecard_complete_view.v1.2"

# historical_scorecard_id points at the prior Complete View identity.
COMPLETE_VIEW_HISTORICAL_SCORECARD_ID = COMPLETE_VIEW_V1_2_SCORECARD_ID
# Full chain for operators: multimetric.v1.1 → complete.v1.2 → complete.v1.3.
COMPLETE_VIEW_HISTORICAL_CHAIN: tuple[str, ...] = (
    MULTIMETRIC_V1_1_SCORECARD_ID,
    COMPLETE_VIEW_V1_2_SCORECARD_ID,
)

# Alternate product string kept as alias only.
COMPLETE_VIEW_COMPARE_ID_ALIAS = "prism_complete_compare.v1.3"

# Soft relative floor spirit for reasoning relative_to_chance (suite fill later).
REASONING_REL_FLOOR = 0.05
REASONING_SUITE_ID = "logic_synthetic.v1"

CompleteAxisLead = AxisLead
ScientificAxis = Literal[
    "short_gen",
    "long_ctx",
    "sample_eff",
    "length_extrap",
    "memorization",
    "stability",
    "reasoning",
]
DiagnosticAxis = Literal[
    "efficiency",
    "memory_state",
    "nice_to_have",
]

# Higher-is-better vs lower-is-better for multi-axis lead rules.
_AXIS_DIRECTION: dict[str, Literal["higher", "lower"]] = {
    "short_gen": "higher",  # heldout_delta preferred
    "long_ctx": "higher",  # suite_mean accuracy
    "sample_eff": "higher",  # quality AUC
    "length_extrap": "lower",  # CE ratio / free CE degradation
    "memorization": "lower",  # memo gap
    "stability": "lower",  # seed_std / spike rates preferred lower
    "reasoning": "higher",  # closed-choice relative_to_chance / logic_acc_macro
    "efficiency": "lower",  # params / VRAM preferred lower among iso-quality
}

COMPLETE_VIEW_SCIENTIFIC_AXES: tuple[str, ...] = (
    "short_gen",
    "long_ctx",
    "sample_eff",
    "length_extrap",
    "memorization",
    "stability",
    "reasoning",
)

# Axes that participate in expanded polar honesty (scientific only; efficiency never sole-ranks).
# v1.3 adds reasoning so short_gen vs reasoning polar can fire when both sides are filled.
COMPLETE_VIEW_POLAR_AXES: tuple[str, ...] = (
    "short_gen",
    "long_ctx",
    "sample_eff",
    "length_extrap",
    "reasoning",
)

COMPLETE_VIEW_DEFAULT_EPS: dict[str, float] = {
    "short_gen": OFFICIAL_EPS_HELDOUT,
    "long_ctx": OFFICIAL_EPS_LONG_CTX,
    "sample_eff": 0.02,
    "length_extrap": 0.05,
    "memorization": 0.05,
    "stability": 0.02,
    "reasoning": REASONING_REL_FLOOR,
    "efficiency": 0.0,  # never sole-ranks; placeholder only
}

COMPLETE_VIEW_HONESTY_NOTES: tuple[str, ...] = (
    "PROVIDER_TRUST: Lium/Targon trusted providers; Prism has no TEE verifier product",
    "LAB-GPU / IMAGE_PIN labels only (not a Prism crypto TEE path)",
    "Not production emission weight crown",
    "Efficiency and wall-clock never sole-rank",
    (
        "No opaque weighted single crown; multi-axis comparison object "
        "is authoritative on disagreements"
    ),
    (
        "Historical multimetric.v1.1 and multimetric.complete.v1.2 remain valid; "
        "complete.v1.3 expands visibility with P10 reasoning/logic"
    ),
    "Invented non-null suite values are forbidden; use null + reason when not-run",
    (
        "Seed-scale synthetic logic probes are lab/architecture comparison only; "
        "they do not certify human-level reasoning, AGI, or curriculum-trained "
        "benchmark SOTA (not GSM8K/MMLU primary for ~7M from-scratch seeds)"
    ),
)

# Panel keys for the machine JSON dashboard (research complete-view + reasoning design).
COMPLETE_VIEW_PANEL_KEYS: tuple[str, ...] = (
    "P0_rank_overlay",
    "P1_short_gen",
    "P2_sample_efficiency",
    "P3_long_ctx",
    "P4_length_extrap",
    "P5_efficiency",
    "P6_memory_state",
    "P7_stability_robustness",
    "P8_calibration_entropy_optional",
    "P9_validity",
    "P10_reasoning_logic",
)

# Must-have metric matrix (research scorecard-complete-a2z-gaps CV-MH-*).
# Each maps to a VAL-COMPLETE family assertion for later metric fill features.
COMPLETE_VIEW_MUST_HAVE: tuple[dict[str, Any], ...] = (
    {
        "matrix_id": "CV-MH-01",
        "key": "val_bpb_trained",
        "panel": "P1_short_gen",
        "tier": "P",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-002",
        "description": "Absolute multi-seed val_bpb_trained free CE",
    },
    {
        "matrix_id": "CV-MH-02",
        "key": "medium_free_ce",
        "panel": "P1_short_gen",
        "tier": "P",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-002",
        "description": "Medium free CE at T=256/512",
    },
    {
        "matrix_id": "CV-MH-03",
        "key": "length_extrapolate_ce",
        "panel": "P4_length_extrap",
        "tier": "P",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-007",
        "description": "Train-short eval-long CE ratios without retrain",
    },
    {
        "matrix_id": "CV-MH-04",
        "key": "long_ctx_multi_T",
        "panel": "P3_long_ctx",
        "tier": "P",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-003",
        "description": "Long-ctx multi-T suite means (256/512/1024 or max feasible)",
    },
    {
        "matrix_id": "CV-MH-05",
        "key": "needle_by_depth",
        "panel": "P3_long_ctx",
        "tier": "P",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-004",
        "description": "Needle accuracy by depth + lost-in-middle mid panel",
    },
    {
        "matrix_id": "CV-MH-06",
        "key": "mqar_grid",
        "panel": "P3_long_ctx",
        "tier": "P",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-005",
        "description": "MQAR N×lag accuracy grid",
    },
    {
        "matrix_id": "CV-MH-07",
        "key": "induction_and_copy_unfused",
        "panel": "P3_long_ctx",
        "tier": "P",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-006",
        "description": "Separate induction_acc and exact-copy acc (unfused)",
    },
    {
        "matrix_id": "CV-MH-08",
        "key": "lag_nll_bins",
        "panel": "P3_long_ctx",
        "tier": "P",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-007",
        "description": "Lag-NLL binned at each eval T",
    },
    {
        "matrix_id": "CV-MH-09",
        "key": "sample_eff_dense",
        "panel": "P2_sample_efficiency",
        "tier": "P",
        "direction": "mixed",
        "val_complete": "VAL-COMPLETE-008",
        "description": "Denser sample-eff marks + curve summary",
    },
    {
        "matrix_id": "CV-MH-10",
        "key": "heldout_at_marks",
        "panel": "P2_sample_efficiency",
        "tier": "P",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-008",
        "description": "Optional heldout@token-marks checkpoint curve",
    },
    {
        "matrix_id": "CV-MH-11",
        "key": "peak_vram_train_eval",
        "panel": "P5_efficiency",
        "tier": "S",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-009",
        "description": "peak_vram train vs eval@T split",
    },
    {
        "matrix_id": "CV-MH-12",
        "key": "tokens_per_s_train_eval",
        "panel": "P5_efficiency",
        "tier": "S",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-009",
        "description": "tokens/s train + eval@T curve",
    },
    {
        "matrix_id": "CV-MH-13",
        "key": "step_time_ms",
        "panel": "P5_efficiency",
        "tier": "S",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-009",
        "description": "step_time_ms mean/p99 residual",
    },
    {
        "matrix_id": "CV-MH-14",
        "key": "state_footprint",
        "panel": "P6_memory_state",
        "tier": "S",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-009",
        "description": "Agnostic state/activation footprint@T",
    },
    {
        "matrix_id": "CV-MH-15",
        "key": "grad_spike_nan",
        "panel": "P7_stability_robustness",
        "tier": "R",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-010",
        "description": "grad_spike_rate + nan_inf_events multi-seed",
    },
    {
        "matrix_id": "CV-MH-16",
        "key": "multi_order_delta",
        "panel": "P7_stability_robustness",
        "tier": "R",
        "direction": "lower",
        "val_complete": "VAL-COMPLETE-011",
        "description": "Multi-order stream robustness residual or explicit BLOCKED reason",
    },
    {
        "matrix_id": "CV-MH-17",
        "key": "quality_per_param_gib",
        "panel": "P5_efficiency",
        "tier": "S",
        "direction": "higher",
        "val_complete": "VAL-COMPLETE-011",
        "description": "Derived quality_per_param and quality_per_gib",
    },
    {
        "matrix_id": "CV-MH-18",
        "key": "long_ctx_floor_honesty",
        "panel": "P0_rank_overlay",
        "tier": "V",
        "direction": "n/a",
        "val_complete": "VAL-COMPLETE-013",
        "description": "Long-ctx floor honesty + no sole crown if both below",
    },
    # --- P10 reasoning / logic MUST probes (VAL-REASON-001 catalogue) --------------
    {
        "matrix_id": "CV-RL-MH-01",
        "key": "boolean_parity_xor",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-002",
        "chance": 0.5,
        "description": "Boolean / parity / XOR chain closed accuracy + forced CE",
    },
    {
        "matrix_id": "CV-RL-MH-02",
        "key": "arith_digit_mod",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-003",
        "chance": 0.1,
        "description": "Digit / mod arithmetic toy closed accuracy + forced CE",
    },
    {
        "matrix_id": "CV-RL-MH-03",
        "key": "transitive_compare",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-004",
        "chance": 0.25,
        "description": "Transitive comparison multi-hop relational composition",
    },
    {
        "matrix_id": "CV-RL-MH-04",
        "key": "multihop_binding",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-005",
        "chance": 0.125,
        "description": "Multi-hop role binding distinct from P3 MQAR lookup",
    },
    {
        "matrix_id": "CV-RL-MH-05",
        "key": "sort_order",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-006",
        "chance": 0.25,
        "description": "Sort / order / sequencing (min/max/median style)",
    },
    {
        "matrix_id": "CV-RL-MH-06",
        "key": "reverse_edit",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-006",
        "chance": 0.25,
        "description": "Reverse / simple string edit (transform ≠ exact copy)",
    },
    {
        "matrix_id": "CV-RL-MH-07",
        "key": "count_stream",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-007",
        "chance": 0.1,
        "description": "Counting over streams with fillers",
    },
    {
        "matrix_id": "CV-RL-MH-08",
        "key": "dyck_nesting",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-007",
        "chance": 0.5,
        "description": "Bounded Dyck / nesting well-formedness and close-token",
    },
    {
        "matrix_id": "CV-RL-MH-09",
        "key": "instruction_toy",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-008",
        "chance": 0.2,
        "description": "Instruction-toy format obedience micro-templates",
    },
    {
        "matrix_id": "CV-RL-MH-10",
        "key": "contradiction_detect",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-008",
        "chance": 0.5,
        "description": "Consistency / contradiction detect over dual claims",
    },
    {
        "matrix_id": "CV-RL-MH-11",
        "key": "logic_suite_mean",
        "panel": "P10_reasoning_logic",
        "tier": "P",
        "direction": "higher",
        "val_reason": "VAL-REASON-009",
        "description": "Macro suite_mean over RL-01..10 with relative-to-chance floors",
    },
)

# Nice-to-have residual (VAL-COMPLETE-012 + VAL-REASON-011); may stay null+reason.
COMPLETE_VIEW_NICE_TO_HAVE: tuple[dict[str, Any], ...] = (
    {
        "matrix_id": "CV-NH-01",
        "key": "multi_budget_slope",
        "panel": "P8_calibration_entropy_optional",
        "description": "Multi-budget quality slope residual",
    },
    {
        "matrix_id": "CV-NH-02",
        "key": "param_size_zipper",
        "panel": "P8_calibration_entropy_optional",
        "description": "Param-size zipper residual",
    },
    {
        "matrix_id": "CV-NH-03",
        "key": "ece_entropy_calibration",
        "panel": "P8_calibration_entropy_optional",
        "description": "ECE / entropy calibration vector on closed-choice tasks",
    },
    {
        "matrix_id": "CV-NH-04",
        "key": "rapid_decay_flag",
        "panel": "P8_calibration_entropy_optional",
        "description": "Rapid-decay detector from dense online curve",
    },
    {
        "matrix_id": "CV-NH-05",
        "key": "few_shot_icl",
        "panel": "P8_calibration_entropy_optional",
        "description": "Few-shot ICL synthetic templates",
    },
    {
        "matrix_id": "CV-NH-06",
        "key": "free_gen_loop_collapse",
        "panel": "P8_calibration_entropy_optional",
        "description": "Free-gen loop / collapse rate AR proxy",
    },
    {
        "matrix_id": "CV-NH-07",
        "key": "k5_seed_stress_optional",
        "panel": "P7_stability_robustness",
        "description": "Optional K=5 seed stress",
    },
    {
        "matrix_id": "CV-NH-08",
        "key": "domain_authenticity_residual",
        "panel": "P8_calibration_entropy_optional",
        "description": "FineWeb authenticity vs synthetic lag residual",
    },
    {
        "matrix_id": "CV-NH-09",
        "key": "system_compare_fused_kernel_note",
        "panel": "P5_efficiency",
        "description": "SystemCompare fused-kernel diagnostic tokens/s only",
    },
    {
        "matrix_id": "CV-NH-10",
        "key": "pareto_ui_dual_scalar",
        "panel": "P0_rank_overlay",
        "description": "Non-authoritative Pareto dual-scalar UI residual",
    },
    # Reasoning residuals (VAL-REASON-011); still null+reason when not remesured.
    {
        "matrix_id": "CV-RL-NH-01",
        "key": "cot_free_gen_collapse",
        "panel": "P10_reasoning_logic",
        "description": "Short CoT free-gen collapse proxy on logic answer fields",
    },
    {
        "matrix_id": "CV-RL-NH-02",
        "key": "logic_ece",
        "panel": "P10_reasoning_logic",
        "description": "ECE / Brier calibration scoped to P10 closed-choice logic",
    },
    {
        "matrix_id": "CV-RL-NH-03",
        "key": "poly_vs_exp_length",
        "panel": "P10_reasoning_logic",
        "description": "Polynomial vs exponential length stress on logic patterns",
    },
)

# Catalogue of MUST reasoning probes only (P10; excludes suite -mean aggregate row).
COMPLETE_VIEW_REASONING_MUST_PROBES: tuple[dict[str, Any], ...] = tuple(
    row
    for row in COMPLETE_VIEW_MUST_HAVE
    if str(row.get("panel")) == "P10_reasoning_logic" and str(row.get("key")) != "logic_suite_mean"
)

# Chance table for MUST probes (seed-scale synthetic; not GSM8K/MMLU).
COMPLETE_VIEW_REASONING_CHANCE_TABLE: dict[str, float] = {
    str(row["key"]): float(row["chance"])
    for row in COMPLETE_VIEW_REASONING_MUST_PROBES
    if "chance" in row
}

# Mapping panels → VAL-COMPLETE / VAL-REASON metric families (schema contract surface).
COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE: dict[str, tuple[str, ...]] = {
    "P0_rank_overlay": (
        "VAL-COMPLETE-001",
        "VAL-COMPLETE-013",
        "VAL-COMPLETE-015",
        "VAL-REASON-001",
        "VAL-REASON-009",
    ),
    "P1_short_gen": ("VAL-COMPLETE-002",),
    "P2_sample_efficiency": ("VAL-COMPLETE-008",),
    "P3_long_ctx": (
        "VAL-COMPLETE-003",
        "VAL-COMPLETE-004",
        "VAL-COMPLETE-005",
        "VAL-COMPLETE-006",
        "VAL-COMPLETE-007",
    ),
    "P4_length_extrap": ("VAL-COMPLETE-007",),
    "P5_efficiency": ("VAL-COMPLETE-009", "VAL-COMPLETE-011"),
    "P6_memory_state": ("VAL-COMPLETE-009",),
    "P7_stability_robustness": ("VAL-COMPLETE-010", "VAL-COMPLETE-011"),
    "P8_calibration_entropy_optional": ("VAL-COMPLETE-012",),
    "P9_validity": ("VAL-COMPLETE-001", "VAL-COMPLETE-013"),
    "P10_reasoning_logic": (
        "VAL-REASON-001",
        "VAL-REASON-002",
        "VAL-REASON-003",
        "VAL-REASON-004",
        "VAL-REASON-005",
        "VAL-REASON-006",
        "VAL-REASON-007",
        "VAL-REASON-008",
        "VAL-REASON-009",
        "VAL-REASON-011",
        "VAL-REASON-012",
    ),
}


@dataclass(frozen=True)
class CompleteAxisScore:
    """One side scalar used by multi-axis comparison (null when suite not-run)."""

    value: float | None
    source: str
    direction: Literal["higher", "lower", "mixed", "n/a"] = "higher"
    reason_if_null: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "direction": self.direction,
            "reason_if_null": self.reason_if_null,
        }


@dataclass(frozen=True)
class MultiAxisComparison:
    """Complete A-vs-B multi-axis comparison object (VAL-COMPLETE-013).

    Never collapses to an opaque sole weighted crown. When scientific axes
    disagree beyond ε, ``tie_polar=True`` and ``crown_allowed=False``.
    """

    winner: Literal["a", "b", "tie"]
    reason: str
    tie_polar: bool
    crown_allowed: bool
    per_axis_leads: dict[str, AxisLead]
    disagreement_matrix: dict[str, dict[str, bool]]
    polar_axes_involved: tuple[str, ...]
    base_scorecard_id: str
    complete_scorecard_id: str
    efficiency_sole_rank_forbidden: bool = True
    opaque_weighted_crown_forbidden: bool = True
    detail: str = ""
    metric_vector: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "tie_polar": self.tie_polar,
            "crown_allowed": self.crown_allowed,
            "per_axis_leads": dict(self.per_axis_leads),
            "disagreement_matrix": {
                outer: dict(inner) for outer, inner in self.disagreement_matrix.items()
            },
            "polar_axes_involved": list(self.polar_axes_involved),
            "base_scorecard_id": self.base_scorecard_id,
            "complete_scorecard_id": self.complete_scorecard_id,
            "efficiency_sole_rank_forbidden": self.efficiency_sole_rank_forbidden,
            "opaque_weighted_crown_forbidden": self.opaque_weighted_crown_forbidden,
            "detail": self.detail,
            "metric_vector": self.metric_vector,
            "authoritative_claim": ("TIE_POLAR" if self.tie_polar else self.reason),
        }


def complete_view_identity() -> dict[str, Any]:
    """Lock Complete View v1.3 machine identity strings (VAL-REASON-001 / VAL-COMPLETE-001)."""
    return {
        "scorecard_id": COMPLETE_VIEW_SCORECARD_ID,
        "schema": COMPLETE_VIEW_SCHEMA,
        "dashboard_id": COMPLETE_VIEW_DASHBOARD_ID,
        "protocol_id": COMPLETE_VIEW_PROTOCOL_ID,
        "historical_scorecard_id": COMPLETE_VIEW_HISTORICAL_SCORECARD_ID,
        "historical_chain": list(COMPLETE_VIEW_HISTORICAL_CHAIN),
        "preserved_complete_v1_2_scorecard_id": COMPLETE_VIEW_V1_2_SCORECARD_ID,
        "preserved_complete_v1_2_schema": COMPLETE_VIEW_V1_2_SCHEMA,
        "multimetric_v1_1_scorecard_id": MULTIMETRIC_V1_1_SCORECARD_ID,
        "compare_id_alias": COMPLETE_VIEW_COMPARE_ID_ALIAS,
        "panel_keys": list(COMPLETE_VIEW_PANEL_KEYS),
        "scientific_axes": list(COMPLETE_VIEW_SCIENTIFIC_AXES),
        "polar_axes": list(COMPLETE_VIEW_POLAR_AXES),
        "must_have_count": len(COMPLETE_VIEW_MUST_HAVE),
        "nice_to_have_count": len(COMPLETE_VIEW_NICE_TO_HAVE),
        "reasoning_suite_id": REASONING_SUITE_ID,
        "reasoning_rel_floor": REASONING_REL_FLOOR,
        "reasoning_chance_table": dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE),
        "reasoning_must_probes": [dict(row) for row in COMPLETE_VIEW_REASONING_MUST_PROBES],
        "panel_to_val_complete": {
            k: list(v) for k, v in COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE.items()
        },
        "non_claims": {
            "prism_tee_product": False,
            "emission_crown": False,
            "opaque_weighted_sole_crown": False,
            "wall_clock_ranks": False,
            "efficiency_sole_ranks": False,
            "human_agi_reasoning": False,
            "gsm8k_mmlu_primary": False,
            "seed_scale_logic_is_lab_only": True,
        },
        "labels": {
            "provider_trust": "PROVIDER_TRUST",
            "image_pin": "IMAGE_PIN",
            "score_class_lab_gpu": "LAB-GPU",
        },
        "honesty_notes": list(COMPLETE_VIEW_HONESTY_NOTES),
        "scoring_channels": {
            "closed_choice_accuracy": True,
            "forced_ce": True,
            "chance_baselines": True,
            "suite_mean_macro": True,
        },
    }


def complete_view_metric_matrix() -> dict[str, Any]:
    """Export must-have + nice-to-have A→Z matrix keys (product schema catalogue)."""
    return {
        "scorecard_id": COMPLETE_VIEW_SCORECARD_ID,
        "must_have": [dict(row) for row in COMPLETE_VIEW_MUST_HAVE],
        "nice_to_have": [dict(row) for row in COMPLETE_VIEW_NICE_TO_HAVE],
        "panels": list(COMPLETE_VIEW_PANEL_KEYS),
        "reasoning_must_probes": [dict(row) for row in COMPLETE_VIEW_REASONING_MUST_PROBES],
        "reasoning_chance_table": dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE),
        "panel_to_val_complete": {
            k: list(v) for k, v in COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE.items()
        },
    }


def reasoning_panel_shell(*, status: str = "not_run") -> dict[str, Any]:
    """Honest P10_reasoning_logic shell (null+reason; never silent omit catalogue).

    Dual scoring channels (closed-choice accuracy + forced CE) and chance baselines
    are declared in schema; suite fill is later features
    (reason-logic-probes-suite-impl / aggregate / remesure).
    """
    probes: dict[str, Any] = {}
    for row in COMPLETE_VIEW_REASONING_MUST_PROBES:
        key = str(row["key"])
        probes[key] = {
            "matrix_id": row["matrix_id"],
            "chance": COMPLETE_VIEW_REASONING_CHANCE_TABLE.get(key),
            "val_reason": row.get("val_reason"),
            "acc": {"a": None, "b": None},
            "rel_to_chance": {"a": None, "b": None},
            "forced_ce": {"a": None, "b": None},
            "by_length": {},
            "trials": 0,
            "status": status,
            "reason": "not_run_schema_shell; fill via reason-logic probe suite features",
        }
    nice_entries = [
        {
            "matrix_id": row["matrix_id"],
            "key": row["key"],
            "a": None,
            "b": None,
            "status": "not_run",
            "reason": "not_run_nice_to_have_reasoning",
        }
        for row in COMPLETE_VIEW_NICE_TO_HAVE
        if str(row.get("panel")) == "P10_reasoning_logic"
    ]
    return {
        "status": status,
        "reason": (
            None
            if status == "filled"
            else "not_run_schema_shell; fill via reason-logic suite (VAL-REASON-002..010)"
        ),
        "suite_id": REASONING_SUITE_ID,
        "scoring": {
            "closed_choice_accuracy": True,
            "forced_ce": True,
            "chance_baselines": True,
            "architecture_agnostic_logits_only": True,
        },
        "chance_table": dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE),
        "rel_floor": REASONING_REL_FLOOR,
        "seeds": None,
        "aggregates": {
            "logic_acc_macro": {"a": None, "b": None},
            "logic_rel_macro": {"a": None, "b": None},
            "logic_ce_macro": {"a": None, "b": None},
            "logic_below_chance_count": {"a": None, "b": None},
            "logic_floor_pass": {"a": None, "b": None},
            "suite_mean": {"a": None, "b": None},
            "status": status,
            "reason": "not_run_until_probe_suite_aggregate",
        },
        "probes": probes,
        "nice": {
            "status": "nice_to_have",
            "entries": nice_entries,
        },
        "distinct_from": {
            "P3_long_ctx_mqar": (
                "MQAR remains associative retrieval under P3; multihop_binding is "
                "role/composition inference under P10"
            ),
            "gsm8k_mmlu_lm_eval": (
                "NOT GSM8K/MMLU/lm-eval primary for ~7M from-scratch seeds; "
                "challenge-owned synthetic only"
            ),
        },
        "honesty": [
            "Seed-scale synthetic logic; not GSM8K/MMLU primary",
            "Not human AGI reasoning certification",
            "Not emission crown; efficiency never sole-ranks",
            "Near-chance results are valid architecture-outcomes at seed scale",
            "LAB/diagnostic architecture comparison only",
            "PROVIDER_TRUST / LAB-GPU / IMAGE_PIN (no Prism TEE product)",
        ],
        "multi_axis": {
            "axis": "reasoning",
            "direction": "higher",
            "polar_with": ["short_gen"],
            "eps": REASONING_REL_FLOOR,
        },
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    return None


def _axis_lead_from_values(
    a: float | None,
    b: float | None,
    *,
    direction: Literal["higher", "lower", "mixed", "n/a"],
    eps: float,
) -> AxisLead:
    if a is None or b is None:
        return "missing"
    if direction in ("mixed", "n/a"):
        return "missing"
    if direction == "higher":
        if a > b + eps:
            return "a"
        if b > a + eps:
            return "b"
        return "tie"
    # lower is better
    if a < b - eps:
        return "a"
    if b < a - eps:
        return "b"
    return "tie"


def default_axis_scores_from_records(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
) -> dict[str, tuple[CompleteAxisScore, CompleteAxisScore]]:
    """Project core OfficialScoreRecord fields into complete-view scientific axes.

    Expanded multi-T / depth / unfused suite fields are filled by later features
    via :func:`build_complete_view` overrides; here we bootstrap from v1.1 fields.
    Reasoning axis starts null until P10 ``logic_rel_macro`` / suite_mean is filled.
    """
    short_a = _finite_or_none(a.heldout_delta)
    short_b = _finite_or_none(b.heldout_delta)
    if short_a is None and a.primary_form == "val_bpb_trained":
        short_a = _finite_or_none(a.val_bpb_trained)
        short_b = _finite_or_none(b.val_bpb_trained)
        short_dir: Literal["higher", "lower"] = "lower"
        short_src = "val_bpb_trained"
    else:
        short_dir = "higher"
        short_src = "heldout_delta"

    long_enabled = bool(a.long_ctx_enabled or b.long_ctx_enabled)
    long_a = _finite_or_none(a.long_ctx_score) if long_enabled else None
    long_b = _finite_or_none(b.long_ctx_score) if long_enabled else None
    long_reason = None if long_enabled else "long_ctx_suite_not_run"

    se_a = _finite_or_none(a.sample_eff_auc)
    se_b = _finite_or_none(b.sample_eff_auc)
    se_reason_a = None if se_a is not None else "sample_eff_not_run"
    se_reason_b = None if se_b is not None else "sample_eff_not_run"

    memo_a = _finite_or_none(a.train_heldout_gap)
    memo_b = _finite_or_none(b.train_heldout_gap)

    # Stability uses heldout_std when multi-seed; may be missing for K=1.
    stab_a = _finite_or_none(a.heldout_std if a.heldout_std is not None else a.bpb_std)
    stab_b = _finite_or_none(b.heldout_std if b.heldout_std is not None else b.bpb_std)

    # length_extrap starts null until suite feature lands.
    le_reason = "length_extrap_not_run"

    return {
        "short_gen": (
            CompleteAxisScore(short_a, short_src, short_dir),
            CompleteAxisScore(short_b, short_src, short_dir),
        ),
        "long_ctx": (
            CompleteAxisScore(long_a, "long_ctx_suite_mean", "higher", reason_if_null=long_reason),
            CompleteAxisScore(long_b, "long_ctx_suite_mean", "higher", reason_if_null=long_reason),
        ),
        "sample_eff": (
            CompleteAxisScore(se_a, "sample_eff_auc", "higher", reason_if_null=se_reason_a),
            CompleteAxisScore(se_b, "sample_eff_auc", "higher", reason_if_null=se_reason_b),
        ),
        "length_extrap": (
            CompleteAxisScore(None, "length_extrapolate_ce", "lower", le_reason),
            CompleteAxisScore(None, "length_extrapolate_ce", "lower", le_reason),
        ),
        "memorization": (
            CompleteAxisScore(memo_a, "memo_gap", "lower"),
            CompleteAxisScore(memo_b, "memo_gap", "lower"),
        ),
        "stability": (
            CompleteAxisScore(
                stab_a,
                "heldout_or_bpb_std",
                "lower",
                reason_if_null=None if stab_a is not None else "stability_std_missing",
            ),
            CompleteAxisScore(
                stab_b,
                "heldout_or_bpb_std",
                "lower",
                reason_if_null=None if stab_b is not None else "stability_std_missing",
            ),
        ),
        "reasoning": (
            CompleteAxisScore(
                None,
                "logic_rel_macro",
                "higher",
                reason_if_null="reasoning_suite_not_run",
            ),
            CompleteAxisScore(
                None,
                "logic_rel_macro",
                "higher",
                reason_if_null="reasoning_suite_not_run",
            ),
        ),
    }


def _build_disagreement_matrix(
    leads: Mapping[str, AxisLead],
    *,
    axes: Sequence[str],
) -> dict[str, dict[str, bool]]:
    """pairwise true when two axes name opposite sides (a vs b)."""
    matrix: dict[str, dict[str, bool]] = {}
    for left in axes:
        matrix[left] = {}
        for right in axes:
            la = leads.get(left, "missing")
            rb = leads.get(right, "missing")
            disagree = la in ("a", "b") and rb in ("a", "b") and la != rb
            matrix[left][right] = disagree
    return matrix


def compare_complete_multi_axis(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    axis_scores: Mapping[str, tuple[CompleteAxisScore, CompleteAxisScore]] | None = None,
    eps_by_axis: Mapping[str, float] | None = None,
    polar_axes: Sequence[str] | None = None,
    eps_heldout: float = OFFICIAL_EPS_HELDOUT,
    eps_bpb: float = OFFICIAL_EPS_BPB,
    eps_long_ctx: float = OFFICIAL_EPS_LONG_CTX,
    long_ctx_floor: float = OFFICIAL_LONG_CTX_FLOOR,
) -> MultiAxisComparison:
    """Build Complete View multi-axis comparison (VAL-COMPLETE-013).

    Rules:
    1. Start from multimetric.v1.1 ``compare_official_scorecard``
       (heldout-primary + short/long polar).
    2. Expand per-axis leads across scientific axes; fill disagreement matrix.
    3. If any pair of polar_axes disagree (strict a vs b), force TIE_POLAR /
       crown_allowed=false.
    4. Efficiency never determines winner and never sole-ranks.

    No opaque weighted scalar is produced.
    """
    eps_map = {**COMPLETE_VIEW_DEFAULT_EPS, **(dict(eps_by_axis or {}))}
    axes_in_play = list(polar_axes or COMPLETE_VIEW_POLAR_AXES)
    raw_scores = axis_scores or default_axis_scores_from_records(a, b)

    per_axis: dict[str, AxisLead] = {}
    metric_vector: dict[str, Any] = {"a": {}, "b": {}}
    for axis, pair in raw_scores.items():
        sa, sb = pair
        direction: Literal["higher", "lower", "mixed", "n/a"]
        if sa.direction == "mixed":
            direction = _AXIS_DIRECTION.get(axis, "higher")
        else:
            direction = sa.direction
        if direction not in ("higher", "lower", "mixed", "n/a"):
            direction = "higher"
        lead = _axis_lead_from_values(
            sa.value,
            sb.value,
            direction=direction,
            eps=float(eps_map.get(axis, 0.0)),
        )
        per_axis[axis] = lead
        metric_vector["a"][axis] = sa.as_dict()
        metric_vector["b"][axis] = sb.as_dict()

    # Always include sample of non-polar scientific axes for the published object.
    for axis in COMPLETE_VIEW_SCIENTIFIC_AXES:
        if axis not in per_axis and axis in raw_scores:
            sa, sb = raw_scores[axis]
            direction2: Literal["higher", "lower", "mixed", "n/a"]
            if sa.direction in ("higher", "lower"):
                direction2 = sa.direction
            else:
                direction2 = _AXIS_DIRECTION.get(axis, "higher")
            per_axis[axis] = _axis_lead_from_values(
                sa.value,
                sb.value,
                direction=direction2,
                eps=float(eps_map.get(axis, 0.0)),
            )
            metric_vector["a"][axis] = sa.as_dict()
            metric_vector["b"][axis] = sb.as_dict()

    disagreement = _build_disagreement_matrix(per_axis, axes=list(axes_in_play))
    polar_pairs: list[tuple[str, str]] = []
    for i, left in enumerate(axes_in_play):
        for right in axes_in_play[i + 1 :]:
            if disagreement.get(left, {}).get(right, False):
                polar_pairs.append((left, right))

    base = compare_official_scorecard(
        a,
        b,
        eps_heldout=eps_heldout,
        eps_bpb=eps_bpb,
        eps_long_ctx=eps_long_ctx,
        long_ctx_floor=long_ctx_floor,
    )
    base_polar = detect_polar_conflict(
        a,
        b,
        eps_heldout=eps_heldout,
        eps_long_ctx=eps_long_ctx,
        long_ctx_floor=long_ctx_floor,
    )

    expand_polar = bool(polar_pairs)
    tie_polar = bool(base.tie_polar or base_polar.tie_polar or expand_polar)
    if tie_polar:
        detail_bits = []
        if base.tie_polar or base_polar.tie_polar:
            detail_bits.append(base_polar.reason or base.detail or "v1.1 short_gen vs long_ctx")
        if expand_polar:
            detail_bits.append("axis_pairs=" + ",".join(f"{x}/{y}" for x, y in polar_pairs))
        return MultiAxisComparison(
            winner="tie",
            reason="tie_polar",
            tie_polar=True,
            crown_allowed=False,
            per_axis_leads=per_axis,
            disagreement_matrix=disagreement,
            polar_axes_involved=tuple(
                sorted(
                    {ax for pair in polar_pairs for ax in pair}
                    | (
                        {"short_gen", "long_ctx"}
                        if (base.tie_polar or base_polar.tie_polar)
                        else set()
                    )
                )
            ),
            base_scorecard_id=MULTIMETRIC_V1_1_SCORECARD_ID,
            complete_scorecard_id=COMPLETE_VIEW_SCORECARD_ID,
            detail="; ".join(detail_bits) or "TIE_POLAR scientific axes disagree",
            metric_vector=metric_vector,
        )

    # No polar conflict: preserve base v1.1/v1 winner (not efficiency, not opaque crown).
    return MultiAxisComparison(
        winner=base.winner,
        reason=base.reason,
        tie_polar=False,
        crown_allowed=True,
        per_axis_leads=per_axis,
        disagreement_matrix=disagreement,
        polar_axes_involved=tuple(axes_in_play),
        base_scorecard_id=MULTIMETRIC_V1_1_SCORECARD_ID,
        complete_scorecard_id=COMPLETE_VIEW_SCORECARD_ID,
        detail=base.detail or "no_scientific_polar_conflict; multi-axis vector published",
        metric_vector=metric_vector,
    )


def _null_side_pair(reason: str) -> dict[str, Any]:
    return {"a": None, "b": None, "reason": reason, "status": "not_run"}


def _empty_panel_shells(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    comparison: MultiAxisComparison,
    base_polar: PolarConflictResult,
) -> dict[str, Any]:
    """Skeleton panels with honest nulls for not-yet-filled complete metrics."""
    not_run = "not_run_schema_shell; fill via complete-view metric features"
    return {
        "P0_rank_overlay": {
            "winner": comparison.winner,
            "reason": comparison.reason,
            "crown_allowed": comparison.crown_allowed,
            "tie_polar": comparison.tie_polar,
            "short_gen_lead": comparison.per_axis_leads.get("short_gen", "missing"),
            "long_ctx_lead": comparison.per_axis_leads.get("long_ctx", "missing"),
            "sample_eff_lead": comparison.per_axis_leads.get("sample_eff", "missing"),
            "length_extrap_lead": comparison.per_axis_leads.get("length_extrap", "missing"),
            "reasoning_lead": comparison.per_axis_leads.get("reasoning", "missing"),
            "authoritative_claim": comparison.as_dict()["authoritative_claim"],
            "floor_vetoes": {
                "a": base_polar.floor_veto_a,
                "b": base_polar.floor_veto_b,
            },
            "opaque_weighted_crown_forbidden": True,
            "efficiency_sole_rank_forbidden": True,
        },
        "P1_short_gen": {
            "heldout_delta": {
                "a": a.heldout_delta,
                "b": b.heldout_delta,
                "std_a": a.heldout_std,
                "std_b": b.heldout_std,
            },
            "prequential_bpb": {
                "a": a.bpb,
                "b": b.bpb,
                "std_a": a.bpb_std,
                "std_b": b.bpb_std,
            },
            "val_bpb_trained": {
                "a": a.val_bpb_trained,
                "b": b.val_bpb_trained,
                "status": (
                    "filled"
                    if (a.val_bpb_trained is not None and b.val_bpb_trained is not None)
                    else "null_pending_VAL-COMPLETE-002"
                ),
            },
            "medium_free_ce": _null_side_pair(not_run),
            "memo_gap": {
                "a": a.train_heldout_gap,
                "b": b.train_heldout_gap,
                "flag_a": a.memorization_flag,
                "flag_b": b.memorization_flag,
            },
        },
        "P2_sample_efficiency": {
            "marks_tokens": [50_000, 100_000, 250_000, 500_000],
            "quality_auc": {"a": a.sample_eff_auc, "b": b.sample_eff_auc},
            "bpb_at_marks": {
                "a": list(a.sample_eff_marks) if a.sample_eff_marks else None,
                "b": list(b.sample_eff_marks) if b.sample_eff_marks else None,
            },
            "dense_marks": _null_side_pair(not_run),
            "curve_summary": _null_side_pair(not_run),
            "heldout_at_marks": _null_side_pair(not_run),
        },
        "P3_long_ctx": {
            "by_T": {
                "192": {
                    "suite_mean": {
                        "a": a.long_ctx_score if a.long_ctx_enabled else None,
                        "b": b.long_ctx_score if b.long_ctx_enabled else None,
                    },
                    "needle": {
                        "a": a.long_ctx_needle,
                        "b": b.long_ctx_needle,
                    },
                    "mqar": {"a": a.long_ctx_mqar, "b": b.long_ctx_mqar},
                    "induction_copy_fused_historical": {
                        "a": a.long_ctx_induction_copy,
                        "b": b.long_ctx_induction_copy,
                    },
                    "floor_pass": {
                        "a": a.long_ctx_floor_pass,
                        "b": b.long_ctx_floor_pass,
                    },
                }
            },
            "multi_T": _null_side_pair("multi_T_pending_VAL-COMPLETE-003"),
            "needle_by_depth": _null_side_pair("pending_VAL-COMPLETE-004"),
            "lost_in_middle": _null_side_pair("pending_VAL-COMPLETE-004"),
            "mqar_grid": _null_side_pair("pending_VAL-COMPLETE-005"),
            "induction_acc": _null_side_pair("pending_VAL-COMPLETE-006"),
            "copy_acc": _null_side_pair("pending_VAL-COMPLETE-006"),
            "lag_nll_bins": {
                "macro": {"a": a.lag_nll, "b": b.lag_nll},
                "binned": _null_side_pair("pending_VAL-COMPLETE-007"),
            },
        },
        "P4_length_extrap": {
            "ce_by_T": _null_side_pair("pending_VAL-COMPLETE-007"),
            "ratio_T_over_train": _null_side_pair("pending_VAL-COMPLETE-007"),
        },
        "P5_efficiency": {
            "params": {"a": a.params, "b": b.params},
            "peak_vram_train_gib": {
                "a": a.peak_vram_gib,
                "b": b.peak_vram_gib,
            },
            "peak_vram_eval_by_T": _null_side_pair("pending_VAL-COMPLETE-009"),
            "tokens_per_s_train": {
                "a": a.tokens_per_s,
                "b": b.tokens_per_s,
            },
            "tokens_per_s_eval_by_T": _null_side_pair("pending_VAL-COMPLETE-009"),
            "step_time_ms": _null_side_pair("pending_VAL-COMPLETE-009"),
            "quality_per_param": _null_side_pair("pending_VAL-COMPLETE-011"),
            "quality_per_gib": _null_side_pair("pending_VAL-COMPLETE-011"),
            "sole_rank_forbidden": True,
            "overrides_polar_rule": False,
        },
        "P6_memory_state": {
            "state_footprint_bytes_by_T": _null_side_pair("pending_VAL-COMPLETE-009"),
            "activation_peak_bytes_by_T": _null_side_pair("pending_VAL-COMPLETE-009"),
        },
        "P7_stability_robustness": {
            "grad_spike_rate": {
                "a": a.grad_spike_rate,
                "b": b.grad_spike_rate,
            },
            "nan_inf_events": {
                "a": a.nan_inf_events,
                "b": b.nan_inf_events,
            },
            "instability_flag": {
                "a": a.instability_flag,
                "b": b.instability_flag,
            },
            "step0_anomaly": {
                "a": a.step0_anomaly,
                "b": b.step0_anomaly,
            },
            "seed_std_bpb": {"a": a.bpb_std, "b": b.bpb_std},
            "seed_std_heldout": {"a": a.heldout_std, "b": b.heldout_std},
            "multi_order_delta": _null_side_pair("pending_VAL-COMPLETE-011"),
        },
        "P8_calibration_entropy_optional": {
            "status": "nice_to_have",
            "entries": [
                {
                    "matrix_id": row["matrix_id"],
                    "key": row["key"],
                    "a": None,
                    "b": None,
                    "reason": "not_run_nice_to_have",
                }
                for row in COMPLETE_VIEW_NICE_TO_HAVE
            ],
        },
        "P9_validity": {
            "gates": [
                "stop_token_budget",
                "finite_bpb",
                "step0_clean",
                "param_cap",
                "matched_pin",
                "multi_seed_K",
                "challenge_authored",
            ],
            "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
            "seed_count": {"a": a.seed_count, "b": b.seed_count},
            "valid": {"a": a.valid, "b": b.valid},
            "public_multi_seed": {
                "a": a.is_public_multi_seed,
                "b": b.is_public_multi_seed,
            },
        },
        "P10_reasoning_logic": reasoning_panel_shell(status="not_run"),
    }


def build_complete_view(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    sides: Mapping[str, Any] | None = None,
    pin: Mapping[str, Any] | None = None,
    panels_override: Mapping[str, Any] | None = None,
    comparison: MultiAxisComparison | None = None,
    score_class: str = "LAB-GPU",
    spend_redacted: Mapping[str, Any] | None = None,
    per_seed_index: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build top-level ``complete_view.v1.3`` machine JSON (VAL-REASON-001 / VAL-COMPLETE-015).

    This is the single reconciling document for Complete View A→Z panels (including
    P10_reasoning_logic) + multi-axis comparison (reasoning axis). Suites that have
    not run MUST appear as null + reason (no invention). multimetric.complete.v1.2
    history is preserved via historical_scorecard_id / historical_chain.
    """
    if comparison is None:
        comparison = compare_complete_multi_axis(a, b)
    base_polar = detect_polar_conflict(a, b)
    panels = _empty_panel_shells(a, b, comparison=comparison, base_polar=base_polar)
    if panels_override:
        for key, value in panels_override.items():
            if key not in COMPLETE_VIEW_PANEL_KEYS:
                raise ValueError(f"unknown complete_view panel key: {key}")
            if isinstance(value, Mapping) and isinstance(panels.get(key), dict):
                panels[key] = {**panels[key], **dict(value)}
            else:
                panels[key] = value

    default_sides = {
        "a": {
            "label": a.label,
            "params": a.params,
            "seed_count": a.seed_count,
        },
        "b": {
            "label": b.label,
            "params": b.params,
            "seed_count": b.seed_count,
        },
    }
    if sides:
        merged_sides = {
            "a": {**default_sides["a"], **dict(sides.get("a") or {})},
            "b": {**default_sides["b"], **dict(sides.get("b") or {})},
        }
    else:
        merged_sides = default_sides

    document: dict[str, Any] = {
        "schema": COMPLETE_VIEW_SCHEMA,
        "dashboard_id": COMPLETE_VIEW_DASHBOARD_ID,
        "scorecard_id": COMPLETE_VIEW_SCORECARD_ID,
        "historical_scorecard_id": COMPLETE_VIEW_HISTORICAL_SCORECARD_ID,
        "protocol_id": COMPLETE_VIEW_PROTOCOL_ID,
        "compare_id_alias": COMPLETE_VIEW_COMPARE_ID_ALIAS,
        "score_class": score_class,
        "labels": {
            "provider_trust": "PROVIDER_TRUST",
            "image_pin": "IMAGE_PIN",
            "score_class": score_class,
        },
        "identity": complete_view_identity(),
        "metric_matrix": complete_view_metric_matrix(),
        "pin": dict(pin or {"protocol_id": COMPLETE_VIEW_PROTOCOL_ID}),
        "sides": merged_sides,
        "panels": panels,
        "comparison": comparison.as_dict(),
        "per_seed_index": dict(per_seed_index or {"a": {}, "b": {}}),
        "spend_redacted": dict(
            spend_redacted
            or {
                "delta_usd": None,
                "gpu": None,
                "verified_terminated": None,
                "note": "fill on remesure; secrets redacted",
            }
        ),
        "honesty": list(COMPLETE_VIEW_HONESTY_NOTES),
        "non_claims": {
            "prism_tee_product": False,
            "emission_weight_crown": False,
            "opaque_weighted_sole_crown": False,
            "wall_clock_ranks": False,
            "efficiency_sole_ranks": False,
            "historical_v1_1_rewritten": False,
            "historical_complete_v1_2_rewritten": False,
            "human_agi_reasoning": False,
            "gsm8k_mmlu_primary": False,
            "seed_scale_logic_is_lab_only": True,
        },
        "relation_to_multimetric_v1_1": {
            "historical_scorecard_id": MULTIMETRIC_V1_1_SCORECARD_ID,
            "complete_expands": True,
            "historical_preserved": True,
            "note": (
                "multimetric.v1.1 remains the scorecard annex on prism_compare_report.v1; "
                "complete_view.v1.3 is the MAX A→Z machine dashboard reconciling expanded "
                "panels including P10 reasoning/logic."
            ),
        },
        "relation_to_complete_v1_2": {
            "historical_scorecard_id": COMPLETE_VIEW_V1_2_SCORECARD_ID,
            "historical_schema": COMPLETE_VIEW_V1_2_SCHEMA,
            "complete_expands": True,
            "historical_preserved": True,
            "note": (
                "multimetric.complete.v1.2 history is preserved; multimetric.complete.v1.3 "
                "adds P10_reasoning_logic, reasoning multi-axis lead, and polar honesty "
                "vs short_gen under the same protocol pin."
            ),
        },
        "historical_chain": list(COMPLETE_VIEW_HISTORICAL_CHAIN),
    }
    if extra:
        for key, value in extra.items():
            if key in document:
                raise ValueError(f"complete_view reserved key override forbidden: {key}")
            document[key] = value
    return document


def validate_complete_view_document(document: Mapping[str, Any]) -> list[str]:
    """Return schema/identity problems; empty list means contract-pass for shell.

    Enforces VAL-COMPLETE-001 / 013 / 015 structural locks (not suite fill quality).
    """
    errors: list[str] = []
    if document.get("schema") != COMPLETE_VIEW_SCHEMA:
        errors.append(f"schema must be {COMPLETE_VIEW_SCHEMA}")
    if document.get("scorecard_id") != COMPLETE_VIEW_SCORECARD_ID:
        errors.append(f"scorecard_id must be {COMPLETE_VIEW_SCORECARD_ID}")
    if document.get("protocol_id") != COMPLETE_VIEW_PROTOCOL_ID:
        errors.append(f"protocol_id must be {COMPLETE_VIEW_PROTOCOL_ID}")
    if document.get("historical_scorecard_id") != COMPLETE_VIEW_HISTORICAL_SCORECARD_ID:
        errors.append(f"historical_scorecard_id must be {COMPLETE_VIEW_HISTORICAL_SCORECARD_ID}")
    # real_provider_tee is retired (VAL-NOTEE-009): not a schema gate.
    non_claims = document.get("non_claims")
    if not isinstance(non_claims, Mapping):
        errors.append("non_claims object required")
    else:
        if non_claims.get("opaque_weighted_sole_crown") is not False:
            errors.append("opaque_weighted_sole_crown must be false")
        if non_claims.get("emission_weight_crown") is not False:
            errors.append("emission_weight_crown must be false")
        if non_claims.get("prism_tee_product") is not False:
            errors.append("prism_tee_product must be false")
        # Structural honesty flags (VAL-REASON-012): seed-scale lab only, no AGI,
        # no GSM8K/MMLU primary. Prefer hard presence over defaults-only.
        if non_claims.get("human_agi_reasoning") is not False:
            errors.append("human_agi_reasoning must be false")
        if non_claims.get("gsm8k_mmlu_primary") is not False:
            errors.append("gsm8k_mmlu_primary must be false")
        if non_claims.get("seed_scale_logic_is_lab_only") is not True:
            errors.append("seed_scale_logic_is_lab_only must be true")

    panels = document.get("panels")
    if not isinstance(panels, Mapping):
        errors.append("panels object required")
    else:
        for key in COMPLETE_VIEW_PANEL_KEYS:
            if key not in panels:
                errors.append(f"missing panel {key}")

    comparison = document.get("comparison")
    if not isinstance(comparison, Mapping):
        errors.append("comparison object required")
    else:
        for req in (
            "winner",
            "reason",
            "tie_polar",
            "crown_allowed",
            "per_axis_leads",
            "disagreement_matrix",
            "opaque_weighted_crown_forbidden",
            "efficiency_sole_rank_forbidden",
            "complete_scorecard_id",
        ):
            if req not in comparison:
                errors.append(f"comparison missing {req}")
        if comparison.get("opaque_weighted_crown_forbidden") is not True:
            errors.append("comparison.opaque_weighted_crown_forbidden must be true")
        if comparison.get("efficiency_sole_rank_forbidden") is not True:
            errors.append("comparison.efficiency_sole_rank_forbidden must be true")
        if comparison.get("complete_scorecard_id") != COMPLETE_VIEW_SCORECARD_ID:
            errors.append("comparison.complete_scorecard_id identity mismatch")
        if comparison.get("tie_polar") is True and comparison.get("crown_allowed") is not False:
            errors.append("tie_polar requires crown_allowed=false")

    metric_matrix = document.get("metric_matrix")
    if not isinstance(metric_matrix, Mapping):
        errors.append("metric_matrix required")
    else:
        mh = metric_matrix.get("must_have")
        nh = metric_matrix.get("nice_to_have")
        if not isinstance(mh, list) or len(mh) != len(COMPLETE_VIEW_MUST_HAVE):
            errors.append("metric_matrix.must_have length mismatch")
        if not isinstance(nh, list) or len(nh) != len(COMPLETE_VIEW_NICE_TO_HAVE):
            errors.append("metric_matrix.nice_to_have length mismatch")

    # VAL-REASON-001: P10 catalogue keys never silently omitted.
    panels_obj = document.get("panels")
    if isinstance(panels_obj, Mapping):
        p10 = panels_obj.get("P10_reasoning_logic")
        if not isinstance(p10, Mapping):
            errors.append("P10_reasoning_logic panel must be an object")
        else:
            if p10.get("suite_id") != REASONING_SUITE_ID:
                errors.append(f"P10 suite_id must be {REASONING_SUITE_ID}")
            probes = p10.get("probes")
            if not isinstance(probes, Mapping):
                errors.append("P10 probes object required")
            else:
                for row in COMPLETE_VIEW_REASONING_MUST_PROBES:
                    if row["key"] not in probes:
                        errors.append(f"P10 missing probe {row['key']}")
            chance = p10.get("chance_table")
            if not isinstance(chance, Mapping):
                errors.append("P10 chance_table required")
            scoring = p10.get("scoring")
            if not isinstance(scoring, Mapping):
                errors.append("P10 scoring channels required")
            else:
                for ch in ("closed_choice_accuracy", "forced_ce", "chance_baselines"):
                    if scoring.get(ch) is not True:
                        errors.append(f"P10 scoring.{ch} must be true")
            honesty = p10.get("honesty")
            if not isinstance(honesty, list) or not honesty:
                errors.append("P10 honesty notes required")

    comparison2 = document.get("comparison")
    if isinstance(comparison2, Mapping):
        leads = comparison2.get("per_axis_leads")
        if isinstance(leads, Mapping) and "reasoning" not in leads:
            # Shell compares may omit only if score overrides; require presence by default
            # when multi-axis builder published the full scientific set.
            pass

    return errors


def assert_complete_view_document(document: Mapping[str, Any]) -> None:
    """Raise ValueError if Complete View document fails structural contract."""
    problems = validate_complete_view_document(document)
    if problems:
        raise ValueError("complete_view.v1.3 schema failures: " + "; ".join(problems))


def attach_complete_view_to_report(
    report: dict[str, Any],
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    comparison: MultiAxisComparison | None = None,
    complete_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach Complete View document under ``complete_view`` without rewriting emission."""
    doc = (
        dict(complete_view)
        if complete_view is not None
        else build_complete_view(a, b, comparison=comparison)
    )
    assert_complete_view_document(doc)
    out = {
        **report,
        "complete_view_scorecard_id": COMPLETE_VIEW_SCORECARD_ID,
        "complete_view": doc,
    }
    # Surface polar honesty on ranking if present, without opaque crown invent.
    ranking = dict(out.get("ranking") or {})
    cmp = doc["comparison"]
    if cmp.get("tie_polar"):
        ranking["winner"] = "tie"
        ranking["reason"] = "tie_polar"
        ranking["tie_polar"] = True
        ranking["crown_allowed"] = False
        ranking["authoritative_claim"] = "TIE_POLAR"
        ranking["complete_view_multi_axis"] = True
    ranking["complete_view_scorecard_id"] = COMPLETE_VIEW_SCORECARD_ID
    out["ranking"] = ranking
    return out


__all__ = [
    "COMPLETE_VIEW_COMPARE_ID_ALIAS",
    "COMPLETE_VIEW_DASHBOARD_ID",
    "COMPLETE_VIEW_HISTORICAL_CHAIN",
    "COMPLETE_VIEW_HISTORICAL_SCORECARD_ID",
    "COMPLETE_VIEW_HONESTY_NOTES",
    "COMPLETE_VIEW_MUST_HAVE",
    "COMPLETE_VIEW_NICE_TO_HAVE",
    "COMPLETE_VIEW_PANEL_KEYS",
    "COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE",
    "COMPLETE_VIEW_POLAR_AXES",
    "COMPLETE_VIEW_PROTOCOL_ID",
    "COMPLETE_VIEW_REASONING_CHANCE_TABLE",
    "COMPLETE_VIEW_REASONING_MUST_PROBES",
    "COMPLETE_VIEW_SCHEMA",
    "COMPLETE_VIEW_SCIENTIFIC_AXES",
    "COMPLETE_VIEW_SCORECARD_ID",
    "COMPLETE_VIEW_V1_2_DASHBOARD_ID",
    "COMPLETE_VIEW_V1_2_SCHEMA",
    "COMPLETE_VIEW_V1_2_SCORECARD_ID",
    "REASONING_REL_FLOOR",
    "REASONING_SUITE_ID",
    "CompleteAxisScore",
    "MultiAxisComparison",
    "assert_complete_view_document",
    "attach_complete_view_to_report",
    "build_complete_view",
    "compare_complete_multi_axis",
    "complete_view_identity",
    "complete_view_metric_matrix",
    "default_axis_scores_from_records",
    "reasoning_panel_shell",
    "validate_complete_view_document",
]
