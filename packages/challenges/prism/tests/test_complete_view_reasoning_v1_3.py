"""Complete View v1.3 reasoning/logic schema + honesty contract tests.

Locks VAL-REASON-001 (identity, P10 catalogue, dual scoring, chance baselines,
non-claims) and VAL-REASON-012 (seed-scale honesty language in protocol docs and
machine honesty notes). Synthetic fixtures only; no real probe fill, no GPU,
no Lium, no REAL TEE claim.
"""

from __future__ import annotations

from pathlib import Path

from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_HISTORICAL_CHAIN,
    COMPLETE_VIEW_HISTORICAL_SCORECARD_ID,
    COMPLETE_VIEW_MUST_HAVE,
    COMPLETE_VIEW_NICE_TO_HAVE,
    COMPLETE_VIEW_PANEL_KEYS,
    COMPLETE_VIEW_POLAR_AXES,
    COMPLETE_VIEW_REASONING_CHANCE_TABLE,
    COMPLETE_VIEW_REASONING_MUST_PROBES,
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCIENTIFIC_AXES,
    COMPLETE_VIEW_SCORECARD_ID,
    COMPLETE_VIEW_V1_2_SCHEMA,
    COMPLETE_VIEW_V1_2_SCORECARD_ID,
    REASONING_REL_FLOOR,
    REASONING_SUITE_ID,
    CompleteAxisScore,
    assert_complete_view_document,
    build_complete_view,
    compare_complete_multi_axis,
    complete_view_identity,
    complete_view_metric_matrix,
    reasoning_panel_shell,
    validate_complete_view_document,
)
from prism_challenge.evaluator.official_comparison import OfficialScoreRecord


def _rec(
    *,
    label: str,
    heldout_delta: float = 0.5,
    bpb: float = 1.2,
    sample_eff_auc: float | None = None,
    long_ctx_enabled: bool = False,
    long_ctx_score: float | None = None,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        valid=True,
        seed_count=3,
        bpb_std=0.01,
        heldout_std=0.02,
        sample_eff_auc=sample_eff_auc,
        long_ctx_enabled=long_ctx_enabled,
        long_ctx_score=long_ctx_score,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


# --- VAL-REASON-001: identity + P10 schema ---------------------------------------


def test_val_reason_001_scorecard_and_schema_identity() -> None:
    identity = complete_view_identity()
    assert COMPLETE_VIEW_SCORECARD_ID == "multimetric.complete.v1.3"
    assert COMPLETE_VIEW_SCHEMA == "complete_view.v1.3"
    assert identity["scorecard_id"] == "multimetric.complete.v1.3"
    assert identity["schema"] == "complete_view.v1.3"
    assert identity["dashboard_id"] == "scorecard_complete_view.v1.3"
    assert identity["protocol_id"] == "prism_official_compare.v1"
    # Preserve complete.v1.2 history (not rewrite).
    assert COMPLETE_VIEW_HISTORICAL_SCORECARD_ID == COMPLETE_VIEW_V1_2_SCORECARD_ID
    assert COMPLETE_VIEW_V1_2_SCORECARD_ID == "multimetric.complete.v1.2"
    assert COMPLETE_VIEW_V1_2_SCHEMA == "complete_view.v1.2"
    assert identity["historical_scorecard_id"] == "multimetric.complete.v1.2"
    assert COMPLETE_VIEW_HISTORICAL_CHAIN == (
        "multimetric.v1.1",
        "multimetric.complete.v1.2",
    )
    assert identity["historical_chain"] == list(COMPLETE_VIEW_HISTORICAL_CHAIN)
    assert "P10_reasoning_logic" in COMPLETE_VIEW_PANEL_KEYS
    assert "P10_reasoning_logic" in identity["panel_keys"]


def test_val_reason_001_synthetic_probe_catalogue_and_chance_table() -> None:
    must_keys = {row["key"] for row in COMPLETE_VIEW_REASONING_MUST_PROBES}
    expected = {
        "boolean_parity_xor",
        "arith_digit_mod",
        "transitive_compare",
        "multihop_binding",
        "sort_order",
        "reverse_edit",
        "count_stream",
        "dyck_nesting",
        "instruction_toy",
        "contradiction_detect",
    }
    assert must_keys == expected
    assert set(COMPLETE_VIEW_REASONING_CHANCE_TABLE) == expected
    assert COMPLETE_VIEW_REASONING_CHANCE_TABLE["boolean_parity_xor"] == 0.5
    assert COMPLETE_VIEW_REASONING_CHANCE_TABLE["arith_digit_mod"] == 0.1
    assert COMPLETE_VIEW_REASONING_CHANCE_TABLE["contradiction_detect"] == 0.5
    # Matrix sell-through: each probe is a must-have row on P10 with val_reason.
    p10_must = [
        row
        for row in COMPLETE_VIEW_MUST_HAVE
        if row["panel"] == "P10_reasoning_logic" and row["key"] != "logic_suite_mean"
    ]
    assert {row["key"] for row in p10_must} == expected
    for row in p10_must:
        assert str(row["val_reason"]).startswith("VAL-REASON-")
        assert "chance" in row
    assert REASONING_SUITE_ID == "logic_synthetic.v1"
    assert REASONING_REL_FLOOR == 0.05

    matrix = complete_view_metric_matrix()
    assert matrix["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert set(matrix["reasoning_chance_table"]) == expected
    # Nice-to-have reasoning residuals never silently omitted from catalogue.
    nice_keys = {row["key"] for row in COMPLETE_VIEW_NICE_TO_HAVE}
    assert "logic_ece" in nice_keys
    assert "cot_free_gen_collapse" in nice_keys


def test_val_reason_001_p10_shell_dual_scoring_and_document() -> None:
    shell = reasoning_panel_shell()
    assert shell["status"] == "not_run"
    assert shell["suite_id"] == REASONING_SUITE_ID
    assert shell["scoring"]["closed_choice_accuracy"] is True
    assert shell["scoring"]["forced_ce"] is True
    assert shell["scoring"]["chance_baselines"] is True
    assert set(shell["probes"]) == set(COMPLETE_VIEW_REASONING_CHANCE_TABLE)
    assert shell["aggregates"]["suite_mean"] == {"a": None, "b": None}
    # Non-primary external batteries must be called out.
    assert "gsm8k" in shell["distinct_from"]["gsm8k_mmlu_lm_eval"].lower()
    assert "mqar" in shell["distinct_from"]["P3_long_ctx_mqar"].lower()

    a = _rec(label="transformer-tiny-1m", heldout_delta=0.4)
    b = _rec(label="mamba-tiny-1m", heldout_delta=0.55)
    doc = build_complete_view(a, b, score_class="fixture")
    assert doc["schema"] == "complete_view.v1.3"
    assert doc["scorecard_id"] == "multimetric.complete.v1.3"
    assert doc["historical_scorecard_id"] == "multimetric.complete.v1.2"
    assert "P10_reasoning_logic" in doc["panels"]
    p10 = doc["panels"]["P10_reasoning_logic"]
    assert p10["status"] == "not_run"
    assert p10["chance_table"] == dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE)
    assert p10["scoring"]["closed_choice_accuracy"] is True
    assert p10["scoring"]["forced_ce"] is True
    assert "real_provider_tee" not in doc
    assert doc["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert doc["non_claims"]["prism_tee_product"] is False
    assert doc["non_claims"]["emission_weight_crown"] is False
    assert doc["non_claims"]["human_agi_reasoning"] is False
    assert doc["non_claims"]["gsm8k_mmlu_primary"] is False
    assert doc["non_claims"]["seed_scale_logic_is_lab_only"] is True
    problems = validate_complete_view_document(doc)
    assert problems == []
    assert_complete_view_document(doc)


def test_val_reason_001_multi_axis_includes_reasoning_and_polar_vs_short_gen() -> None:
    assert "reasoning" in COMPLETE_VIEW_SCIENTIFIC_AXES
    assert "reasoning" in COMPLETE_VIEW_POLAR_AXES

    a = _rec(label="A", heldout_delta=0.95)  # A better short_gen
    b = _rec(label="B", heldout_delta=0.20)
    axis_scores = {
        "short_gen": (
            CompleteAxisScore(0.95, "heldout_delta", "higher"),
            CompleteAxisScore(0.20, "heldout_delta", "higher"),
        ),
        "long_ctx": (
            CompleteAxisScore(None, "long_ctx", "higher", "not_run"),
            CompleteAxisScore(None, "long_ctx", "higher", "not_run"),
        ),
        "sample_eff": (
            CompleteAxisScore(None, "auc", "higher", "not_run"),
            CompleteAxisScore(None, "auc", "higher", "not_run"),
        ),
        "length_extrap": (
            CompleteAxisScore(None, "ratio", "lower", "not_run"),
            CompleteAxisScore(None, "ratio", "lower", "not_run"),
        ),
        "reasoning": (
            # higher better: B better on logic relative_to_chance → polar vs short_gen A
            CompleteAxisScore(0.02, "logic_rel_macro", "higher"),
            CompleteAxisScore(0.40, "logic_rel_macro", "higher"),
        ),
    }
    cmp = compare_complete_multi_axis(a, b, axis_scores=axis_scores)
    assert cmp.per_axis_leads["short_gen"] == "a"
    assert cmp.per_axis_leads["reasoning"] == "b"
    assert cmp.disagreement_matrix["short_gen"]["reasoning"] is True
    assert cmp.tie_polar is True
    assert cmp.crown_allowed is False
    assert "reasoning" in cmp.polar_axes_involved
    assert cmp.as_dict()["complete_scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID

    # Default shell (not filled) leaves reasoning missing without inventing polar.
    cmp_shell = compare_complete_multi_axis(a, b)
    assert cmp_shell.per_axis_leads.get("reasoning") == "missing"
    assert doc_lead_ok(cmp_shell)


def doc_lead_ok(cmp) -> bool:
    return "reasoning" in cmp.per_axis_leads


# --- VAL-REASON-012: honesty language --------------------------------------------


def test_val_reason_012_machine_honesty_notes_seed_scale_only() -> None:
    identity = complete_view_identity()
    joined = " ".join(identity["honesty_notes"]).lower()
    assert "seed-scale" in joined or "seed scale" in joined
    assert "gsm8k" in joined or "mmlu" in joined
    assert "human" in joined or "agi" in joined
    assert identity["non_claims"]["seed_scale_logic_is_lab_only"] is True
    assert identity["non_claims"]["human_agi_reasoning"] is False
    assert identity["non_claims"]["gsm8k_mmlu_primary"] is False

    a = _rec(label="A")
    b = _rec(label="B")
    doc = build_complete_view(a, b)
    p10_honesty = " ".join(doc["panels"]["P10_reasoning_logic"]["honesty"]).lower()
    assert "synthetic" in p10_honesty
    assert "gsm8k" in p10_honesty or "mmlu" in p10_honesty
    assert "agi" in p10_honesty or "human" in p10_honesty
    assert "lab" in p10_honesty or "diagnostic" in p10_honesty
    assert "emission" in p10_honesty or "crown" in p10_honesty
    # Protocol-level notes further insist lab-only.
    doc_honesty = " ".join(doc["honesty"]).lower()
    assert "lab" in doc_honesty or "architecture comparison" in doc_honesty
    assert "gsm8k" in doc_honesty or "mmlu" in doc_honesty


def test_val_reason_012_docs_state_seed_scale_lab_only_not_human_agi() -> None:
    protocol = Path("docs/official-comparison.md").read_text(encoding="utf-8")
    operators = Path("docs/operators.md").read_text(encoding="utf-8")
    lower = protocol.lower()
    assert "complete_view.v1.3" in protocol
    assert "multimetric.complete.v1.3" in protocol
    assert "p10_reasoning_logic" in lower
    assert "multimetric.complete.v1.2" in protocol  # history preserved
    assert "reasoning" in lower
    # Honesty: not human AGI / not GSM8K primary / lab comparison only.
    assert "gsm8k" in lower or "mmlu" in lower
    assert "human" in lower or "agi" in lower
    assert "lab" in lower or "diagnostic" in lower or "architecture comparison" in lower
    assert "seed-scale" in lower or "seed scale" in lower or "~7m" in lower
    # Operators pointer updated.
    op_lower = operators.lower()
    assert "complete_view.v1.3" in operators or "multimetric.complete.v1.3" in operators
    assert "p10" in op_lower or "reasoning" in op_lower
    assert "blocked" in op_lower


def test_val_reason_001_missing_p10_probe_fails_validation() -> None:
    a = _rec(label="A")
    b = _rec(label="B")
    doc = build_complete_view(a, b)
    p10 = dict(doc["panels"]["P10_reasoning_logic"])
    probes = dict(p10["probes"])
    del probes["boolean_parity_xor"]
    p10["probes"] = probes
    panels = dict(doc["panels"])
    panels["P10_reasoning_logic"] = p10
    bad = dict(doc)
    bad["panels"] = panels
    problems = validate_complete_view_document(bad)
    assert any("boolean_parity_xor" in e for e in problems)


def test_validate_prefers_structural_honesty_non_claims_flags() -> None:
    """human_agi / gsm8k primary / seed-scale lab flags are hard structural locks."""
    a = _rec(label="A")
    b = _rec(label="B")
    doc = build_complete_view(a, b)
    assert_complete_view_document(doc)

    for flag, bad_val, needle in (
        ("human_agi_reasoning", True, "human_agi_reasoning"),
        ("gsm8k_mmlu_primary", True, "gsm8k_mmlu_primary"),
        ("seed_scale_logic_is_lab_only", False, "seed_scale_logic_is_lab_only"),
    ):
        broken = dict(doc)
        nc = dict(doc["non_claims"])
        nc[flag] = bad_val
        broken["non_claims"] = nc
        problems = validate_complete_view_document(broken)
        assert any(needle in e for e in problems), (flag, problems)

    # Omitting any of the three is also rejected (prefer presence over defaults).
    missing = dict(doc)
    nc2 = dict(doc["non_claims"])
    del nc2["seed_scale_logic_is_lab_only"]
    missing["non_claims"] = nc2
    problems2 = validate_complete_view_document(missing)
    assert any("seed_scale_logic_is_lab_only" in e for e in problems2)


def test_docs_typo_near_chance_on_both_sides() -> None:
    protocol = Path("docs/official-comparison.md").read_text(encoding="utf-8")
    assert "Near-chance on both-sides" in protocol
    assert "Near-chance us both-sides" not in protocol
