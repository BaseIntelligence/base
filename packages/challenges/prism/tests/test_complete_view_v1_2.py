"""Complete View v1.3 schema + multi-axis comparison contract tests.

Covers VAL-COMPLETE-001 (identity + matrix), VAL-COMPLETE-013 (multi-axis,
TIE_POLAR honesty, no opaque crown), and VAL-COMPLETE-015 (single reconciling
document). Synthetic fixtures only: no NVIDIA, no Lium, no REAL TEE claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_HISTORICAL_SCORECARD_ID,
    COMPLETE_VIEW_MUST_HAVE,
    COMPLETE_VIEW_NICE_TO_HAVE,
    COMPLETE_VIEW_PANEL_KEYS,
    COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE,
    COMPLETE_VIEW_PROTOCOL_ID,
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
    CompleteAxisScore,
    assert_complete_view_document,
    attach_complete_view_to_report,
    build_complete_view,
    compare_complete_multi_axis,
    complete_view_identity,
    complete_view_metric_matrix,
    validate_complete_view_document,
)
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_LONG_CTX_FLOOR,
    OfficialScoreRecord,
)
from prism_challenge.evaluator.official_comparison import (
    SCORECARD_ID as MULTIMETRIC_V1_1,
)


def _rec(
    *,
    label: str,
    bpb: float = 1.5,
    heldout_delta: float | None = 0.5,
    seed_count: int = 3,
    bpb_std: float | None = 0.01,
    heldout_std: float | None = 0.02,
    memorization_flag: bool = False,
    train_heldout_gap: float | None = 0.15,
    step0_anomaly: bool = False,
    valid: bool = True,
    long_ctx_score: float | None = None,
    long_ctx_enabled: bool = False,
    long_ctx_needle: float | None = None,
    long_ctx_mqar: float | None = None,
    long_ctx_induction_copy: float | None = None,
    lag_nll: float | None = None,
    sample_eff_auc: float | None = None,
    sample_eff_marks: tuple[float, ...] | None = None,
    params: int | None = 1_000_000,
    peak_vram_gib: float | None = 0.5,
    tokens_per_s: float | None = 1000.0,
    nan_inf_events: int | None = 0,
    grad_spike_rate: float | None = 0.0,
    val_bpb_trained: float | None = None,
) -> OfficialScoreRecord:
    floor_pass: bool | None = None
    if long_ctx_enabled and long_ctx_score is not None and math.isfinite(long_ctx_score):
        floor_pass = float(long_ctx_score) >= OFFICIAL_LONG_CTX_FLOOR
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        val_bpb_trained=val_bpb_trained,
        memorization_flag=memorization_flag,
        train_heldout_gap=train_heldout_gap,
        step0_anomaly=step0_anomaly,
        valid=valid,
        seed_count=seed_count,
        bpb_std=bpb_std,
        heldout_std=heldout_std,
        long_ctx_score=long_ctx_score,
        long_ctx_needle=long_ctx_needle,
        long_ctx_mqar=long_ctx_mqar,
        long_ctx_induction_copy=long_ctx_induction_copy,
        lag_nll=lag_nll,
        long_ctx_enabled=long_ctx_enabled,
        long_ctx_floor_pass=floor_pass,
        sample_eff_auc=sample_eff_auc,
        sample_eff_marks=sample_eff_marks,
        params=params,
        peak_vram_gib=peak_vram_gib,
        tokens_per_s=tokens_per_s,
        nan_inf_events=nan_inf_events,
        grad_spike_rate=grad_spike_rate,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


# --- VAL-COMPLETE-001: identity + matrix ------------------------------------------


def test_val_complete_001_identity_strings_locked() -> None:
    identity = complete_view_identity()
    assert identity["scorecard_id"] == "multimetric.complete.v1.3"
    assert identity["schema"] == "complete_view.v1.3"
    assert identity["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert identity["schema"] == COMPLETE_VIEW_SCHEMA
    assert identity["protocol_id"] == COMPLETE_VIEW_PROTOCOL_ID
    assert identity["protocol_id"] == "prism_official_compare.v1"
    # Immediate historical is complete.v1.2; multimetric.v1.1 remains in the chain.
    assert identity["historical_scorecard_id"] == COMPLETE_VIEW_HISTORICAL_SCORECARD_ID
    assert identity["historical_scorecard_id"] == "multimetric.complete.v1.2"
    assert identity["multimetric_v1_1_scorecard_id"] == MULTIMETRIC_V1_1
    assert identity["multimetric_v1_1_scorecard_id"] == "multimetric.v1.1"
    assert identity["historical_chain"] == [
        "multimetric.v1.1",
        "multimetric.complete.v1.2",
    ]
    assert identity["dashboard_id"] == "scorecard_complete_view.v1.3"
    assert identity["non_claims"]["prism_tee_product"] is False
    assert identity["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert identity["non_claims"]["opaque_weighted_sole_crown"] is False
    assert identity["non_claims"]["emission_crown"] is False
    assert identity["non_claims"]["human_agi_reasoning"] is False
    assert identity["non_claims"]["gsm8k_mmlu_primary"] is False
    assert identity["non_claims"]["seed_scale_logic_is_lab_only"] is True
    assert "complete_view.v1.3" in COMPLETE_VIEW_SCHEMA
    assert "P10_reasoning_logic" in identity["panel_keys"]
    assert "reasoning" in identity["scientific_axes"]
    assert "reasoning" in identity["polar_axes"]
    assert set(COMPLETE_VIEW_PANEL_KEYS) == set(identity["panel_keys"])


def test_val_complete_001_metric_matrix_must_have_and_nice_to_have() -> None:
    matrix = complete_view_metric_matrix()
    assert matrix["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    must_ids = {row["matrix_id"] for row in matrix["must_have"]}
    nice_ids = {row["matrix_id"] for row in matrix["nice_to_have"]}
    assert must_ids == {row["matrix_id"] for row in COMPLETE_VIEW_MUST_HAVE}
    assert nice_ids == {row["matrix_id"] for row in COMPLETE_VIEW_NICE_TO_HAVE}
    # Core MAX complete keys from gap research present.
    must_keys = {row["key"] for row in matrix["must_have"]}
    for key in (
        "val_bpb_trained",
        "long_ctx_multi_T",
        "needle_by_depth",
        "mqar_grid",
        "induction_and_copy_unfused",
        "lag_nll_bins",
        "length_extrapolate_ce",
        "sample_eff_dense",
        "peak_vram_train_eval",
        "state_footprint",
        "grad_spike_nan",
        "multi_order_delta",
        "quality_per_param_gib",
    ):
        assert key in must_keys
    # Every must-have maps to a VAL-COMPLETE or VAL-REASON family id on the row.
    for row in matrix["must_have"]:
        if "val_complete" in row:
            assert str(row["val_complete"]).startswith("VAL-COMPLETE-")
        else:
            assert str(row["val_reason"]).startswith("VAL-REASON-")
        assert row["panel"] in COMPLETE_VIEW_PANEL_KEYS
    # Panel → VAL-COMPLETE mapping is complete for all panels.
    for panel in COMPLETE_VIEW_PANEL_KEYS:
        assert panel in COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE
        assert COMPLETE_VIEW_PANEL_TO_VAL_COMPLETE[panel]


def test_val_complete_001_document_schema_validates_shell() -> None:
    a = _rec(label="transformer-tiny-1m", heldout_delta=0.40, bpb=1.2)
    b = _rec(label="mamba-tiny-1m", heldout_delta=0.55, bpb=1.1)
    doc = build_complete_view(a, b, score_class="fixture")
    assert doc["schema"] == COMPLETE_VIEW_SCHEMA
    assert doc["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    problems = validate_complete_view_document(doc)
    assert problems == []
    assert_complete_view_document(doc)
    # All panels present; nice-to-have residual not silently omitted.
    for key in COMPLETE_VIEW_PANEL_KEYS:
        assert key in doc["panels"]
    nice = doc["panels"]["P8_calibration_entropy_optional"]
    assert nice["status"] == "nice_to_have"
    assert len(nice["entries"]) == len(COMPLETE_VIEW_NICE_TO_HAVE)
    assert all(e.get("reason") for e in nice["entries"])
    # Relation to historical complete.v1.2 + multimetric.v1.1 preserved.
    assert doc["historical_scorecard_id"] == "multimetric.complete.v1.2"
    assert doc["relation_to_complete_v1_2"]["historical_preserved"] is True
    assert doc["relation_to_multimetric_v1_1"]["historical_preserved"] is True
    assert "P10_reasoning_logic" in doc["panels"]
    assert "real_provider_tee" not in doc
    assert doc["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert doc["non_claims"]["prism_tee_product"] is False
    assert doc["non_claims"]["opaque_weighted_sole_crown"] is False
    assert doc["non_claims"]["emission_weight_crown"] is False


# --- VAL-COMPLETE-013: multi-axis comparison + polar honesty ----------------------


def test_val_complete_013_no_opaque_crown_and_multi_axis_keys() -> None:
    a = _rec(
        label="A",
        heldout_delta=0.9,
        bpb=1.4,
        long_ctx_enabled=True,
        long_ctx_score=0.35,
        sample_eff_auc=0.50,
    )
    b = _rec(
        label="B",
        heldout_delta=0.2,
        bpb=1.5,
        long_ctx_enabled=True,
        long_ctx_score=0.30,
        sample_eff_auc=0.48,
    )
    cmp = compare_complete_multi_axis(a, b)
    payload = cmp.as_dict()
    assert payload["opaque_weighted_crown_forbidden"] is True
    assert payload["efficiency_sole_rank_forbidden"] is True
    assert "per_axis_leads" in payload
    assert "disagreement_matrix" in payload
    assert payload["complete_scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert payload["base_scorecard_id"] == MULTIMETRIC_V1_1
    # No sole weighted scalar crown key is authoritative output.
    assert "weighted_sole_score" not in payload
    assert "opaque_crown" not in payload
    assert set(payload["per_axis_leads"]) >= {
        "short_gen",
        "long_ctx",
        "sample_eff",
        "reasoning",
    }


def test_val_complete_013_tie_polar_when_short_and_long_axes_conflict() -> None:
    """Classic multimetric polar: short-gen favors B, long-ctx favors A → TIE_POLAR."""
    a = _rec(
        label="transformer",
        heldout_delta=0.30,  # worse short
        bpb=1.3,
        long_ctx_enabled=True,
        long_ctx_score=0.40,  # better long (above floor)
        long_ctx_needle=0.45,
        long_ctx_mqar=0.20,
        sample_eff_auc=0.54,
    )
    b = _rec(
        label="mamba",
        heldout_delta=0.90,  # better short
        bpb=1.1,
        long_ctx_enabled=True,
        long_ctx_score=0.18,  # worse long
        long_ctx_needle=0.20,
        long_ctx_mqar=0.08,
        sample_eff_auc=0.56,
    )
    cmp = compare_complete_multi_axis(a, b)
    assert cmp.tie_polar is True
    assert cmp.crown_allowed is False
    assert cmp.winner == "tie"
    assert cmp.reason == "tie_polar"
    assert cmp.per_axis_leads["short_gen"] == "b"
    assert cmp.per_axis_leads["long_ctx"] == "a"
    assert cmp.disagreement_matrix["short_gen"]["long_ctx"] is True
    assert cmp.as_dict()["authoritative_claim"] == "TIE_POLAR"


def test_val_complete_013_expanded_polar_on_sample_eff_vs_short_gen() -> None:
    """Expanded Complete View polar: short_gen vs sample_eff scientific disagreement."""
    a = _rec(
        label="A",
        heldout_delta=0.95,  # A better short
        bpb=1.2,
        long_ctx_enabled=False,
        sample_eff_auc=0.40,  # worse sample-eff
    )
    b = _rec(
        label="B",
        heldout_delta=0.20,
        bpb=1.3,
        long_ctx_enabled=False,
        sample_eff_auc=0.80,  # B better sample-eff
    )
    cmp = compare_complete_multi_axis(a, b)
    assert cmp.per_axis_leads["short_gen"] == "a"
    assert cmp.per_axis_leads["sample_eff"] == "b"
    assert cmp.disagreement_matrix["short_gen"]["sample_eff"] is True
    assert cmp.tie_polar is True
    assert cmp.crown_allowed is False


def test_val_complete_013_preserves_v1_when_no_scientific_polar() -> None:
    a = _rec(
        label="A",
        heldout_delta=0.90,
        bpb=1.1,
        long_ctx_enabled=True,
        long_ctx_score=0.40,
        sample_eff_auc=0.60,
    )
    b = _rec(
        label="B",
        heldout_delta=0.20,
        bpb=1.4,
        long_ctx_enabled=True,
        long_ctx_score=0.30,
        sample_eff_auc=0.50,
    )
    cmp = compare_complete_multi_axis(a, b)
    assert cmp.tie_polar is False
    assert cmp.crown_allowed is True
    assert cmp.winner == "a"
    assert cmp.per_axis_leads["short_gen"] == "a"
    assert cmp.per_axis_leads["long_ctx"] == "a"


def test_val_complete_013_axis_scores_override_length_extrap_polar() -> None:
    a = _rec(label="A", heldout_delta=0.8, bpb=1.2)
    b = _rec(label="B", heldout_delta=0.3, bpb=1.3)
    axis_scores = {
        "short_gen": (
            CompleteAxisScore(0.8, "heldout_delta", "higher"),
            CompleteAxisScore(0.3, "heldout_delta", "higher"),
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
            # lower better: A worse at length extrap
            CompleteAxisScore(1.50, "ratio", "lower"),
            CompleteAxisScore(1.05, "ratio", "lower"),
        ),
    }
    cmp = compare_complete_multi_axis(a, b, axis_scores=axis_scores)
    assert cmp.per_axis_leads["short_gen"] == "a"
    assert cmp.per_axis_leads["length_extrap"] == "b"
    assert cmp.tie_polar is True
    assert cmp.crown_allowed is False


# --- VAL-COMPLETE-015: single machine JSON ---------------------------------------


def test_val_complete_015_single_document_reconciles_panels_and_comparison(
    tmp_path: Path,
) -> None:
    a = _rec(
        label="transformer-tiny-1m",
        heldout_delta=3.46,
        bpb=0.122,
        params=6_963_840,
        long_ctx_enabled=True,
        long_ctx_score=0.134,
        sample_eff_auc=0.544,
    )
    b = _rec(
        label="mamba-tiny-1m",
        heldout_delta=4.70,
        bpb=0.118,
        params=6_672_256,
        long_ctx_enabled=True,
        long_ctx_score=0.093,
        sample_eff_auc=0.554,
    )
    doc = build_complete_view(
        a,
        b,
        score_class="LAB-GPU",
        pin={
            "token_budget": 500_000,
            "seq_len_train": 128,
            "seeds": [1337, 2027, 4242],
            "shared_lr": 0.004,
        },
    )
    assert_complete_view_document(doc)
    assert set(doc["panels"]) == set(COMPLETE_VIEW_PANEL_KEYS)
    assert "comparison" in doc
    assert doc["comparison"]["opaque_weighted_crown_forbidden"] is True
    # Machine-only write path: serialize as complete_view.v1.3.json
    out = tmp_path / "complete_view.v1.3.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["schema"] == "complete_view.v1.3"
    assert reloaded["scorecard_id"] == "multimetric.complete.v1.3"
    assert reloaded["historical_scorecard_id"] == "multimetric.complete.v1.2"
    assert validate_complete_view_document(reloaded) == []
    # Expected polar from known A long better / B short better pattern.
    assert reloaded["comparison"]["tie_polar"] is True
    assert reloaded["comparison"]["crown_allowed"] is False


def test_val_complete_015_attach_to_report_uses_provider_trust_labels() -> None:
    a = _rec(label="A", heldout_delta=0.5)
    b = _rec(label="B", heldout_delta=0.4)
    report = {
        "schema": "prism_compare_report.v1",
        "protocol_id": COMPLETE_VIEW_PROTOCOL_ID,
        "ranking": {"winner": "a", "reason": "primary_heldout"},
        "score_class": "fixture",
        "labels": {"provider_trust": "PROVIDER_TRUST", "prism_tee_product": False},
    }
    out = attach_complete_view_to_report(report, a, b)
    assert out["complete_view_scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert out["complete_view"]["schema"] == COMPLETE_VIEW_SCHEMA
    assert "real_provider_tee" not in out["complete_view"]
    assert out["complete_view"]["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert out["complete_view"]["non_claims"]["prism_tee_product"] is False
    # Historical report still carries its own fields; complete_view is additive.
    assert out["schema"] == "prism_compare_report.v1"


def test_validate_rejects_opaque_crown_and_wrong_identity() -> None:
    a = _rec(label="A")
    b = _rec(label="B")
    doc = build_complete_view(a, b)
    bad = dict(doc)
    bad["scorecard_id"] = "not-complete"
    assert any("scorecard_id" in e for e in validate_complete_view_document(bad))

    bad2 = dict(doc)
    cmp = dict(doc["comparison"])
    cmp["opaque_weighted_crown_forbidden"] = False
    bad2["comparison"] = cmp
    assert any("opaque_weighted" in e for e in validate_complete_view_document(bad2))

    bad3 = dict(doc)
    nc = dict(doc["non_claims"])
    nc["prism_tee_product"] = True
    bad3["non_claims"] = nc
    assert any("prism_tee_product" in e for e in validate_complete_view_document(bad3))
    # Retired key must NOT be a schema gate even if someone stashes PASS on it.
    bad4 = dict(doc)
    bad4["real_provider_tee"] = "REAL-PROVIDER PASS"
    assert validate_complete_view_document(bad4) == []


def test_unknown_panel_override_rejected() -> None:
    a = _rec(label="A")
    b = _rec(label="B")
    with pytest.raises(ValueError, match="unknown complete_view panel"):
        build_complete_view(a, b, panels_override={"P_invented": {}})
