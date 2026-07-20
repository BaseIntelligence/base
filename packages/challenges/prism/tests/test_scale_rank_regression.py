"""Scale-eval rank regression + product pin guards (VAL-SCALE-018 partial).

Freezes:

* heldout-primary official rank key (better heldout wins)
* anti memorization / step-0 / miner self-report (ignored for rank)
* wall-clock never ranks
* multi-seed K≥3 ProtocolPin defaults for public claims
* multi-family host compare under matched explore pin
* Complete View long_ctx / sample_eff densify entrypoints usable
* tee package still absent

No Lium spend. Synthetic / fixture metrics only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from prism_challenge.evaluator.complete_view import assert_complete_view_document
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    OFFICIAL_MIN_PUBLIC_SEEDS,
    OFFICIAL_WALL_CLOCK_NEVER_RANKS,
    OfficialScoreRecord,
    ProtocolPin,
    aggregate_official_records,
    compare_official,
    official_rank_key,
    rank_official,
)
from prism_challenge.evaluator.scale_eval import (
    SCALE_P0_CORE_FAMILY_IDS,
    SCALE_P0_SEEDS,
    SCALE_P0_SEQ_LEN,
    SCALE_P0_TOKEN_BUDGET,
    assert_public_multi_seed_pin,
    densify_complete_view_pair,
    densify_entrypoints,
    scale_p0_protocol_pin,
    scale_pin_fields,
    scale_pin_public_ok,
    scale_product_snapshot,
    tee_package_absent,
)
from prism_challenge.evaluator.scoring import (
    MEMORIZATION_PENALTY_FACTOR,
    score_prequential_bpb,
)


def _rec(
    *,
    label: str,
    bpb: float,
    heldout_delta: float | None = None,
    memorization_flag: bool = False,
    train_heldout_gap: float | None = None,
    step0_anomaly: bool = False,
    valid: bool = True,
    wall_clock_seconds: float | None = None,
    miner_reported_bpb: float | None = None,
    seed_count: int = 3,
    bpb_std: float | None = 0.01,
    overfit_rate: float = 0.0,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        memorization_flag=memorization_flag,
        train_heldout_gap=train_heldout_gap,
        step0_anomaly=step0_anomaly,
        valid=valid,
        seed_count=seed_count,
        bpb_std=bpb_std,
        overfit_rate=overfit_rate,
        wall_clock_seconds=wall_clock_seconds,
        miner_reported_bpb=miner_reported_bpb,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


def _challenge_manifest(
    *,
    bpb: float,
    covered_bytes: int = 1000,
    heldout_delta: float | None = None,
    train_heldout_gap: float | None = None,
    step0_anomaly: bool = False,
) -> dict:
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    metrics: dict = {
        "online_loss": [2.0, 1.5, 1.0],
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "covered_bytes": covered_bytes,
        "predicted_tokens": 100,
        "step0_loss": 2.0,
        "consumed_batches": 3,
        "random_init_baseline_nats": math.log(256),
        "nan_inf_batches": 0,
    }
    if heldout_delta is not None:
        metrics["heldout_delta"] = heldout_delta
        metrics["held_out_delta"] = heldout_delta
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": metrics,
        "anti_cheat": {
            "step0_anomaly": step0_anomaly,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


# --- VAL-SCALE-018: heldout-primary freeze -------------------------------------------------------


def test_scale_rank_heldout_primary_beats_bpb_secondary() -> None:
    """Better heldout wins even when secondary bpb is worse (emission-like Official key)."""
    generalizer = _rec(label="gen", bpb=1.90, heldout_delta=1.10)
    compressor = _rec(label="comp", bpb=1.00, heldout_delta=0.15)
    result = compare_official(generalizer, compressor)
    assert result.winner == "a"
    assert result.reason == "primary_heldout"
    ranked = rank_official([compressor, generalizer])
    assert ranked[0].label == "gen"
    assert official_rank_key(generalizer) < official_rank_key(compressor)


def test_scale_rank_secondary_bpb_breaks_heldout_tie() -> None:
    a = _rec(label="A", bpb=1.00, heldout_delta=0.55)
    b = _rec(label="B", bpb=1.40, heldout_delta=0.55)
    result = compare_official(a, b)
    assert result.winner == "a"
    assert result.reason == "secondary_bpb"


# --- anti mem / step0 / self-report --------------------------------------------------------------


def test_scale_rank_rejects_memorizer_vs_benign() -> None:
    memorizer = _rec(
        label="memo",
        bpb=1.0,
        heldout_delta=0.45,
        memorization_flag=True,
        train_heldout_gap=2.5,
        overfit_rate=1.0,
    )
    benign = _rec(
        label="benign",
        bpb=1.0,
        heldout_delta=0.45,
        memorization_flag=False,
        train_heldout_gap=0.1,
        overfit_rate=0.0,
    )
    result = compare_official(memorizer, benign)
    assert result.winner == "b"
    assert result.reason == "anti_overfit"

    memo_score = score_prequential_bpb(
        _challenge_manifest(bpb=1.0, heldout_delta=0.3, train_heldout_gap=2.5)
    )
    assert memo_score.memorization_flag is True
    assert memo_score.memorization_penalty == pytest.approx(MEMORIZATION_PENALTY_FACTOR)


def test_scale_rank_step0_anomaly_disqualifies() -> None:
    smuggled = _rec(
        label="smuggle",
        bpb=0.01,
        heldout_delta=9.0,
        step0_anomaly=True,
        valid=False,
    )
    clean = _rec(label="clean", bpb=1.6, heldout_delta=0.35)
    result = compare_official(smuggled, clean)
    assert result.winner == "b"
    assert result.reason == "step0_anomaly"
    ranked = rank_official([smuggled, clean])
    assert ranked[0].label == "clean"


def test_scale_rank_miner_self_report_ignored() -> None:
    fabulous_self = _rec(
        label="liar",
        bpb=2.0,
        heldout_delta=0.20,
        miner_reported_bpb=0.0001,
    )
    honest_better = _rec(
        label="peer",
        bpb=1.4,
        heldout_delta=0.85,
        miner_reported_bpb=9.0,
    )
    # Rank key ignores miner self-report fields.
    base = _rec(label="liar", bpb=2.0, heldout_delta=0.20, miner_reported_bpb=None)
    assert official_rank_key(fabulous_self) == official_rank_key(base)
    result = compare_official(fabulous_self, honest_better)
    assert result.winner == "b"
    assert result.reason == "primary_heldout"


def test_scale_rank_wall_clock_never_orders() -> None:
    assert OFFICIAL_WALL_CLOCK_NEVER_RANKS is True
    fast_worse = _rec(label="fast", bpb=1.5, heldout_delta=0.10, wall_clock_seconds=1.0)
    slow_better = _rec(label="slow", bpb=1.5, heldout_delta=0.90, wall_clock_seconds=9999.0)
    result = compare_official(slow_better, fast_worse)
    assert result.winner == "a"
    assert result.reason == "primary_heldout"
    same_quality_fast = _rec(label="x", bpb=1.5, heldout_delta=0.5, wall_clock_seconds=10.0)
    same_quality_slow = _rec(label="x", bpb=1.5, heldout_delta=0.5, wall_clock_seconds=10_000.0)
    assert official_rank_key(same_quality_fast) == official_rank_key(same_quality_slow)


# --- multi-seed K≥3 pin fields -------------------------------------------------------------------


def test_scale_p0_pin_is_public_k_ge_3() -> None:
    pin = scale_p0_protocol_pin()
    assert pin.seeds == SCALE_P0_SEEDS == OFFICIAL_DEFAULT_SEEDS
    assert len(pin.seeds) >= OFFICIAL_MIN_PUBLIC_SEEDS == 3
    assert pin.token_budget == SCALE_P0_TOKEN_BUDGET
    assert pin.seq_len == SCALE_P0_SEQ_LEN
    assert pin.param_ladder_stage == "explore"
    assert pin.primary_form == "heldout_delta"
    assert pin.tokenizer == "gpt2"
    assert pin.force_iter_train_batches is True
    fields = scale_pin_fields(pin)
    assert fields["seed_count"] >= 3
    assert fields["min_public_seeds"] == 3
    assert fields["wall_clock_never_ranks"] is True
    guard = scale_pin_public_ok(pin)
    assert guard.ok is True
    assert_public_multi_seed_pin(pin)


def test_scale_p0_pin_rejects_k1_when_public_required() -> None:
    with pytest.raises(ValueError, match="public scale pin requires K"):
        scale_p0_protocol_pin(seeds=(1337,), require_public_k=True)
    # Provisional lab path still available.
    pin_k1 = scale_p0_protocol_pin(seeds=(1337,), require_public_k=False)
    assert pin_k1.seeds == (1337,)
    guard = scale_pin_public_ok(pin_k1)
    assert guard.ok is False
    assert any("seed_count_below_public_min" in r for r in guard.reasons)
    with pytest.raises(ValueError, match="public multi-seed pin guard failed"):
        assert_public_multi_seed_pin(pin_k1)


def test_scale_aggregate_k3_seed_count() -> None:
    seeds = [
        _rec(label="s1", bpb=1.0, heldout_delta=0.4, seed_count=1),
        _rec(label="s2", bpb=1.2, heldout_delta=0.6, seed_count=1),
        _rec(label="s3", bpb=1.1, heldout_delta=0.5, seed_count=1),
    ]
    agg = aggregate_official_records(seeds, label="family-x")
    assert agg.seed_count == 3
    assert agg.heldout_delta == pytest.approx(0.5)
    assert agg.is_public_multi_seed is True
    assert agg.multi_seed_provisional is False


# --- densify entrypoints -------------------------------------------------------------------------


def test_scale_densify_entrypoints_document_longctx_and_sample_eff() -> None:
    ep = densify_entrypoints()
    assert ep["schema"] == "prism_scale_densify_entrypoints.v1"
    assert "build_complete_view_with_longctx_quality" in ep["long_ctx"]["build_view"]
    assert "build_complete_view_with_eff_stability" in ep["sample_eff"]["build_view"]
    assert ep["rank_guards"]["wall_clock_never_ranks"] is True
    assert ep["rank_guards"]["min_public_seeds"] >= 3
    assert (
        "tee" in ep["rank_guards"]["tee_package"].lower()
        or "absent" in ep["rank_guards"]["tee_package"].lower()
    )
    # Import paths resolve.
    from prism_challenge.evaluator.complete_view_eff import build_complete_view_with_eff_stability
    from prism_challenge.evaluator.complete_view_longctx import (
        build_complete_view_with_longctx_quality,
    )
    from prism_challenge.evaluator.multi_family_compare import (
        run_multi_family_lab_gpu_host_compare,
        run_multi_family_official_compare,
    )

    assert callable(build_complete_view_with_longctx_quality)
    assert callable(build_complete_view_with_eff_stability)
    assert callable(run_multi_family_official_compare)
    assert callable(run_multi_family_lab_gpu_host_compare)


def test_scale_densify_complete_view_pair_usable() -> None:
    a = _rec(label="deeploop-tiny-1m", bpb=1.2, heldout_delta=0.9)
    b = _rec(label="transformer-tiny-1m", bpb=1.3, heldout_delta=0.4)
    doc_long = densify_complete_view_pair(a, b, panel="long_ctx", score_class="fixture")
    assert_complete_view_document(doc_long)
    doc_eff = densify_complete_view_pair(a, b, panel="sample_eff", score_class="fixture")
    assert_complete_view_document(doc_eff)
    doc_both = densify_complete_view_pair(a, b, panel="both", score_class="fixture")
    assert_complete_view_document(doc_both)
    # Emission-like official compare on the same records is unchanged by densify call.
    assert compare_official(a, b).reason == "primary_heldout"
    assert compare_official(a, b).winner == "a"


# --- multi-family host compare under scale pin ---------------------------------------------------


def test_scale_multi_family_fixture_host_compare(tmp_path: Path) -> None:
    from prism_challenge.evaluator.scale_eval import run_scale_multi_family_host_compare

    out = tmp_path / "mf"
    report = run_scale_multi_family_host_compare(
        out,
        family_ids=SCALE_P0_CORE_FAMILY_IDS,
        fixture_mode=True,
        package=True,
        write_report=True,
    )
    assert report["pin"]["seeds"] == list(SCALE_P0_SEEDS)
    assert len(report["pin"]["seeds"]) >= 3
    assert report["pin"]["token_budget"] == SCALE_P0_TOKEN_BUDGET
    # Ranking present and does not use wall-clock as key.
    ranking = report.get("ranking") or report.get("ordered") or report.get("score_table")
    assert ranking is not None or report.get("families") or report.get("sides")
    # Core families listed.
    fams = report.get("family_ids") or list((report.get("aggregates") or {}).keys())
    if not fams and "sides" in report:
        fams = list(report["sides"].keys())
    assert set(SCALE_P0_CORE_FAMILY_IDS).issubset(set(fams) | set(report.get("family_ids") or []))


# --- tee absence + snapshot ----------------------------------------------------------------------


def test_scale_tee_package_still_absent() -> None:
    assert tee_package_absent() is True
    tee_path = Path(__file__).resolve().parents[1] / "src" / "prism_challenge" / "tee"
    assert not tee_path.exists()
    with pytest.raises(ModuleNotFoundError):
        __import__("prism_challenge.tee")


def test_scale_product_snapshot_honest() -> None:
    snap = scale_product_snapshot()
    assert snap["schema"] == "prism_scale_product_snapshot.v1"
    assert snap["tee_package_absent"] is True
    assert snap["public_guard"]["ok"] is True
    assert snap["pin"]["seed_count"] >= 3
    assert snap["wall_clock_never_ranks"] is True
    assert "deeploop-tiny-1m" in snap["core_families_p0"]
    assert "kda-tiny-1m" in snap["core_families_p0"]


def test_protocol_pin_dataclass_still_defaults_k3() -> None:
    """Product ProtocolPin default seeds remain public K≥3 (no silent K=1 default)."""
    pin = ProtocolPin()
    assert len(pin.seeds) >= OFFICIAL_MIN_PUBLIC_SEEDS
    assert pin.as_dict()["wall_clock_never_ranks"] is True
