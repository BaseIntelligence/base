"""Official Comparison Protocol v1 ranking (VAL-COMP-002..007, 011).

Synthetic metrics only: no multi-hour train, no live Swarm, no Prism TEE product claim.
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    OFFICIAL_DEFAULT_TOKEN_BUDGET,
    OFFICIAL_EPS_BPB,
    OFFICIAL_EPS_HELDOUT,
    OFFICIAL_PARAM_CAP,
    OFFICIAL_WALL_CLOCK_NEVER_RANKS,
    PROTOCOL_ID,
    CompareResult,
    OfficialScoreRecord,
    ProtocolPin,
    aggregate_official_records,
    compare_official,
    official_rank_key,
    official_record_from_manifest,
    official_record_from_score,
    protocol_budget_constants,
    rank_official,
)
from prism_challenge.evaluator.scoring import (
    MEMORIZATION_PENALTY_FACTOR,
    score_prequential_bpb,
)


def _challenge_manifest(
    *,
    bpb: float,
    covered_bytes: int = 1000,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    val_bpb_random_init: float | None = None,
    train_heldout_gap: float | None = None,
    memorization_flag: bool | None = None,
    step0_anomaly: bool = False,
    wall_clock_seconds: float | None = None,
    online_loss: list[float] | None = None,
) -> dict:
    """Minimal challenge-owned v2 manifest for score_prequential_bpb recompute."""
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    metrics: dict = {
        "online_loss": online_loss if online_loss is not None else [2.0, 1.5, 1.0],
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
    if val_bpb_trained is not None:
        metrics["val_bpb_trained"] = val_bpb_trained
    if val_bpb_random_init is not None:
        metrics["val_bpb_random_init"] = val_bpb_random_init
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    if memorization_flag is not None:
        metrics["memorization_flag"] = memorization_flag
    payload: dict = {
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
    if wall_clock_seconds is not None:
        payload["compute"] = {
            "schema": "prism_compute.v1",
            "gpu_count": 1,
            "world_size": 1,
            "nproc_per_node": 1,
            "device": "cpu",
            "wall_clock_seconds": wall_clock_seconds,
        }
    return payload


def _record(
    *,
    label: str,
    bpb: float,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    primary_form: str = "heldout_delta",
    memorization_flag: bool = False,
    train_heldout_gap: float | None = None,
    step0_anomaly: bool = False,
    valid: bool = True,
    wall_clock_seconds: float | None = None,
    miner_reported_bpb: float | None = None,
    seed_count: int = 1,
    bpb_std: float | None = None,
    overfit_rate: float = 0.0,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        primary_form=primary_form,  # type: ignore[arg-type]
        heldout_delta=heldout_delta,
        val_bpb_trained=val_bpb_trained,
        memorization_flag=memorization_flag,
        train_heldout_gap=train_heldout_gap,
        step0_anomaly=step0_anomaly,
        valid=valid,
        seed_count=seed_count,
        bpb_std=bpb_std,
        overfit_rate=overfit_rate,
        wall_clock_seconds=wall_clock_seconds,
        miner_reported_bpb=miner_reported_bpb,
    )


# --- VAL-COMP-002: primary ranking axis is held-out generalization ---------------


def test_official_primary_heldout_beats_identical_train_bpb() -> None:
    """Identical secondary bpb, different held-out quality → better held-out wins."""
    better_gen = _record(label="A", bpb=1.50, heldout_delta=0.80)
    worse_gen = _record(label="B", bpb=1.50, heldout_delta=0.10)
    result = compare_official(better_gen, worse_gen)
    assert result.winner == "a"
    assert result.reason == "primary_heldout"
    # Invert order still prefers better held-out.
    reverse = compare_official(worse_gen, better_gen)
    assert reverse.winner == "b"
    assert reverse.reason == "primary_heldout"


def test_official_primary_heldout_overrides_worse_train_bpb() -> None:
    """Held-out primary must beat secondary when deltas diverge beyond eps (docs invert)."""
    strong_heldout_weaker_bpb = _record(label="generalizer", bpb=1.80, heldout_delta=1.20)
    weak_heldout_stronger_bpb = _record(label="compressor", bpb=1.00, heldout_delta=0.10)
    result = compare_official(strong_heldout_weaker_bpb, weak_heldout_stronger_bpb)
    assert result.winner == "a"
    assert result.reason == "primary_heldout"
    # Emission path is now Official-like too (VAL-RESLAB-006): held-out primary wins scalar rank.
    score_generalizer = score_prequential_bpb(
        _challenge_manifest(bpb=1.80, heldout_delta=1.20)
    )
    score_compressor = score_prequential_bpb(_challenge_manifest(bpb=1.00, heldout_delta=0.10))
    assert score_generalizer.final_score > score_compressor.final_score


# --- VAL-COMP-003: prequential bpb secondary and always Prism-recomputed ---------


def test_official_secondary_bpb_breaks_near_tie_on_heldout() -> None:
    a = _record(label="A", bpb=1.00, heldout_delta=0.50)
    b = _record(label="B", bpb=1.20, heldout_delta=0.50)
    result = compare_official(a, b)
    assert result.winner == "a"
    assert result.reason == "secondary_bpb"


def test_miner_self_report_bpb_ignored_for_official_rank() -> None:
    """Miner self-reports fabulous bpb → still ranks by Prism-recomputed metrics only."""
    # Challenge manifest: mediocre bpb + mediocre held-out.
    mediocre = official_record_from_manifest(
        _challenge_manifest(bpb=2.0, heldout_delta=0.20),
        label="honest",
        miner_reported={"bpb": 0.01, "final_score": 0.99},
    )
    # Stronger challenge-owned record with no self-report.
    better = official_record_from_manifest(
        _challenge_manifest(bpb=1.5, heldout_delta=0.80),
        label="peer",
        miner_reported={"bpb": 9.0, "final_score": 0.01},
    )
    assert mediocre.miner_reported_bpb == pytest.approx(0.01)
    assert mediocre.bpb == pytest.approx(2.0)  # recomputed, not miner 0.01
    result = compare_official(mediocre, better)
    assert result.winner == "b"
    assert result.reason == "primary_heldout"
    # Rank key must ignore miner self-report: attach absurd self-report without changing key.
    inflated = _record(
        label="honest",
        bpb=2.0,
        heldout_delta=0.20,
        miner_reported_bpb=0.0001,
    )
    base = _record(label="honest", bpb=2.0, heldout_delta=0.20, miner_reported_bpb=None)
    assert official_rank_key(inflated) == official_rank_key(base)


# --- VAL-COMP-004: anti-overfit memorization residual -----------------------------


def test_official_memorizer_ranks_worse_than_benign_same_primary_secondary() -> None:
    memorizer = _record(
        label="memo",
        bpb=1.0,
        heldout_delta=0.40,
        memorization_flag=True,
        train_heldout_gap=2.5,
        overfit_rate=1.0,
    )
    benign = _record(
        label="benign",
        bpb=1.0,
        heldout_delta=0.40,
        memorization_flag=False,
        train_heldout_gap=0.1,
        overfit_rate=0.0,
    )
    result = compare_official(memorizer, benign)
    assert result.winner == "b"
    assert result.reason == "anti_overfit"

    # Via real scoring path: gap generates penalty + flag that flows into official record.
    memo_score = score_prequential_bpb(
        _challenge_manifest(bpb=1.0, heldout_delta=0.3, train_heldout_gap=2.5)
    )
    benign_score = score_prequential_bpb(
        _challenge_manifest(bpb=1.0, heldout_delta=0.3, train_heldout_gap=0.1)
    )
    assert memo_score.memorization_flag is True
    assert memo_score.memorization_penalty == pytest.approx(MEMORIZATION_PENALTY_FACTOR)
    memo_rec = official_record_from_score(memo_score, label="memo")
    benign_rec = official_record_from_score(benign_score, label="benign")
    assert compare_official(memo_rec, benign_rec).winner == "b"


# --- VAL-COMP-005: step-0 anomaly still fail-closes -------------------------------


def test_official_step0_anomaly_disqualifies_despite_fabulous_metrics() -> None:
    smuggled = _record(
        label="smuggle",
        bpb=0.05,
        heldout_delta=5.0,
        step0_anomaly=True,
        valid=False,
    )
    clean = _record(label="clean", bpb=1.5, heldout_delta=0.3)
    result = compare_official(smuggled, clean)
    assert result.winner == "b"
    assert result.reason == "step0_anomaly"

    # Via score_prequential_bpb: anomaly zeros anti_cheat_multiplier → final_score 0 surface.
    anomalous = score_prequential_bpb(
        _challenge_manifest(bpb=0.05, heldout_delta=5.0, step0_anomaly=True)
    )
    assert anomalous.anomaly is True
    assert anomalous.anti_cheat_multiplier == 0.0
    assert anomalous.final_score == pytest.approx(0.0)
    rec = official_record_from_score(anomalous, label="smuggle")
    assert rec.step0_anomaly is True
    assert rec.valid is False
    assert compare_official(rec, clean).winner == "b"


# --- VAL-COMP-006: matched budget & tokenizer denominators surface ---------------


def test_protocol_budget_constants_surface_matched_pin() -> None:
    constants = protocol_budget_constants()
    assert constants["protocol_id"] == PROTOCOL_ID
    # Official default pin follows promote ceiling (350M); dual ladder visible.
    assert constants["param_cap"] == OFFICIAL_PARAM_CAP == 350_000_000
    assert constants["explore_max_parameters"] == 124_000_000
    assert constants["promote_max_parameters"] == 350_000_000
    assert constants["param_ladder_stage"] == "promote"
    assert constants["token_budget"] == OFFICIAL_DEFAULT_TOKEN_BUDGET
    assert constants["seeds"] == list(OFFICIAL_DEFAULT_SEEDS)
    assert len(constants["seeds"]) >= 3
    assert constants["tokenizer"] == "gpt2"
    assert constants["wall_clock_never_ranks"] is True
    assert constants["eps_heldout"] == OFFICIAL_EPS_HELDOUT
    assert constants["eps_bpb"] == OFFICIAL_EPS_BPB

    pin_a = ProtocolPin(seeds=(1337, 2027, 4242), token_budget=500_000)
    pin_b = ProtocolPin(seeds=(1337, 2027, 4242), token_budget=500_000)
    assert pin_a.as_dict() == pin_b.as_dict()
    # Two seeds under the same pin share budget knobs.
    assert pin_a.token_budget == pin_b.token_budget
    assert pin_a.seq_len == pin_b.seq_len
    assert pin_a.val_byte_budget == pin_b.val_byte_budget
    assert pin_a.param_cap == pin_b.param_cap


# --- VAL-COMP-007: deterministic compare_official matrix --------------------------


@pytest.mark.parametrize(
    ("a", "b", "winner", "reason"),
    [
        (
            _record(label="A", bpb=1.0, heldout_delta=0.9),
            _record(label="B", bpb=1.0, heldout_delta=0.1),
            "a",
            "primary_heldout",
        ),
        (
            _record(label="A", bpb=1.0, heldout_delta=0.1),
            _record(label="B", bpb=1.0, heldout_delta=0.9),
            "b",
            "primary_heldout",
        ),
        (
            _record(label="A", bpb=0.9, heldout_delta=0.5),
            _record(label="B", bpb=1.2, heldout_delta=0.5),
            "a",
            "secondary_bpb",
        ),
        (
            _record(label="A", bpb=1.2, heldout_delta=0.5),
            _record(label="B", bpb=0.9, heldout_delta=0.5),
            "b",
            "secondary_bpb",
        ),
        (
            _record(
                label="A",
                bpb=1.0,
                heldout_delta=0.5,
                memorization_flag=True,
                overfit_rate=1.0,
            ),
            _record(label="B", bpb=1.0, heldout_delta=0.5, memorization_flag=False),
            "b",
            "anti_overfit",
        ),
        (
            _record(label="A", bpb=1.0, heldout_delta=0.5, step0_anomaly=True, valid=False),
            _record(label="B", bpb=2.0, heldout_delta=0.1),
            "b",
            "step0_anomaly",
        ),
        (
            _record(label="A", bpb=1.0, heldout_delta=0.5),
            _record(label="B", bpb=1.0 + OFFICIAL_EPS_BPB / 10, heldout_delta=0.5),
            "tie",
            "tie",
        ),
        (
            _record(
                label="A",
                bpb=1.0,
                heldout_delta=0.5,
                seed_count=3,
                bpb_std=0.01,
            ),
            _record(
                label="B",
                bpb=1.0,
                heldout_delta=0.5,
                seed_count=3,
                bpb_std=0.10,
            ),
            "a",
            "multi_seed_residual",
        ),
    ],
)
def test_compare_official_determinism_matrix(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    winner: str,
    reason: str,
) -> None:
    first = compare_official(a, b)
    second = compare_official(a, b)
    assert first == second
    assert isinstance(first, CompareResult)
    assert first.winner == winner
    assert first.reason == reason
    # Antisymmetry for non-ties
    reverse = compare_official(b, a)
    if winner == "tie":
        assert reverse.winner == "tie"
    elif winner == "a":
        assert reverse.winner == "b"
    else:
        assert reverse.winner == "a"


def test_official_rank_key_sort_total_order() -> None:
    rows = [
        _record(label="mid", bpb=1.2, heldout_delta=0.4),
        _record(label="best", bpb=1.5, heldout_delta=0.9),
        _record(label="worst", bpb=1.0, heldout_delta=0.1),
        _record(label="dead", bpb=0.1, heldout_delta=9.0, step0_anomaly=True, valid=False),
    ]
    ranked = rank_official(rows)
    assert [r.label for r in ranked] == ["best", "mid", "worst", "dead"]


# --- VAL-COMP-011: wall-clock never ranks -----------------------------------------


def test_wall_clock_never_orders_official_winner() -> None:
    assert OFFICIAL_WALL_CLOCK_NEVER_RANKS is True
    # Same scientific label; wall-clock differs only on the diagnostic field.
    fast = _record(label="same", bpb=1.5, heldout_delta=0.6, wall_clock_seconds=10.0)
    slow = _record(label="same", bpb=1.5, heldout_delta=0.6, wall_clock_seconds=10_000.0)
    result = compare_official(fast, slow)
    assert result.winner == "tie"
    assert official_rank_key(fast) == official_rank_key(slow)
    # Compare also ties when labels differ; wall-clock still ignored.
    fast_named = _record(label="fast", bpb=1.5, heldout_delta=0.6, wall_clock_seconds=10.0)
    slow_named = _record(label="slow", bpb=1.5, heldout_delta=0.6, wall_clock_seconds=10_000.0)
    assert compare_official(fast_named, slow_named).winner == "tie"
    # Different quality, different wall: still quality decides (not wall).
    better_but_slower = _record(
        label="better", bpb=1.5, heldout_delta=0.9, wall_clock_seconds=9999.0
    )
    worse_but_faster = _record(label="worse", bpb=1.5, heldout_delta=0.1, wall_clock_seconds=1.0)
    assert compare_official(better_but_slower, worse_but_faster).winner == "a"
    assert compare_official(better_but_slower, worse_but_faster).reason == "primary_heldout"


def test_wall_clock_from_compute_block_attached_but_ignored() -> None:
    rec = official_record_from_manifest(
        _challenge_manifest(bpb=1.0, heldout_delta=0.5, wall_clock_seconds=42.0),
        label="run",
    )
    assert rec.wall_clock_seconds == pytest.approx(42.0)
    peer = official_record_from_manifest(
        _challenge_manifest(bpb=1.0, heldout_delta=0.5, wall_clock_seconds=999.0),
        label="run",
    )
    # Labels equal → keys equal for diagnostics-only wall_clock divergence.
    assert official_rank_key(rec) == official_rank_key(peer)
    assert compare_official(rec, peer).winner == "tie"


# --- near-tie ε on primary -----------------------------------------------------------------------


def test_heldout_delta_within_eps_defers_to_secondary() -> None:
    # Within ε → primary near-tie → secondary decides.
    a = _record(label="A", bpb=1.0, heldout_delta=0.500)
    b = _record(label="B", bpb=1.2, heldout_delta=0.500 + OFFICIAL_EPS_HELDOUT / 2)
    result = compare_official(a, b)
    assert result.reason == "secondary_bpb"
    assert result.winner == "a"


# --- multi-seed aggregate ------------------------------------------------------------------------


def test_aggregate_official_records_means_clean_seeds() -> None:
    seeds = [
        _record(label="s1", bpb=1.0, heldout_delta=0.5),
        _record(label="s2", bpb=1.2, heldout_delta=0.7),
        _record(label="s3", bpb=0.05, heldout_delta=9.0, step0_anomaly=True, valid=False),
    ]
    agg = aggregate_official_records(seeds, label="side-a")
    assert agg.valid is True
    assert agg.seed_count == 2
    assert agg.bpb == pytest.approx(1.1)
    assert agg.heldout_delta == pytest.approx(0.6)
    assert agg.step0_anomaly is False
