from __future__ import annotations

import math

from prism_challenge.evaluator.scoring import (
    LEADERBOARD_TIE_EPSILON,
    LeaderboardRow,
    leaderboard_rank_key,
    rank_leaderboard,
    score_prequential_bpb,
)


def _manifest(
    *,
    bpb: float,
    covered_bytes: int = 4096,
    predicted_tokens: int = 1000,
    heldout_delta: float | None = None,
    val_bpb_trained: float | None = None,
    val_bpb_random_init: float | None = None,
    train_heldout_gap: float | None = None,
) -> dict:
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    metrics: dict = {
        "online_loss": [2.0, 1.5, 1.0],
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "covered_bytes": covered_bytes,
        "predicted_tokens": predicted_tokens,
        "step0_loss": 2.0,
        "consumed_batches": 3,
        "random_init_baseline_nats": math.log(256),
        "nan_inf_batches": 0,
    }
    if heldout_delta is not None:
        metrics["heldout_delta"] = heldout_delta
    if val_bpb_trained is not None:
        metrics["val_bpb_trained"] = val_bpb_trained
    if val_bpb_random_init is not None:
        metrics["val_bpb_random_init"] = val_bpb_random_init
    if train_heldout_gap is not None:
        metrics["train_heldout_gap"] = train_heldout_gap
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": metrics,
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


def _row(sub: str, final_score: float, accepted_at: str, hotkey: str = "hk") -> LeaderboardRow:
    return LeaderboardRow(
        submission_id=sub, hotkey=hotkey, final_score=final_score, accepted_at=accepted_at
    )


# --- VAL-SCORE-012 / VAL-RESLAB-006: leaderboard orders by emission rank (held-out primary) ------


def test_scoring_leaderboard_orders_by_emission_rank_even_against_commit_time() -> None:
    # A better emission-rank score (held-out primary band) committed LATER still outranks a worse
    # score committed earlier: the primary axis is emission final_score, not commit time.
    from prism_challenge.evaluator.scoring import heldout_to_primary_score

    worse_early = _row(
        "worse-heldout", heldout_to_primary_score(0.1), "2024-01-01T00:00:00+00:00"
    )
    better_late = _row(
        "better-heldout", heldout_to_primary_score(1.0), "2024-01-02T00:00:00+00:00"
    )
    ranked = rank_leaderboard([worse_early, better_late])
    assert [r.submission_id for r in ranked] == ["better-heldout", "worse-heldout"]


# --- VAL-SCORE-019: deterministic final tie-break (earliest-commit-wins) -------------------------


def test_scoring_leaderboard_equal_final_score_breaks_by_earliest_commit() -> None:
    late = _row("sub-late", 0.5, "2024-01-02T00:00:00+00:00")
    early = _row("sub-early", 0.5, "2024-01-01T00:00:00+00:00")
    ranked = rank_leaderboard([late, early])
    assert [r.submission_id for r in ranked] == ["sub-early", "sub-late"]


def test_scoring_leaderboard_tie_break_is_reproducible_across_input_orders() -> None:
    late = _row("sub-late", 0.5, "2024-01-02T00:00:00+00:00")
    early = _row("sub-early", 0.5, "2024-01-01T00:00:00+00:00")
    forward = [r.submission_id for r in rank_leaderboard([late, early])]
    backward = [r.submission_id for r in rank_leaderboard([early, late])]
    assert forward == backward == ["sub-early", "sub-late"]


def test_scoring_leaderboard_near_equal_within_epsilon_breaks_by_commit() -> None:
    # final_scores differ by far less than the tie epsilon -> treated as a tie -> earliest commit.
    late = _row("sub-late", 0.5 + LEADERBOARD_TIE_EPSILON / 100.0, "2024-01-02T00:00:00+00:00")
    early = _row("sub-early", 0.5, "2024-01-01T00:00:00+00:00")
    ranked = rank_leaderboard([late, early])
    assert [r.submission_id for r in ranked] == ["sub-early", "sub-late"]


def test_scoring_leaderboard_equal_score_and_commit_breaks_by_submission_id() -> None:
    same_b = _row("sub-b", 0.5, "2024-01-01T00:00:00+00:00")
    same_a = _row("sub-a", 0.5, "2024-01-01T00:00:00+00:00")
    ranked = rank_leaderboard([same_b, same_a])
    assert [r.submission_id for r in ranked] == ["sub-a", "sub-b"]


def test_scoring_leaderboard_heldout_primary_not_collapsed_by_epsilon() -> None:
    # VAL-RESLAB-006: equal secondary bpb, the LARGER held-out delta wins even when committed later.
    # The tie epsilon must stay below the primary resolution so a genuine held-out difference
    # still orders ahead of the commit-time tie-break.
    big_delta = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.8))
    small_delta = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.1))
    big_late = _row("big-delta-late", big_delta.final_score, "2024-01-02T00:00:00+00:00")
    small_early = _row("small-delta-early", small_delta.final_score, "2024-01-01T00:00:00+00:00")
    ranked = rank_leaderboard([big_late, small_early])
    assert [r.submission_id for r in ranked] == ["big-delta-late", "small-delta-early"]


def test_scoring_leaderboard_rank_key_is_total_order_tuple() -> None:
    key = leaderboard_rank_key(_row("sub", 0.5, "2024-01-01T00:00:00+00:00"))
    assert isinstance(key, tuple)
    # higher final_score -> strictly smaller (better) primary key component.
    better = leaderboard_rank_key(_row("sub", 0.9, "2024-01-01T00:00:00+00:00"))
    assert better[0] < key[0]


# --- VAL-SCORE-013: determinism (identical inputs => identical score, bit-exact) -----------------


def test_scoring_determinism_identical_manifest_identical_score() -> None:
    first = score_prequential_bpb(
        _manifest(
            bpb=1.234,
            heldout_delta=0.4,
            val_bpb_trained=2.6,
            val_bpb_random_init=3.0,
            train_heldout_gap=0.2,
        )
    )
    second = score_prequential_bpb(
        _manifest(
            bpb=1.234,
            heldout_delta=0.4,
            val_bpb_trained=2.6,
            val_bpb_random_init=3.0,
            train_heldout_gap=0.2,
        )
    )
    # Bit-exact (not merely within tolerance): scoring is a pure function of the manifest.
    assert first.bpb == second.bpb
    assert first.final_score == second.final_score
    assert first.heldout_delta == second.heldout_delta
    assert first.metrics_payload() == second.metrics_payload()


# --- VAL-SCORE-004: two tokenizers that learn equivalently get comparable bpb --------------------


def test_scoring_two_tokenizers_comparable_bpb() -> None:
    # Same locked train bytes, equivalent learning, but different tokenizers: gpt2 / llama / a
    # custom tokenizer produce different TOKEN counts (and slightly different code-length), yet the
    # byte denominator keeps bpb within a small tolerance -> ordering is not decided by tokenizer.
    covered = 8192
    gpt2 = score_prequential_bpb(_manifest(bpb=1.00, covered_bytes=covered, predicted_tokens=2000))
    llama = score_prequential_bpb(_manifest(bpb=1.02, covered_bytes=covered, predicted_tokens=2400))
    custom = score_prequential_bpb(
        _manifest(bpb=0.98, covered_bytes=covered, predicted_tokens=3100)
    )
    tolerance = 0.05
    assert abs(gpt2.bpb - llama.bpb) <= tolerance
    assert abs(gpt2.bpb - custom.bpb) <= tolerance
    assert abs(llama.bpb - custom.bpb) <= tolerance
    # The byte denominator is identical across the three tokenizers.
    assert gpt2.covered_bytes == llama.covered_bytes == custom.covered_bytes == covered
