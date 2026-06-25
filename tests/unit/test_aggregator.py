from __future__ import annotations

import pytest

from base.master.aggregator import (
    CHAIN_U16_MAX,
    ZeroMinerWeightError,
    aggregate_challenge_weights,
    build_zero_miner_weights,
    normalize_weights,
)
from base.schemas.weights import ChallengeWeightsResult

# Recon facts (netuid 100, finney) used as test fixtures.
# Source: .omo/evidence/recon-facts.md (captured 2026-06-25T09:23:49Z).
RECON_MIN_ALLOWED_WEIGHTS = 1
RECON_MAX_WEIGHT_LIMIT = 65535  # u16 max -> 65535/65535 == 1.0 fraction


def test_chain_u16_max_matches_recon() -> None:
    assert CHAIN_U16_MAX == RECON_MAX_WEIGHT_LIMIT == 65535


def test_normalize_weights_clamps_invalid_values() -> None:
    assert normalize_weights({"a": 2, "b": -1, "c": float("nan"), "d": 2}) == {
        "a": 0.5,
        "d": 0.5,
    }


def test_aggregate_normalizes_emissions_and_ignores_unknown_hotkeys() -> None:
    results = [
        ChallengeWeightsResult(
            slug="a", emission_percent=40, weights={"hk1": 1, "missing": 3}
        ),
        ChallengeWeightsResult(slug="b", emission_percent=60, weights={"hk2": 2}),
    ]
    final = aggregate_challenge_weights(results, {"hk1": 1, "hk2": 2})
    assert final.uids == [1, 2]
    assert round(sum(final.weights), 8) == 1.0
    assert round(final.weights[0], 8) == round(1 / 7, 8)
    assert round(final.weights[1], 8) == round(6 / 7, 8)


def test_prism_and_agent_challenge_emissions_normalize_by_successful_total() -> None:
    results = [
        ChallengeWeightsResult(
            slug="prism", emission_percent=30, weights={"prism-hotkey": 1}
        ),
        ChallengeWeightsResult(
            slug="agent-challenge", emission_percent=15, weights={"agent-hotkey": 1}
        ),
        ChallengeWeightsResult(
            slug="other-active", emission_percent=5, weights={"other-hotkey": 1}
        ),
        ChallengeWeightsResult(
            slug="failed-active",
            emission_percent=50,
            weights={"failed-hotkey": 1},
            ok=False,
        ),
    ]

    source_emissions = {result.slug: result.emission_percent for result in results}
    assert source_emissions["prism"] == 30
    assert source_emissions["agent-challenge"] == 15

    final = aggregate_challenge_weights(
        results,
        {
            "prism-hotkey": 30,
            "agent-hotkey": 15,
            "other-hotkey": 5,
            "failed-hotkey": 50,
        },
    )

    assert final.uids == [5, 15, 30]
    assert [round(weight, 8) for weight in final.weights] == [
        round(5 / 50, 8),
        round(15 / 50, 8),
        round(30 / 50, 8),
    ]
    assert final.hotkey_weights == {
        "prism-hotkey": 30 / 50,
        "agent-hotkey": 15 / 50,
        "other-hotkey": 5 / 50,
    }


def test_failed_challenge_contributes_zero() -> None:
    results = [
        ChallengeWeightsResult(slug="a", emission_percent=50, weights={"hk1": 1}),
        ChallengeWeightsResult(
            slug="b", emission_percent=50, weights={"hk2": 1}, ok=False
        ),
    ]
    final = aggregate_challenge_weights(results, {"hk1": 1, "hk2": 2})
    assert final.uids == [1]
    assert final.weights == [1.0]


def test_aggregate_falls_back_to_uid_zero_without_active_challenges() -> None:
    final = aggregate_challenge_weights([], {})

    assert final.uids == [0]
    assert final.weights == [1.0]
    assert final.hotkey_weights == {}


def test_aggregate_falls_back_to_uid_zero_for_empty_challenge_weights() -> None:
    results = [ChallengeWeightsResult(slug="a", emission_percent=100, weights={})]

    final = aggregate_challenge_weights(results, {"validator": 0})

    assert final.uids == [0]
    assert final.weights == [1.0]
    assert final.hotkey_weights == {}


def test_aggregate_falls_back_to_uid_zero_for_uid_zero_only_weights() -> None:
    results = [
        ChallengeWeightsResult(slug="a", emission_percent=100, weights={"validator": 1})
    ]

    final = aggregate_challenge_weights(results, {"validator": 0})

    assert final.uids == [0]
    assert final.weights == [1.0]
    assert final.hotkey_weights == {}


def test_zero_miner_branch1_min_le_one_keeps_uid_zero_self_burn() -> None:
    vector = build_zero_miner_weights(
        min_allowed_weights=RECON_MIN_ALLOWED_WEIGHTS,
        max_weight_limit=RECON_MAX_WEIGHT_LIMIT,
        available_uids=[0, 5, 9],
    )
    assert vector == {0: 1.0}


def test_zero_miner_branch1_min_le_one_without_self_vote_uses_other_uid() -> None:
    vector = build_zero_miner_weights(
        min_allowed_weights=RECON_MIN_ALLOWED_WEIGHTS,
        max_weight_limit=RECON_MAX_WEIGHT_LIMIT,
        available_uids=[0, 7],
        allow_burn_self_vote=False,
    )
    assert vector == {7: 1.0}


def test_zero_miner_branch2_min_gt_one_pads_with_burn_and_distributes() -> None:
    vector = build_zero_miner_weights(
        min_allowed_weights=3,
        max_weight_limit=RECON_MAX_WEIGHT_LIMIT,
        available_uids=[0, 5, 7, 9],
    )
    assert set(vector) == {0, 5, 7}
    assert round(sum(vector.values()), 8) == 1.0
    assert all(round(w, 8) == round(1 / 3, 8) for w in vector.values())
    max_fraction = RECON_MAX_WEIGHT_LIMIT / CHAIN_U16_MAX
    assert all(w <= max_fraction + 1e-12 for w in vector.values())


def test_zero_miner_branch2_low_max_weight_limit_forces_more_entries() -> None:
    vector = build_zero_miner_weights(
        min_allowed_weights=1,
        max_weight_limit=20000,
        available_uids=[0, 3, 5, 8, 11],
    )
    max_fraction = 20000 / CHAIN_U16_MAX
    assert len(vector) == 4
    assert round(sum(vector.values()), 8) == 1.0
    assert all(w <= max_fraction + 1e-12 for w in vector.values())


def test_zero_miner_branch3_aborts_when_not_enough_uids_for_min() -> None:
    with pytest.raises(ZeroMinerWeightError):
        build_zero_miner_weights(
            min_allowed_weights=3,
            max_weight_limit=RECON_MAX_WEIGHT_LIMIT,
            available_uids=[0],
        )


def test_zero_miner_branch3_aborts_when_max_weight_limit_unsatisfiable() -> None:
    with pytest.raises(ZeroMinerWeightError):
        build_zero_miner_weights(
            min_allowed_weights=1,
            max_weight_limit=20000,
            available_uids=[0, 4],
        )


def test_zero_miner_branch3_aborts_on_nonpositive_max_weight_limit() -> None:
    with pytest.raises(ZeroMinerWeightError):
        build_zero_miner_weights(
            min_allowed_weights=1,
            max_weight_limit=0,
            available_uids=[0, 1, 2],
        )


def test_aggregate_zero_miner_min_gt_one_distributes_across_metagraph() -> None:
    results = [ChallengeWeightsResult(slug="a", emission_percent=100, weights={})]
    final = aggregate_challenge_weights(
        results,
        {"validator": 0, "hkA": 5, "hkB": 7},
        min_allowed_weights=3,
        max_weight_limit=RECON_MAX_WEIGHT_LIMIT,
    )
    assert final.uids == [0, 5, 7]
    assert round(sum(final.weights), 8) == 1.0
    assert final.hotkey_weights == {}


def test_aggregate_zero_miner_min_gt_one_aborts_when_impossible() -> None:
    results = [ChallengeWeightsResult(slug="a", emission_percent=100, weights={})]
    with pytest.raises(ZeroMinerWeightError):
        aggregate_challenge_weights(
            results,
            {"validator": 0},
            min_allowed_weights=3,
            max_weight_limit=RECON_MAX_WEIGHT_LIMIT,
        )


def test_aggregate_with_miners_present_ignores_chain_guard_params() -> None:
    results = [
        ChallengeWeightsResult(slug="a", emission_percent=40, weights={"hk1": 1}),
        ChallengeWeightsResult(slug="b", emission_percent=60, weights={"hk2": 2}),
    ]
    final = aggregate_challenge_weights(
        results,
        {"hk1": 1, "hk2": 2},
        min_allowed_weights=99,
        max_weight_limit=1,
    )
    assert final.uids == [1, 2]
    assert round(sum(final.weights), 8) == 1.0
