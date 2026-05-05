from __future__ import annotations

from platform_network.master.aggregator import (
    aggregate_challenge_weights,
    normalize_weights,
)
from platform_network.schemas.weights import ChallengeWeightsResult


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
