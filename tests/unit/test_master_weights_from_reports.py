"""Weights are sourced from validator-reported challenge ``get_weights`` only.

Covers VAL-WEIGHTS-001..017: ``/v1/weights/latest`` recomputes from each active
challenge's ``get_weights`` on every read, zero reported results collapse to the
zero-miner burn fallback (never fabricated miner values), a failing challenge
aborts the recompute, and the preserved aggregator semantics (per-challenge
normalize, emission normalize across OK challenges, hotkey->UID with UID 0 /
unknown hotkeys dropped, final sum-to-1, and the chain-guarded burn fallback).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi.testclient import TestClient

from base.bittensor.metagraph_cache import MetagraphCache
from base.master.aggregator import (
    CHAIN_U16_MAX,
    ZeroMinerWeightError,
    aggregate_challenge_weights,
    build_zero_miner_weights,
    normalize_emissions,
    normalize_weights,
)
from base.master.app_admin import create_admin_app
from base.master.challenge_client import ChallengeClient
from base.master.registry import ChallengeRegistry
from base.master.service import MasterWeightService
from base.schemas.challenge import (
    ChallengeCreate,
    ChallengeStatus,
    RuntimeOperationResponse,
)
from base.schemas.weights import ChallengeWeightsResult

FIXED_NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


class StubMetagraphCache:
    """Minimal ``MetagraphCache`` stand-in returning a fixed hotkey->UID map."""

    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = dict(mapping)
        self._updated_at = 0.0

    def get(self, *, force: bool = False) -> dict[str, int]:
        return dict(self._mapping)


class RecordingChallengeClient:
    """Records every ``get_weights`` call and returns canned per-hotkey results.

    Simulates validator-reported results surfacing through each challenge's
    ``/internal/v1/get_weights``; failing slugs surface ``ok=False`` exactly as
    the real client does when a challenge cannot provide weights.
    """

    def __init__(
        self,
        weights_by_slug: dict[str, dict[str, float]],
        *,
        failing_slugs: tuple[str, ...] = (),
    ) -> None:
        self.weights_by_slug = weights_by_slug
        self.failing_slugs = set(failing_slugs)
        self.calls: list[str] = []

    async def get_weights(
        self,
        *,
        slug: str,
        base_url: str,
        token: str,
        emission_percent: float,
    ) -> ChallengeWeightsResult:
        self.calls.append(slug)
        if slug in self.failing_slugs:
            return ChallengeWeightsResult(
                slug=slug,
                emission_percent=emission_percent,
                weights={},
                ok=False,
                error="reported results unavailable",
            )
        return ChallengeWeightsResult(
            slug=slug,
            emission_percent=emission_percent,
            weights=dict(self.weights_by_slug.get(slug, {})),
            ok=True,
        )


class FakeRuntimeController:
    async def pull(self, slug: str) -> RuntimeOperationResponse:
        return RuntimeOperationResponse(slug=slug, operation="pull", status="ok")

    async def restart(self, slug: str) -> RuntimeOperationResponse:
        return RuntimeOperationResponse(slug=slug, operation="restart", status="ok")

    async def status(self, slug: str) -> RuntimeOperationResponse:
        return RuntimeOperationResponse(slug=slug, operation="status", status="ok")


def _make_active_registry(
    challenges: list[tuple[str, str]],
) -> ChallengeRegistry:
    """Build a registry of active challenges as ``(slug, emission_percent)``."""

    registry = ChallengeRegistry()
    for slug, emission in challenges:
        registry.create(
            ChallengeCreate(
                slug=slug,
                name=slug.title(),
                image=f"ghcr.io/baseintelligence/{slug}:1.0.0",
                version="1.0.0",
                emission_percent=emission,  # type: ignore[arg-type]
                status=ChallengeStatus.ACTIVE,
                internal_base_url=f"http://challenge-{slug}:8000",
            )
        )
    return registry


def _client(
    registry: ChallengeRegistry,
    service: MasterWeightService,
) -> TestClient:
    return TestClient(
        create_admin_app(
            registry=registry,
            runtime_controller=FakeRuntimeController(),
            weight_service=service,
            netuid=42,
            chain_endpoint="wss://chain.example:9944",
            now_fn=lambda: FIXED_NOW,
        )
    )


def _service(
    *,
    mapping: dict[str, int],
    weights_by_slug: dict[str, dict[str, float]],
    failing_slugs: tuple[str, ...] = (),
) -> tuple[MasterWeightService, RecordingChallengeClient]:
    recorder = RecordingChallengeClient(weights_by_slug, failing_slugs=failing_slugs)
    service = MasterWeightService(
        metagraph_cache=cast(MetagraphCache, StubMetagraphCache(mapping)),
        challenge_client=cast(ChallengeClient, recorder),
    )
    return service, recorder


# --- A. Source of weights: validator-reported, no pre-computed path -----------


def test_weights_latest_recomputes_from_get_weights_each_read() -> None:
    """VAL-WEIGHTS-001."""
    registry = _make_active_registry([("agent-challenge", "40"), ("prism", "60")])
    service, recorder = _service(
        mapping={"hkA": 1, "hkB": 2},
        weights_by_slug={
            "agent-challenge": {"hkA": 1.0},
            "prism": {"hkB": 1.0},
        },
    )
    client = _client(registry, service)

    first = client.get("/v1/weights/latest")
    assert first.status_code == 200
    # One get_weights fetch per active challenge on the first read.
    assert sorted(recorder.calls) == ["agent-challenge", "prism"]

    body = first.json()
    assert body["uids"] == [1, 2]
    assert [round(w, 8) for w in body["weights"]] == [0.4, 0.6]
    assert body["hotkey_weights"] == {"hkA": 0.4, "hkB": 0.6}
    assert sorted(item["slug"] for item in body["source_challenges"]) == [
        "agent-challenge",
        "prism",
    ]

    second = client.get("/v1/weights/latest")
    assert second.status_code == 200
    # Each read triggers a fresh fetch: the recorded call count increases.
    assert len(recorder.calls) == 4


def test_zero_reported_results_yields_zero_miner_burn_not_fabricated() -> None:
    """VAL-WEIGHTS-002."""
    registry = _make_active_registry([("agent-challenge", "40"), ("prism", "60")])
    service, recorder = _service(
        mapping={"hkA": 1, "hkB": 2},
        weights_by_slug={"agent-challenge": {}, "prism": {}},
    )
    client = _client(registry, service)

    response = client.get("/v1/weights/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["uids"] == [0]
    assert body["weights"] == [1.0]
    assert body["hotkey_weights"] == {}
    assert recorder.calls  # weights still come from a real get_weights fetch


def test_no_active_challenges_burn_fallback() -> None:
    """VAL-WEIGHTS-003."""
    final = aggregate_challenge_weights([], {})
    assert final.uids == [0]
    assert final.weights == [1.0]
    assert final.hotkey_weights == {}


def test_only_uid_zero_reported_still_burn_fallback() -> None:
    """VAL-WEIGHTS-004."""
    results = [
        ChallengeWeightsResult(slug="a", emission_percent=100, weights={"validator": 1})
    ]
    final = aggregate_challenge_weights(results, {"validator": 0})
    assert final.uids == [0]
    assert final.weights == [1.0]
    assert final.hotkey_weights == {}


def test_failed_get_weights_aborts_recompute_with_bad_gateway() -> None:
    """VAL-WEIGHTS-005."""
    registry = _make_active_registry([("agent-challenge", "40"), ("prism", "60")])
    service, _recorder = _service(
        mapping={"hkA": 1, "hkB": 2},
        weights_by_slug={"agent-challenge": {"hkA": 1.0}, "prism": {"hkB": 1.0}},
        failing_slugs=("prism",),
    )
    client = _client(registry, service)

    response = client.get("/v1/weights/latest")

    assert response.status_code == 502
    body = response.json()
    assert "prism" in body["detail"]
    # No partial/stale weights vector is served.
    assert "uids" not in body
    assert "weights" not in body


# --- B. Preserved aggregator semantics ----------------------------------------


def test_per_challenge_normalize_before_emission_weighting() -> None:
    """VAL-WEIGHTS-006."""
    assert normalize_weights({"a": 2, "b": 2}) == {"a": 0.5, "b": 0.5}

    scaled = aggregate_challenge_weights(
        [
            ChallengeWeightsResult(slug="x", emission_percent=50, weights={"hk": 5}),
            ChallengeWeightsResult(slug="y", emission_percent=50, weights={"hk": 1}),
        ],
        {"hk": 1},
    )
    # Absolute magnitudes inside a challenge do not bias the result.
    assert scaled.uids == [1]
    assert round(scaled.weights[0], 8) == 1.0


def test_invalid_raw_weights_dropped_before_normalization() -> None:
    """VAL-WEIGHTS-007."""
    cleaned = normalize_weights({"a": 2, "b": -1, "c": float("nan"), "d": 2})
    assert cleaned == {"a": 0.5, "d": 0.5}
    assert round(sum(cleaned.values()), 8) == 1.0


def test_emissions_normalize_across_ok_challenges_only() -> None:
    """VAL-WEIGHTS-008."""
    results = [
        ChallengeWeightsResult(slug="a", emission_percent=30, weights={"hk1": 1}),
        ChallengeWeightsResult(slug="b", emission_percent=15, weights={"hk2": 1}),
        ChallengeWeightsResult(slug="c", emission_percent=5, weights={"hk3": 1}),
        ChallengeWeightsResult(
            slug="d", emission_percent=50, weights={"hk4": 1}, ok=False
        ),
    ]
    shares = normalize_emissions(results)
    assert "d" not in shares
    assert round(shares["a"], 8) == round(30 / 50, 8)
    assert round(shares["b"], 8) == round(15 / 50, 8)
    assert round(shares["c"], 8) == round(5 / 50, 8)
    assert round(sum(shares.values()), 8) == 1.0


def test_unknown_hotkeys_dropped() -> None:
    """VAL-WEIGHTS-009."""
    results = [
        ChallengeWeightsResult(
            slug="a", emission_percent=100, weights={"hk1": 1, "missing": 3}
        )
    ]
    final = aggregate_challenge_weights(results, {"hk1": 1})
    # "missing" maps to no uid, so its share of the absolute emission burns.
    assert final.uids == [0, 1]
    weights = dict(zip(final.uids, final.weights, strict=True))
    assert round(weights[1], 8) == 0.25
    assert round(weights[0], 8) == 0.75
    assert "missing" not in final.hotkey_weights


def test_uid_zero_dropped_from_miner_vector() -> None:
    """VAL-WEIGHTS-010 (uid-0 hotkey is never a miner; its share burns to uid 0)."""
    results = [
        ChallengeWeightsResult(
            slug="a",
            emission_percent=100,
            weights={"validator": 1, "miner1": 1, "miner2": 1},
        )
    ]
    final = aggregate_challenge_weights(
        results, {"validator": 0, "miner1": 1, "miner2": 2}
    )
    # miner1/miner2 keep their absolute 1/3 shares; the uid-0 hotkey is not a
    # miner so its 1/3 share burns to uid 0 (which therefore appears in uids).
    assert final.uids == [0, 1, 2]
    weights = dict(zip(final.uids, final.weights, strict=True))
    assert round(weights[0], 8) == round(1 / 3, 8)
    assert round(weights[1], 8) == round(1 / 3, 8)
    assert round(weights[2], 8) == round(1 / 3, 8)
    assert "validator" not in final.hotkey_weights
    assert set(final.hotkey_weights) == {"miner1", "miner2"}


def test_final_vector_sums_to_one_and_is_uid_sorted() -> None:
    """VAL-WEIGHTS-011."""
    results = [
        ChallengeWeightsResult(
            slug="a", emission_percent=100, weights={"hk2": 1, "hk1": 1, "hk3": 1}
        )
    ]
    final = aggregate_challenge_weights(results, {"hk1": 9, "hk2": 3, "hk3": 5})
    assert final.uids == sorted(final.uids)
    assert final.uids == [3, 5, 9]
    assert round(sum(final.weights), 8) == 1.0


def test_multi_challenge_cross_emission_weighting() -> None:
    """VAL-WEIGHTS-012."""
    results = [
        ChallengeWeightsResult(slug="a", emission_percent=40, weights={"hk1": 1}),
        ChallengeWeightsResult(slug="b", emission_percent=60, weights={"hk2": 2}),
    ]
    final = aggregate_challenge_weights(results, {"hk1": 1, "hk2": 2})
    assert final.uids == [1, 2]
    assert [round(w, 8) for w in final.weights] == [0.4, 0.6]
    assert round(sum(final.weights), 8) == 1.0


def test_zero_miner_burn_single_self_burn() -> None:
    """VAL-WEIGHTS-013."""
    vector = build_zero_miner_weights(
        min_allowed_weights=1,
        max_weight_limit=CHAIN_U16_MAX,
        available_uids=[0, 5, 9],
    )
    assert vector == {0: 1.0}


def test_zero_miner_burn_padded_distribution() -> None:
    """VAL-WEIGHTS-014."""
    vector = build_zero_miner_weights(
        min_allowed_weights=3,
        max_weight_limit=CHAIN_U16_MAX,
        available_uids=[0, 5, 7, 9],
    )
    assert set(vector) == {0, 5, 7}
    assert all(round(w, 8) == round(1 / 3, 8) for w in vector.values())
    assert round(sum(vector.values()), 8) == 1.0


def test_zero_miner_low_cap_forces_additional_entries() -> None:
    """VAL-WEIGHTS-015."""
    cap = 20000
    vector = build_zero_miner_weights(
        min_allowed_weights=1,
        max_weight_limit=cap,
        available_uids=[0, 3, 5, 8, 11],
    )
    assert len(vector) == 4
    assert round(sum(vector.values()), 8) == 1.0
    assert all(w <= cap / CHAIN_U16_MAX + 1e-12 for w in vector.values())


def test_zero_miner_aborts_when_unsatisfiable() -> None:
    """VAL-WEIGHTS-016."""
    with pytest.raises(ZeroMinerWeightError):
        build_zero_miner_weights(min_allowed_weights=3, available_uids=[0])
    with pytest.raises(ZeroMinerWeightError):
        build_zero_miner_weights(
            min_allowed_weights=1, max_weight_limit=20000, available_uids=[0, 4]
        )
    with pytest.raises(ZeroMinerWeightError):
        build_zero_miner_weights(
            min_allowed_weights=1, max_weight_limit=0, available_uids=[0, 1, 2]
        )


def test_chain_guard_params_ignored_when_miners_present() -> None:
    """VAL-WEIGHTS-017."""
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
    assert math.isclose(sum(final.weights), 1.0, abs_tol=1e-9)
