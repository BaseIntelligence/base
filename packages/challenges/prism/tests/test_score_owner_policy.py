"""Deterministic Prism score-owner / emission policy (VAL-WEIGHT-094).

Gateway-era ``test_prism_master_gate_weights`` imported deleted ``llm_review``.
This suite replaces the collectable score-owner matrix without LLM residual.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.repository import PrismRepository
from prism_challenge.weights import SCORE_OWNER_POLICY_VERSION, get_weights

EPOCH_SECONDS = 60


async def _new_repository(tmp_path: Path, name: str) -> PrismRepository:
    database = Database(tmp_path / name)
    await database.init()
    return PrismRepository(database, epoch_seconds=EPOCH_SECONDS)


async def _seed_architecture(
    repository: PrismRepository,
    *,
    architecture_id: str,
    owner_hotkey: str,
    q_arch_best: float,
    created_at: str,
    family_hash: str | None = None,
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO architecture_families("
            "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
            "owner_submission_id, canonical_submission_id, q_arch_best, display_name, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                architecture_id,
                family_hash or f"fh-{architecture_id}",
                f"fp-{architecture_id}",
                f"bp-{architecture_id}",
                owner_hotkey,
                f"sub-{architecture_id}",
                f"sub-{architecture_id}",
                q_arch_best,
                f"arch-{architecture_id}",
                created_at,
                created_at,
            ),
        )


async def _seed_training_variant(
    repository: PrismRepository,
    *,
    variant_id: str,
    architecture_id: str,
    owner_hotkey: str,
    q_recipe: float,
    created_at: str,
    is_current_best: int = 1,
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
            "metric_mean, metric_std, is_current_best, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                variant_id,
                architecture_id,
                f"th-{variant_id}",
                owner_hotkey,
                f"sub-{variant_id}",
                q_recipe,
                q_recipe,
                0.0,
                is_current_best,
                created_at,
                created_at,
            ),
        )


def test_score_owner_policy_version_is_explicit() -> None:
    assert SCORE_OWNER_POLICY_VERSION == "score-owner.architecture-training.v1"
    assert SCORE_OWNER_POLICY_VERSION.startswith("score-owner.")


async def test_get_weights_empty_store_returns_empty(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-empty.sqlite3")
    assert await get_weights(repository, EPOCH_SECONDS) == {}


async def test_get_weights_splits_between_distinct_owners(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-split.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="train-owner",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights["arch-owner"] == pytest.approx(0.50)
    assert weights["train-owner"] == pytest.approx(0.50)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Exactly one positive share per hotkey after renormalization.
    assert all(weight > 0.0 for weight in weights.values())
    assert len(weights) == 2


async def test_get_weights_same_owner_takes_full_pool(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-same.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="solo",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="solo",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights == {"solo": pytest.approx(1.0)}


async def test_get_weights_honors_db_configured_custom_split(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-custom.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="train-owner",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await repository.store_runtime_config(
        config_key="reward_pools",
        value={"architecture": 0.7, "training": 0.3},
        updated_by="ops",
    )

    runtime_config = await repository.runtime_config(PrismSettings(), official=True)
    weights = await get_weights(
        repository,
        EPOCH_SECONDS,
        architecture_weight=runtime_config.reward_pools.architecture,
        training_weight=runtime_config.reward_pools.training,
    )

    assert weights["arch-owner"] == pytest.approx(0.70)
    assert weights["train-owner"] == pytest.approx(0.30)
    assert sum(weights.values()) == pytest.approx(1.0)


async def test_get_weights_crown_is_global_not_duplicate_owners(tmp_path: Path) -> None:
    # Per documented policy: one cross-epoch architecture crown (tie-break earliest),
    # plus the best training variant on that frontrunner only — never multi-owner payout.
    repository = await _new_repository(tmp_path, "weights-crossepoch.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-old",
        owner_hotkey="old-owner",
        q_arch_best=0.95,
        created_at="2023-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-old",
        architecture_id="arch-old",
        owner_hotkey="old-train",
        q_recipe=0.9,
        created_at="2023-01-01T00:00:00+00:00",
    )
    await _seed_architecture(
        repository,
        architecture_id="arch-new",
        owner_hotkey="new-owner",
        q_arch_best=0.50,
        created_at="2025-06-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-new",
        architecture_id="arch-new",
        owner_hotkey="new-train",
        q_recipe=0.4,
        created_at="2025-06-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert set(weights) == {"old-owner", "old-train"}
    assert weights["old-owner"] == pytest.approx(0.50)
    assert weights["old-train"] == pytest.approx(0.50)
    assert "new-owner" not in weights
    assert "new-train" not in weights


async def test_get_weights_nonpositive_crown_burns(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-zero-crown.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.0,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="train-owner",
        q_recipe=0.0,
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert await get_weights(repository, EPOCH_SECONDS) == {}


async def test_get_weights_negative_score_never_receives_advantage(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-negative.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-neg",
        owner_hotkey="neg-owner",
        q_arch_best=-1.0,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_architecture(
        repository,
        architecture_id="arch-pos",
        owner_hotkey="pos-owner",
        q_arch_best=0.5,
        created_at="2024-02-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-pos",
        architecture_id="arch-pos",
        owner_hotkey="pos-train",
        q_recipe=0.4,
        created_at="2024-02-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert "neg-owner" not in weights
    assert set(weights) == {"pos-owner", "pos-train"}
    assert all(weight > 0.0 for weight in weights.values())


async def test_get_weights_missing_training_variant_gives_arch_owner_full(
    tmp_path: Path,
) -> None:
    repository = await _new_repository(tmp_path, "weights-no-variant.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights == {"arch-owner": pytest.approx(1.0)}


async def test_get_weights_tiebreak_picks_earliest_architecture(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-tie.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-b",
        owner_hotkey="later-owner",
        q_arch_best=0.8,
        created_at="2024-06-01T00:00:00+00:00",
    )
    await _seed_architecture(
        repository,
        architecture_id="arch-a",
        owner_hotkey="earlier-owner",
        q_arch_best=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-a",
        architecture_id="arch-a",
        owner_hotkey="earlier-train",
        q_recipe=0.7,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-b",
        architecture_id="arch-b",
        owner_hotkey="later-train",
        q_recipe=0.9,
        created_at="2024-06-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    # Tie on q_arch_best -> earliest created architecture keeps the crown exclusively.
    assert set(weights) == {"earlier-owner", "earlier-train"}
    assert "later-owner" not in weights
    assert "later-train" not in weights


async def test_get_weights_order_independent(tmp_path: Path) -> None:
    """Insert order must not change the score-owner matrix (documented ranking)."""

    async def build(name: str, reverse: bool) -> dict[str, float]:
        repository = await _new_repository(tmp_path, name)
        seeds = [
            ("arch-low", "low-owner", 0.4, "2024-03-01T00:00:00+00:00"),
            ("arch-high", "high-owner", 0.9, "2024-01-01T00:00:00+00:00"),
        ]
        if reverse:
            seeds = list(reversed(seeds))
        for architecture_id, owner, score, created in seeds:
            await _seed_architecture(
                repository,
                architecture_id=architecture_id,
                owner_hotkey=owner,
                q_arch_best=score,
                created_at=created,
            )
        await _seed_training_variant(
            repository,
            variant_id=f"var-{seeds[0][0]}",
            architecture_id="arch-high",
            owner_hotkey="high-train",
            q_recipe=0.7,
            created_at="2024-01-02T00:00:00+00:00",
        )
        return await get_weights(repository, EPOCH_SECONDS)

    forward = await build("order-fwd.sqlite3", reverse=False)
    reverse = await build("order-rev.sqlite3", reverse=True)
    assert forward == reverse
    assert set(forward) == {"high-owner", "high-train"}


async def test_get_weights_is_current_best_training_variant(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-training-best.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-old",
        architecture_id="arch-1",
        owner_hotkey="old-train",
        q_recipe=0.5,
        created_at="2024-01-01T00:00:00+00:00",
        is_current_best=0,
    )
    await _seed_training_variant(
        repository,
        variant_id="var-new",
        architecture_id="arch-1",
        owner_hotkey="new-train",
        q_recipe=0.8,
        created_at="2024-02-01T00:00:00+00:00",
        is_current_best=1,
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert set(weights) == {"arch-owner", "new-train"}
    assert "old-train" not in weights


def test_get_weights_endpoint_shape_under_internal_auth(tmp_path: Path) -> None:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'weights-shape.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/internal/v1/get_weights").status_code == 401
        response = client.get(
            "/internal/v1/get_weights",
            headers={"Authorization": "Bearer secret", "X-Base-Challenge-Slug": "prism"},
        )
    assert response.status_code == 200
    body = response.json()
    assert {"challenge_slug", "epoch", "weights"} <= set(body)
    assert body["challenge_slug"] == "prism"
    assert isinstance(body["epoch"], int)
    assert isinstance(body["weights"], dict)
    assert all(isinstance(value, float) for value in body["weights"].values())
