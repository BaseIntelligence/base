"""VAL-RESLAB-004/005/008/009: 0.50/0.50 defaults + provisional/promote crowns."""

from __future__ import annotations

from pathlib import Path

import pytest
from base.challenge_sdk.schemas import RawWeightPushRequest

from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator.param_ladder import (
    CROWN_STATUS_CONFIRMED,
    CROWN_STATUS_PROVISIONAL,
    CROWN_STATUS_REVOKED,
    STAGE_EXPLORE,
    STAGE_PROMOTE,
    crown_status_is_weight_eligible,
    promote_path_decision,
    resolve_crown_status_transition,
    resolve_package_pin,
)
from prism_challenge.raw_weight_push import (
    RawWeightPushClient,
    build_weights_loader,
    maybe_build_push_client_from_settings,
)
from prism_challenge.repository import PrismRepository
from prism_challenge.runtime_config import runtime_policy_defaults
from prism_challenge.weights import (
    DEFAULT_ARCHITECTURE_REWARD_WEIGHT,
    DEFAULT_TRAINING_REWARD_WEIGHT,
    get_weights,
)

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
    crown_status: str = "none",
    param_ladder_stage: str = "explore",
    package_pin: str = "",
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO architecture_families("
            "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
            "owner_submission_id, canonical_submission_id, q_arch_best, display_name, "
            "created_at, updated_at, crown_status, param_ladder_stage, package_pin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                crown_status,
                param_ladder_stage,
                package_pin or f"pin-{architecture_id}",
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
    crown_status: str = "none",
    param_ladder_stage: str = "explore",
    package_pin: str = "",
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
            "metric_mean, metric_std, is_current_best, created_at, updated_at, "
            "crown_status, param_ladder_stage, package_pin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                crown_status,
                param_ladder_stage,
                package_pin or f"pin-{variant_id}",
            ),
        )


def test_default_split_constants_are_half_half() -> None:
    assert DEFAULT_ARCHITECTURE_REWARD_WEIGHT == pytest.approx(0.50)
    assert DEFAULT_TRAINING_REWARD_WEIGHT == pytest.approx(0.50)
    assert DEFAULT_ARCHITECTURE_REWARD_WEIGHT + DEFAULT_TRAINING_REWARD_WEIGHT == pytest.approx(1.0)


def test_settings_reward_weights_default_half_half() -> None:
    settings = PrismSettings()
    assert settings.architecture_reward_weight == pytest.approx(0.50)
    assert settings.training_reward_weight == pytest.approx(0.50)


def test_runtime_reward_pools_default_half_half() -> None:
    pools = runtime_policy_defaults(PrismSettings())["reward_pools"]
    assert pools["architecture"] == pytest.approx(0.50)
    assert pools["training"] == pytest.approx(0.50)


def test_build_weights_loader_defaults_half_half() -> None:
    # Signature defaults must match settings / runtime pools (VAL-RESLAB-008).
    import inspect

    sig = inspect.signature(build_weights_loader)
    assert sig.parameters["architecture_weight"].default == pytest.approx(0.50)
    assert sig.parameters["training_weight"].default == pytest.approx(0.50)


def test_promote_path_decision_confirm_and_revoke() -> None:
    assert (
        promote_path_decision(
            provisional_stage=STAGE_EXPLORE,
            promote_stage=STAGE_PROMOTE,
            promote_valid=True,
            promote_beats_provisional=True,
        )
        == "confirm"
    )
    assert (
        promote_path_decision(
            provisional_stage=STAGE_EXPLORE,
            promote_stage=STAGE_PROMOTE,
            promote_valid=True,
            promote_beats_provisional=False,
        )
        == "revoke"
    )
    assert (
        promote_path_decision(
            provisional_stage=STAGE_EXPLORE,
            promote_stage=STAGE_PROMOTE,
            promote_valid=False,
        )
        == "revoke"
    )


def test_resolve_crown_status_explore_provisional() -> None:
    pin = resolve_package_pin(family_hash="fh-a", package_pin="pkg-1")
    status = resolve_crown_status_transition(
        previous_status="none",
        previous_stage=None,
        previous_pin=None,
        incoming_stage=STAGE_EXPLORE,
        incoming_pin=pin,
        score_valid=True,
    )
    assert status == CROWN_STATUS_PROVISIONAL
    assert crown_status_is_weight_eligible(status)


def test_resolve_crown_status_promote_confirm_same_pin() -> None:
    pin = "pkg-family-1"
    confirmed = resolve_crown_status_transition(
        previous_status=CROWN_STATUS_PROVISIONAL,
        previous_stage=STAGE_EXPLORE,
        previous_pin=pin,
        incoming_stage=STAGE_PROMOTE,
        incoming_pin=pin,
        score_valid=True,
        score_beats_previous=True,
    )
    assert confirmed == CROWN_STATUS_CONFIRMED
    assert crown_status_is_weight_eligible(confirmed)


def test_resolve_crown_status_promote_revoke_loss() -> None:
    pin = "pkg-family-1"
    revoked = resolve_crown_status_transition(
        previous_status=CROWN_STATUS_PROVISIONAL,
        previous_stage=STAGE_EXPLORE,
        previous_pin=pin,
        incoming_stage=STAGE_PROMOTE,
        incoming_pin=pin,
        score_valid=True,
        score_beats_previous=False,
    )
    assert revoked == CROWN_STATUS_REVOKED
    assert not crown_status_is_weight_eligible(revoked)


def test_resolve_crown_status_promote_invalid_revokes_provisional() -> None:
    pin = "pkg-family-1"
    revoked = resolve_crown_status_transition(
        previous_status=CROWN_STATUS_PROVISIONAL,
        previous_stage=STAGE_EXPLORE,
        previous_pin=pin,
        incoming_stage=STAGE_PROMOTE,
        incoming_pin=pin,
        score_valid=False,
        score_beats_previous=False,
    )
    assert revoked == CROWN_STATUS_REVOKED


async def test_provisional_crown_populates_weights_map(tmp_path: Path) -> None:
    """VAL-RESLAB-004: qualifying explore provisional may occupy architecture/training crowns."""
    repository = await _new_repository(tmp_path, "provisional-weights.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-prov",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_PROVISIONAL,
        param_ladder_stage=STAGE_EXPLORE,
        package_pin="pin-prov",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-prov",
        architecture_id="arch-prov",
        owner_hotkey="train-owner",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_PROVISIONAL,
        param_ladder_stage=STAGE_EXPLORE,
        package_pin="pin-prov",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights["arch-owner"] == pytest.approx(0.50)
    assert weights["train-owner"] == pytest.approx(0.50)
    assert sum(weights.values()) == pytest.approx(1.0)


async def test_promote_confirm_keeps_weights(tmp_path: Path) -> None:
    """VAL-RESLAB-005 confirm: promote lasting crown stays weight-eligible at 0.50/0.50."""
    repository = await _new_repository(tmp_path, "confirm-weights.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-ok",
        owner_hotkey="arch-owner",
        q_arch_best=0.95,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_CONFIRMED,
        param_ladder_stage=STAGE_PROMOTE,
        package_pin="pin-ok",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-ok",
        architecture_id="arch-ok",
        owner_hotkey="train-owner",
        q_recipe=0.9,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_CONFIRMED,
        param_ladder_stage=STAGE_PROMOTE,
        package_pin="pin-ok",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)
    assert set(weights) == {"arch-owner", "train-owner"}
    assert weights["arch-owner"] == pytest.approx(0.50)


async def test_promote_revoke_removes_dead_provisional_from_weights(tmp_path: Path) -> None:
    """VAL-RESLAB-005 revoke: get_weights must not keep dead provisional after fail/loss."""
    repository = await _new_repository(tmp_path, "revoke-weights.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-dead",
        owner_hotkey="dead-owner",
        q_arch_best=0.99,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_REVOKED,
        param_ladder_stage=STAGE_PROMOTE,
        package_pin="pin-dead",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-dead",
        architecture_id="arch-dead",
        owner_hotkey="dead-train",
        q_recipe=0.99,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_REVOKED,
        param_ladder_stage=STAGE_PROMOTE,
        package_pin="pin-dead",
    )

    assert await get_weights(repository, EPOCH_SECONDS) == {}

    # A surviving different-family provisional takes the map instead.
    await _seed_architecture(
        repository,
        architecture_id="arch-live",
        owner_hotkey="live-owner",
        q_arch_best=0.5,
        created_at="2024-02-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_PROVISIONAL,
        param_ladder_stage=STAGE_EXPLORE,
        package_pin="pin-live",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-live",
        architecture_id="arch-live",
        owner_hotkey="live-train",
        q_recipe=0.4,
        created_at="2024-02-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_PROVISIONAL,
        param_ladder_stage=STAGE_EXPLORE,
        package_pin="pin-live",
    )
    weights = await get_weights(repository, EPOCH_SECONDS)
    assert set(weights) == {"live-owner", "live-train"}
    assert "dead-owner" not in weights
    assert "dead-train" not in weights


async def test_best_architecture_skips_revoked(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "best-skip-revoked.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="high-revoked",
        owner_hotkey="revoked-hk",
        q_arch_best=0.99,
        created_at="2024-01-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_REVOKED,
    )
    await _seed_architecture(
        repository,
        architecture_id="low-ok",
        owner_hotkey="ok-hk",
        q_arch_best=0.4,
        created_at="2024-02-01T00:00:00+00:00",
        crown_status=CROWN_STATUS_PROVISIONAL,
    )
    best = await repository.best_architecture()
    assert best is not None
    assert best["id"] == "low-ok"
    assert best["crown_status"] == CROWN_STATUS_PROVISIONAL


@pytest.mark.asyncio
async def test_raw_weight_map_still_base_contract_legal(tmp_path: Path) -> None:
    """VAL-RESLAB-009: pushed weights remain hotkey→non-negative finite under Base ingress."""
    database = Database(tmp_path / "raw-push.sqlite3")
    await database.init()
    client = RawWeightPushClient(
        database=database,
        challenge_slug="prism",
        master_base_url="http://master.test",
        shared_token="tok",
    )
    payload, raw = client._build_payload(
        weights={"5CkeyArch": 0.5, "5CkeyTrain": 0.5},
        epoch=1,
        revision=1,
        nonce="n-reslab",
        now=__import__("datetime").datetime.now(__import__("datetime").UTC).replace(microsecond=0),
    )
    again = RawWeightPushRequest.model_validate_json(raw)
    assert again.payload_digest == payload.payload_digest
    assert all(value >= 0.0 for value in again.weights.values())
    assert all(isinstance(key, str) and key for key in again.weights)
    assert "uids" not in raw.decode("utf-8")


def test_maybe_build_push_client_uses_settings_half_half(tmp_path: Path) -> None:
    database_path = tmp_path / "push-settings.sqlite3"

    class _Settings:
        raw_weight_push_enabled = True
        master_base_url = "http://master.test"
        worker_plane = type("WP", (), {"master_base_url": None})()
        slug = "prism"
        epoch_seconds = 3600
        # Defaults must be 0.50 when omitted; explicit settings already half/half.
        architecture_reward_weight = 0.5
        training_reward_weight = 0.5
        raw_weight_push_interval_seconds = 5.0

        def internal_token(self) -> str:
            return "tok"

    # Lazy: construct after Database.init would happen at app start; here just settings path.
    import anyio

    async def _run() -> None:
        db = Database(database_path)
        await db.init()
        client = maybe_build_push_client_from_settings(
            settings=_Settings(), database=db, repository=object()
        )
        assert client is not None
        assert client.weights_fn is not None

    anyio.run(_run)
