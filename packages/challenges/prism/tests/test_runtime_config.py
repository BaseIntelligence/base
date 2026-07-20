from __future__ import annotations

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator.dataset import FINEWEB_EDU_SUBSETS
from prism_challenge.evaluator.modes import (
    FULL_SCALE_PHASE_1_TOKEN_TARGET,
    FULL_SCALE_PHASE_2_TOKEN_TARGET,
    GPU_PROXY_TOKEN_TARGET,
)
from prism_challenge.repository import PrismRepository
from prism_challenge.runtime_config import RuntimeConfigError, runtime_policy_defaults


@pytest.fixture
async def repository(tmp_path):
    database = Database(tmp_path / "runtime-config.sqlite3")
    await database.init()
    return PrismRepository(database, epoch_seconds=60)


async def test_sql_runtime_config_overrides_scoring_weights(repository: PrismRepository) -> None:
    settings = PrismSettings(arch_weight=0.7, recipe_weight=0.3)
    await repository.store_runtime_config(
        config_key="score_weights",
        value={"final_architecture_weight": 0.25, "final_recipe_weight": 0.75},
        updated_by="ops",
    )

    runtime_config = await repository.runtime_config(settings)

    assert runtime_config.score_weights.final_architecture_weight == 0.25
    assert runtime_config.score_weights.final_recipe_weight == 0.75


async def test_missing_sql_runtime_config_falls_back_to_settings_defaults(
    repository: PrismRepository,
) -> None:
    settings = PrismSettings(arch_weight=0.6, recipe_weight=0.4, fineweb_sample_count=7)

    runtime_config = await repository.runtime_config(settings)

    assert runtime_config.score_weights.final_architecture_weight == 0.6
    assert runtime_config.score_weights.final_recipe_weight == 0.4
    assert runtime_config.dataset_configs.fineweb_sample_count == 7


async def test_runtime_config_rows_include_audit_fields(repository: PrismRepository) -> None:
    await repository.store_runtime_config(
        config_key="gpu_policy",
        value={"max_gpu_count": 4, "actual_gpu_count": 2},
        updated_by="validator-admin",
        schema_version=2,
        effective_from="2026-05-25T00:00:00+00:00",
    )

    rows = await repository.active_runtime_config_rows(at="2026-05-25T00:00:01+00:00")

    assert rows[0]["config_key"] == "gpu_policy"
    assert rows[0]["updated_by"] == "validator-admin"
    assert rows[0]["schema_version"] == 2
    assert rows[0]["updated_at"]
    assert rows[0]["effective_from"] == "2026-05-25T00:00:00+00:00"
    assert rows[0]["enabled"] == 1


async def test_invalid_sql_gpu_count_fails_closed_for_official_config(
    repository: PrismRepository,
) -> None:
    await repository.store_runtime_config(
        config_key="gpu_policy",
        value={"max_gpu_count": 9, "actual_gpu_count": 1},
        updated_by="ops",
    )

    with pytest.raises(RuntimeConfigError, match="max_gpu_count"):
        await repository.runtime_config(PrismSettings(), official=True)


async def test_invalid_sql_gpu_count_can_fallback_for_explicit_local_safe_path(
    repository: PrismRepository,
) -> None:
    await repository.store_runtime_config(
        config_key="gpu_policy",
        value={"max_gpu_count": 9, "actual_gpu_count": 1},
        updated_by="ops",
    )

    runtime_config = await repository.runtime_config(PrismSettings(), official=False)

    assert runtime_config.gpu_policy.max_gpu_count == 8
    assert runtime_config.gpu_policy.actual_gpu_count == 1


async def test_invalid_sql_weight_sum_fails_closed(repository: PrismRepository) -> None:
    await repository.store_runtime_config(
        config_key="reward_pools",
        value={"architecture": 0.9, "training": 0.9},
        updated_by="ops",
    )

    with pytest.raises(RuntimeConfigError, match="sum to 1.0"):
        await repository.runtime_config(PrismSettings(), official=True)


def test_runtime_execution_mode_defaults_match_evaluator_and_fineweb_contracts() -> None:
    targets = runtime_policy_defaults(PrismSettings())["execution_mode_targets"]
    gpu_proxy = targets["gpu_proxy_eval"]
    full_scale = targets["full_scale_eval"]

    assert "local_cpu_smoke" not in targets
    assert gpu_proxy["max_tokens"] == GPU_PROXY_TOKEN_TARGET
    assert gpu_proxy["dataset_subset"] == "sample-10BT"
    assert gpu_proxy["dataset_tokens"] == FINEWEB_EDU_SUBSETS["sample-10BT"]["token_count"]
    assert gpu_proxy["max_tokens"] == FINEWEB_EDU_SUBSETS["sample-10BT"]["token_count"]
    assert full_scale["max_tokens"] == FULL_SCALE_PHASE_1_TOKEN_TARGET
    assert full_scale["phase_1_max_tokens"] == FULL_SCALE_PHASE_1_TOKEN_TARGET
    assert full_scale["phase_2_max_tokens"] == FULL_SCALE_PHASE_2_TOKEN_TARGET
    assert full_scale["phase_1_dataset_subset"] == "sample-10BT"
    assert full_scale["phase_2_dataset_subset"] == "sample-100BT"
    assert full_scale["phase_1_dataset_tokens"] == FINEWEB_EDU_SUBSETS["sample-10BT"]["token_count"]
    assert (
        full_scale["phase_2_dataset_tokens"] == FINEWEB_EDU_SUBSETS["sample-100BT"]["token_count"]
    )
