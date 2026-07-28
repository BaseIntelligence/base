"""Lium training client/scheduler factories from LiumTrainingSettings."""

from __future__ import annotations

from pathlib import Path

import pytest

from base.compute.lium import LiumClient
from base.compute.lium_capacity import LiumCapacityScheduler
from base.compute.lium_training_wiring import (
    DEFAULT_LIUM_TRAINING_IMAGE,
    build_lium_capacity_scheduler,
    build_lium_training_client,
    resolve_lium_training_api_key,
    run_lium_capacity_tick,
    try_build_lium_capacity_scheduler,
)
from base.compute.worker_deployment import WORKER_IMAGE
from base.config.settings import LiumTrainingSettings, Settings


def test_disabled_returns_none() -> None:
    """Given enabled=False, When build client, Then None."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=False, api_key="k"))
    assert build_lium_training_client(settings) is None


def test_lium_training_enabled_without_key_fail_closed() -> None:
    """Given enabled=True without key, When build, Then raises clear error."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=True))
    with pytest.raises(ValueError, match="api_key|api_key_file|lium_training"):
        build_lium_training_client(settings)

    settings_empty = Settings(
        lium_training=LiumTrainingSettings(enabled=True, api_key="   ")
    )
    with pytest.raises(ValueError, match="api_key|api_key_file|lium_training"):
        build_lium_training_client(settings_empty)


def test_enabled_with_key_returns_locked_client() -> None:
    """Given enabled + api_key, When build, Then training-locked LiumClient."""
    settings = Settings(
        lium_training=LiumTrainingSettings(enabled=True, api_key="test-lium-key")
    )
    client = build_lium_training_client(settings)
    assert isinstance(client, LiumClient)
    assert client._training_gpu_lock is True  # noqa: SLF001
    # Key must never appear in repr
    assert "test-lium-key" not in repr(client)


def test_enabled_with_key_file_returns_locked_client(tmp_path: Path) -> None:
    """Given enabled + api_key_file, When build, Then training-locked client."""
    key_path = tmp_path / "lium.key"
    key_path.write_text("  file-lium-key\n", encoding="utf-8")
    settings = Settings(
        lium_training=LiumTrainingSettings(enabled=True, api_key_file=key_path)
    )
    client = build_lium_training_client(settings)
    assert isinstance(client, LiumClient)
    assert client._training_gpu_lock is True  # noqa: SLF001
    assert "file-lium-key" not in repr(client)


def test_resolve_lium_training_api_key_prefers_inline_over_file(tmp_path: Path) -> None:
    """Given both inline and file, When resolve, Then inline wins (read_secret)."""
    key_path = tmp_path / "lium.key"
    key_path.write_text("file-key", encoding="utf-8")
    lt = LiumTrainingSettings(api_key="inline-key", api_key_file=key_path)
    assert resolve_lium_training_api_key(lt) == "inline-key"


def test_build_lium_capacity_scheduler_maps_settings() -> None:
    """Given settings + factory, When build scheduler, Then fields mapped."""
    settings = Settings(
        lium_training=LiumTrainingSettings(
            enabled=True,
            api_key="sched-key",
            concurrency_cap=5,
            pod_name_prefix="train-x-",
            max_price_per_hour=2.25,
            max_lifetime_hours=6.0,
        )
    )
    scheduler = build_lium_capacity_scheduler(settings)
    assert isinstance(scheduler, LiumCapacityScheduler)
    assert scheduler._concurrency_cap == 5  # noqa: SLF001
    assert scheduler._pod_name_prefix == "train-x-"  # noqa: SLF001
    assert scheduler._max_price_per_hour == 2.25  # noqa: SLF001
    assert scheduler._max_lifetime_hours == 6.0  # noqa: SLF001
    assert scheduler._image == DEFAULT_LIUM_TRAINING_IMAGE  # noqa: SLF001
    assert DEFAULT_LIUM_TRAINING_IMAGE == WORKER_IMAGE
    assert "prism-train:latest" not in scheduler._image  # noqa: SLF001

    client = scheduler._client_factory()  # noqa: SLF001
    assert isinstance(client, LiumClient)
    assert client._training_gpu_lock is True  # noqa: SLF001


def test_build_lium_capacity_scheduler_disabled_fail_closed() -> None:
    """Given enabled=False, When build scheduler, Then raises (no silent empty)."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=False))
    with pytest.raises(ValueError, match="lium_training|enabled"):
        build_lium_capacity_scheduler(settings)


def test_try_build_lium_capacity_scheduler_disabled_returns_none() -> None:
    """Given enabled=False, When try_build, Then None (no raise)."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=False))
    assert try_build_lium_capacity_scheduler(settings) is None


def test_try_build_lium_capacity_scheduler_enabled_builds() -> None:
    """Given enabled + key, When try_build, Then scheduler instance."""
    settings = Settings(
        lium_training=LiumTrainingSettings(enabled=True, api_key="try-key")
    )
    scheduler = try_build_lium_capacity_scheduler(settings)
    assert isinstance(scheduler, LiumCapacityScheduler)


def test_try_build_lium_capacity_scheduler_missing_key_returns_none() -> None:
    """Given enabled without key, When try_build, Then None (log + soft fail)."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=True))
    assert try_build_lium_capacity_scheduler(settings) is None


async def test_run_lium_capacity_tick_none_is_noop() -> None:
    """Given None scheduler, When tick helper, Then no raise."""
    await run_lium_capacity_tick(None)
