"""Unit tests for LiumTrainingSettings (master-owned Prism Lium training guards)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from base.config.loader import load_settings
from base.config.settings import LiumTrainingSettings, Settings


def test_lium_training_settings_defaults() -> None:
    settings = LiumTrainingSettings()
    assert settings.enabled is False
    assert settings.api_key is None
    assert settings.api_key_file is None
    assert settings.max_price_per_hour == 1.50
    assert settings.max_lifetime_hours == 4.0
    assert settings.concurrency_cap == 3
    assert settings.daily_spend_ceiling_usd == 50.0
    assert settings.queue_poll_seconds == 30
    assert settings.max_queue_age_hours == 48.0
    assert settings.pod_name_prefix == "prism-train-"
    assert settings.ssh_public_key_file is None


def test_lium_training_settings_rejects_non_positive_price() -> None:
    with pytest.raises(ValidationError):
        LiumTrainingSettings(max_price_per_hour=0)
    with pytest.raises(ValidationError):
        LiumTrainingSettings(max_price_per_hour=-1.0)


def test_lium_training_settings_rejects_lifetime_below_one() -> None:
    with pytest.raises(ValidationError):
        LiumTrainingSettings(max_lifetime_hours=0.5)
    with pytest.raises(ValidationError):
        LiumTrainingSettings(max_lifetime_hours=0)


def test_lium_training_settings_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValidationError):
        LiumTrainingSettings(concurrency_cap=0)
    with pytest.raises(ValidationError):
        LiumTrainingSettings(concurrency_cap=-1)


def test_lium_training_settings_rejects_non_positive_spend_ceiling() -> None:
    with pytest.raises(ValidationError):
        LiumTrainingSettings(daily_spend_ceiling_usd=0)
    with pytest.raises(ValidationError):
        LiumTrainingSettings(daily_spend_ceiling_usd=-10.0)


def test_lium_training_settings_rejects_non_positive_poll() -> None:
    with pytest.raises(ValidationError):
        LiumTrainingSettings(queue_poll_seconds=0)
    with pytest.raises(ValidationError):
        LiumTrainingSettings(queue_poll_seconds=-5)


def test_lium_training_settings_rejects_empty_prefix() -> None:
    with pytest.raises(ValidationError):
        LiumTrainingSettings(pod_name_prefix="")
    with pytest.raises(ValidationError):
        LiumTrainingSettings(pod_name_prefix="   ")


def test_lium_training_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # Convention: BASE_ prefix + nested path joined by __ (see loader._apply_env).
    monkeypatch.setenv("BASE_LIUM_TRAINING__ENABLED", "true")
    monkeypatch.setenv("BASE_LIUM_TRAINING__MAX_PRICE_PER_HOUR", "2.25")
    monkeypatch.setenv("BASE_LIUM_TRAINING__CONCURRENCY_CAP", "2")
    loaded = load_settings()
    assert loaded.lium_training.enabled is True
    assert loaded.lium_training.max_price_per_hour == 2.25
    assert loaded.lium_training.concurrency_cap == 2


def test_lium_training_settings_attached_to_root() -> None:
    root = Settings()
    assert isinstance(root.lium_training, LiumTrainingSettings)
    assert root.lium_training.enabled is False
