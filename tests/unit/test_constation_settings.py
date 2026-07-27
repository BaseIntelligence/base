"""Unit tests for ConstationSettings (production constation orchestration config)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from base.config.loader import load_settings
from base.config.settings import ConstationSettings, Settings


def test_constation_settings_defaults() -> None:
    settings = ConstationSettings()
    assert settings.enabled is False
    assert settings.gap_budget_seconds == 30.0
    assert settings.sidecar_internal_port == 8787
    assert settings.sidecar_scheme == "http"


def test_constation_settings_requires_positive_gap() -> None:
    with pytest.raises(ValidationError):
        ConstationSettings(gap_budget_seconds=0)


def test_constation_settings_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError):
        ConstationSettings(min_interval_seconds=20.0, max_interval_seconds=5.0)


def test_constation_settings_rejects_bad_port() -> None:
    with pytest.raises(ValidationError):
        ConstationSettings(sidecar_internal_port=0)
    with pytest.raises(ValidationError):
        ConstationSettings(sidecar_internal_port=70000)


def test_constation_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # Convention: BASE_ prefix + nested path joined by __ (see loader._apply_env).
    monkeypatch.setenv("BASE_CONSTATION__GAP_BUDGET_SECONDS", "45.5")
    loaded = load_settings()
    assert loaded.constation.gap_budget_seconds == 45.5


def test_constation_settings_attached_to_root() -> None:
    root = Settings()
    assert isinstance(root.constation, ConstationSettings)
    assert root.constation.enabled is False
