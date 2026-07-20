from __future__ import annotations

import logging
from pathlib import Path

import pytest

from prism_challenge import app as app_module
from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, configure_logging


def _captured_basicconfig_level(monkeypatch: pytest.MonkeyPatch, settings: PrismSettings) -> object:
    """Return the numeric ``level`` ``configure_logging`` passes to ``logging.basicConfig``.

    Spying on ``basicConfig`` keeps the assertion deterministic and never mutates the real root
    logger (whose handler set pytest owns during a test), so it also documents that
    configure_logging routes solely through the stdlib entrypoint.
    """
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))
    configure_logging(settings)
    return captured.get("level")


def test_log_level_setting_default_is_info() -> None:
    assert PrismSettings(shared_token="x").log_level == "INFO"


def test_log_level_setting_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_LOG_LEVEL", "DEBUG")
    assert PrismSettings(shared_token="x").log_level == "DEBUG"


def test_configure_logging_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    level = _captured_basicconfig_level(monkeypatch, PrismSettings(shared_token="x"))
    assert level == logging.INFO


def test_configure_logging_respects_custom_level(monkeypatch: pytest.MonkeyPatch) -> None:
    level = _captured_basicconfig_level(
        monkeypatch, PrismSettings(shared_token="x", log_level="warning")
    )
    assert level == logging.WARNING


def test_configure_logging_invalid_level_falls_back_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level = _captured_basicconfig_level(
        monkeypatch, PrismSettings(shared_token="x", log_level="NOT_A_LEVEL")
    )
    assert level == logging.INFO


def test_configure_logging_installs_info_handler_on_bare_root() -> None:
    """On a handler-less root (the deploy entrypoint), configure_logging installs an INFO handler.

    pytest re-attaches its capture handler for the call phase, so the bare-root state is emulated by
    clearing handlers inside the test body and restoring them afterwards.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers[:] = []
    try:
        configure_logging(PrismSettings(shared_token="x", log_level="WARNING"))
        assert root.level == logging.WARNING
        assert root.handlers  # a handler was installed on the previously-bare root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_configure_logging_is_noop_when_root_already_has_handlers() -> None:
    """basicConfig (no force) must not displace an entrypoint's existing handlers or its level."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers[:] = []
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    root.setLevel(logging.WARNING)
    try:
        configure_logging(PrismSettings(shared_token="x", log_level="DEBUG"))
        assert root.handlers == [sentinel]
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_create_app_configures_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[PrismSettings] = []
    monkeypatch.setattr(app_module, "configure_logging", lambda s: calls.append(s))
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="x",
    )

    create_app(settings)

    assert calls == [settings]
