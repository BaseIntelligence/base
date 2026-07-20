"""Root logging is configured at both service entrypoints (worker + API).

Uvicorn installs no root handler, so without an explicit ``configure_root_logging``
call the worker service emits ZERO logs and the API swallows all application INFO.
These tests lock in that both entrypoints configure stdlib root logging at the
``CHALLENGE_LOG_LEVEL`` level (default INFO), and that the level is configurable.
"""

from __future__ import annotations

import importlib
import logging
import sys

import pytest

from agent_challenge.core.config import settings
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.observability import configure_root_logging, resolve_log_level


def test_resolve_log_level_names_numbers_and_fallback() -> None:
    assert resolve_log_level("INFO") == logging.INFO
    assert resolve_log_level("info") == logging.INFO
    assert resolve_log_level("DEBUG") == logging.DEBUG
    assert resolve_log_level("WARNING") == logging.WARNING
    assert resolve_log_level("10") == logging.DEBUG
    assert resolve_log_level(logging.ERROR) == logging.ERROR
    # Unknown names fall back to INFO rather than raising.
    assert resolve_log_level("nonsense") == logging.INFO


def test_log_level_setting_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHALLENGE_LOG_LEVEL", raising=False)
    assert ChallengeSettings().log_level == "INFO"


def test_log_level_setting_reads_challenge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHALLENGE_LOG_LEVEL", "DEBUG")
    resolved = ChallengeSettings()
    assert resolved.log_level == "DEBUG"
    assert resolve_log_level(resolved.log_level) == logging.DEBUG


def test_configure_root_logging_sets_level_and_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    # A sibling test or production import may pin this logger's level above
    # INFO; force inheritance so the root-level configuration under test is
    # what getEffectiveLevel() reports.
    app_logger = logging.getLogger("agent_challenge.evaluation.worker")
    original_app_level = app_logger.level
    try:
        app_logger.setLevel(logging.NOTSET)
        monkeypatch.setattr(settings, "log_level", "INFO")
        configure_root_logging(settings)
        assert root.level == logging.INFO
        assert root.handlers, "a root handler must be installed so records are emitted"
        # A representative application logger now propagates at INFO.
        assert app_logger.getEffectiveLevel() <= logging.INFO
    finally:
        app_logger.setLevel(original_app_level)
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_configure_root_logging_honours_configured_level(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    try:
        monkeypatch.setattr(settings, "log_level", "WARNING")
        configure_root_logging(settings)
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_worker_main_configures_root_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.evaluation import worker as worker_mod

    calls: list[object] = []
    monkeypatch.setattr(worker_mod, "configure_root_logging", lambda cfg: calls.append(cfg))

    async def _noop_loop(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(worker_mod, "run_worker_loop", _noop_loop)
    monkeypatch.setattr(sys, "argv", ["agent-challenge-worker", "--once"])

    worker_mod.main()
    assert calls, "worker main() must configure root logging before the loop starts"


def test_api_app_import_configures_root_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_challenge.api.app as api_app

    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    try:
        monkeypatch.setattr(settings, "log_level", "INFO")
        # Simulate a fresh process: no root handler and a non-INFO level.
        root.handlers[:] = []
        root.setLevel(logging.WARNING)
        importlib.reload(api_app)
        assert root.level == logging.INFO
        assert root.handlers, "importing the API app must install a root handler"
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
