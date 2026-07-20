"""Durable full agent-log persistence + the broker-mode stream-disabled warning.

Three operator-facing guarantees:

* the COMPLETE evaluated-agent stdout/stderr for an attempt is written to a
  durable file under the data volume and is retrievable after the run (the
  ``TaskResult`` only keeps a size-capped tail);
* when real-time log streaming is DISABLED in broker mode, a single WARNING is
  emitted so operators know the live SSE feed is dark;
* the best-effort log streamer logs (does not silently swallow) transport
  failures at WARNING.
"""

from __future__ import annotations

import logging
import urllib.error
from pathlib import Path

import pytest

from agent_challenge.evaluation import runner as runner_mod
from agent_challenge.evaluation.own_runner import log_streamer as log_streamer_mod
from agent_challenge.evaluation.own_runner.log_streamer import LogStreamer
from agent_challenge.evaluation.runner import (
    agent_execution_log_paths,
    persist_agent_execution_logs,
    read_agent_execution_logs,
)

_RUNNER_LOGGER = "agent_challenge.evaluation.runner"
_STREAMER_LOGGER = "agent_challenge.evaluation.own_runner.log_streamer"


# ---------------------------------------------------------------------------
# Full stdout/stderr persistence (no truncation) + retrieval
# ---------------------------------------------------------------------------
def test_persist_and_read_full_agent_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner_mod.settings, "data_dir", str(tmp_path))
    # Well beyond the 64k broker/DB tail cap: the durable file keeps it all.
    stdout = "agent stdout line\n" * 100_000
    stderr = "agent stderr line\n" * 20_000

    persist_agent_execution_logs(4242, stdout, stderr)

    stdout_path, stderr_path = agent_execution_log_paths(4242)
    assert stdout_path == tmp_path / "agent-logs" / "4242.stdout.log"
    assert stderr_path == tmp_path / "agent-logs" / "4242.stderr.log"
    assert stdout_path.read_text(encoding="utf-8") == stdout
    assert stderr_path.read_text(encoding="utf-8") == stderr
    assert read_agent_execution_logs(4242) == (stdout, stderr)


def test_read_missing_agent_logs_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod.settings, "data_dir", str(tmp_path))
    assert read_agent_execution_logs(999) is None


def test_persist_preserves_benchmark_result_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod.settings, "data_dir", str(tmp_path))
    stdout = 'noise\nBASE_BENCHMARK_RESULT={"score": 1.0}\nmore noise\n'

    persist_agent_execution_logs(1, stdout, "")

    persisted = read_agent_execution_logs(1)
    assert persisted is not None
    assert 'BASE_BENCHMARK_RESULT={"score": 1.0}' in persisted[0]


def test_persist_agent_logs_best_effort_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Point data_dir at a regular FILE so mkdir under it fails with OSError.
    bad = tmp_path / "not-a-dir"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setattr(runner_mod.settings, "data_dir", str(bad))

    with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
        persist_agent_execution_logs(7, "out", "err")  # must not raise

    assert any("failed to persist agent execution logs" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Broker-mode "streaming disabled" WARNING (once)
# ---------------------------------------------------------------------------
def test_stream_env_warns_once_when_disabled_in_broker_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runner_mod.settings, "docker_backend", "broker")
    monkeypatch.setattr(runner_mod.settings, "terminal_bench_log_stream_url", None)
    monkeypatch.setattr(runner_mod, "_stream_disabled_warned", False)

    with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
        assert runner_mod._terminal_bench_stream_env(5) == {}
        assert runner_mod._terminal_bench_stream_env(6) == {}

    disabled = [r for r in caplog.records if "streaming is DISABLED" in r.getMessage()]
    assert len(disabled) == 1  # warned exactly once despite two calls


def test_stream_env_warns_in_broker_mode_when_token_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runner_mod.settings, "docker_backend", "broker")
    monkeypatch.setattr(
        runner_mod.settings, "terminal_bench_log_stream_url", "http://challenge:8000/"
    )
    monkeypatch.setattr(runner_mod.settings, "shared_token", None)
    monkeypatch.setattr(runner_mod.settings, "shared_token_file", None)
    monkeypatch.setattr(runner_mod, "_stream_disabled_warned", False)

    with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
        assert runner_mod._terminal_bench_stream_env(5) == {}

    assert any("streaming is DISABLED" in r.getMessage() for r in caplog.records)


def test_stream_env_no_warning_when_not_broker_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runner_mod.settings, "docker_backend", "cli")
    monkeypatch.setattr(runner_mod.settings, "terminal_bench_log_stream_url", None)
    monkeypatch.setattr(runner_mod, "_stream_disabled_warned", False)

    with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
        assert runner_mod._terminal_bench_stream_env(5) == {}

    assert not [r for r in caplog.records if "streaming is DISABLED" in r.getMessage()]


# ---------------------------------------------------------------------------
# Streamer transport failures are logged at WARNING (not silently swallowed)
# ---------------------------------------------------------------------------
def test_streamer_logs_warning_on_transport_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom(request: object, timeout: float | None = None) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _boom)
    streamer = LogStreamer(
        base_url="http://challenge:8000",
        attempt_id=42,
        token="scoped-token",
        slug="agent-challenge",
    )

    with caplog.at_level(logging.WARNING, logger=_STREAMER_LOGGER):
        # Best-effort: the failure is logged but never propagates.
        streamer.emit([{"kind": "log", "stream": "agent", "message": "hi"}])

    assert any("log-stream POST" in r.getMessage() for r in caplog.records)
