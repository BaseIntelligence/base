"""Phase 3/4 own_runner log production + real-time streaming (no docker).

Covers the pieces that turn a finished trial into complete, correctly-channeled
logs and stream them in real time WITHOUT changing the parity reward path:

* :class:`LogStreamer` env wiring + best-effort POST (mocked transport);
* ``build_log_events`` / ``trial_log_channels`` channel mapping;
* the orchestrator's best-effort ``trial_listener`` hook;
* ``_persist_trial`` writing the per-channel files + a derived ``score`` key;
* the per-attempt SCOPED stream token (never the raw shared token);
* the dispatcher's ``_terminal_bench_stream_env`` injection.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from agent_challenge.evaluation import runner as runner_mod
from agent_challenge.evaluation.own_runner import log_streamer as log_streamer_mod
from agent_challenge.evaluation.own_runner.log_streamer import (
    LogStreamer,
    build_incremental_log_event,
    build_log_events,
)
from agent_challenge.evaluation.own_runner.orchestrator import (
    JobConfig,
    TaskSpec,
    TrialId,
    TrialJobOrchestrator,
    TrialOutcome,
    _bind_incremental,
    trial_log_channels,
)
from agent_challenge.evaluation.own_runner.verifier_runner import _read_test_stdout
from agent_challenge.evaluation.own_runner_backend import _build_incremental_emitter
from agent_challenge.sdk.auth import (
    mint_attempt_stream_token,
    verify_attempt_stream_token,
)

# ---------------------------------------------------------------------------
# LogStreamer.from_env
# ---------------------------------------------------------------------------
_FULL_ENV = {
    "BASE_LOG_STREAM_URL": "http://challenge:8000/",
    "BASE_LOG_STREAM_ATTEMPT_ID": "42",
    "BASE_LOG_STREAM_TOKEN": "scoped-token",
    "BASE_LOG_STREAM_SLUG": "agent-challenge",
}


def test_from_env_builds_streamer_and_url() -> None:
    streamer = LogStreamer.from_env(_FULL_ENV)
    assert streamer is not None
    assert streamer.base_url == "http://challenge:8000"  # trailing slash trimmed
    assert streamer.attempt_id == 42
    assert streamer.url == "http://challenge:8000/internal/v1/evaluations/42/events"


@pytest.mark.parametrize("missing", sorted(_FULL_ENV))
def test_from_env_disabled_when_any_var_missing(missing: str) -> None:
    env = {key: value for key, value in _FULL_ENV.items() if key != missing}
    assert LogStreamer.from_env(env) is None


def test_from_env_disabled_on_non_integer_attempt() -> None:
    env = {**_FULL_ENV, "BASE_LOG_STREAM_ATTEMPT_ID": "not-an-int"}
    assert LogStreamer.from_env(env) is None


def test_emit_posts_ndjson_with_scoped_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_urlopen(request: object, timeout: float | None = None) -> _Resp:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["data"] = request.data  # type: ignore[attr-defined]
        captured["headers"] = request.headers  # type: ignore[attr-defined]
        captured["method"] = request.get_method()  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _fake_urlopen)

    streamer = LogStreamer.from_env(_FULL_ENV)
    assert streamer is not None
    streamer.emit([{"kind": "log", "stream": "agent", "message": "hi"}])

    assert captured["method"] == "POST"
    assert captured["url"] == "http://challenge:8000/internal/v1/evaluations/42/events"
    # NDJSON body, one compact JSON object per line.
    assert json.loads(captured["data"].decode("utf-8")) == {  # type: ignore[union-attr]
        "kind": "log",
        "stream": "agent",
        "message": "hi",
    }
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer scoped-token"  # type: ignore[index]
    assert headers["X-base-challenge-slug"] == "agent-challenge"  # type: ignore[index]


def test_emit_swallows_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(request: object, timeout: float | None = None) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _boom)
    streamer = LogStreamer.from_env(_FULL_ENV)
    assert streamer is not None
    # Best-effort: a transport failure must never propagate.
    streamer.emit([{"kind": "log", "stream": "agent", "message": "hi"}])


def test_emit_noop_on_empty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(request: object, timeout: float | None = None) -> None:
        raise AssertionError("urlopen must not be called for an empty batch")

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _fail)
    streamer = LogStreamer.from_env(_FULL_ENV)
    assert streamer is not None
    streamer.emit([])


# ---------------------------------------------------------------------------
# build_log_events / trial_log_channels
# ---------------------------------------------------------------------------
def test_build_log_events_one_per_nonempty_channel() -> None:
    events = build_log_events(
        trial_name="t__attempt-0",
        task_id="hello-world",
        status="completed",
        channels={"agent": "a", "harness": "  ", "test_stdout": "out"},
    )
    streams = [event["stream"] for event in events]
    assert streams == ["agent", "test_stdout"]  # blank harness dropped
    assert all(event["kind"] == "log" for event in events)
    assert all(event["task_id"] == "hello-world" for event in events)


def test_trial_log_channels_maps_agent_harness_test_stdout() -> None:
    outcome = TrialOutcome(
        task_name="hello-world",
        trial_name="hello-world__attempt-0",
        status="completed",
        rewards={"reward": 1.0},
        agent_output="agent says hi",
        verifier_stdout="2 passed",
        verifier_return_code=0,
    )
    channels = trial_log_channels(outcome)
    assert channels["agent"] == "agent says hi"
    assert channels["test_stdout"] == "2 passed"
    # harness is the synthesized trial.log summary.
    assert "status: completed" in channels["harness"]
    assert "reward: 1.0" in channels["harness"]


def test_trial_log_channels_appends_exception_to_harness() -> None:
    outcome = TrialOutcome(
        task_name="hello-world",
        trial_name="hello-world__attempt-0",
        status="failed",
        rewards=None,
        errored=True,
        reason_code="harbor_agent_timeout_error",
        error_text="boom traceback",
    )
    channels = trial_log_channels(outcome)
    assert "boom traceback" in channels["harness"]
    assert "agent" not in channels  # no agent output captured


# ---------------------------------------------------------------------------
# build_incremental_log_event (real-time live-pane delta)
# ---------------------------------------------------------------------------
def test_build_incremental_log_event_shape_and_agent_stream() -> None:
    event = build_incremental_log_event(
        trial_name="t__attempt-0",
        task_id="hello-world",
        stream="agent",
        message="fresh pane output",
    )
    assert event == {
        "kind": "log",
        "trial_name": "t__attempt-0",
        "task_id": "hello-world",
        "stream": "agent",
        "message": "fresh pane output",
    }
    # A mid-trial delta has no terminal status.
    assert "status" not in event


def test_build_incremental_log_event_includes_status_when_given() -> None:
    event = build_incremental_log_event(
        trial_name="t__attempt-0",
        task_id="hello-world",
        stream="agent",
        message="x",
        status="running",
    )
    assert event["status"] == "running"


# ---------------------------------------------------------------------------
# Backend incremental emitter + orchestrator per-trial binding
# ---------------------------------------------------------------------------
async def test_incremental_emitter_posts_agent_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_urlopen(request: object, timeout: float | None = None) -> _Resp:
        captured["data"] = request.data  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _fake_urlopen)

    emitter = _build_incremental_emitter(LogStreamer.from_env(_FULL_ENV))
    assert emitter is not None
    # Runs the blocking POST off the event loop (asyncio.to_thread).
    await emitter("trial__attempt-0", "hello-world", "fresh pane text")

    body = json.loads(captured["data"].decode("utf-8"))  # type: ignore[union-attr]
    assert body["kind"] == "log"
    assert body["stream"] == "agent"
    assert body["message"] == "fresh pane text"
    assert body["trial_name"] == "trial__attempt-0"
    assert body["task_id"] == "hello-world"


def test_incremental_emitter_none_when_streamer_absent() -> None:
    assert _build_incremental_emitter(None) is None


async def test_incremental_emitter_redacts_gateway_token_and_miner_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The live agent-pane delta stream must be routed through LogRedactor before
    # emit, so the scoped gateway token + miner-env values cannot leak into the
    # live feed if CVM streaming is enabled (final-outcome path already redacted).
    from agent_challenge.evaluation.own_runner.redaction import (
        REDACTED_GATEWAY_TOKEN,
        REDACTED_MINER_ENV,
        LogRedactor,
    )

    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_urlopen(request: object, timeout: float | None = None) -> _Resp:
        captured["data"] = request.data  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _fake_urlopen)

    redactor = LogRedactor(
        gateway_token="scoped-gw-token-9f2a", miner_env_values=["miner-secret-xyz"]
    )
    emitter = _build_incremental_emitter(LogStreamer.from_env(_FULL_ENV), redactor)
    assert emitter is not None
    await emitter(
        "trial__attempt-0",
        "hello-world",
        "pane leaking scoped-gw-token-9f2a and miner-secret-xyz",
    )

    body = json.loads(captured["data"].decode("utf-8"))  # type: ignore[union-attr]
    assert "scoped-gw-token-9f2a" not in body["message"]
    assert "miner-secret-xyz" not in body["message"]
    assert REDACTED_GATEWAY_TOKEN in body["message"]
    assert REDACTED_MINER_ENV in body["message"]


async def test_incremental_emitter_passthrough_when_redactor_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An inactive redactor (no secrets) must not alter the delta text.
    from agent_challenge.evaluation.own_runner.redaction import LogRedactor

    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_urlopen(request: object, timeout: float | None = None) -> _Resp:
        captured["data"] = request.data  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr(log_streamer_mod.urllib.request, "urlopen", _fake_urlopen)

    emitter = _build_incremental_emitter(LogStreamer.from_env(_FULL_ENV), LogRedactor())
    assert emitter is not None
    await emitter("trial__attempt-0", "hello-world", "ordinary pane output")

    body = json.loads(captured["data"].decode("utf-8"))  # type: ignore[union-attr]
    assert body["message"] == "ordinary pane output"


async def test_bind_incremental_tags_delta_with_trial_identity() -> None:
    seen: list[tuple[str, str, str]] = []

    async def _emitter(trial_name: str, task_id: str, delta: str) -> None:
        seen.append((trial_name, task_id, delta))

    bound = _bind_incremental(_emitter, TrialId("my-task", 3), TaskSpec("my-task"))
    assert bound is not None
    await bound("chunk")
    assert seen == [(TrialId("my-task", 3).trial_name, "my-task", "chunk")]


def test_bind_incremental_none_passthrough() -> None:
    assert _bind_incremental(None, TrialId("t", 0), TaskSpec("t")) is None


# ---------------------------------------------------------------------------
# Orchestrator trial_listener hook + _persist_trial channels + score key
# ---------------------------------------------------------------------------
def _stub_runner(outcome: TrialOutcome):
    async def _run(trial_id: TrialId, task: TaskSpec) -> TrialOutcome:
        return outcome

    return _run


async def test_trial_listener_invoked_after_persist(tmp_path: Path) -> None:
    outcome = TrialOutcome(
        task_name="task",
        trial_name=TrialId("task", 0).trial_name,
        status="completed",
        rewards={"reward": 1.0},
        agent_output="hi",
        verifier_stdout="ok",
    )
    seen: list[tuple[str, str]] = []

    async def _listener(trial_id: TrialId, got: TrialOutcome) -> None:
        seen.append((trial_id.trial_name, got.status))

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1),
        job_dir=tmp_path / "job",
        trial_runner=_stub_runner(outcome),
        trial_listener=_listener,
    )
    await orch.run([TaskSpec("task")])
    assert seen == [(TrialId("task", 0).trial_name, "completed")]


async def test_trial_listener_failure_never_breaks_run(tmp_path: Path) -> None:
    outcome = TrialOutcome(
        task_name="task",
        trial_name=TrialId("task", 0).trial_name,
        status="completed",
        rewards={"reward": 1.0},
    )

    async def _listener(trial_id: TrialId, got: TrialOutcome) -> None:
        raise RuntimeError("listener exploded")

    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1),
        job_dir=tmp_path / "job",
        trial_runner=_stub_runner(outcome),
        trial_listener=_listener,
    )
    result = await orch.run([TaskSpec("task")])
    assert result.status == "completed"
    assert result.score == 1.0


async def test_persist_trial_writes_all_log_channels_and_score(tmp_path: Path) -> None:
    outcome = TrialOutcome(
        task_name="task",
        trial_name=TrialId("task", 0).trial_name,
        status="completed",
        rewards={"reward": 1.0},
        agent_output="agent log body",
        verifier_stdout="3 passed",
        verifier_return_code=0,
        error_text="non-fatal note",
    )
    orch = TrialJobOrchestrator(
        config=JobConfig(n_attempts=1),
        job_dir=tmp_path / "job",
        trial_runner=_stub_runner(outcome),
    )
    await orch.run([TaskSpec("task")])

    trial_dir = tmp_path / "job" / "trials" / TrialId("task", 0).trial_name
    assert (trial_dir / "agent" / "agent.log").read_text() == "agent log body"
    assert (trial_dir / "verifier" / "test-stdout.txt").read_text() == "3 passed"
    assert (trial_dir / "exception.txt").read_text() == "non-fatal note"
    assert "status: completed" in (trial_dir / "trial.log").read_text()

    # result.json carries a host-readable derived ``score`` AND the verbatim
    # ``rewards`` (parity path); the log channel fields are NOT in result.json.
    data = json.loads((trial_dir / "result.json").read_text())
    assert data["score"] == 1.0
    assert data["rewards"] == {"reward": 1.0}
    assert "verifier_stdout" not in data


# ---------------------------------------------------------------------------
# verifier stdout capture (Phase 3a)
# ---------------------------------------------------------------------------
def test_read_test_stdout_reads_copied_out_file(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "test-stdout.txt").write_text("== 5 passed ==", encoding="utf-8")
    assert _read_test_stdout(verifier_dir) == "== 5 passed =="


def test_read_test_stdout_missing_returns_none(tmp_path: Path) -> None:
    assert _read_test_stdout(tmp_path / "verifier") is None


# ---------------------------------------------------------------------------
# Per-attempt scoped token (never the raw shared token)
# ---------------------------------------------------------------------------
def test_scoped_token_is_deterministic_and_attempt_bound() -> None:
    token = mint_attempt_stream_token("shared-secret", 7)
    assert token == mint_attempt_stream_token("shared-secret", 7)
    # Bound to the attempt id.
    assert token != mint_attempt_stream_token("shared-secret", 8)
    # Bound to the shared secret.
    assert token != mint_attempt_stream_token("other-secret", 7)
    # Never leaks the shared token itself.
    assert "shared-secret" not in token


def test_verify_scoped_token_rejects_cross_attempt() -> None:
    token = mint_attempt_stream_token("shared-secret", 7)
    assert verify_attempt_stream_token("shared-secret", 7, token) is True
    assert verify_attempt_stream_token("shared-secret", 8, token) is False
    assert verify_attempt_stream_token("shared-secret", 7, "garbage") is False


# ---------------------------------------------------------------------------
# Dispatcher: streaming env injection + non-privileged DooD broker limits
# ---------------------------------------------------------------------------
def test_stream_env_empty_when_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod.settings, "terminal_bench_log_stream_url", None)
    assert runner_mod._terminal_bench_stream_env(5) == {}


def test_stream_env_injects_scoped_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_mod.settings, "terminal_bench_log_stream_url", "http://challenge:8000/"
    )
    monkeypatch.setattr(runner_mod.settings, "shared_token", "test-token")
    monkeypatch.setattr(runner_mod.settings, "slug", "agent-challenge")

    env = runner_mod._terminal_bench_stream_env(5)
    assert env["BASE_LOG_STREAM_URL"] == "http://challenge:8000"
    assert env["BASE_LOG_STREAM_ATTEMPT_ID"] == "5"
    assert env["BASE_LOG_STREAM_SLUG"] == "agent-challenge"
    assert env["BASE_LOG_STREAM_TOKEN"] == mint_attempt_stream_token("test-token", 5)
    # The raw shared token is never handed to the job.
    assert env["BASE_LOG_STREAM_TOKEN"] != "test-token"


def test_broker_limits_are_non_privileged_dood() -> None:
    limits = runner_mod._terminal_bench_broker_limits()
    assert limits.read_only is True
    assert limits.privileged is False
    assert any(mount.startswith("/tmp:") for mount in limits.tmpfs)
    assert limits.cap_drop == ("ALL",)
    assert "no-new-privileges" in limits.security_opt
