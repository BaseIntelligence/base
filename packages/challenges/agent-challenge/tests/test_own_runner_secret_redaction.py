"""Secret-hygiene isolation tests for the in-CVM orchestrator.

Behavioral, black-box tests for the secret-handling isolation invariants the M2
in-CVM orchestrator MUST preserve:

* provider ``*_API_KEY`` (and provider base-url/model) env vars are stripped from
  the environment handed to the agent (VAL-ORCH-018);
* ONLY the LLM gateway allowlist reaches the agent (VAL-ORCH-019);
* the scoped gateway token is redacted from captured stdout/stderr/logs and the
  persisted per-trial output (VAL-ORCH-020);
* miner-supplied env values that surface in task output are redacted from the
  captured logs and persisted output (VAL-ORCH-021);
* a golden digest mismatch fails closed: the task is not executed (no agent, no
  verifier) and yields a failed/aborted result, never a fabricated score
  (VAL-ORCH-017).

These run offline (no docker) with injected preparer/verifier/agent seams.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner.isolation import (
    AGENT_ENV_ALLOWLIST,
    filter_agent_env,
)
from agent_challenge.evaluation.own_runner.orchestrator import (
    PreparedTrial,
    TaskSpec,
    TrialId,
)
from agent_challenge.evaluation.own_runner.redaction import (
    REDACTED_GATEWAY_TOKEN,
    REDACTED_MINER_ENV,
    LogRedactor,
)
from agent_challenge.evaluation.own_runner.taskdefs import DigestMismatch, compute_task_digest
from agent_challenge.evaluation.own_runner.verifier_runner import VerifierOutcome
from agent_challenge.evaluation.own_runner_backend import (
    _build_default_preparer,
    _resolve_agent_gateway_env,
    run_own_runner_job,
)

# VAL-ACAT-013: allowlisted agent env (no Base gateway residuals).
_AGENT_ENV = {
    "LLM_COST_LIMIT": "5",
    "OPENROUTER_API_KEY": "or-measured-key",
}


class _FakeEnv:
    def __init__(self) -> None:
        self.removed = False

    async def exec(self, command: str, **kwargs: Any) -> Any:
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": None})()

    def remove(self) -> None:
        self.removed = True


# --------------------------------------------------------------------------- #
# VAL-ORCH-018 — provider *_API_KEY (and base-url/model) stripped
# --------------------------------------------------------------------------- #
def test_filter_agent_env_strips_provider_api_keys() -> None:
    raw = {
        **_AGENT_ENV,
        "BASE_LLM_GATEWAY_URL": "https://master-gateway.test/llm/v1",
        "BASE_GATEWAY_TOKEN": "scoped",
        "OPENAI_API_KEY": "sk-openai",
        "ANTHROPIC_API_KEY": "sk-anthropic",
        "SOMEPROVIDER_API_KEY": "leak",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "gpt-4",
    }
    filtered = filter_agent_env(raw)
    assert filtered == _AGENT_ENV
    for leaked in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SOMEPROVIDER_API_KEY",
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
    ):
        assert leaked not in filtered
    for provider_cfg in ("OPENAI_BASE_URL", "OPENAI_MODEL"):
        assert provider_cfg not in filtered


def test_resolve_agent_gateway_env_excludes_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in _AGENT_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("HF_API_TOKEN", "hf-leak")
    env = _resolve_agent_gateway_env()
    assert env is not None
    assert set(env) <= AGENT_ENV_ALLOWLIST
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "HF_API_TOKEN" not in env
    assert "BASE_GATEWAY_TOKEN" not in (env or {})


# --------------------------------------------------------------------------- #
# VAL-ORCH-019 — only the gateway allowlist reaches the agent
# --------------------------------------------------------------------------- #
def _make_task_root(tmp_path: Path, task_id: str = "iso-task") -> Path:
    root = tmp_path / "cache" / task_id
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    (root / "instruction.md").write_text("do it\n")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (root / "task.toml").write_text(
        '[task]\nname = "iso/task"\n\n[environment]\ndocker_image = "python:3.12-slim"\n'
    )
    return root


def _manifest_for(task_root: Path, task_id: str) -> dict[str, Any]:
    return {"tasks": {task_id: {"content_digest_sha256": compute_task_digest(task_root)}}}


class _FakeBuilder:
    def __init__(self) -> None:
        self.env = _FakeEnv()

    def prepare(self, task: Any) -> Any:
        return type("Built", (), {"env": self.env})()


async def test_preparer_hands_agent_only_the_allowlist(tmp_path: Path) -> None:
    task_id = "iso-task"
    task_root = _make_task_root(tmp_path, task_id)
    dirty_agent_env = {
        **_AGENT_ENV,
        "BASE_GATEWAY_TOKEN": "scoped",
        "OPENAI_API_KEY": "sk-leak",
        "MINER_SECRET": "should-not-pass",
        "PATH": "/usr/bin",
    }
    preparer = _build_default_preparer(
        task_ids=[task_id],
        cache_root=task_root.parent,
        digest_manifest=_manifest_for(task_root, task_id),
        digest_manifest_path=None,
        builder=_FakeBuilder(),  # type: ignore[arg-type]
        agent_env=dirty_agent_env,
    )
    prepared = await preparer(TrialId(task_name=task_id, attempt=0), TaskSpec(task_name=task_id))
    assert isinstance(prepared, PreparedTrial)
    assert prepared.agent_env == _AGENT_ENV
    assert set(prepared.agent_env or {}) <= AGENT_ENV_ALLOWLIST


# --------------------------------------------------------------------------- #
# LogRedactor unit behavior (VAL-ORCH-020 / VAL-ORCH-021 primitive)
# --------------------------------------------------------------------------- #
def test_redactor_redacts_gateway_token() -> None:
    redactor = LogRedactor(gateway_token="scoped-gw-token-9f2a")
    out = redactor.redact("curl -H 'Authorization: Bearer scoped-gw-token-9f2a' ...")
    assert "scoped-gw-token-9f2a" not in out
    assert REDACTED_GATEWAY_TOKEN in out


def test_redactor_redacts_miner_env_values() -> None:
    redactor = LogRedactor(miner_env_values=["miner-secret-xyz"])
    out = redactor.redact("echo miner-secret-xyz")
    assert "miner-secret-xyz" not in out
    assert REDACTED_MINER_ENV in out


def test_redactor_noop_when_no_secrets() -> None:
    redactor = LogRedactor()
    assert redactor.active is False
    assert redactor.redact("nothing secret here") == "nothing secret here"
    assert redactor.redact(None) is None


def test_redactor_redacts_outcome_log_channels() -> None:
    from agent_challenge.evaluation.own_runner.orchestrator import TrialOutcome

    redactor = LogRedactor(gateway_token="TOK123", miner_env_values=["MINERVAL"])
    outcome = TrialOutcome(
        task_name="t",
        trial_name="t__attempt-0",
        status="completed",
        agent_output="agent used TOK123 and MINERVAL",
        verifier_stdout="verifier saw TOK123",
        error_text="err with MINERVAL",
    )
    redacted = redactor.redact_outcome(outcome)
    assert "TOK123" not in (redacted.agent_output or "")
    assert "MINERVAL" not in (redacted.agent_output or "")
    assert "TOK123" not in (redacted.verifier_stdout or "")
    assert "MINERVAL" not in (redacted.error_text or "")
    # non-log fields are preserved.
    assert redacted.task_name == "t"
    assert redacted.status == "completed"


# --------------------------------------------------------------------------- #
# VAL-ORCH-020 — gateway token redacted from captured logs + persisted output
# --------------------------------------------------------------------------- #
class _TokenLeakAgent:
    TOKEN = "or-measured-key"

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        return f"agent leaking token {self.TOKEN} to stdout"


async def _preparer_factory(tmp_path: Path) -> Any:
    async def _preparer(trial_id: TrialId, task: TaskSpec) -> PreparedTrial:
        return PreparedTrial(
            environment=_FakeEnv(),
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    return _preparer


async def test_gateway_token_redacted_end_to_end(tmp_path: Path) -> None:
    async def _verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
            verifier_stdout=f"verifier stdout mentioning {_TokenLeakAgent.TOKEN}",
        )

    job_dir = tmp_path / "job"
    result = await run_own_runner_job(
        task_ids=["task"],
        job_dir=job_dir,
        agent_class=_TokenLeakAgent,
        preparer=await _preparer_factory(tmp_path),
        verifier=_verifier,
        agent_env=dict(_AGENT_ENV),
        n_attempts=1,
        miner_env={"OPENROUTER_API_KEY": _TokenLeakAgent.TOKEN},
    )

    outcome = result.trial_outcomes[0]
    assert _TokenLeakAgent.TOKEN not in (outcome.agent_output or "")
    assert _TokenLeakAgent.TOKEN not in (outcome.verifier_stdout or "")

    trial_dir = next((job_dir / "trials").iterdir())
    agent_blob = "\n".join(p.read_text() for p in (trial_dir / "agent").rglob("*") if p.is_file())
    assert _TokenLeakAgent.TOKEN not in agent_blob
    # Token may be redacted as miner env and/or gateway residual marker.
    assert REDACTED_GATEWAY_TOKEN in agent_blob or REDACTED_MINER_ENV in agent_blob
    verifier_blob = (trial_dir / "verifier" / "test-stdout.txt").read_text()
    assert _TokenLeakAgent.TOKEN not in verifier_blob


# --------------------------------------------------------------------------- #
# Live incremental log stream is routed through the redactor (no live leak)
# --------------------------------------------------------------------------- #
async def test_incremental_emitter_wired_with_active_redactor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_challenge.evaluation.own_runner_backend as backend
    from agent_challenge.evaluation.own_runner import log_streamer as log_streamer_mod
    from agent_challenge.evaluation.own_runner.log_streamer import LogStreamer

    # No real network: swallow the best-effort POSTs.
    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(
        log_streamer_mod.urllib.request,
        "urlopen",
        lambda request, timeout=None: _Resp(),
    )

    captured: dict[str, Any] = {}
    real_builder = backend._build_incremental_emitter

    def _spy(log_streamer: Any, redactor: Any = None) -> Any:
        captured["redactor"] = redactor
        return real_builder(log_streamer, redactor)

    monkeypatch.setattr(backend, "_build_incremental_emitter", _spy)

    async def _verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
        )

    streamer = LogStreamer(base_url="http://challenge.test", attempt_id=1, token="tok", slug="s")
    await run_own_runner_job(
        task_ids=["task"],
        job_dir=tmp_path / "job",
        agent_class=_TokenLeakAgent,
        preparer=await _preparer_factory(tmp_path),
        verifier=_verifier,
        agent_env=dict(_AGENT_ENV),
        miner_env={"MINER_KEY": "miner-secret-abc"},
        log_streamer=streamer,
        n_attempts=1,
    )

    redactor = captured.get("redactor")
    assert redactor is not None, "the incremental emitter must receive a redactor"
    assert redactor.active is True
    assert "miner-secret-abc" not in redactor.redact("echo miner-secret-abc")


# --------------------------------------------------------------------------- #
# VAL-ORCH-021 — miner-env values redacted from captured logs + persisted output
# --------------------------------------------------------------------------- #
class _MinerSecretAgent:
    SECRET = "miner-secret-value-7c1d"

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        return f"agent echoing {self.SECRET}"


async def test_miner_env_values_redacted_end_to_end(tmp_path: Path) -> None:
    async def _verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
        )

    job_dir = tmp_path / "job"
    result = await run_own_runner_job(
        task_ids=["task"],
        job_dir=job_dir,
        agent_class=_MinerSecretAgent,
        preparer=await _preparer_factory(tmp_path),
        verifier=_verifier,
        agent_env=dict(_AGENT_ENV),
        miner_env={"MINER_KEY": _MinerSecretAgent.SECRET},
        n_attempts=1,
    )

    outcome = result.trial_outcomes[0]
    assert _MinerSecretAgent.SECRET not in (outcome.agent_output or "")

    trial_dir = next((job_dir / "trials").iterdir())
    agent_blob = "\n".join(p.read_text() for p in (trial_dir / "agent").rglob("*") if p.is_file())
    assert _MinerSecretAgent.SECRET not in agent_blob
    assert REDACTED_MINER_ENV in agent_blob


# --------------------------------------------------------------------------- #
# VAL-ORCH-017 — golden digest mismatch fails closed (no agent/verifier run)
# --------------------------------------------------------------------------- #
async def test_digest_mismatch_fails_closed_no_execution(tmp_path: Path) -> None:
    task_id = "iso-task"
    task_root = _make_task_root(tmp_path, task_id)
    wrong_manifest = {"tasks": {task_id: {"content_digest_sha256": "0" * 64}}}

    verifier_called = False

    async def _spy_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        nonlocal verifier_called
        verifier_called = True
        return VerifierOutcome(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None, rewards={"r": 1}
        )

    class _NeverAgent:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("agent must not be constructed on a digest mismatch")

    with pytest.raises(DigestMismatch):
        await run_own_runner_job(
            task_ids=[task_id],
            job_dir=tmp_path / "job",
            cache_root=task_root.parent,
            digest_manifest=wrong_manifest,
            agent_class=_NeverAgent,
            verifier=_spy_verifier,
            n_attempts=1,
        )

    assert verifier_called is False
    # fail-closed: no trials directory / no persisted (fabricated) score.
    assert not (tmp_path / "job" / "trials").exists()
