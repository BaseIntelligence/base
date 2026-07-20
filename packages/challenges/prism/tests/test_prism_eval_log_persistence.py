from __future__ import annotations

import logging
from pathlib import Path

import pytest
from base.challenge_sdk.executor import DockerRunResult

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import (
    ContainerEvaluationError,
    PrismContainerEvaluator,
)
from prism_challenge.evaluator.interface import PrismContext

CONTRACT_CODE = "def build_model(ctx): pass\ndef get_recipe(ctx): return {}"


def _evaluator(tmp_path: Path, **overrides: object) -> PrismContainerEvaluator:
    settings = PrismSettings(
        shared_token="secret",
        docker_backend="broker",
        docker_broker_url="http://broker",
        docker_broker_token="token",
        eval_log_dir=tmp_path / "eval-logs",
        base_eval_artifact_root=tmp_path / "artifacts",
        **overrides,
    )
    return PrismContainerEvaluator(settings=settings, ctx=PrismContext(sequence_length=16))


def _patch_run(monkeypatch: pytest.MonkeyPatch, result: DockerRunResult) -> None:
    def fake_run(self: object, spec: object, timeout_seconds: float) -> DockerRunResult:
        return result

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)


def test_resolved_eval_log_dir_defaults_next_to_db() -> None:
    settings = PrismSettings(
        shared_token="x", database_url="sqlite+aiosqlite:////data/prism.sqlite3"
    )
    assert settings.resolved_eval_log_dir == Path("/data/eval-logs")


def test_resolved_eval_log_dir_honors_override(tmp_path: Path) -> None:
    settings = PrismSettings(shared_token="x", eval_log_dir=tmp_path / "custom-logs")
    assert settings.resolved_eval_log_dir == tmp_path / "custom-logs"


def test_full_stdout_stderr_persisted_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A large stream (well beyond any snippet) plus the metrics line at the end proves the COMPLETE
    # stdout is persisted verbatim (no truncation) while scoring still parses its metrics.
    big_stdout = "x" * (2 * 1024 * 1024) + '\nPRISM_METRICS_JSON={"covered_bytes":250.0}\n'
    big_stderr = "warning noise line\n" * 4096
    _patch_run(monkeypatch, DockerRunResult("prism-eval", big_stdout, big_stderr, 0))
    evaluator = _evaluator(tmp_path)

    result = evaluator.evaluate(
        submission_id="sub-ok",
        code=CONTRACT_CODE,
        code_hash="code",
        arch_hash="arch",
        backend="base_gpu",
        attempt=3,
    )

    # Scoring/metrics behavior is unchanged (persist is additive).
    assert result.metrics == {"covered_bytes": 250.0}
    log_dir = evaluator.settings.resolved_eval_log_dir
    stdout_file = log_dir / "sub-ok.attempt-3.stdout.log"
    stderr_file = log_dir / "sub-ok.attempt-3.stderr.log"
    assert stdout_file.read_text(encoding="utf-8") == big_stdout
    assert stderr_file.read_text(encoding="utf-8") == big_stderr


def test_full_stdout_stderr_persisted_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(
        monkeypatch,
        DockerRunResult(
            "prism-eval", "partial stdout before crash\n", "Traceback\nRuntimeError: boom", 1
        ),
    )
    evaluator = _evaluator(tmp_path)

    with pytest.raises(ContainerEvaluationError):
        evaluator.evaluate(
            submission_id="sub-fail",
            code=CONTRACT_CODE,
            code_hash="code",
            arch_hash="arch",
            backend="base_gpu",
            attempt=1,
        )

    log_dir = evaluator.settings.resolved_eval_log_dir
    assert (log_dir / "sub-fail.attempt-1.stdout.log").read_text(
        encoding="utf-8"
    ) == "partial stdout before crash\n"
    assert "RuntimeError: boom" in (log_dir / "sub-fail.attempt-1.stderr.log").read_text(
        encoding="utf-8"
    )


def test_stdout_stderr_persisted_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, DockerRunResult("prism-eval", "hung stdout", "", 124, timed_out=True))
    evaluator = _evaluator(
        tmp_path,
        base_eval_budget_seconds=2,
        base_eval_watchdog_grace_seconds=1,
        base_eval_timeout_seconds=5,
    )

    with pytest.raises(ContainerEvaluationError):
        evaluator.evaluate(
            submission_id="sub-timeout",
            code=CONTRACT_CODE,
            code_hash="code",
            arch_hash="arch",
            backend="base_gpu",
            attempt=2,
        )

    log_dir = evaluator.settings.resolved_eval_log_dir
    assert (log_dir / "sub-timeout.attempt-2.stdout.log").read_text(
        encoding="utf-8"
    ) == "hung stdout"


def test_persist_eval_logs_logs_info_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    evaluator = _evaluator(tmp_path)
    with caplog.at_level(logging.INFO, logger="prism_challenge.evaluator.container"):
        paths = evaluator._persist_eval_logs("sub-info", 5, DockerRunResult("c", "out", "err", 0))

    assert paths is not None
    stdout_path, stderr_path = paths
    assert stdout_path.read_text(encoding="utf-8") == "out"
    assert stderr_path.read_text(encoding="utf-8") == "err"
    assert any(
        "persisted eval logs" in record.message and "returncode=0" in record.message
        for record in caplog.records
    )


def test_persist_eval_logs_is_best_effort_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    evaluator = _evaluator(tmp_path)

    def boom_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", boom_mkdir)

    with caplog.at_level(logging.WARNING, logger="prism_challenge.evaluator.container"):
        result = evaluator._persist_eval_logs("sub-x", 1, DockerRunResult("c", "out", "err", 0))

    assert result is None
    assert not evaluator.settings.resolved_eval_log_dir.exists()
    assert any("failed to persist eval logs" in record.message for record in caplog.records)
