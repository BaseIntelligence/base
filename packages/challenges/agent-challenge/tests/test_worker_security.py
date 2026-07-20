from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.evaluation.worker import run_worker_once
from agent_challenge.models import (
    AgentSubmission,
    AnalyzerReport,
    EvaluationJob,
    OwnerActionAudit,
    RequestNonce,
    TaskResult,
)
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.security import SignedRequestAuth
from agent_challenge.swe_forge import FALLBACK_TASK_IDS

NOW = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
BROKER_TOKEN = "broker-token-secret"
SIGNATURE_SECRET = "miner-signature-secret"
SIGNATURE_NONCE = "miner-nonce-secret"
SIGNATURE_HASH = "miner-body-hash-secret"
SIGNATURE_MESSAGE = "POST\n/submissions\nsecret canonical request"


class CapturingBrokerExecutor:
    def __init__(self) -> None:
        self.specs = []

    def run(self, spec, timeout_seconds: int):  # type: ignore[no-untyped-def]
        self.specs.append(spec)
        if spec.labels["base.task"] == "analyzer":
            return DockerRunResult(
                container_name="analyzer",
                stdout="analyzer ok",
                stderr="",
                returncode=0,
            )
        return DockerRunResult(
            container_name="broker-terminal-bench",
            stdout=('harbor done\nBASE_BENCHMARK_RESULT={"score": 0.25, "status": "completed"}'),
            stderr="",
            returncode=0,
        )


@pytest.fixture
def owner_auth_override() -> AsyncIterator[None]:
    calls = 0

    async def authenticate() -> SignedRequestAuth:
        nonlocal calls
        calls += 1
        return SignedRequestAuth(
            hotkey="owner-hotkey",
            signature=f"owner-signature-{calls}",
            nonce=f"owner-nonce-{calls}",
            timestamp=NOW.isoformat(),
            body_sha256=hashlib.sha256(f"owner-body-{calls}".encode()).hexdigest(),
            canonical_request=f"owner-request-{calls}",
        )

    app.dependency_overrides[routes.owner_signed_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.owner_signed_auth, None)


async def test_invalid_signed_submission_rejects_before_enqueue_or_dispatch(
    client,
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    enqueue_calls = 0
    dispatch_calls = 0

    async def fail_if_enqueued(*_args: object, **_kwargs: object) -> None:
        nonlocal enqueue_calls
        enqueue_calls += 1
        raise AssertionError("invalid signed submission reached enqueue")

    class DispatchTrap:
        def run(self, *_args: object, **_kwargs: object) -> None:
            nonlocal dispatch_calls
            dispatch_calls += 1
            raise AssertionError("invalid signed submission reached broker dispatch")

    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.auth.security.verify_substrate_signature",
        lambda *_: False,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.queue_submission_analysis",
        fail_if_enqueued,
    )

    archive_bytes = build_zip({"agent.py": "print('ok')\n"})
    response = await client.post(
        "/submissions",
        json={
            "name": "invalid-signed-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
        headers={
            "X-Hotkey": "miner-hotkey",
            "X-Signature": "invalid-signature",
            "X-Nonce": "invalid-nonce",
            "X-Timestamp": datetime.now(UTC).isoformat(),
        },
    )
    iteration = await run_worker_once(worker_id="security-worker", executor=DispatchTrap())

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid signed request"}
    assert iteration.summary is None
    assert enqueue_calls == 0
    assert dispatch_calls == 0
    async with database_session() as session:
        assert await session.scalar(select(func.count(AgentSubmission.id))) == 0
        assert await session.scalar(select(func.count(EvaluationJob.id))) == 0
        assert await session.scalar(select(func.count(TaskResult.id))) == 0
        assert await session.scalar(select(func.count(RequestNonce.id))) == 0


async def test_worker_broker_path_scrubs_token_and_signature_metadata_and_keeps_analyzer_evidence(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    configure_terminal_bench_broker(monkeypatch, artifact_root=tmp_path / "artifacts")
    artifact_dir = tmp_path / "agent"
    artifact_dir.mkdir()
    (artifact_dir / "agent.py").write_text(
        f"TASK_ID = {FALLBACK_TASK_IDS[0]!r}\n"
        "def solve():\n"
        "    if 'test_expected_behavior' in __name__:\n"
        "        return 42\n",
        encoding="utf-8",
    )
    async with database_session() as session:
        submission = signed_submission(artifact_dir)
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id="worker-security-job",
            submission_id=submission.id,
            status="queued",
            selected_tasks_json=json.dumps(
                [
                    {
                        "task_id": "hello-world",
                        "benchmark": "terminal_bench",
                        "docker_image": "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
                        "prompt": "Run Harbor",
                        "metadata": {"task_id": "hello-world"},
                    }
                ],
                sort_keys=True,
            ),
            total_tasks=1,
        )
        session.add(job)
        await session.commit()
        submission_id = submission.id

    executor = CapturingBrokerExecutor()
    iteration = await run_worker_once(worker_id="security-worker", executor=executor)

    assert iteration.summary is not None
    async with database_session() as session:
        failed_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "worker-security-job")
        )
    assert failed_job is not None
    assert iteration.summary.status == "completed", failed_job.error
    assert [spec.labels["base.task"] for spec in executor.specs] == [
        "analyzer",
        "hello-world",
    ]
    benchmark_spec = executor.specs[1]
    assert benchmark_spec.env == {
        "BASE_AGENT_PATH": "/workspace/agent",
        "BASE_BENCHMARK_DATASET": "terminal-bench/terminal-bench-2-1",
        "HOME": "/tmp",
        "XDG_CACHE_HOME": "/tmp/.cache",
    }
    assert_no_untrusted_secret(
        json.dumps(serializable_spec_payload(benchmark_spec), sort_keys=True)
    )
    assert_no_untrusted_secret("\n".join(benchmark_spec.command))

    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "worker-security-job")
        )
        task_result = await session.scalar(select(TaskResult))
        report = await session.scalar(select(AnalyzerReport))

    assert submission is not None
    assert submission.signature == SIGNATURE_SECRET
    assert submission.signature_nonce == SIGNATURE_NONCE
    assert submission.signature_payload_sha256 == SIGNATURE_HASH
    assert submission.signature_message == SIGNATURE_MESSAGE
    assert job is not None
    assert_no_untrusted_secret(job.container_config_json)
    assert task_result is not None
    assert_no_untrusted_secret(task_result.stdout)
    assert_no_untrusted_secret(task_result.stderr)
    assert report is not None
    assert_no_untrusted_secret(report.report_json)
    report_payload = json.loads(report.report_json)
    assert "hardcoding_detected" in report_payload["reason_codes"]
    evidence = report_payload["evidence"]
    assert evidence
    assert all(item["path"] == "agent.py" for item in evidence)
    assert all(1 <= item["line_start"] <= item["line_end"] for item in evidence)
    assert all(item["snippet"] and len(item["snippet"]) <= 240 for item in evidence)
    assert {item["reason_code"] for item in evidence} >= {
        "benchmark_task_id_literal",
        "branch_on_test_name",
    }


async def test_owner_controls_after_worker_completion_are_append_only_and_preserve_evidence(
    client,
    database_session,
    monkeypatch,
    owner_auth_override,
    tmp_path,
) -> None:
    configure_terminal_bench_broker(monkeypatch, artifact_root=tmp_path / "artifacts")
    artifact_dir = tmp_path / "agent"
    artifact_dir.mkdir()
    (artifact_dir / "agent.py").write_text("def solve():\n    return 'ok'\n", encoding="utf-8")
    async with database_session() as session:
        submission = signed_submission(artifact_dir, agent_hash="signed-worker-clean")
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id="worker-owner-job",
            submission_id=submission.id,
            status="queued",
            selected_tasks_json=json.dumps(
                [
                    {
                        "task_id": "hello-world",
                        "benchmark": "terminal_bench",
                        "docker_image": "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
                        "prompt": "Run Harbor",
                        "metadata": {"task_id": "hello-world"},
                    }
                ],
                sort_keys=True,
            ),
            total_tasks=1,
        )
        session.add(job)
        await session.commit()
        submission_id = submission.id

    iteration = await run_worker_once(
        worker_id="security-worker",
        executor=CapturingBrokerExecutor(),
    )
    assert iteration.summary is not None
    async with database_session() as session:
        failed_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "worker-owner-job")
        )
    assert failed_job is not None
    assert iteration.summary.status == "completed", failed_job.error

    async with database_session() as session:
        original_submission = await session.get(AgentSubmission, submission_id)
        original_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "worker-owner-job")
        )
        original_report = await session.scalar(select(AnalyzerReport))
        original_result = await session.scalar(select(TaskResult))

    assert original_submission is not None
    assert original_job is not None
    assert original_report is not None
    assert original_result is not None
    submission_snapshot = {
        "signature": original_submission.signature,
        "signature_nonce": original_submission.signature_nonce,
        "signature_timestamp": original_submission.signature_timestamp,
        "signature_payload_sha256": original_submission.signature_payload_sha256,
        "signature_message": original_submission.signature_message,
        "artifact_uri": original_submission.artifact_uri,
        "artifact_path": original_submission.artifact_path,
        "zip_sha256": original_submission.zip_sha256,
    }
    job_snapshot = immutable_job_snapshot(original_job)
    report_snapshot = immutable_report_snapshot(original_report)
    result_snapshot = immutable_task_result_snapshot(original_result)

    override = await client.post(
        f"/owner/submissions/{submission_id}/override",
        json={"status": "overridden_invalid", "reason": "security review"},
    )
    revalidate = await client.post(
        f"/owner/submissions/{submission_id}/revalidate",
        json={"reason": "rerun after review"},
    )

    assert override.status_code == 200
    assert revalidate.status_code == 200
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        original_job_after = await session.get(EvaluationJob, original_job.id)
        original_report_after = await session.get(AnalyzerReport, original_report.id)
        original_result_after = await session.get(TaskResult, original_result.id)
        audits = (
            (await session.execute(select(OwnerActionAudit).order_by(OwnerActionAudit.id)))
            .scalars()
            .all()
        )
        jobs = (
            (await session.execute(select(EvaluationJob).order_by(EvaluationJob.id)))
            .scalars()
            .all()
        )

    assert submission is not None
    assert {key: getattr(submission, key) for key in submission_snapshot} == submission_snapshot
    assert original_job_after is not None
    assert immutable_job_snapshot(original_job_after) == job_snapshot
    assert original_report_after is not None
    assert immutable_report_snapshot(original_report_after) == report_snapshot
    assert original_result_after is not None
    assert immutable_task_result_snapshot(original_result_after) == result_snapshot
    assert [audit.action for audit in audits] == ["override", "revalidate"]
    assert [audit.reason for audit in audits] == ["security review", "rerun after review"]
    assert audits[0].after_effective_status == "overridden_invalid"
    assert audits[1].before_effective_status == "overridden_invalid"
    assert len(jobs) == 2
    assert jobs[0].job_id == "worker-owner-job"
    assert jobs[1].triggered_by_hotkey == "owner-hotkey"
    assert jobs[1].trigger_reason == "revalidate"


def build_zip(files: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in files.items():
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def configure_terminal_bench_broker(monkeypatch, *, artifact_root: object | None = None) -> None:
    for settings_path in (
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
    ):
        monkeypatch.setattr(f"{settings_path}.validator_role", "master")
        monkeypatch.setattr(f"{settings_path}.analyzer_similarity_enabled", False)
    for settings_path in (
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
    ):
        monkeypatch.setattr(f"{settings_path}.benchmark_backend", "terminal_bench")
        monkeypatch.setattr(f"{settings_path}.terminal_bench_task_ids", ("hello-world",))
        monkeypatch.setattr(f"{settings_path}.evaluation_task_count", 1)
        monkeypatch.setattr(f"{settings_path}.evaluation_concurrency", 1)
        monkeypatch.setattr(f"{settings_path}.evaluation_timeout_seconds", 120)
        if artifact_root is not None:
            monkeypatch.setattr(f"{settings_path}.artifact_root", str(artifact_root))
        monkeypatch.setattr(f"{settings_path}.docker_enabled", True)
        monkeypatch.setattr(f"{settings_path}.docker_backend", "broker")
        monkeypatch.setattr(f"{settings_path}.docker_broker_url", "https://platform-broker.test")
        monkeypatch.setattr(f"{settings_path}.docker_broker_token", BROKER_TOKEN)
        monkeypatch.setattr(f"{settings_path}.docker_broker_token_file", None)
        monkeypatch.setattr(
            f"{settings_path}.docker_allowed_images",
            (
                "ghcr.io/baseintelligence/agent-challenge-analyzer:1.0",
                "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
            ),
        )
        monkeypatch.setattr(f"{settings_path}.docker_network", "default")
        monkeypatch.setattr(
            f"{settings_path}.harbor_runner_image",
            "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        )
        monkeypatch.setattr(f"{settings_path}.harbor_forward_env_vars", ())


def signed_submission(
    artifact_dir,  # type: ignore[no-untyped-def]
    *,
    agent_hash: str = "signed-worker-agent",
) -> AgentSubmission:
    return AgentSubmission(
        miner_hotkey="signed-miner-hotkey",
        name="signed-worker-agent",
        agent_hash=agent_hash,
        artifact_uri=str(artifact_dir),
        artifact_path=str(artifact_dir),
        status="queued",
        raw_status="queued",
        effective_status="queued",
        zip_sha256="zip-sha256-secret",
        signature=SIGNATURE_SECRET,
        signature_nonce=SIGNATURE_NONCE,
        signature_timestamp=NOW.isoformat(),
        signature_payload_sha256=SIGNATURE_HASH,
        signature_message=SIGNATURE_MESSAGE,
    )


def serializable_spec_payload(spec) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "image": spec.image,
        "command": list(spec.command),
        "mounts": [
            {"source": str(mount.source), "target": mount.target, "read_only": mount.read_only}
            for mount in spec.mounts
        ],
        "workdir": spec.workdir,
        "env": dict(spec.env),
        "labels": dict(spec.labels),
    }


def assert_no_untrusted_secret(payload: str) -> None:
    for secret in (
        BROKER_TOKEN,
        SIGNATURE_SECRET,
        SIGNATURE_NONCE,
        SIGNATURE_HASH,
        SIGNATURE_MESSAGE,
        "CHALLENGE_DOCKER_BROKER_TOKEN",
        "X-Signature",
        "signature_nonce",
        "signature_message",
    ):
        assert secret not in payload


def immutable_job_snapshot(job: EvaluationJob) -> dict[str, object]:
    return {
        "status": job.status,
        "score": job.score,
        "passed_tasks": job.passed_tasks,
        "total_tasks": job.total_tasks,
        "error": job.error,
        "rules_version": job.rules_version,
        "image_digest": job.image_digest,
        "container_config_json": job.container_config_json,
        "verdict": job.verdict,
        "reason_codes_json": job.reason_codes_json,
        "logs_ref": job.logs_ref,
        "attempt_count": job.attempt_count,
        "last_error": job.last_error,
    }


def immutable_report_snapshot(report: AnalyzerReport) -> dict[str, object]:
    return {
        "rules_version": report.rules_version,
        "verdict": report.verdict,
        "reason_codes_json": report.reason_codes_json,
        "report_json": report.report_json,
        "logs_ref": report.logs_ref,
    }


def immutable_task_result_snapshot(result: TaskResult) -> dict[str, object]:
    return {
        "task_id": result.task_id,
        "docker_image": result.docker_image,
        "status": result.status,
        "score": result.score,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
