from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_challenge.analyzer import container
from agent_challenge.analyzer.container import (
    AnalyzerContainerConfigError,
    build_analyzer_container_plan,
    configure_analyzer_container_job,
    persist_analyzer_container_evidence,
)
from agent_challenge.evaluation import create_evaluation_job, run_evaluation_job
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.swe_forge import SweForgeTask


class FakeExecutor:
    def __init__(self) -> None:
        self.specs = []

    def run(self, spec, timeout_seconds: int):
        self.specs.append((spec, timeout_seconds))
        return DockerRunResult(
            container_name=spec.labels["base.task"], stdout="ok", stderr="", returncode=0
        )


def test_strict_analyzer_spec_defaults(tmp_path, monkeypatch):
    artifact_path, rules_dir, output_dir = _fixture_paths(tmp_path)
    monkeypatch.setattr("agent_challenge.analyzer.container.settings.docker_cpus", 4.0)
    monkeypatch.setattr("agent_challenge.analyzer.container.settings.docker_memory", "8g")
    monkeypatch.setattr("agent_challenge.analyzer.container.settings.docker_memory_swap", "8g")
    monkeypatch.setattr("agent_challenge.analyzer.container.settings.docker_network", "default")
    monkeypatch.setattr(
        "agent_challenge.analyzer.container.settings.evaluation_timeout_seconds",
        3600,
    )

    job = _job()
    submission = _submission(artifact_path)
    plan = build_analyzer_container_plan(
        submission,
        job,
        rules_dir=rules_dir,
        output_dir=output_dir,
    )

    assert plan.timeout_seconds == 3600
    assert plan.spec.image == "ghcr.io/baseintelligence/agent-challenge-analyzer:1.0"
    assert plan.spec.image.startswith("ghcr.io/baseintelligence/")
    assert plan.spec.limits.cpus == 4.0
    assert plan.spec.limits.memory == "8g"
    assert plan.spec.limits.memory_swap == "4g"
    assert plan.spec.limits.pids_limit == 512
    assert plan.spec.limits.network == "none"
    assert plan.spec.limits.read_only is True
    assert plan.spec.limits.user == "65532:65532"
    assert plan.spec.limits.cap_drop == ("ALL",)
    assert plan.spec.limits.security_opt == ("no-new-privileges",)
    assert plan.spec.limits.tmpfs == ("/tmp:rw,noexec,nosuid,size=512m",)

    mounts = {mount.target: mount for mount in plan.spec.mounts}
    assert mounts[container.ARTIFACT_TARGET].source == artifact_path.resolve()
    assert mounts[container.ARTIFACT_TARGET].read_only is True
    assert mounts[container.RULES_TARGET].source == rules_dir.resolve()
    assert mounts[container.RULES_TARGET].read_only is True
    assert mounts[container.OUTPUT_TARGET].source == output_dir.resolve()
    assert mounts[container.OUTPUT_TARGET].read_only is False
    assert all("/.env" not in str(mount.source) for mount in plan.spec.mounts)
    assert all("docker.sock" not in str(mount.source) for mount in plan.spec.mounts)


def test_configure_analyzer_job_persists_container_evidence(tmp_path):
    artifact_path, rules_dir, output_dir = _fixture_paths(tmp_path)
    job = _job()
    submission = _submission(artifact_path)

    plan = configure_analyzer_container_job(
        job,
        submission,
        rules_dir=rules_dir,
        output_dir=output_dir,
    )

    payload = json.loads(job.container_config_json)
    assert job.image_digest == plan.spec.image
    assert job.rules_version == plan.rules_version
    assert payload["timeout_seconds"] == 3600
    assert payload["limits"]["network"] == "none"
    assert payload["mounts"] == [
        {
            "source": str(artifact_path.resolve()),
            "target": container.ARTIFACT_TARGET,
            "read_only": True,
        },
        {
            "source": str(rules_dir.resolve()),
            "target": container.RULES_TARGET,
            "read_only": True,
        },
        {
            "source": str(output_dir.resolve()),
            "target": container.OUTPUT_TARGET,
            "read_only": False,
        },
    ]


def test_container_result_evidence_records_exit_timeout_and_logs(tmp_path):
    artifact_path, rules_dir, output_dir = _fixture_paths(tmp_path)
    job = _job()
    plan = build_analyzer_container_plan(
        _submission(artifact_path),
        job,
        rules_dir=rules_dir,
        output_dir=output_dir,
    )
    result = DockerRunResult(
        container_name="analyzer-job-1",
        stdout="partial",
        stderr="timeout",
        returncode=124,
        timed_out=True,
    )

    persist_analyzer_container_evidence(job, plan, result=result, logs_ref="logs/job-1.txt")

    payload = json.loads(job.container_config_json)
    assert payload["result"] == {
        "container_name": "analyzer-job-1",
        "returncode": 124,
        "timed_out": True,
    }
    assert job.logs_ref == "logs/job-1.txt"
    assert json.loads(job.reason_codes_json) == ["analyzer_container_timed_out"]


def test_unsupported_security_option_fails_closed(tmp_path, monkeypatch):
    artifact_path, rules_dir, output_dir = _fixture_paths(tmp_path)

    @dataclass(frozen=True)
    class IncompleteDockerLimits:
        cpus: float = 4.0
        memory: str = "8g"
        network: str = "none"

    monkeypatch.setattr(container, "DockerLimits", IncompleteDockerLimits)

    with pytest.raises(AnalyzerContainerConfigError, match="required security fields"):
        build_analyzer_container_plan(
            _submission(artifact_path),
            _job(),
            rules_dir=rules_dir,
            output_dir=output_dir,
        )


async def test_run_evaluation_job_persists_analyzer_container_config(
    database_session, monkeypatch, tmp_path
):
    artifact_path, rules_dir, _output_dir = _fixture_paths(tmp_path)
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [SweForgeTask(task_id="task-a", docker_image="baseintelligence/swe-forge:task-a")],
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend",
        "swe_forge",
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 1)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_concurrency", 1)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.analyzer.container._rules_dir", lambda _=None: rules_dir)

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hotkey-a",
            name="agent-a",
            agent_hash="abc123",
            artifact_uri=str(artifact_path),
            artifact_path=str(artifact_path),
        )
        session.add(submission)
        await session.flush()
        job = await create_evaluation_job(session, submission)

        executor = FakeExecutor()
        summary = await run_evaluation_job(session, job.job_id, executor=executor)

        assert summary.status == "completed"
        payload = json.loads(job.container_config_json)
        assert len(executor.specs) == 2
        assert executor.specs[0][0].labels["base.task"] == "analyzer"
        assert executor.specs[1][0].labels["base.task"] == "task-a"
        assert payload["result"] == {
            "container_name": "analyzer",
            "returncode": 0,
            "timed_out": False,
        }
        assert payload["limits"]["network"] == "none"
        assert payload["mounts"][0]["source"] == str(artifact_path.resolve())
        assert job.image_digest == "ghcr.io/baseintelligence/agent-challenge-analyzer:1.0"
        assert payload["rules_version"]
        assert job.rules_version


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_dir = tmp_path / "artifact-store" / ("a" * 64)
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "agent.zip"
    artifact_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    rules_dir = tmp_path / ".rules"
    rules_dir.mkdir()
    (rules_dir / "acceptance.md").write_text("accept safe agents\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    return artifact_path, rules_dir, output_dir


def _submission(artifact_path: Path) -> AgentSubmission:
    return AgentSubmission(
        id=1,
        miner_hotkey="hotkey-a",
        name="agent-a",
        agent_hash="abc123",
        artifact_uri=str(artifact_path),
        artifact_path=str(artifact_path),
    )


def _job() -> EvaluationJob:
    return EvaluationJob(
        id=1,
        job_id="job-a",
        submission_id=1,
        status="queued",
        selected_tasks_json="[]",
    )
