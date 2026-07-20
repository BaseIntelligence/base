from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from agent_challenge.core.config import settings
from agent_challenge.core.models import AgentSubmission, EvaluationJob
from agent_challenge.rules import load_rules
from agent_challenge.sdk.executors import DockerLimits, DockerMount, DockerRunResult, DockerRunSpec

ANALYZER_IMAGE = "ghcr.io/baseintelligence/agent-challenge-analyzer:1.0"
ARTIFACT_TARGET = "/workspace/artifact/agent.zip"
ARTIFACT_PACKAGE_TARGET = "/workspace/artifact/package"
RULES_TARGET = "/workspace/rules"
OUTPUT_TARGET = "/workspace/output"
REQUIRED_SECURITY_LIMIT_FIELDS = frozenset(
    {
        "read_only",
        "user",
        "tmpfs",
        "ulimits",
        "cap_drop",
        "security_opt",
        "init",
    }
)


class AnalyzerContainerConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalyzerContainerPlan:
    spec: DockerRunSpec
    timeout_seconds: int
    output_dir: Path
    rules_version: str


def build_analyzer_container_plan(
    submission: AgentSubmission,
    job: EvaluationJob,
    *,
    rules_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    image: str = ANALYZER_IMAGE,
) -> AnalyzerContainerPlan:
    artifact_path = _submission_artifact_path(submission)
    resolved_rules_dir = _rules_dir(rules_dir)
    resolved_output_dir = _output_dir(submission, job, output_dir)

    if not artifact_path.exists():
        raise AnalyzerContainerConfigError(f"artifact path does not exist: {artifact_path}")
    if not resolved_rules_dir.is_dir():
        raise AnalyzerContainerConfigError(f"rules directory does not exist: {resolved_rules_dir}")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    rules = load_rules(resolved_rules_dir.parent)
    artifact_target = ARTIFACT_PACKAGE_TARGET if artifact_path.is_dir() else ARTIFACT_TARGET
    spec = DockerRunSpec(
        image=image,
        command=(
            "bash",
            "-lc",
            _analyzer_preflight_script(artifact_target),
        ),
        mounts=(
            DockerMount(source=artifact_path, target=artifact_target, read_only=True),
            DockerMount(source=resolved_rules_dir, target=RULES_TARGET, read_only=True),
            DockerMount(source=resolved_output_dir, target=OUTPUT_TARGET, read_only=False),
        ),
        workdir="/workspace",
        env={},
        labels={
            "base.job": job.job_id,
            "base.task": "analyzer",
            "base.agent": submission.agent_hash[:32],
            "base.component": "analyzer",
        },
        limits=_strict_analyzer_limits(),
    )
    return AnalyzerContainerPlan(
        spec=spec,
        timeout_seconds=settings.evaluation_timeout_seconds,
        output_dir=resolved_output_dir,
        rules_version=rules.rules_version,
    )


def configure_analyzer_container_job(
    job: EvaluationJob,
    submission: AgentSubmission,
    *,
    rules_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> AnalyzerContainerPlan:
    plan = build_analyzer_container_plan(
        submission,
        job,
        rules_dir=rules_dir,
        output_dir=output_dir,
    )
    persist_analyzer_container_evidence(job, plan)
    job.rules_version = plan.rules_version
    return plan


def persist_analyzer_container_evidence(
    job: EvaluationJob,
    plan: AnalyzerContainerPlan,
    *,
    result: DockerRunResult | None = None,
    logs_ref: str | None = None,
) -> None:
    job.image_digest = plan.spec.image
    job.container_config_json = container_config_json(plan, result=result)
    if logs_ref is not None:
        job.logs_ref = logs_ref
    if result is not None:
        reason_codes: list[str] = []
        if result.timed_out:
            reason_codes.append("analyzer_container_timed_out")
        elif result.returncode != 0:
            reason_codes.append("analyzer_container_failed")
        if reason_codes:
            job.reason_codes_json = json.dumps(reason_codes, sort_keys=True)


def container_config_json(
    plan: AnalyzerContainerPlan,
    *,
    result: DockerRunResult | None = None,
) -> str:
    payload: dict[str, Any] = {
        "image": plan.spec.image,
        "command": list(plan.spec.command),
        "workdir": plan.spec.workdir,
        "env": dict(plan.spec.env),
        "labels": dict(plan.spec.labels),
        "timeout_seconds": plan.timeout_seconds,
        "mounts": [
            {
                "source": str(mount.source.resolve(strict=False)),
                "target": mount.target,
                "read_only": mount.read_only,
            }
            for mount in plan.spec.mounts
        ],
        "limits": _limits_dict(plan.spec.limits),
        "output_dir": str(plan.output_dir.resolve(strict=False)),
        "rules_version": plan.rules_version,
    }
    if result is not None:
        payload["result"] = {
            "container_name": result.container_name,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _strict_analyzer_limits() -> DockerLimits:
    _require_supported_security_fields()
    return DockerLimits(
        cpus=settings.docker_cpus,
        memory=settings.docker_memory,
        memory_swap="4g",
        pids_limit=512,
        network="none",
        read_only=True,
        user=settings.docker_user or "65532:65532",
        tmpfs=("/tmp:rw,noexec,nosuid,size=512m",),
        ulimits=("nofile=1024:1024",),
        cap_drop=("ALL",),
        security_opt=("no-new-privileges",),
        init=True,
    )


def _require_supported_security_fields() -> None:
    if not is_dataclass(DockerLimits):
        raise AnalyzerContainerConfigError("DockerLimits does not expose required security fields")
    supported = {field.name for field in fields(DockerLimits)}
    missing = sorted(REQUIRED_SECURITY_LIMIT_FIELDS - supported)
    if missing:
        raise AnalyzerContainerConfigError(
            "DockerLimits does not expose required security fields: " + ", ".join(missing)
        )


def _limits_dict(limits: DockerLimits) -> dict[str, Any]:
    return {
        "cpus": limits.cpus,
        "memory": limits.memory,
        "memory_swap": limits.memory_swap,
        "pids_limit": limits.pids_limit,
        "network": limits.network,
        "read_only": limits.read_only,
        "user": limits.user,
        "tmpfs": list(limits.tmpfs),
        "ulimits": list(limits.ulimits),
        "cap_drop": list(limits.cap_drop),
        "security_opt": list(limits.security_opt),
        "init": limits.init,
    }


def _submission_artifact_path(submission: AgentSubmission) -> Path:
    raw_path = submission.artifact_path or submission.artifact_uri
    if not raw_path:
        raise AnalyzerContainerConfigError("submission has no artifact path")
    return Path(raw_path).expanduser().resolve(strict=False)


def _rules_dir(rules_dir: Path | str | None) -> Path:
    if rules_dir is not None:
        return Path(rules_dir).expanduser().resolve(strict=False)
    return (Path(__file__).parents[3] / ".rules").resolve(strict=False)


def _output_dir(
    submission: AgentSubmission,
    job: EvaluationJob,
    output_dir: Path | str | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve(strict=False)
    artifact_path = _submission_artifact_path(submission)
    base_dir = artifact_path if artifact_path.is_dir() else artifact_path.parent
    return (base_dir / "analyzer-output" / job.job_id).resolve(strict=False)


def _analyzer_preflight_script(artifact_target: str) -> str:
    return (
        "test -e "
        f"{artifact_target} && test -d {RULES_TARGET} && test -w {OUTPUT_TARGET} && "
        f"printf '%s\\n' '{{\"status\":\"container_ready\"}}' > {OUTPUT_TARGET}/preflight.json"
    )
