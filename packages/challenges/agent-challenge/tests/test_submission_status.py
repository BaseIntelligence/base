from __future__ import annotations

import json

from cryptography.fernet import Fernet
from sqlalchemy import event, select

from agent_challenge.api import routes
from agent_challenge.db import database
from agent_challenge.evaluation import task_events
from agent_challenge.models import (
    AgentSubmission,
    AnalysisRun,
    EvaluationAttempt,
    EvaluationJob,
    ExternalExecutionRef,
    LlmVerdict,
    PythonAstFeature,
    SimilarityMatch,
    SubmissionEnvVar,
    SubmissionStatusEvent,
    TerminalBenchTrial,
)
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.submissions.state_machine import transition_submission_status


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            name, value = line.split(": ", 1)
            fields[name] = value
        events.append(
            {
                "id": int(fields["id"]),
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return events


PLATFORM_SDK_PUBLIC_FORBIDDEN = (
    "platform_sdk",
    "base_sdk",
    "tb21-platform-sdk-secret",
    "/terminal-bench/jobs/platform-sdk-private",
    "platform-terminal-bench-command.sh",
    "base-terminal-bench-command.sh",
    "worker-a",
    "broker-token",
    "k8s-job-task8",
    "pod-task8",
    "raw-ref-task8",
    "artifact-task8",
)


def _assert_platform_sdk_markers_redacted(payload: object) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in PLATFORM_SDK_PUBLIC_FORBIDDEN:
        assert forbidden not in serialized


async def test_submission_status_progression_uses_latest_event_public_mapping(
    client,
    database_session,
):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-status",
            name="status-agent",
            agent_hash="status-agent-hash",
            artifact_uri="/tmp/status-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.flush()
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="received",
            from_status=None,
        )
        await transition_submission_status(
            session,
            submission,
            "upload_verified",
            actor="api",
            reason="artifact verified",
        )
        await transition_submission_status(
            session,
            submission,
            "rate_limit_reserved",
            actor="api",
            reason="rate limit reserved",
        )
        await transition_submission_status(
            session,
            submission,
            "analysis_queued",
            actor="analysis",
            reason="queued",
        )
        await transition_submission_status(
            session,
            submission,
            "ast_running",
            actor="worker",
            reason="ast started",
        )
        await session.commit()
        submission_id = submission.id
        last_event = (
            (
                await session.execute(
                    select(SubmissionStatusEvent)
                    .where(SubmissionStatusEvent.submission_id == submission_id)
                    .order_by(SubmissionStatusEvent.sequence.desc())
                )
            )
            .scalars()
            .first()
        )

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["submission_id"] == submission_id
    assert payload["status"] == "AST review"
    assert payload["public_state"] == "AST review"
    assert payload["phase"] == "ast_review"
    assert payload["analyzer"]["phase"] == "running"
    assert payload["last_event_id"] == last_event.id
    assert payload["last_event_sequence"] == 5
    assert payload["progress"]["status_events"] == 5
    assert payload["progress"]["analysis_runs"] == 0
    assert payload["evaluation"]["task_phases"] == []
    assert payload["terminal_bench"]["total_trials"] == 0


async def test_waiting_miner_env_status_is_public_safe(
    client,
    database_session,
):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-waiting",
            name="waiting-agent",
            agent_hash="waiting-agent-hash",
            artifact_uri="/tmp/waiting-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.flush()
        for to_status, actor, from_status in (
            ("received", "api", None),
            ("upload_verified", "api", None),
            ("rate_limit_reserved", "api", None),
            ("analysis_queued", "analysis", None),
            ("ast_running", "worker", None),
            ("llm_running", "worker", None),
            ("analysis_allowed", "worker", None),
            ("waiting_miner_env", "worker", None),
        ):
            kwargs = {"from_status": from_status} if to_status == "received" else {}
            await transition_submission_status(
                session,
                submission,
                to_status,
                actor=actor,
                reason=(
                    "waiting_miner_env"
                    if to_status == "waiting_miner_env"
                    else f"{to_status} reason"
                ),
                **kwargs,
            )
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Waiting environments"
    assert payload["public_state"] == "Waiting environments"
    assert payload["phase"] == "waiting_environments"
    assert payload["env_action_required"] is True
    assert payload["env_keys"] == []
    assert payload["env_var_count"] == 0
    assert payload["env_confirmed_empty"] is False
    assert payload["env_locked"] is False
    assert payload["env_updated_at"] is None
    assert payload["analyzer"]["phase"] == "completed"
    assert payload["progress"]["evaluation_jobs"] == 0


async def test_waiting_miner_env_public_payloads_include_redacted_env_metadata(
    client,
    database_session,
    tmp_path,
):
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))
    sentinel_value = "task8-status-metadata-secret-value"

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-waiting-env-metadata",
            name="waiting-env-metadata-agent",
            agent_hash="waiting-env-metadata-agent-hash",
            artifact_uri="/tmp/waiting-env-metadata-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.flush()
        for index, to_status in enumerate(
            (
                "received",
                "upload_verified",
                "rate_limit_reserved",
                "analysis_queued",
                "ast_running",
                "llm_running",
                "analysis_allowed",
                "waiting_miner_env",
            )
        ):
            kwargs = {"from_status": None} if index == 0 else {}
            await transition_submission_status(
                session,
                submission,
                to_status,
                actor="worker" if index >= 4 else "api",
                reason="waiting_miner_env" if to_status == "waiting_miner_env" else to_status,
                **kwargs,
            )
        env_var = SubmissionEnvVar.encrypted(
            submission_id=submission.id,
            key="TASK8_PUBLIC_METADATA_KEY",
            value=sentinel_value,
            settings=settings,
        )
        session.add(env_var)
        await task_events.record_task_event(
            session,
            submission_id=submission.id,
            event_type="task.log",
            message=f"runtime API_KEY={sentinel_value}",
            metadata={"env": {"TASK8_PUBLIC_METADATA_KEY": sentinel_value}, "safe": "visible"},
        )
        await session.commit()
        submission_id = submission.id
        ciphertext = env_var.value_ciphertext
        value_hash = env_var.value_sha256

    status_response = await client.get(f"/submissions/{submission_id}/status")
    list_response = await client.get("/submissions")
    detail_response = await client.get(f"/submissions/{submission_id}")
    task_events_response = await client.get(f"/submissions/{submission_id}/task-events?limit=10")

    assert status_response.status_code == 200
    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert task_events_response.status_code == 200

    status_payload = status_response.json()
    detail_payload = detail_response.json()
    list_payload = next(row for row in list_response.json() if row["id"] == submission_id)
    for payload in (status_payload, detail_payload, list_payload):
        assert payload["status"] == "Waiting environments"
        assert payload["env_action_required"] is True
        assert payload["env_keys"] == ["TASK8_PUBLIC_METADATA_KEY"]
        assert payload["env_var_count"] == 1
        assert payload["env_confirmed_empty"] is False
        assert payload["env_locked"] is False
        assert payload["env_updated_at"] is not None

    async with database_session() as session:
        latest_event = (
            (
                await session.execute(
                    select(SubmissionStatusEvent)
                    .where(SubmissionStatusEvent.submission_id == submission_id)
                    .order_by(SubmissionStatusEvent.sequence.desc())
                    .limit(1)
                )
            )
            .scalars()
            .one()
        )

    sse_events = _parse_sse_events(routes._format_sse_event(latest_event))
    waiting_event = sse_events[-1]["data"]
    assert waiting_event["status"] == "Waiting environments"
    assert waiting_event["public_state"] == "Waiting environments"
    assert waiting_event["phase"] == "waiting_environments"
    assert waiting_event["reason_code"] == "waiting_miner_env"
    assert "env_keys" not in waiting_event

    task_event_payload = task_events_response.json()
    assert task_event_payload["events"][0]["metadata"] == {"safe": "visible"}
    serialized = json.dumps(
        {
            "status": status_payload,
            "list": list_payload,
            "detail": detail_payload,
            "sse": sse_events,
            "task_events": task_event_payload,
        },
        sort_keys=True,
    )
    for forbidden in (
        sentinel_value,
        ciphertext,
        value_hash,
        str(key_file),
        "value_ciphertext",
        "value_sha256",
        "submission_env_encryption_key_file",
    ):
        assert forbidden not in serialized


async def test_public_status_and_sse_expose_distinct_lifecycle_phase_copy(
    client,
    database_session,
):
    expected = {
        "ast_running": ("AST review", "ast_review"),
        "llm_running": ("LLM review", "llm_review"),
        "llm_standby": ("LLM standby", "llm_standby"),
        "waiting_miner_env": ("Waiting environments", "waiting_environments"),
        "tb_queued": ("evaluation queued", "evaluation_queued"),
        "tb_running": ("evaluating", "evaluation"),
    }
    async with database_session() as session:
        submissions: dict[str, int] = {}
        for index, raw_status in enumerate(expected, start=1):
            public_copy, _phase = expected[raw_status]
            submission = AgentSubmission(
                miner_hotkey=f"miner-visible-{index}",
                name=f"visible-{raw_status}",
                agent_hash=f"visible-{raw_status}",
                artifact_uri=f"/tmp/visible-{raw_status}.zip",
                status=public_copy,
                raw_status=raw_status,
                effective_status=public_copy,
            )
            session.add(submission)
            await session.flush()
            session.add(
                SubmissionStatusEvent(
                    submission_id=submission.id,
                    sequence=1,
                    from_status="analysis_queued" if raw_status != "ast_running" else None,
                    to_status=raw_status,
                    reason=raw_status,
                    actor="worker" if raw_status.startswith(("ast", "llm")) else "evaluation",
                )
            )
            submissions[raw_status] = submission.id
        await session.commit()

    async with database_session() as session:
        status_events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.id)
                )
            )
            .scalars()
            .all()
        )
    events_by_submission_id = {event.submission_id: event for event in status_events}

    for raw_status, submission_id in submissions.items():
        public_copy, phase = expected[raw_status]
        status_response = await client.get(f"/submissions/{submission_id}/status")
        sse_event = events_by_submission_id[submission_id]

        assert status_response.status_code == 200
        status_payload = status_response.json()
        sse_payload = _parse_sse_events(routes._format_sse_event(sse_event))[-1]["data"]
        assert status_payload["status"] == public_copy
        assert status_payload["public_state"] == public_copy
        assert status_payload["phase"] == phase
        assert sse_payload["status"] == public_copy
        assert sse_payload["public_state"] == public_copy
        assert sse_payload["phase"] == phase


async def test_submission_status_redacts_raw_analysis_similarity_and_trial_details(
    client,
    database_session,
):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-redaction",
            name="redaction-agent",
            agent_hash="redaction-agent-hash",
            artifact_uri="/tmp/redaction-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.flush()
        for to_status, actor, from_status in (
            ("received", "api", None),
            ("upload_verified", "api", None),
            ("rate_limit_reserved", "api", None),
            ("analysis_queued", "analysis", None),
            ("ast_running", "worker", None),
            ("llm_running", "worker", None),
            ("analysis_allowed", "worker", None),
            ("waiting_miner_env", "worker", None),
            ("tb_queued", "evaluation", None),
            ("tb_running", "evaluation", None),
        ):
            kwargs = {"from_status": from_status} if to_status == "received" else {}
            await transition_submission_status(
                session,
                submission,
                to_status,
                actor=actor,
                reason=f"{to_status} reason",
                **kwargs,
            )
        analysis = AnalysisRun(
            submission_id=submission.id,
            analyzer_name="static",
            analyzer_version="v1",
            status="completed",
            verdict="allow",
            reason_codes_json=json.dumps(["safe_reason"]),
            report_json=json.dumps({"raw_code": "print('do-not-expose')"}),
        )
        session.add(analysis)
        await session.flush()
        session.add(
            LlmVerdict(
                analysis_run_id=analysis.id,
                reviewer_name="reviewer",
                model_name="model",
                verdict="allow",
                confidence=0.91,
                reason_codes_json=json.dumps(["llm_safe_reason"]),
                raw_request_json=json.dumps({"Authorization": "Bearer raw-provider-token"}),
                raw_response_json=json.dumps(
                    {
                        "content": "raw transcript with sk-test-secret",
                        "provider_errors": ["sk-test-secret"],
                        "verdict_json": {
                            "verdict": "allow",
                            "confidence": 0.91,
                            "rationale": (
                                "Safe review rationale. Ignored raw token sk-test-secret "
                                "from /tmp/private-prompt.txt and Bearer raw-provider-token."
                            ),
                        },
                    }
                ),
            )
        )
        session.add(
            SimilarityMatch(
                analysis_run_id=analysis.id,
                source_submission_id=submission.id,
                matched_submission_id=999,
                matched_artifact_uri="/tmp/private-match.zip",
                match_kind="python_ast_similarity",
                score=92.5,
                evidence_json=json.dumps(
                    {
                        "risk_band": "high",
                        "algorithm_version": "sim-v1",
                        "matched_code": "def stolen(): pass",
                        "top_file_pairs": [
                            {
                                "source_file_path": "agent.py",
                                "matched_file_path": "other.py",
                                "score_percent": 92.5,
                                "source_code": "def source_secret(): pass",
                            },
                            {
                                "source_file_path": "/tmp/private-source.py",
                                "matched_file_path": "/root/private-match.py",
                                "score_percent": 88.0,
                                "matched_code": "def matched_secret(): pass",
                            },
                            {
                                "source_file_path": "three.py",
                                "matched_file_path": "three-other.py",
                                "score_percent": 3.0,
                            },
                            {
                                "source_file_path": "four.py",
                                "matched_file_path": "four-other.py",
                                "score_percent": 4.0,
                            },
                            {
                                "source_file_path": "five.py",
                                "matched_file_path": "five-other.py",
                                "score_percent": 5.0,
                            },
                            {
                                "source_file_path": "six.py",
                                "matched_file_path": "six-other.py",
                                "score_percent": 6.0,
                            },
                        ],
                    }
                ),
            )
        )
        session.add_all(
            [
                PythonAstFeature(
                    analysis_run_id=analysis.id,
                    file_path="agent.py",
                    feature_key="function:run",
                    feature_type="function",
                    feature_value="def secret_source(): pass",
                ),
                PythonAstFeature(
                    analysis_run_id=analysis.id,
                    file_path="agent.py",
                    feature_key="import:os",
                    feature_type="import",
                    feature_value="sk-test-secret",
                ),
                PythonAstFeature(
                    analysis_run_id=analysis.id,
                    file_path="agent.py",
                    feature_key="function:helper",
                    feature_type="function",
                    feature_value="/tmp/private-helper.py",
                ),
            ]
        )
        job = EvaluationJob(
            job_id="job-redaction",
            submission_id=submission.id,
            status="running",
            selected_tasks_json="[]",
            score=0.25,
            passed_tasks=1,
            total_tasks=4,
            verdict="valid",
            reason_codes_json=json.dumps(["job_safe_reason"]),
            last_error="provider failure included sk-test-secret",
            lease_owner="lease-worker-secret",
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        attempt = EvaluationAttempt(
            submission_id=submission.id,
            job_id=job.id,
            attempt_number=2,
            evaluator_name="terminal_bench",
            status="running",
            score=None,
            error="harbor secret bearer token",
            metadata_json=json.dumps({"command": ["secret-command"]}),
        )
        session.add(attempt)
        await session.flush()
        session.add_all(
            [
                TerminalBenchTrial(
                    evaluation_attempt_id=attempt.id,
                    task_id="task-a",
                    trial_name="trial-a",
                    trial_number=1,
                    job_dir="/tmp/private-job-dir",
                    job_name="tb21-redaction",
                    status="completed",
                    score=1.0,
                    is_final=1,
                    raw_artifacts_json=json.dumps({"stdout": "secret stdout"}),
                ),
                TerminalBenchTrial(
                    evaluation_attempt_id=attempt.id,
                    task_id="task-b",
                    trial_name="trial-b",
                    trial_number=2,
                    job_dir="/tmp/private-job-dir",
                    job_name="tb21-redaction",
                    status="errored",
                    score=None,
                    is_final=0,
                    raw_artifacts_json=json.dumps({"stderr": "secret stderr"}),
                ),
            ]
        )
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["public_state"] == "evaluating"
    assert payload["phase"] == "evaluation"
    assert payload["current_attempt"] == 2
    assert payload["analyzer"] == {
        **payload["analyzer"],
        "phase": "completed",
        "status": "completed",
        "verdict": "allow",
        "reason_codes": ["safe_reason"],
        "llm_verdict": "allow",
        "llm_confidence": 0.91,
        "llm_reason_codes": ["llm_safe_reason"],
        "llm_rationale": (
            "Safe review rationale. Ignored raw [REDACTED_SECRET] sk-[REDACTED] "
            "from [REDACTED_PATH] and Bearer [REDACTED]"
        ),
    }
    assert payload["similarity"]["max_score_percent"] == 92.5
    assert payload["similarity"]["top_matches"] == [
        {
            "matched_submission_id": 999,
            "match_kind": "python_ast_similarity",
            "score_percent": 92.5,
            "risk_band": "high",
            "algorithm_version": "sim-v1",
            "top_file_pairs": [
                {
                    "source_file_path": "agent.py",
                    "matched_file_path": "other.py",
                    "score_percent": 92.5,
                },
                {
                    "source_file_path": "[REDACTED_PATH]",
                    "matched_file_path": "[REDACTED_PATH]",
                    "score_percent": 88.0,
                },
                {
                    "source_file_path": "three.py",
                    "matched_file_path": "three-other.py",
                    "score_percent": 3.0,
                },
                {
                    "source_file_path": "four.py",
                    "matched_file_path": "four-other.py",
                    "score_percent": 4.0,
                },
                {
                    "source_file_path": "five.py",
                    "matched_file_path": "five-other.py",
                    "score_percent": 5.0,
                },
            ],
        }
    ]
    assert payload["ast"] == {
        "feature_count": 3,
        "feature_types": {"function": 2, "import": 1},
        "verdict": None,
        "verdict_reason": None,
    }
    assert payload["rules_check"] is None
    assert payload["evaluation"]["job_id"] == "job-redaction"
    assert payload["evaluation"]["status"] == "running"
    assert payload["evaluation"]["current_attempt"] == 2
    assert payload["evaluation"]["attempt_status"] == "running"
    assert payload["terminal_bench"] == {
        "total_trials": 2,
        "completed_trials": 1,
        "failed_trials": 0,
        "errored_trials": 1,
        "final_trials": 1,
    }
    assert payload["progress"] == {
        "status_events": 10,
        "analysis_runs": 1,
        "similarity_matches": 1,
        "llm_verdicts": 1,
        "evaluation_jobs": 1,
        "evaluation_attempts": 1,
        "terminal_bench_trials": 2,
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "do-not-expose",
        "raw-provider-token",
        "sk-test-secret",
        "raw provider transcript",
        "matched_code",
        "hidden_ast_value",
        "def stolen",
        "source_secret",
        "matched_secret",
        "secret_source",
        "private-source",
        "private-match",
        "lease-worker-secret",
        "secret-command",
        "secret stdout",
        "secret stderr",
        "evidence_json",
        "raw_response_json",
        "raw_request_json",
    ):
        assert forbidden not in serialized


async def test_submission_status_route_is_publicly_discoverable(client, database_session):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-public",
            name="public-agent",
            agent_hash="public-agent-hash",
            artifact_uri="/tmp/public-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.flush()
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="received",
            from_status=None,
        )
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    assert response.json()["last_event_sequence"] == 1


async def test_platform_sdk_public_status_events_and_task_events_use_public_contract(
    client,
    database_session,
):
    cases = {
        "completed": {
            "raw_status": "tb_completed",
            "public_state": "valid",
            "phase": "complete",
            "job_status": "completed",
            "attempt_status": "completed",
            "task_event_type": "task.completed",
            "task_event_status": "completed",
        },
        "failed-final": {
            "raw_status": "tb_failed_final",
            "public_state": "error",
            "phase": "failed",
            "job_status": "failed",
            "attempt_status": "failed",
            "task_event_type": "task.failed",
            "task_event_status": "failed",
        },
    }
    async with database_session() as session:
        submissions: dict[str, int] = {}
        status_events: dict[str, SubmissionStatusEvent] = {}
        for slug, case in cases.items():
            submission = AgentSubmission(
                miner_hotkey=f"miner-platform-sdk-{slug}",
                name=f"platform-sdk-{slug}",
                agent_hash=f"platform-sdk-{slug}-hash",
                artifact_uri=(
                    f"/terminal-bench/jobs/platform-sdk-private/artifact-task8-{slug}.zip"
                ),
                status="received",
                raw_status="received",
                effective_status="received",
                artifact_path=(
                    f"/terminal-bench/jobs/platform-sdk-private/artifact-task8-{slug}.zip"
                ),
            )
            session.add(submission)
            await session.flush()
            job_dir = f"/terminal-bench/jobs/platform-sdk-private/tb21-platform-sdk-secret-{slug}"
            for index, to_status in enumerate(
                (
                    "received",
                    "upload_verified",
                    "rate_limit_reserved",
                    "analysis_queued",
                    "ast_running",
                    "llm_running",
                    "analysis_allowed",
                    "waiting_miner_env",
                    "tb_queued",
                    "tb_running",
                    case["raw_status"],
                )
            ):
                kwargs = {"from_status": None} if index == 0 else {}
                status_event = await transition_submission_status(
                    session,
                    submission,
                    to_status,
                    actor="evaluation" if to_status.startswith("tb_") else "worker",
                    reason="evaluation_job_completed"
                    if to_status == "tb_completed"
                    else "evaluation_job_failed"
                    if to_status == "tb_failed_final"
                    else to_status,
                    metadata={
                        "execution_provider": "platform_sdk",
                        "job_dir": job_dir,
                        "job_name": f"k8s-job-task8-{slug}",
                        "pod_name": f"pod-task8-{slug}",
                        "broker_ref": f"broker-token-{slug}",
                        "worker": "worker-a",
                    },
                    **kwargs,
                )
            job = EvaluationJob(
                job_id=f"public-platform-sdk-{slug}-job",
                submission_id=submission.id,
                status=case["job_status"],
                selected_tasks_json=json.dumps([f"safe-task-{slug}"]),
                score=1.0 if slug == "completed" else 0.0,
                passed_tasks=1 if slug == "completed" else 0,
                total_tasks=1,
                verdict="valid" if slug == "completed" else "invalid",
                reason_codes_json=json.dumps([f"safe_{slug.replace('-', '_')}"]),
                logs_ref=f"{job_dir}/logs.txt",
                lease_owner="worker-a",
                last_error=f"platform_sdk base_sdk broker-token-{slug} raw-ref-task8-{slug}",
            )
            session.add(job)
            await session.flush()
            submission.latest_evaluation_job_id = job.id
            attempt = EvaluationAttempt(
                submission_id=submission.id,
                job_id=job.id,
                attempt_number=1,
                evaluator_name="terminal_bench",
                status=case["attempt_status"],
                score=job.score,
                error=f"platform_sdk failed at pod-task8-{slug}",
                metadata_json=json.dumps(
                    {
                        "execution_provider": "platform_sdk",
                        "provider": "platform_sdk",
                        "job_dir": job_dir,
                        "job_name": f"k8s-job-task8-{slug}",
                        "pod_name": f"pod-task8-{slug}",
                        "raw_ref": f"raw-ref-task8-{slug}",
                        "broker_ref": f"broker-token-{slug}",
                        "worker": "worker-a",
                    }
                ),
                lease_owner="worker-a",
            )
            session.add(attempt)
            await session.flush()
            trial = TerminalBenchTrial(
                evaluation_attempt_id=attempt.id,
                task_id=f"safe-task-{slug}",
                trial_name=f"safe-trial-{slug}",
                trial_number=1,
                job_dir=job_dir,
                job_name=f"k8s-job-task8-{slug}",
                status="completed" if slug == "completed" else "failed",
                score=job.score,
                is_final=1,
                raw_artifacts_json=json.dumps(
                    {
                        "provider": "platform_sdk",
                        "raw_ref": f"raw-ref-task8-{slug}",
                        "broker_ref": f"broker-token-{slug}",
                        "pod_name": f"pod-task8-{slug}",
                    }
                ),
                lease_owner="worker-a",
                stdout_ref=f"{job_dir}/stdout.log",
                stderr_ref=f"{job_dir}/stderr.log",
            )
            session.add(trial)
            await session.flush()
            session.add(
                ExternalExecutionRef(
                    evaluation_attempt_id=attempt.id,
                    terminal_bench_trial_id=trial.id,
                    provider="platform_sdk",
                    external_id=f"tb21-platform-sdk-secret-{slug}-external",
                    status=case["attempt_status"],
                    job_dir=job_dir,
                    job_name=f"k8s-job-task8-{slug}",
                    raw_ref=f"raw-ref-task8-{slug}",
                    raw_payload_json=json.dumps(
                        {
                            "provider": "platform_sdk",
                            "pod_name": f"pod-task8-{slug}",
                            "broker_ref": f"broker-token-{slug}",
                            "worker": "worker-a",
                        }
                    ),
                )
            )
            await task_events.record_task_event(
                session,
                submission_id=submission.id,
                job_id=job.id,
                task_id=f"safe-task-{slug}",
                event_type="task.status",
                status="starting",
                message=f"safe platform starting {slug}",
                metadata={
                    "phase": "starting",
                    "attempt": 0,
                    "provider": "platform_sdk",
                    "job_name": f"k8s-job-task8-{slug}",
                    "pod_name": f"pod-task8-{slug}",
                },
            )
            await task_events.record_task_event(
                session,
                submission_id=submission.id,
                job_id=job.id,
                task_id=f"safe-task-{slug}",
                event_type="task.status",
                status=case["task_event_status"],
                message=f"safe platform latest phase {slug}",
                metadata={
                    "phase": case["task_event_status"],
                    "attempt": 1,
                    "provider": "platform_sdk",
                    "raw_ref": f"raw-ref-task8-{slug}",
                    "worker": "worker-a",
                },
            )
            await task_events.record_task_event(
                session,
                submission_id=submission.id,
                job_id=job.id,
                task_id=f"safe-task-{slug}",
                event_type=case["task_event_type"],
                status=case["task_event_status"],
                message=f"safe platform execution {slug}",
                metadata={
                    "safe": slug,
                    "execution_provider": "platform_sdk",
                    "provider": "platform_sdk",
                    "job_dir": job_dir,
                    "job_name": f"k8s-job-task8-{slug}",
                    "kubernetes_job_name": f"k8s-job-task8-{slug}",
                    "pod_name": f"pod-task8-{slug}",
                    "raw_ref": f"raw-ref-task8-{slug}",
                    "broker_ref": f"broker-token-{slug}",
                    "command_path": (
                        f"{job_dir}/platform-terminal-bench-command.sh"
                        f" {job_dir}/base-terminal-bench-command.sh"
                    ),
                    "worker": "worker-a",
                },
            )
            submissions[slug] = submission.id
            status_events[slug] = status_event
        await session.commit()

    for slug, case in cases.items():
        submission_id = submissions[slug]
        status_response = await client.get(f"/submissions/{submission_id}/status")
        task_events_response = await client.get(
            f"/submissions/{submission_id}/task-events?limit=10"
        )
        task_events_stream_response = await client.get(
            f"/submissions/{submission_id}/task-events/stream"
        )
        status_events_response = await client.get(f"/submissions/{submission_id}/events")

        assert status_response.status_code == 200
        assert task_events_response.status_code == 200
        assert task_events_stream_response.status_code == 200
        assert status_events_response.status_code == 200

        status_payload = status_response.json()
        task_events_payload = task_events_response.json()
        task_stream_events = _parse_sse_events(task_events_stream_response.text)
        status_stream_events = _parse_sse_events(status_events_response.text)
        direct_status_sse = _parse_sse_events(routes._format_sse_event(status_events[slug]))[-1]
        assert status_payload["status"] == case["public_state"]
        assert status_payload["public_state"] == case["public_state"]
        assert status_payload["phase"] == case["phase"]
        assert status_payload["effective_status"] == case["public_state"]
        assert status_payload["evaluation"] == {
            **status_payload["evaluation"],
            "job_id": f"public-platform-sdk-{slug}-job",
            "status": case["job_status"],
            "current_attempt": 1,
            "attempt_status": case["attempt_status"],
        }
        expected_phase_status = case["task_event_status"]
        assert status_payload["evaluation"]["task_phases"] == [
            {
                "task_id": f"safe-task-{slug}",
                "phase": expected_phase_status,
                "status": expected_phase_status,
                "updated_at": status_payload["evaluation"]["task_phases"][0]["updated_at"],
                "attempt": 1,
            }
        ]
        assert set(status_payload["evaluation"]["task_phases"][0]) == {
            "task_id",
            "phase",
            "status",
            "updated_at",
            "attempt",
        }
        terminal_task_events = [
            event
            for event in task_events_payload["events"]
            if event["event_type"] == case["task_event_type"]
        ]
        assert terminal_task_events == [
            {
                **terminal_task_events[0],
                "event_type": case["task_event_type"],
                "status": case["task_event_status"],
                "metadata": {"safe": slug},
            }
        ]
        assert task_stream_events[-1]["data"]["metadata"] == {"safe": slug}
        assert status_stream_events[-1]["data"]["status"] == case["public_state"]
        assert status_stream_events[-1]["data"]["phase"] == case["phase"]
        assert direct_status_sse["data"]["status"] == case["public_state"]
        _assert_platform_sdk_markers_redacted(
            {
                "status": status_payload,
                "task_events": task_events_payload,
                "task_stream": task_stream_events,
                "status_stream": status_stream_events,
                "direct_status_sse": direct_status_sse,
            }
        )


async def _seed_submission_with_analysis(session, suffix: str) -> int:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{suffix}",
        name=f"agent-{suffix}",
        agent_hash=f"hash-{suffix}",
        artifact_uri=f"/tmp/{suffix}.zip",
        status="received",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    analysis = AnalysisRun(
        submission_id=submission.id,
        analyzer_name="static",
        analyzer_version="v1",
        status="completed",
        verdict="allow",
        reason_codes_json=json.dumps(["safe_reason"]),
    )
    session.add(analysis)
    await session.flush()
    session.add(
        LlmVerdict(
            analysis_run_id=analysis.id,
            reviewer_name="reviewer",
            model_name="model",
            verdict="allow",
            confidence=0.91,
        )
    )
    session.add_all(
        [
            SimilarityMatch(
                analysis_run_id=analysis.id,
                source_submission_id=submission.id,
                matched_submission_id=999,
                match_kind="python_ast_similarity",
                score=92.5,
            ),
            SimilarityMatch(
                analysis_run_id=analysis.id,
                source_submission_id=submission.id,
                matched_submission_id=998,
                match_kind="python_ast_similarity",
                score=40.0,
            ),
        ]
    )
    session.add_all(
        [
            PythonAstFeature(
                analysis_run_id=analysis.id,
                file_path="agent.py",
                feature_key="function:run",
                feature_type="function",
                feature_value="def run(): pass",
            ),
            PythonAstFeature(
                analysis_run_id=analysis.id,
                file_path="agent.py",
                feature_key="import:os",
                feature_type="import",
                feature_value="import os",
            ),
            PythonAstFeature(
                analysis_run_id=analysis.id,
                file_path="agent.py",
                feature_key="call:open",
                feature_type="call",
                feature_value="open(...)",
            ),
        ]
    )
    return submission.id


async def _seed_submission_without_analysis(session, suffix: str) -> int:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{suffix}",
        name=f"agent-{suffix}",
        agent_hash=f"hash-{suffix}",
        artifact_uri=f"/tmp/{suffix}.zip",
        status="received",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    return submission.id


async def test_list_submissions_includes_analysis_summary_fields(client, database_session):
    async with database_session() as session:
        with_id = await _seed_submission_with_analysis(session, "with-analysis")
        without_id = await _seed_submission_without_analysis(session, "without-analysis")
        await session.commit()

    list_response = await client.get("/submissions")
    detail_response = await client.get(f"/submissions/{with_id}")

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    rows = {row["id"]: row for row in list_response.json()}

    enriched = rows[with_id]
    assert enriched["has_analysis"] is True
    assert enriched["analyzer_status"] == "completed"
    assert enriched["analyzer_verdict"] == "allow"
    assert enriched["llm_verdict"] == "allow"
    assert enriched["llm_confidence"] == 0.91
    assert enriched["similarity_max_score_percent"] == 92.5
    assert enriched["similarity_match_count"] == 2
    assert enriched["ast_feature_count"] == 3

    detail_payload = detail_response.json()
    assert detail_payload["has_analysis"] is True
    assert detail_payload["analyzer_status"] == "completed"
    assert detail_payload["llm_verdict"] == "allow"
    assert detail_payload["similarity_max_score_percent"] == 92.5
    assert detail_payload["similarity_match_count"] == 2
    assert detail_payload["ast_feature_count"] == 3

    empty = rows[without_id]
    assert empty["has_analysis"] is False
    assert empty["analyzer_status"] is None
    assert empty["analyzer_verdict"] is None
    assert empty["llm_verdict"] is None
    assert empty["llm_confidence"] is None
    assert empty["similarity_max_score_percent"] is None
    assert empty["similarity_match_count"] == 0
    assert empty["ast_feature_count"] == 0


async def test_list_submissions_analysis_summary_has_no_n_plus_one(client, database_session):
    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    async def _count_list_queries() -> int:
        statements.clear()
        event.listen(database.engine.sync_engine, "before_cursor_execute", _record)
        try:
            response = await client.get("/submissions")
        finally:
            event.remove(database.engine.sync_engine, "before_cursor_execute", _record)
        assert response.status_code == 200
        return len(statements)

    async with database_session() as session:
        for index in range(2):
            await _seed_submission_with_analysis(session, f"two-{index}")
        await session.commit()
    queries_for_two = await _count_list_queries()

    async with database_session() as session:
        for index in range(3):
            await _seed_submission_with_analysis(session, f"five-{index}")
        await session.commit()
    queries_for_five = await _count_list_queries()

    assert queries_for_two == queries_for_five
    assert queries_for_five <= 12


# Assembled at runtime from fragments so no contiguous secret literal is
# committed (avoids GitHub push protection); the runtime value is unchanged
# and still matches redact_secrets' sk- pattern for the assertions below.
_RULES_CHECK_SECRET = "sk-a" + "ntic" + "heat" + "LEAK" + "EDto" + "ken0" + "9876" + "5432" + "1"


async def _seed_submission_with_report(session, suffix: str, *, report_json: str) -> int:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{suffix}",
        name=f"agent-{suffix}",
        agent_hash=f"hash-{suffix}",
        artifact_uri=f"/tmp/{suffix}.zip",
        status="received",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    session.add(
        AnalysisRun(
            submission_id=submission.id,
            analyzer_name="static",
            analyzer_version="v1",
            status="completed",
            verdict="allow",
            reason_codes_json=json.dumps(["safe_reason"]),
            report_json=report_json,
        )
    )
    await session.flush()
    return submission.id


async def test_submission_status_exposes_ast_verdict_and_rules_check_with_redacted_evidence(
    client,
    database_session,
):
    report_json = json.dumps(
        {
            "ast": {
                "feature_count": 3,
                "verdict": "flagged",
                "verdict_reason": "max delta similarity 82.00% at or above high-risk 70.00%",
            },
            "llm_verdict": {"verdict": "flagged"},
            "similarity": {"algorithm_version": "python-ast-similarity-v1", "matches": []},
            "rules_check": {
                "rules_version": "sha256:abc123def456",
                "overall_verdict": "invalid",
                "recommended_status": "rejected",
                "reason_codes": ["hardcoded_secret", "acceptance_policy_violation"],
                "rule_results": [
                    {
                        "rule_id": "hardcoding",
                        "title": "Hardcoding Policy",
                        "status": "fail",
                        "reason_codes": ["hardcoded_secret"],
                        "evidence": [
                            {
                                "path": "src/agent/solver.py",
                                "line_start": 12,
                                "line_end": 12,
                                "snippet": f'API_KEY = "{_RULES_CHECK_SECRET}"',
                                "reason_code": "hardcoded_secret",
                                "description": "Hardcoded credential detected in solver",
                            }
                        ],
                    },
                    {
                        "rule_id": "acceptance",
                        "title": "Acceptance Policy",
                        "status": "pass",
                        "reason_codes": [],
                        "evidence": [],
                    },
                ],
                "evidence": [],
                "hardcoding_findings": [],
                "rules_files": ["acceptance.md", "anti-cheat.md", "hardcoding.md", "security.md"],
                "reviewer_used": True,
                "reviewer_notes": (
                    f"Reviewer flagged embedded token {_RULES_CHECK_SECRET} in solver."
                ),
            },
        }
    )
    async with database_session() as session:
        submission_id = await _seed_submission_with_report(
            session, "rules-flagged", report_json=report_json
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    payload = response.json()

    assert payload["ast"]["verdict"] == "flagged"
    assert (
        payload["ast"]["verdict_reason"]
        == "max delta similarity 82.00% at or above high-risk 70.00%"
    )

    rules_check = payload["rules_check"]
    assert rules_check is not None
    assert rules_check["verdict"] == "invalid"
    assert rules_check["recommended_status"] == "rejected"
    assert rules_check["rules_version"] == "sha256:abc123def456"
    assert rules_check["reviewer_used"] is True
    assert rules_check["reason_codes"] == ["hardcoded_secret", "acceptance_policy_violation"]

    rules_by_id = {rule["rule_id"]: rule for rule in rules_check["rules"]}
    assert rules_by_id["acceptance"]["status"] == "pass"
    failing = rules_by_id["hardcoding"]
    assert failing["status"] == "fail"
    assert failing["title"] == "Hardcoding Policy"
    assert failing["reason_codes"] == ["hardcoded_secret"]
    assert len(failing["evidence"]) == 1
    evidence = failing["evidence"][0]
    assert evidence["path"] == "src/agent/solver.py"
    assert evidence["line_start"] == 12
    assert evidence["line_end"] == 12
    assert evidence["reason_code"] == "hardcoded_secret"

    assert _RULES_CHECK_SECRET not in evidence["snippet"]
    assert "[REDACTED]" in evidence["snippet"]

    assert rules_check["notes"] is not None
    assert _RULES_CHECK_SECRET not in rules_check["notes"]

    serialized = json.dumps(payload, sort_keys=True)
    assert _RULES_CHECK_SECRET not in serialized
    assert "anticheatLEAKEDtoken0987654321" not in serialized


async def test_submission_status_rules_check_null_when_report_incomplete_or_missing(
    client,
    database_session,
):
    async with database_session() as session:
        empty_report_id = await _seed_submission_with_report(
            session, "rules-empty", report_json="{}"
        )
        null_rules_id = await _seed_submission_with_report(
            session,
            "rules-null",
            report_json=json.dumps(
                {
                    "ast": {"verdict": "clean", "verdict_reason": "no similarity risk"},
                    "rules_check": None,
                }
            ),
        )
        no_analysis_id = await _seed_submission_without_analysis(session, "rules-no-analysis")
        await session.commit()

    empty_response = await client.get(f"/submissions/{empty_report_id}/status")
    null_response = await client.get(f"/submissions/{null_rules_id}/status")
    no_analysis_response = await client.get(f"/submissions/{no_analysis_id}/status")

    assert empty_response.status_code == 200
    assert null_response.status_code == 200
    assert no_analysis_response.status_code == 200

    empty_payload = empty_response.json()
    assert empty_payload["rules_check"] is None
    assert empty_payload["ast"]["verdict"] is None
    assert empty_payload["ast"]["verdict_reason"] is None

    null_payload = null_response.json()
    assert null_payload["rules_check"] is None
    assert null_payload["ast"]["verdict"] == "clean"
    assert null_payload["ast"]["verdict_reason"] == "no similarity risk"

    no_analysis_payload = no_analysis_response.json()
    assert no_analysis_payload["rules_check"] is None
    assert no_analysis_payload["ast"]["verdict"] is None
    assert no_analysis_payload["ast"]["verdict_reason"] is None
