from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from _routing import public_route_paths
from sqlalchemy import select

from agent_challenge.app import app
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
    SubmissionFamily,
    TaskResult,
    TerminalBenchTrial,
)
from agent_challenge.submissions.state_machine import transition_submission_status
from agent_challenge.submissions.versioning import normalize_submission_name
from agent_challenge.swe_forge import SweForgeTask

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
FORBIDDEN_PUBLIC_STRINGS = (
    "sk-test-secret",
    "/tmp/private-job-dir",
    "Bearer raw-provider-token",
    "def secret_source",
    "lease-worker-secret",
    "broker-ref-secret",
    "matched_code",
    "ast-feature-secret",
    "private-match",
)
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


def _assert_platform_sdk_public_payload_is_redacted(payload: object) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in PLATFORM_SDK_PUBLIC_FORBIDDEN:
        assert forbidden not in serialized


def test_frontend_matrix_routes_are_publicly_decorated():
    public_paths = public_route_paths(app)

    assert {
        "/benchmarks",
        "/benchmarks/tasks",
        "/submissions",
        "/submissions/count",
        "/submissions/{submission_id}",
        "/submissions/{submission_id}/versions",
        "/submissions/{submission_id}/status",
        "/submissions/{submission_id}/review/tee",
        "/submissions/{submission_id}/task-events",
        "/submissions/{submission_id}/task-events/stream",
        "/submissions/{submission_id}/events",
        "/agents/{agent_hash}/evaluation",
        "/leaderboard",
    }.issubset(public_paths)
    assert "/internal/v1/bridge/submissions" not in public_paths


async def test_benchmark_routes_expose_frontend_contract_fields(client, monkeypatch):
    monkeypatch.setattr("agent_challenge.api.routes.settings.benchmark_backend", "swe_forge")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.swe_forge_tree_url",
        "gh://platform/public-swe-forge",
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.evaluation_concurrency", 3)
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend",
        "swe_forge",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [
            SweForgeTask(
                task_id="task-alpha",
                docker_image="baseintelligence/swe-forge:task-alpha",
            ),
            SweForgeTask(
                task_id="task-beta",
                docker_image="baseintelligence/swe-forge:task-beta",
            ),
        ],
    )

    response = await client.get("/benchmarks")
    tasks_response = await client.get("/benchmarks/tasks")

    assert response.status_code == 200
    assert response.json() == {
        "backend": "swe_forge",
        "dataset": "gh://platform/public-swe-forge",
        "task_count": 2,
        "evaluation_concurrency": 3,
    }
    assert tasks_response.status_code == 200
    assert tasks_response.json() == [
        {
            "task_id": "task-alpha",
            "benchmark": "swe_forge",
            "docker_image": "baseintelligence/swe-forge:task-alpha",
            "prompt": "",
        },
        {
            "task_id": "task-beta",
            "benchmark": "swe_forge",
            "docker_image": "baseintelligence/swe-forge:task-beta",
            "prompt": "",
        },
    ]


async def test_frontend_submission_status_and_evaluation_routes_are_public_safe(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id, agent_hash = await _create_rich_frontend_fixture(session)
        await session.commit()

    count_response = await client.get("/submissions/count")
    list_response = await client.get("/submissions")
    detail_response = await client.get(f"/submissions/{submission_id}")
    status_response = await client.get(f"/submissions/{submission_id}/status")
    evaluation_response = await client.get(f"/agents/{agent_hash}/evaluation")
    versions_response = await client.get(f"/submissions/{submission_id}/versions")

    assert count_response.status_code == 200
    assert count_response.json() == {"count": 2}
    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert status_response.status_code == 200
    assert evaluation_response.status_code == 200
    assert versions_response.status_code == 200

    list_payload = list_response.json()
    detail_payload = detail_response.json()
    status_payload = status_response.json()
    evaluation_payload = evaluation_response.json()
    versions_payload = versions_response.json()

    assert len(list_payload) == 2
    assert list_payload[0] == detail_payload
    assert list_payload[1]["version_label"] == "v1"
    assert list_payload[1]["is_latest_version"] is False
    assert detail_payload == {
        **detail_payload,
        "id": submission_id,
        "miner_hotkey": "miner-rich",
        "name": "rich-agent",
        "display_name": "Rich Agent",
        "family_id": "family-rich-public",
        "version_number": 2,
        "version_label": "v2",
        "version_count": 2,
        "is_latest_version": True,
        "latest_submission_id": submission_id,
        "agent_hash": agent_hash,
        "zip_sha256": "zip-rich-agent-hash",
        "status": "evaluating",
        "effective_status": "evaluating",
        "score": 0.82,
    }
    assert detail_payload["submitted_at"] is not None
    assert detail_payload["created_at"] is not None
    assert detail_payload["latest_evaluation"] == {
        **detail_payload["latest_evaluation"],
        "job_id": "job-rich",
        "status": "running",
        "score": 0.82,
        "passed_tasks": 2,
        "total_tasks": 3,
        "verdict": "valid",
        "rules_version": "rules-v1",
    }

    assert status_payload == {
        **status_payload,
        "submission_id": submission_id,
        "name": "rich-agent",
        "display_name": "Rich Agent",
        "family_id": "family-rich-public",
        "version_number": 2,
        "version_label": "v2",
        "version_count": 2,
        "is_latest_version": True,
        "latest_submission_id": submission_id,
        "agent_hash": agent_hash,
        "status": "evaluating",
        "public_state": "evaluating",
        "phase": "evaluation",
        "current_attempt": 2,
    }
    assert status_payload["last_event_id"] is not None
    assert status_payload["last_event_sequence"] == 10
    assert status_payload["submitted_at"] is not None
    assert status_payload["updated_at"] is not None
    assert status_payload["analyzer"] == {
        **status_payload["analyzer"],
        "phase": "completed",
        "status": "completed",
        "verdict": "allow",
        "reason_codes": ["safe_reason"],
        "llm_verdict": "allow",
        "llm_confidence": 0.91,
        "llm_reason_codes": ["llm_safe_reason"],
        "llm_rationale": "No policy issue found for [REDACTED_PATH] using sk-[REDACTED].",
    }
    assert status_payload["analyzer"]["started_at"] is not None
    assert status_payload["analyzer"]["finished_at"] is not None
    assert status_payload["similarity"] == {
        "max_score_percent": 92.5,
        "match_count": 1,
        "top_matches": [
            {
                "matched_submission_id": 999,
                "match_kind": "python_ast_similarity",
                "score_percent": 92.5,
                "risk_band": "high",
                "algorithm_version": "sim-v1",
                "top_file_pairs": [
                    {
                        "source_file_path": "agent.py",
                        "matched_file_path": "public-match.py",
                        "score_percent": 92.5,
                    },
                    {
                        "source_file_path": "[REDACTED_PATH]",
                        "matched_file_path": "[REDACTED_PATH]",
                        "score_percent": 90.0,
                    },
                ],
            }
        ],
    }
    assert status_payload["ast"] == {
        "feature_count": 2,
        "feature_types": {"call": 1, "function": 1},
        "verdict": None,
        "verdict_reason": None,
    }
    assert status_payload["rules_check"] is None
    expected_task_rows = [
        {
            "task_id": "task-alpha",
            "display_name": "task-alpha",
            "source": "benchmark",
            "phase": "completed",
            "status": "completed",
            "updated_at": status_payload["evaluation"]["task_rows"][0]["updated_at"],
            "attempt": None,
            "has_result": True,
        },
        {
            "task_id": "task-beta",
            "display_name": "task-beta",
            "source": "benchmark",
            "phase": "failed",
            "status": "failed",
            "updated_at": status_payload["evaluation"]["task_rows"][1]["updated_at"],
            "attempt": None,
            "has_result": True,
        },
    ]
    assert status_payload["evaluation"]["task_phases"] == []
    assert status_payload["evaluation"]["task_rows"] == expected_task_rows
    assert status_payload["evaluation"] == {
        "job_id": "job-rich",
        "status": "running",
        "score": 0.82,
        "passed_tasks": 2,
        "total_tasks": 3,
        "verdict": "valid",
        "reason_codes": ["job_safe_reason"],
        "current_attempt": 2,
        "attempt_status": "running",
        "task_phases": [],
        "task_rows": expected_task_rows,
    }
    assert status_payload["terminal_bench"] == {
        "total_trials": 2,
        "completed_trials": 1,
        "failed_trials": 0,
        "errored_trials": 1,
        "final_trials": 1,
    }
    assert status_payload["progress"] == {
        "status_events": 10,
        "analysis_runs": 1,
        "similarity_matches": 1,
        "llm_verdicts": 1,
        "evaluation_jobs": 1,
        "evaluation_attempts": 1,
        "terminal_bench_trials": 2,
    }

    assert evaluation_payload == {
        **evaluation_payload,
        "job_id": "job-rich",
        "submission_id": submission_id,
        "name": "rich-agent",
        "display_name": "Rich Agent",
        "family_id": "family-rich-public",
        "version_number": 2,
        "version_label": "v2",
        "version_count": 2,
        "is_latest_version": True,
        "latest_submission_id": submission_id,
        "agent_hash": agent_hash,
        "zip_sha256": "zip-rich-agent-hash",
        "status": "running",
        "effective_status": "evaluating",
        "score": 0.82,
        "passed_tasks": 2,
        "total_tasks": 3,
        "verdict": "valid",
        "rules_version": "rules-v1",
    }
    assert evaluation_payload["task_phases"] == []
    assert evaluation_payload["task_rows"] == [
        {
            **expected_task_rows[0],
            "updated_at": evaluation_payload["task_rows"][0]["updated_at"],
        },
        {
            **expected_task_rows[1],
            "updated_at": evaluation_payload["task_rows"][1]["updated_at"],
        },
    ]
    assert evaluation_payload["tasks"] == [
        {
            "task_id": "task-alpha",
            "docker_image": "baseintelligence/swe-forge:task-alpha",
            "status": "passed",
            "score": 1.0,
            "returncode": 0,
            "duration_seconds": 12.5,
            "failure_reason": None,
            "detail_log": None,
        },
        {
            "task_id": "task-beta",
            "docker_image": "baseintelligence/swe-forge:task-beta",
            "status": "failed",
            "score": 0.0,
            "returncode": 1,
            "duration_seconds": 8.25,
            "failure_reason": "task log with Bearer [REDACTED] and [REDACTED_SECRET]",
            "detail_log": (
                "Task: task-beta\n"
                "Status: failed\n"
                "Score: 0.0000\n"
                "Return code: 1\n"
                "Duration seconds: 8.250\n\n"
                "Error log:\n"
                "task log with Bearer [REDACTED] and [REDACTED_SECRET]"
            ),
        },
    ]
    for task in evaluation_payload["tasks"]:
        assert {"stdout", "stderr", "logs_ref", "raw_artifacts_json"}.isdisjoint(task)

    assert [version["version_label"] for version in versions_payload] == ["v1", "v2"]
    assert [version["version_number"] for version in versions_payload] == [1, 2]
    assert [version["version_count"] for version in versions_payload] == [2, 2]
    assert [version["agent_hash"] for version in versions_payload] == [
        "rich-agent-v1-hash",
        agent_hash,
    ]
    assert versions_payload[1]["id"] == submission_id
    assert versions_payload[0]["id"] != submission_id
    assert [version["family_id"] for version in versions_payload] == [
        "family-rich-public",
        "family-rich-public",
    ]
    assert [version["display_name"] for version in versions_payload] == [
        "Rich Agent",
        "Rich Agent",
    ]
    assert [version["is_latest_version"] for version in versions_payload] == [False, True]
    assert [version["latest_submission_id"] for version in versions_payload] == [
        submission_id,
        submission_id,
    ]
    assert versions_payload[1] == {
        **versions_payload[1],
        "id": submission_id,
        "name": "rich-agent",
        "agent_hash": agent_hash,
        "zip_sha256": "zip-rich-agent-hash",
        "status": "evaluating",
        "effective_status": "evaluating",
        "score": 0.82,
    }

    _assert_public_payload_is_redacted(
        {
            "count": count_response.json(),
            "list": list_payload,
            "detail": detail_payload,
            "status": status_payload,
            "evaluation": evaluation_payload,
            "versions": versions_payload,
        }
    )


async def test_platform_sdk_frontend_status_evaluation_and_events_are_public_safe(
    client,
    database_session,
):
    cases = (
        {
            "slug": "completed",
            "raw_status": "tb_completed",
            "public_state": "valid",
            "phase": "complete",
            "job_status": "completed",
            "attempt_status": "completed",
            "event_type": "task.completed",
            "task_status": "completed",
            "task_result_status": "passed",
            "score": 1.0,
            "passed_tasks": 1,
            "returncode": 0,
        },
        {
            "slug": "failed-final",
            "raw_status": "tb_failed_final",
            "public_state": "error",
            "phase": "failed",
            "job_status": "failed",
            "attempt_status": "failed",
            "event_type": "task.failed",
            "task_status": "failed",
            "task_result_status": "failed",
            "score": 0.0,
            "passed_tasks": 0,
            "returncode": 1,
        },
    )
    async with database_session() as session:
        created = []
        for case in cases:
            submission_id, agent_hash = await _create_platform_sdk_frontend_fixture(
                session,
                slug=case["slug"],
                raw_status=case["raw_status"],
                public_state=case["public_state"],
                phase=case["phase"],
                job_status=case["job_status"],
                attempt_status=case["attempt_status"],
                event_type=case["event_type"],
            )
            created.append((case, submission_id, agent_hash))
        await session.commit()

    public_payloads: dict[str, object] = {}
    for case, submission_id, agent_hash in created:
        list_response = await client.get("/submissions")
        detail_response = await client.get(f"/submissions/{submission_id}")
        versions_response = await client.get(f"/submissions/{submission_id}/versions")
        status_response = await client.get(f"/submissions/{submission_id}/status")
        evaluation_response = await client.get(f"/agents/{agent_hash}/evaluation")
        task_events_response = await client.get(
            f"/submissions/{submission_id}/task-events?limit=10"
        )
        task_stream_response = await client.get(f"/submissions/{submission_id}/task-events/stream")
        status_stream_response = await client.get(f"/submissions/{submission_id}/events")

        assert list_response.status_code == 200
        assert detail_response.status_code == 200
        assert versions_response.status_code == 200
        assert status_response.status_code == 200
        assert evaluation_response.status_code == 200
        assert task_events_response.status_code == 200
        assert task_stream_response.status_code == 200
        assert status_stream_response.status_code == 200

        list_payload = next(row for row in list_response.json() if row["id"] == submission_id)
        detail_payload = detail_response.json()
        versions_payload = versions_response.json()
        status_payload = status_response.json()
        evaluation_payload = evaluation_response.json()
        task_events_payload = task_events_response.json()
        task_stream_events = _parse_sse_events(task_stream_response.text)
        status_stream_events = _parse_sse_events(status_stream_response.text)
        expected_job_id = f"frontend-public-{case['slug']}-job"
        expected_task_id = f"safe-platform-sdk-{submission_id}"

        assert list_payload["status"] == case["public_state"]
        assert detail_payload["status"] == case["public_state"]
        assert detail_payload["effective_status"] == case["public_state"]
        assert versions_payload[0]["status"] == case["public_state"]
        assert status_payload["status"] == case["public_state"]
        assert status_payload["public_state"] == case["public_state"]
        assert status_payload["phase"] == case["phase"]
        assert status_payload["effective_status"] == case["public_state"]
        expected_task_phase = {
            "task_id": expected_task_id,
            "phase": case["task_status"],
            "status": case["task_status"],
            "updated_at": status_payload["evaluation"]["task_phases"][0]["updated_at"],
            "attempt": 1,
        }
        assert status_payload["evaluation"] == {
            **status_payload["evaluation"],
            "job_id": expected_job_id,
            "status": case["job_status"],
            "score": case["score"],
            "passed_tasks": case["passed_tasks"],
            "total_tasks": 1,
            "current_attempt": 1,
            "attempt_status": case["attempt_status"],
            "task_phases": [expected_task_phase],
        }
        assert evaluation_payload == {
            **evaluation_payload,
            "job_id": expected_job_id,
            "submission_id": submission_id,
            "agent_hash": agent_hash,
            "status": case["job_status"],
            "effective_status": case["public_state"],
            "score": case["score"],
            "passed_tasks": case["passed_tasks"],
            "total_tasks": 1,
            "verdict": "valid" if case["public_state"] == "valid" else "invalid",
            "rules_version": "rules-platform-sdk-public",
        }

        assert evaluation_payload["task_phases"] == [
            {
                **expected_task_phase,
                "updated_at": evaluation_payload["task_phases"][0]["updated_at"],
            }
        ]
        assert set(evaluation_payload["task_phases"][0]) == {
            "task_id",
            "phase",
            "status",
            "updated_at",
            "attempt",
        }
        assert evaluation_payload["tasks"] == [
            {
                "task_id": expected_task_id,
                "docker_image": "baseintelligence/public-runner:task8",
                "status": case["task_result_status"],
                "score": case["score"],
                "returncode": case["returncode"],
                "duration_seconds": 3.5,
                "failure_reason": None
                if case["task_result_status"] == "passed"
                else "[REDACTED_SECRET] [REDACTED_SECRET]",
                "detail_log": None
                if case["task_result_status"] == "passed"
                else (
                    f"Task: {expected_task_id}\n"
                    "Status: failed\n"
                    "Score: 0.0000\n"
                    "Return code: 1\n"
                    "Duration seconds: 3.500\n\n"
                    "Error log:\n"
                    "[REDACTED_SECRET] [REDACTED_SECRET]\n\n"
                    "Output log:\n"
                    "base [REDACTED_SECRET]"
                ),
            }
        ]
        terminal_task_events = [
            event
            for event in task_events_payload["events"]
            if event["event_type"] == case["event_type"]
        ]
        assert terminal_task_events == [
            {
                **terminal_task_events[0],
                "task_id": expected_task_id,
                "event_type": case["event_type"],
                "status": case["task_status"],
                "metadata": {"safe": case["public_state"]},
            }
        ]
        assert task_stream_events[-1]["data"]["metadata"] == {"safe": case["public_state"]}
        assert status_stream_events[-1]["data"]["status"] == case["public_state"]
        assert status_stream_events[-1]["data"]["phase"] == case["phase"]
        public_payloads[str(submission_id)] = {
            "list": list_payload,
            "detail": detail_payload,
            "versions": versions_payload,
            "status": status_payload,
            "evaluation": evaluation_payload,
            "task_events": task_events_payload,
            "task_stream": task_stream_events,
            "status_stream": status_stream_events,
        }

    _assert_platform_sdk_public_payload_is_redacted(public_payloads)


async def test_frontend_task_rows_include_queued_phase_result_and_redacted_selected_tasks(
    client,
    database_session,
):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-task-rows",
            name="task-rows-agent",
            agent_hash="task-rows-agent-hash",
            artifact_uri="/terminal-bench/jobs/platform-sdk-private/artifact-task8.zip",
            artifact_path="/terminal-bench/jobs/platform-sdk-private/artifact-task8.zip",
            status="tb_queued",
            raw_status="tb_queued",
            effective_status="evaluation queued",
        )
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id="task-rows-job",
            submission_id=submission.id,
            status="queued",
            selected_tasks_json=json.dumps(
                [
                    {
                        "task_id": "safe-task-alpha",
                        "docker_image": "ghcr.io/baseintelligence/private-runner:secret",
                        "benchmark": "terminal_bench",
                        "metadata": {
                            "provider": "platform_sdk",
                            "job_name": "k8s-job-task8-alpha",
                            "pod_name": "pod-task8-alpha",
                            "raw_ref": "raw-ref-task8-alpha",
                            "command": "platform-terminal-bench-command.sh",
                        },
                    },
                    "safe-task-beta",
                ]
            ),
            total_tasks=2,
            logs_ref="/terminal-bench/jobs/platform-sdk-private/logs.txt",
            lease_owner="worker-a",
            last_error="platform_sdk raw-ref-task8-alpha broker-token",
            created_at=NOW + timedelta(minutes=30),
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        await session.commit()
        submission_id = submission.id
        agent_hash = submission.agent_hash

    status_response = await client.get(f"/submissions/{submission_id}/status")
    evaluation_response = await client.get(f"/agents/{agent_hash}/evaluation")

    assert status_response.status_code == 200
    assert evaluation_response.status_code == 200
    status_rows = status_response.json()["evaluation"]["task_rows"]
    evaluation_rows = evaluation_response.json()["task_rows"]
    assert status_rows == evaluation_rows
    assert status_rows == [
        {
            "task_id": "safe-task-alpha",
            "display_name": "safe-task-alpha",
            "source": "terminal_bench",
            "phase": "assigned",
            "status": "assigned",
            "updated_at": status_rows[0]["updated_at"],
            "attempt": None,
            "has_result": False,
        },
        {
            "task_id": "safe-task-beta",
            "display_name": "safe-task-beta",
            "source": "benchmark",
            "phase": "assigned",
            "status": "assigned",
            "updated_at": status_rows[1]["updated_at"],
            "attempt": None,
            "has_result": False,
        },
    ]
    assert all(
        set(row)
        == {
            "task_id",
            "display_name",
            "source",
            "phase",
            "status",
            "updated_at",
            "attempt",
            "has_result",
        }
        for row in status_rows
    )
    _assert_platform_sdk_public_payload_is_redacted(
        {"status_task_rows": status_rows, "evaluation_task_rows": evaluation_rows}
    )

    async with database_session() as session:
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "task-rows-job")
        )
        assert job is not None
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            task_id="safe-task-alpha",
            event_type="task.status",
            status="running",
            message="safe public running phase",
            metadata={
                "phase": "running",
                "attempt": 2,
                "provider": "platform_sdk",
                "raw_ref": "raw-ref-task8-alpha",
                "worker": "worker-a",
            },
        )
        session.add(
            TaskResult(
                job_id=job.id,
                task_id="safe-task-beta",
                docker_image="ghcr.io/baseintelligence/private-runner:secret",
                status="passed",
                score=1.0,
                returncode=0,
                stdout="platform_sdk broker-token",
                stderr="pod-task8 raw-ref-task8-alpha",
                duration_seconds=4.0,
            )
        )
        await session.commit()

    status_response = await client.get(f"/submissions/{submission_id}/status")
    evaluation_response = await client.get(f"/agents/{agent_hash}/evaluation")

    assert status_response.status_code == 200
    assert evaluation_response.status_code == 200
    status_rows = status_response.json()["evaluation"]["task_rows"]
    evaluation_rows = evaluation_response.json()["task_rows"]
    assert status_rows == evaluation_rows
    assert status_rows == [
        {
            "task_id": "safe-task-alpha",
            "display_name": "safe-task-alpha",
            "source": "terminal_bench",
            "phase": "running",
            "status": "running",
            "updated_at": status_rows[0]["updated_at"],
            "attempt": 2,
            "has_result": False,
        },
        {
            "task_id": "safe-task-beta",
            "display_name": "safe-task-beta",
            "source": "benchmark",
            "phase": "completed",
            "status": "completed",
            "updated_at": status_rows[1]["updated_at"],
            "attempt": None,
            "has_result": True,
        },
    ]
    _assert_platform_sdk_public_payload_is_redacted(
        {"status_task_rows": status_rows, "evaluation_task_rows": evaluation_rows}
    )


async def test_platform_sdk_evaluation_exposes_running_task_phase_before_results(
    client,
    database_session,
):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-running-platform-sdk",
            name="running-platform-sdk",
            agent_hash="running-platform-sdk-hash",
            artifact_uri="/terminal-bench/jobs/platform-sdk-private/artifact-task8-running.zip",
            status="tb_running",
            raw_status="tb_running",
            effective_status="evaluating",
            artifact_path="/terminal-bench/jobs/platform-sdk-private/artifact-task8-running.zip",
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
                "tb_queued",
                "tb_running",
            )
        ):
            kwargs = {"from_status": None} if index == 0 else {}
            await transition_submission_status(
                session,
                submission,
                to_status,
                actor="evaluation" if to_status.startswith("tb_") else "worker",
                reason="evaluation_job_running" if to_status == "tb_running" else to_status,
                metadata={
                    "execution_provider": "platform_sdk",
                    "job_name": "k8s-job-task8-running",
                    "pod_name": "pod-task8-running",
                    "worker": "worker-a",
                },
                **kwargs,
            )
        job = EvaluationJob(
            job_id="frontend-public-running-job",
            submission_id=submission.id,
            status="running",
            selected_tasks_json=json.dumps(["safe-platform-sdk-running"]),
            total_tasks=1,
            logs_ref="/terminal-bench/jobs/platform-sdk-private/logs.txt",
            lease_owner="worker-a",
            last_error="platform_sdk raw-ref-task8-running",
            created_at=NOW + timedelta(minutes=20),
            started_at=NOW + timedelta(minutes=20, seconds=1),
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        await task_events.record_task_event(
            session,
            submission_id=submission.id,
            job_id=job.id,
            task_id="safe-platform-sdk-running",
            event_type="task.status",
            status="running",
            message="safe public running phase",
            metadata={
                "phase": "running",
                "attempt": 2,
                "provider": "platform_sdk",
                "raw_ref": "raw-ref-task8-running",
                "pod_name": "pod-task8-running",
            },
        )
        await session.commit()
        submission_id = submission.id
        agent_hash = submission.agent_hash

    status_response = await client.get(f"/submissions/{submission_id}/status")
    evaluation_response = await client.get(f"/agents/{agent_hash}/evaluation")

    assert status_response.status_code == 200
    assert evaluation_response.status_code == 200
    status_payload = status_response.json()
    evaluation_payload = evaluation_response.json()
    expected_phase = {
        "task_id": "safe-platform-sdk-running",
        "phase": "running",
        "status": "running",
        "updated_at": evaluation_payload["task_phases"][0]["updated_at"],
        "attempt": 2,
    }
    expected_task_row = {
        "task_id": "safe-platform-sdk-running",
        "display_name": "safe-platform-sdk-running",
        "source": "benchmark",
        "phase": "running",
        "status": "running",
        "updated_at": evaluation_payload["task_rows"][0]["updated_at"],
        "attempt": 2,
        "has_result": False,
    }
    assert status_payload["evaluation"]["task_phases"] == [expected_phase]
    assert status_payload["evaluation"]["task_rows"] == [
        {
            **expected_task_row,
            "updated_at": status_payload["evaluation"]["task_rows"][0]["updated_at"],
        }
    ]
    assert evaluation_payload["task_phases"] == [expected_phase]
    assert evaluation_payload["task_rows"] == [expected_task_row]
    assert evaluation_payload["tasks"] == []
    assert set(evaluation_payload["task_phases"][0]) == {
        "task_id",
        "phase",
        "status",
        "updated_at",
        "attempt",
    }
    _assert_platform_sdk_public_payload_is_redacted(
        {"status": status_payload, "evaluation": evaluation_payload}
    )


async def test_submissions_route_is_bounded_to_latest_100_newest_first(
    client,
    database_session,
):
    async with database_session() as session:
        for index in range(105):
            session.add(
                AgentSubmission(
                    miner_hotkey=f"miner-{index}",
                    name=f"agent-{index}",
                    agent_hash=f"hash-bounded-{index}",
                    artifact_uri=f"/tmp/artifact-{index}.zip",
                    status="received",
                    raw_status="received",
                    effective_status="received",
                    zip_sha256=f"zip-bounded-{index}",
                    submitted_at=NOW + timedelta(seconds=index),
                    created_at=NOW + timedelta(seconds=index),
                    signature="private-signature",
                    signature_nonce="private-nonce",
                    signature_payload_sha256="private-payload-hash",
                    signature_message="private canonical request",
                )
            )
        await session.commit()

    count_response = await client.get("/submissions/count")
    response = await client.get("/submissions")

    assert count_response.status_code == 200
    assert count_response.json() == {"count": 105}
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 100
    assert [row["agent_hash"] for row in rows[:3]] == [
        "hash-bounded-104",
        "hash-bounded-103",
        "hash-bounded-102",
    ]
    assert rows[-1]["agent_hash"] == "hash-bounded-5"
    assert "hash-bounded-4" not in {row["agent_hash"] for row in rows}


async def test_leaderboard_returns_best_scoring_row_per_hotkey(client, database_session):
    async with database_session() as session:
        await _create_scoring_submission(
            session,
            hotkey="miner-a",
            agent_hash="hash-miner-a-low",
            score=0.4,
            created_at=NOW,
        )
        await _create_scoring_submission(
            session,
            hotkey="miner-a",
            agent_hash="hash-miner-a-best",
            score=0.9,
            created_at=NOW + timedelta(seconds=1),
        )
        await _create_scoring_submission(
            session,
            hotkey="miner-b",
            agent_hash="hash-miner-b-overridden",
            score=0.7,
            effective_status="overridden_valid",
            verdict="invalid",
            created_at=NOW + timedelta(seconds=2),
        )
        await _create_scoring_submission(
            session,
            hotkey="miner-c",
            agent_hash="hash-miner-c-invalid",
            score=1.0,
            effective_status="overridden_invalid",
            created_at=NOW + timedelta(seconds=3),
        )
        await _create_scoring_submission(
            session,
            hotkey="miner-d",
            agent_hash="hash-miner-d-stale",
            score=0.99,
            raw_status="analysis_rejected",
            created_at=NOW + timedelta(seconds=4),
        )
        await session.commit()

    response = await client.get("/leaderboard")

    assert response.status_code == 200
    rows = response.json()
    assert rows == [
        {
            "miner_hotkey": "miner-a",
            "submission_id": rows[0]["submission_id"],
            "name": "agent-hash-miner-a-best",
            "agent_hash": "hash-miner-a-best",
            "display_name": "agent-hash-miner-a-best",
            "family_id": None,
            "version_number": None,
            "version_label": None,
            "version_count": None,
            "is_latest_version": False,
            "latest_submission_id": None,
            "score": 0.9,
            "passed_tasks": 2,
            "total_tasks": 3,
        },
        {
            "miner_hotkey": "miner-b",
            "submission_id": rows[1]["submission_id"],
            "name": "agent-hash-miner-b-overridden",
            "agent_hash": "hash-miner-b-overridden",
            "display_name": "agent-hash-miner-b-overridden",
            "family_id": None,
            "version_number": None,
            "version_label": None,
            "version_count": None,
            "is_latest_version": False,
            "latest_submission_id": None,
            "score": 0.7,
            "passed_tasks": 2,
            "total_tasks": 3,
        },
    ]


async def _create_platform_sdk_frontend_fixture(
    session,
    *,
    slug: str,
    raw_status: str,
    public_state: str,
    phase: str,
    job_status: str,
    attempt_status: str,
    event_type: str,
) -> tuple[int, str]:
    _ = phase
    submission = AgentSubmission(
        miner_hotkey=f"miner-frontend-platform-sdk-{slug}",
        name=f"frontend-platform-sdk-{slug}",
        agent_hash=f"frontend-platform-sdk-{slug}-hash",
        artifact_uri=(f"/terminal-bench/jobs/platform-sdk-private/artifact-task8-{slug}.zip"),
        status="received",
        raw_status="received",
        effective_status="received",
        zip_sha256=f"zip-frontend-platform-sdk-{slug}",
        zip_size_bytes=123,
        artifact_path=(f"/terminal-bench/jobs/platform-sdk-private/artifact-task8-{slug}.zip"),
        submitted_at=NOW + timedelta(minutes=10),
        created_at=NOW + timedelta(minutes=10),
        signature="platform-sdk-signature-secret",
        signature_nonce="platform-sdk-nonce-secret",
        signature_payload_sha256="platform-sdk-payload-secret",
        signature_message="platform_sdk hidden signature payload",
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
            raw_status,
        )
    ):
        kwargs = {"from_status": None} if index == 0 else {}
        await transition_submission_status(
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
                "command_path": f"{job_dir}/platform-terminal-bench-command.sh",
                "worker": "worker-a",
            },
            **kwargs,
        )

    job = EvaluationJob(
        job_id=f"frontend-public-{slug}-job",
        submission_id=submission.id,
        status=job_status,
        selected_tasks_json="[]",
        score=1.0 if public_state == "valid" else 0.0,
        passed_tasks=1 if public_state == "valid" else 0,
        total_tasks=1,
        verdict="valid" if public_state == "valid" else "invalid",
        rules_version="rules-platform-sdk-public",
        reason_codes_json=json.dumps([f"safe_{slug.replace('-', '_')}"]),
        error=f"platform_sdk broker-token-{slug}",
        logs_ref=f"{job_dir}/logs.txt",
        lease_owner="worker-a",
        last_error=f"raw-ref-task8-{slug} pod-task8-{slug}",
        created_at=NOW + timedelta(minutes=10, seconds=20),
        started_at=NOW + timedelta(minutes=10, seconds=21),
        finished_at=NOW + timedelta(minutes=10, seconds=30),
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    task_id = f"safe-platform-sdk-{submission.id}"
    session.add(
        TaskResult(
            job_id=job.id,
            task_id=task_id,
            docker_image="baseintelligence/public-runner:task8",
            status="passed" if public_state == "valid" else "failed",
            score=job.score,
            returncode=0 if public_state == "valid" else 1,
            stdout=f"platform_sdk broker-token-{slug}",
            stderr=f"pod-task8-{slug} raw-ref-task8-{slug}",
            duration_seconds=3.5,
        )
    )
    attempt = EvaluationAttempt(
        submission_id=submission.id,
        job_id=job.id,
        attempt_number=1,
        evaluator_name="terminal_bench",
        status=attempt_status,
        score=job.score,
        error=f"platform_sdk failed at k8s-job-task8-{slug}",
        metadata_json=json.dumps(
            {
                "execution_provider": "platform_sdk",
                "provider": "platform_sdk",
                "job_dir": job_dir,
                "job_name": f"k8s-job-task8-{slug}",
                "pod_name": f"pod-task8-{slug}",
                "raw_ref": f"raw-ref-task8-{slug}",
                "broker_ref": f"broker-token-{slug}",
                "command_path": f"{job_dir}/platform-terminal-bench-command.sh",
                "worker": "worker-a",
            }
        ),
        lease_owner="worker-a",
        started_at=NOW + timedelta(minutes=10, seconds=22),
        finished_at=NOW + timedelta(minutes=10, seconds=30),
    )
    session.add(attempt)
    await session.flush()
    trial = TerminalBenchTrial(
        evaluation_attempt_id=attempt.id,
        task_id=task_id,
        trial_name=f"safe-trial-{slug}",
        trial_number=1,
        job_dir=job_dir,
        job_name=f"k8s-job-task8-{slug}",
        status="completed" if public_state == "valid" else "failed",
        score=job.score,
        is_final=1,
        raw_artifacts_json=json.dumps(
            {
                "provider": "platform_sdk",
                "raw_ref": f"raw-ref-task8-{slug}",
                "broker_ref": f"broker-token-{slug}",
                "command_path": "platform-terminal-bench-command.sh",
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
            status=attempt_status,
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
        task_id=task_id,
        event_type="task.status",
        status="running",
        message=f"safe public platform SDK phase {slug}",
        metadata={
            "phase": "running",
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
        task_id=task_id,
        event_type="task.status",
        status="completed" if public_state == "valid" else "failed",
        message=f"safe public platform SDK latest phase {slug}",
        metadata={
            "phase": "completed" if public_state == "valid" else "failed",
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
        task_id=task_id,
        event_type=event_type,
        status="completed" if public_state == "valid" else "failed",
        message=f"safe public platform SDK event {slug}",
        metadata={
            "safe": public_state,
            "execution_provider": "platform_sdk",
            "provider": "platform_sdk",
            "job_dir": job_dir,
            "job_name": f"k8s-job-task8-{slug}",
            "kubernetes_job_name": f"k8s-job-task8-{slug}",
            "pod_name": f"pod-task8-{slug}",
            "raw_ref": f"raw-ref-task8-{slug}",
            "broker_ref": f"broker-token-{slug}",
            "command_path": f"{job_dir}/platform-terminal-bench-command.sh",
            "worker": "worker-a",
        },
    )
    await session.flush()
    return submission.id, submission.agent_hash


async def _create_rich_frontend_fixture(session) -> tuple[int, str]:
    family = SubmissionFamily(
        public_family_id="family-rich-public",
        owner_hotkey="miner-rich",
        display_name="Rich Agent",
        normalized_name=normalize_submission_name("Rich Agent"),
        version_count=2,
    )
    session.add(family)
    await session.flush()

    previous_submission = AgentSubmission(
        miner_hotkey="miner-rich",
        name="rich-agent",
        agent_hash="rich-agent-v1-hash",
        artifact_uri="/tmp/private-job-dir/rich-agent-v1.zip",
        submission_family_id=family.id,
        version_number=1,
        version_label="v1",
        canonical_artifact_hash="zip-rich-agent-v1-hash",
        is_latest_version=False,
        status="tb_completed",
        raw_status="tb_completed",
        effective_status="valid",
        zip_sha256="zip-rich-agent-v1-hash",
        zip_size_bytes=100,
        artifact_path="/tmp/private-job-dir/rich-agent-v1.zip",
        submitted_at=NOW - timedelta(minutes=5),
        created_at=NOW - timedelta(minutes=5),
        signature="previous-signature-secret",
        signature_nonce="previous-nonce-secret",
        signature_payload_sha256="previous-payload-secret",
        signature_message="def secret_source(): return 'v1'",
    )
    session.add(previous_submission)
    await session.flush()

    submission = AgentSubmission(
        miner_hotkey="miner-rich",
        name="rich-agent",
        agent_hash="rich-agent-hash",
        artifact_uri="/tmp/private-job-dir/rich-agent.zip",
        submission_family_id=family.id,
        version_number=2,
        version_label="v2",
        canonical_artifact_hash="zip-rich-agent-hash",
        is_latest_version=True,
        status="received",
        raw_status="received",
        effective_status="received",
        zip_sha256="zip-rich-agent-hash",
        zip_size_bytes=123,
        artifact_path="/tmp/private-job-dir/rich-agent.zip",
        submitted_at=NOW,
        created_at=NOW,
        signature="signature-secret",
        signature_nonce="nonce-secret",
        signature_timestamp=NOW.isoformat(),
        signature_payload_sha256="payload-secret",
        signature_message="def secret_source(): return 'hidden'",
    )
    session.add(submission)
    await session.flush()
    family.latest_submission_id = submission.id

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

    job = EvaluationJob(
        job_id="job-rich",
        submission_id=submission.id,
        status="running",
        selected_tasks_json=json.dumps(["task-alpha", "task-beta"]),
        score=0.82,
        passed_tasks=2,
        total_tasks=3,
        verdict="valid",
        rules_version="rules-v1",
        reason_codes_json=json.dumps(["job_safe_reason"]),
        error="provider failure included sk-test-secret",
        logs_ref="broker-ref-secret/logs.txt",
        lease_owner="lease-worker-secret",
        last_error="Bearer raw-provider-token",
        created_at=NOW + timedelta(seconds=20),
        started_at=NOW + timedelta(seconds=21),
        finished_at=None,
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id

    analysis = AnalysisRun(
        submission_id=submission.id,
        job_id=job.id,
        analyzer_name="static",
        analyzer_version="v1",
        status="completed",
        verdict="allow",
        reason_codes_json=json.dumps(["safe_reason"]),
        report_json=json.dumps({"raw_source": "def secret_source(): pass"}),
        logs_ref="/tmp/private-job-dir/analyzer.log",
        lease_owner="lease-worker-secret",
        started_at=NOW + timedelta(seconds=10),
        finished_at=NOW + timedelta(seconds=11),
    )
    session.add(analysis)
    await session.flush()
    session.add_all(
        [
            LlmVerdict(
                analysis_run_id=analysis.id,
                reviewer_name="reviewer",
                model_name="model",
                verdict="allow",
                confidence=0.91,
                reason_codes_json=json.dumps(["llm_safe_reason"]),
                prompt_ref="/tmp/private-job-dir/prompt.txt",
                raw_request_json=json.dumps({"Authorization": "Bearer raw-provider-token"}),
                raw_response_json=json.dumps(
                    {
                        "content": "sk-test-secret",
                        "verdict_json": {
                            "verdict": "allow",
                            "confidence": 0.91,
                            "rationale": (
                                "No policy issue found for /tmp/private-job-dir/prompt.txt "
                                "using sk-test-secret."
                            ),
                        },
                    }
                ),
            ),
            SimilarityMatch(
                analysis_run_id=analysis.id,
                source_submission_id=submission.id,
                matched_submission_id=999,
                matched_artifact_uri="/tmp/private-job-dir/matched.zip",
                match_kind="python_ast_similarity",
                score=92.5,
                evidence_json=json.dumps(
                    {
                        "risk_band": "high",
                        "algorithm_version": "sim-v1",
                        "matched_code": "def secret_source(): pass",
                        "top_file_pairs": [
                            {
                                "source_file_path": "agent.py",
                                "matched_file_path": "public-match.py",
                                "score_percent": 92.5,
                                "matched_code": "def matched_code_secret(): pass",
                            },
                            {
                                "source_file_path": "/tmp/private-job-dir/source.py",
                                "matched_file_path": "/root/private-match.py",
                                "score_percent": 90.0,
                            },
                        ],
                    }
                ),
            ),
            PythonAstFeature(
                analysis_run_id=analysis.id,
                file_path="agent.py",
                feature_key="call:open",
                feature_type="call",
                feature_value="ast-feature-secret",
            ),
            PythonAstFeature(
                analysis_run_id=analysis.id,
                file_path="agent.py",
                feature_key="function:run",
                feature_type="function",
                feature_value="def secret_source(): pass",
            ),
            TaskResult(
                job_id=job.id,
                task_id="task-alpha",
                docker_image="baseintelligence/swe-forge:task-alpha",
                status="passed",
                score=1.0,
                returncode=0,
                stdout="stdout with sk-test-secret and def secret_source",
                stderr="",
                duration_seconds=12.5,
            ),
            TaskResult(
                job_id=job.id,
                task_id="task-beta",
                docker_image="baseintelligence/swe-forge:task-beta",
                status="failed",
                score=0.0,
                returncode=1,
                stdout="",
                stderr="stderr with Bearer raw-provider-token and broker-ref-secret",
                duration_seconds=8.25,
            ),
        ]
    )
    attempt = EvaluationAttempt(
        submission_id=submission.id,
        job_id=job.id,
        attempt_number=2,
        evaluator_name="terminal_bench",
        status="running",
        score=None,
        error="broker-ref-secret",
        metadata_json=json.dumps({"job_dir": "/tmp/private-job-dir"}),
        lease_owner="lease-worker-secret",
        started_at=NOW + timedelta(seconds=22),
    )
    session.add(attempt)
    await session.flush()
    session.add_all(
        [
            TerminalBenchTrial(
                evaluation_attempt_id=attempt.id,
                task_id="task-alpha",
                trial_name="trial-alpha",
                trial_number=1,
                job_dir="/tmp/private-job-dir",
                job_name="tb-rich-alpha",
                status="completed",
                score=1.0,
                is_final=1,
                raw_artifacts_json=json.dumps(
                    {"stdout": "sk-test-secret", "broker_ref": "broker-ref-secret"}
                ),
                lease_owner="lease-worker-secret",
                stdout_ref="/tmp/private-job-dir/stdout.log",
                stderr_ref="/tmp/private-job-dir/stderr.log",
            ),
            TerminalBenchTrial(
                evaluation_attempt_id=attempt.id,
                task_id="task-beta",
                trial_name="trial-beta",
                trial_number=2,
                job_dir="/tmp/private-job-dir",
                job_name="tb-rich-beta",
                status="errored",
                score=None,
                is_final=0,
                raw_artifacts_json=json.dumps({"stderr": "Bearer raw-provider-token"}),
                lease_owner="lease-worker-secret",
            ),
        ]
    )
    await session.flush()
    return submission.id, submission.agent_hash


async def _create_scoring_submission(
    session,
    *,
    hotkey: str,
    agent_hash: str,
    score: float,
    created_at: datetime,
    raw_status: str = "tb_completed",
    effective_status: str = "valid",
    job_status: str = "completed",
    verdict: str | None = "valid",
) -> int:
    submission = AgentSubmission(
        miner_hotkey=hotkey,
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status=raw_status,
        raw_status=raw_status,
        effective_status=effective_status,
        zip_sha256=f"zip-{agent_hash}",
        zip_size_bytes=123,
        artifact_path=f"/tmp/{agent_hash}.zip",
        submitted_at=created_at,
        created_at=created_at,
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status=job_status,
        selected_tasks_json="[]",
        score=score,
        passed_tasks=2,
        total_tasks=3,
        verdict=verdict,
        rules_version="rules-v1",
        created_at=created_at,
        started_at=created_at + timedelta(seconds=1),
        finished_at=created_at + timedelta(seconds=2),
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    return submission.id


def _assert_public_payload_is_redacted(payload: object) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_PUBLIC_STRINGS:
        assert forbidden not in serialized
    for forbidden_field in (
        "signature",
        "signature_nonce",
        "signature_payload_sha256",
        "signature_message",
        "raw_status",
        "submission_family_id",
        "normalized_name",
        "canonical_artifact_hash",
        "artifact_path",
        "artifact_uri",
        "stdout",
        "stderr",
        "logs_ref",
        "raw_artifacts_json",
        "lease_owner",
        "raw_response_json",
        "raw_request_json",
        "evidence_json",
        "matched_artifact_uri",
        "feature_value",
    ):
        assert forbidden_field not in serialized


async def test_frontend_dualflag_status_task_rows_from_evalrun_plan(
    client,
    database_session,
    monkeypatch,
):
    """Frontend contract: dual-flag evaluation.task_rows from EvalRun plan (N=3)."""

    import hashlib
    from datetime import timedelta as _td

    import agent_challenge.api.routes as api_routes
    from agent_challenge.canonical import eval_wire as ew
    from agent_challenge.evaluation.plan_scoring import (
        build_score_record_from_eval_plan,
        canonical_eval_plan_json,
    )
    from agent_challenge.models import EvalRun

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    eval_run_id = "eval-fe-df-rows"
    task_ids = [f"fe-task-{i:03d}" for i in range(3)]
    plan = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": f"submission-{eval_run_id}",
        "submission_version": 1,
        "authorizing_review_digest": "b" * 64,
        "agent_hash": "a" * 64,
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry.example/task@sha256:" + "3" * 64,
                "task_config_sha256": "4" * 64,
            }
            for task_id in task_ids
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "5" * 64,
            "compose_hash": "6" * 64,
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "6" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("6" * 64)).hexdigest(),
            "measurement": {
                "mrtd": "1" * 96,
                "rtmr0": "2" * 96,
                "rtmr1": "3" * 96,
                "rtmr2": "4" * 96,
                "os_image_hash": "5" * 64,
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": f"key-nonce-{eval_run_id}",
        "score_nonce": f"score-nonce-{eval_run_id}",
        "run_token_sha256": "7" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    plan_json = canonical_eval_plan_json(plan)
    trials = {task_id: [1.0 if i == 0 else 0.0] for i, task_id in enumerate(task_ids)}
    record = build_score_record_from_eval_plan(plan, trials)
    score_json = ew.canonical_json_v1(record).decode("utf-8")

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-fe-df-rows",
            name="fe-df-rows-agent",
            agent_hash=hashlib.sha256(eval_run_id.encode()).hexdigest(),
            artifact_uri=f"/tmp/{eval_run_id}.zip",
            status="queued",
            raw_status="queued",
            effective_status="queued",
            submitted_at=NOW,
            created_at=NOW,
        )
        session.add(submission)
        await session.flush()
        run = EvalRun(
            eval_run_id=eval_run_id,
            submission_id=submission.id,
            submission_version=1,
            authorizing_review_digest="b" * 64,
            plan_json=plan_json,
            plan_sha256=hashlib.sha256(plan_json.encode()).hexdigest(),
            token_sha256=hashlib.sha256(f"token-{eval_run_id}".encode()).hexdigest(),
            phase="eval_expired",
            verified=False,
            reward_eligible=False,
            result_available=True,
            score=ew.decode_score_f64be(record["final"]["job_score_f64be"]),
            passed_tasks=record["final"]["passed_tasks"],
            total_tasks=record["final"]["total_tasks"],
            canonical_score_record_json=score_json,
            canonical_score_record_sha256=ew.score_record_digest(record),
            issued_at=NOW,
            expires_at=NOW + _td(days=3650),
            created_at=NOW,
            updated_at=NOW,
        )
        session.add(run)
        await session.commit()
        submission_id = submission.id

    status_response = await client.get(f"/submissions/{submission_id}/status")
    assert status_response.status_code == 200
    payload = status_response.json()
    evaluation = payload["evaluation"]
    assert evaluation is not None
    assert evaluation["status"] == "eval_expired"
    assert evaluation["job_id"] == eval_run_id
    assert evaluation["total_tasks"] == 3
    assert evaluation["passed_tasks"] == 1
    assert evaluation["score"] == ew.decode_score_f64be(record["final"]["job_score_f64be"])
    rows = evaluation["task_rows"]
    assert len(rows) == 3
    assert [row["task_id"] for row in rows] == task_ids
    assert rows[0]["has_result"] is True
    assert rows[0]["phase"] == "completed"
    assert rows[1]["has_result"] is True
    assert rows[1]["phase"] == "failed"
    assert rows[2]["has_result"] is True
    assert rows[2]["phase"] == "failed"
    # Dual-flag status must stay free of private provider/lease material.
    _assert_public_payload_is_redacted(payload)
    _assert_platform_sdk_public_payload_is_redacted(payload)


async def test_frontend_dualflag_eval_prepared_task_rows_without_score_record(
    client,
    database_session,
    monkeypatch,
):
    """Frontend contract: prepared dual-flag run still surfaces planned rows."""

    import hashlib
    from datetime import timedelta as _td

    import agent_challenge.api.routes as api_routes
    from agent_challenge.canonical import eval_wire as ew
    from agent_challenge.evaluation.plan_scoring import canonical_eval_plan_json
    from agent_challenge.models import EvalRun

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    eval_run_id = "eval-fe-df-prepared"
    task_ids = [f"fe-prep-{i:03d}" for i in range(3)]
    plan = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": f"submission-{eval_run_id}",
        "submission_version": 1,
        "authorizing_review_digest": "b" * 64,
        "agent_hash": "a" * 64,
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry.example/task@sha256:" + "3" * 64,
                "task_config_sha256": "4" * 64,
            }
            for task_id in task_ids
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "5" * 64,
            "compose_hash": "6" * 64,
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "6" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("6" * 64)).hexdigest(),
            "measurement": {
                "mrtd": "1" * 96,
                "rtmr0": "2" * 96,
                "rtmr1": "3" * 96,
                "rtmr2": "4" * 96,
                "os_image_hash": "5" * 64,
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": f"key-nonce-{eval_run_id}",
        "score_nonce": f"score-nonce-{eval_run_id}",
        "run_token_sha256": "7" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    plan_json = canonical_eval_plan_json(plan)

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-fe-df-prepared",
            name="fe-df-prepared-agent",
            agent_hash=hashlib.sha256(eval_run_id.encode()).hexdigest(),
            artifact_uri=f"/tmp/{eval_run_id}.zip",
            status="queued",
            raw_status="queued",
            effective_status="queued",
            submitted_at=NOW,
            created_at=NOW,
        )
        session.add(submission)
        await session.flush()
        run = EvalRun(
            eval_run_id=eval_run_id,
            submission_id=submission.id,
            submission_version=1,
            authorizing_review_digest="b" * 64,
            plan_json=plan_json,
            plan_sha256=hashlib.sha256(plan_json.encode()).hexdigest(),
            token_sha256=hashlib.sha256(f"token-{eval_run_id}".encode()).hexdigest(),
            phase="eval_prepared",
            verified=False,
            reward_eligible=False,
            result_available=False,
            score=None,
            passed_tasks=None,
            total_tasks=None,
            canonical_score_record_json=None,
            issued_at=NOW,
            expires_at=NOW + _td(days=3650),
            created_at=NOW,
            updated_at=NOW,
        )
        session.add(run)
        await session.commit()
        submission_id = submission.id

    status_response = await client.get(f"/submissions/{submission_id}/status")
    assert status_response.status_code == 200
    evaluation = status_response.json()["evaluation"]
    assert evaluation["status"] == "eval_prepared"
    assert evaluation["score"] == 0.0
    assert evaluation["passed_tasks"] == 0
    assert evaluation["total_tasks"] == 0
    rows = evaluation["task_rows"]
    assert len(rows) == 3
    assert [row["task_id"] for row in rows] == task_ids
    assert all(row["has_result"] is False for row in rows)
    assert all(row["phase"] == "assigned" for row in rows)
    _assert_public_payload_is_redacted(status_response.json())
