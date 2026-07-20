from __future__ import annotations

import json
from datetime import UTC, datetime

from cryptography.fernet import Fernet

from agent_challenge.api.routes import _task_event_replay_item
from agent_challenge.evaluation import task_events
from agent_challenge.models import AgentSubmission, SubmissionEnvVar, TaskLogEvent
from agent_challenge.sdk.config import DEFAULT_SECRET_REDACTION, ChallengeSettings


def test_submission_env_key_file_is_redacted_from_safe_config(tmp_path):
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")

    settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))
    safe = settings.safe_model_dump()

    assert safe["submission_env_encryption_key_file"] == DEFAULT_SECRET_REDACTION
    assert str(key_file) not in str(safe)
    assert str(key_file) not in repr(settings)


async def test_submission_env_storage_is_absent_from_public_status(
    client,
    database_session,
    tmp_path,
):
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))
    plaintext = "generated-sensitive-sentinel-value"

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-env-redaction",
            name="env-redaction-agent",
            agent_hash="env-redaction-agent-hash",
            artifact_uri="/tmp/env-redaction-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
        )
        session.add(submission)
        await session.flush()
        env_var = SubmissionEnvVar.encrypted(
            submission_id=submission.id,
            key="AGENT_API_KEY",
            value=plaintext,
            settings=settings,
        )
        session.add(env_var)
        await session.commit()
        submission_id = submission.id
        ciphertext = env_var.value_ciphertext
        value_hash = env_var.value_sha256

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    serialized = json.dumps(response.json(), sort_keys=True)
    for forbidden in (
        plaintext,
        ciphertext,
        value_hash,
        str(key_file),
        "AGENT_API_KEY",
        "value_ciphertext",
        "value_sha256",
        "submission_env_encryption_key_file",
    ):
        assert forbidden not in serialized


def test_task_event_metadata_redacts_env_keys_and_values():
    event = TaskLogEvent(
        id=1,
        submission_id=1,
        sequence=1,
        event_type="task.running",
        message="running",
        truncated=False,
        cap_reached=False,
        created_at=datetime.now(UTC),
        metadata_json=json.dumps(
            {
                "env": {"AGENT_API_KEY": "plain-secret-value"},
                "environment": {"OPENAI_API_KEY": "plain-openai-value"},
                "harbor_forward_env_vars": ["AGENT_API_KEY"],
                "safe": "visible",
            }
        ),
    )

    payload = _task_event_replay_item(event).model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["metadata"] == {"safe": "visible"}
    for forbidden in (
        "plain-secret-value",
        "plain-openai-value",
        "AGENT_API_KEY",
        "OPENAI_API_KEY",
        "harbor_forward_env_vars",
        "environment",
    ):
        assert forbidden not in serialized


async def test_raw_runtime_env_sentinel_is_redacted_from_public_task_events(
    client,
    database_session,
):
    sentinel_value = "task7-runtime-public-redaction-sentinel"
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-env-runtime-redaction",
            name="env-runtime-redaction-agent",
            agent_hash="env-runtime-redaction-agent-hash",
            artifact_uri="/tmp/env-runtime-redaction-agent.zip",
            status="tb_running",
            raw_status="tb_running",
            effective_status="evaluating",
        )
        session.add(submission)
        await session.flush()
        await task_events.record_task_event(
            session,
            submission_id=submission.id,
            event_type="task.log",
            stream="stderr",
            message=f"runtime reported API_KEY={sentinel_value}",
            metadata={"env": {"TASK7_SENTINEL": sentinel_value}, "safe": "visible"},
        )
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=10")

    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload, sort_keys=True)
    assert sentinel_value not in serialized
    assert "TASK7_SENTINEL" not in serialized
    assert "API_KEY=[REDACTED]" in serialized
    assert payload["events"][0]["metadata"] == {"safe": "visible"}
