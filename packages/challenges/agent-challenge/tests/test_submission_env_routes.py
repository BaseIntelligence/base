from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import AgentSubmission, EvaluationJob, SubmissionEnvVar
from agent_challenge.security import SignedRequestAuth

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


@dataclass
class MinerAuthState:
    hotkey: str = "miner-env-owner"
    calls: int = 0


@pytest.fixture
def miner_auth_override() -> MinerAuthState:
    state = MinerAuthState()

    async def authenticate() -> SignedRequestAuth:
        state.calls += 1
        return SignedRequestAuth(
            hotkey=state.hotkey,
            signature=f"miner-signature-{state.calls}",
            nonce=f"miner-nonce-{state.calls}",
            timestamp=NOW.isoformat(),
            body_sha256=hashlib.sha256(f"miner-body-{state.calls}".encode()).hexdigest(),
            canonical_request=f"miner-request-{state.calls}",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield state
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


@pytest.fixture
def env_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.submission_env_encryption_key_file",
        str(key_file),
    )
    return key_file


async def create_waiting_submission(database_session, *, hotkey: str = "miner-env-owner") -> int:
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=hotkey,
            name="env-agent",
            agent_hash=f"env-agent-hash-{hotkey}",
            artifact_uri="/tmp/env-agent.zip",
            status="Waiting environments",
            raw_status="waiting_miner_env",
            effective_status="Waiting environments",
        )
        session.add(submission)
        await session.commit()
        return submission.id


async def test_miner_env_put_locks_and_returns_redacted_metadata(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
    monkeypatch,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id = await create_waiting_submission(database_session)
    first_value = "first-sensitive-value"
    second_value = "second-sensitive-value"

    response = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": first_value, "OPENROUTER_API_KEY": second_value}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["submission_id"] == submission_id
    assert payload["keys"] == ["API_TOKEN", "OPENROUTER_API_KEY"]
    assert payload["count"] == 2
    assert payload["locked"] is True
    assert payload["env_confirmed_empty"] is False
    serialized = json.dumps(payload, sort_keys=True)
    assert first_value not in serialized
    assert second_value not in serialized
    assert str(env_key_file) not in serialized

    # The env is locked on submit, so a later replace is rejected without echo.
    replacement_value = "replacement-sensitive-value"
    replacement = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"EVAL_RUN_TOKEN": replacement_value}},
    )
    redacted = await client.get(f"/submissions/{submission_id}/env")

    assert replacement.status_code == 409
    assert replacement_value not in json.dumps(replacement.json(), sort_keys=True)
    assert redacted.status_code == 200
    redacted_payload = redacted.json()
    assert redacted_payload["keys"] == ["API_TOKEN", "OPENROUTER_API_KEY"]
    assert redacted_payload["count"] == 2
    redacted_serialized = json.dumps(redacted_payload, sort_keys=True)
    assert replacement_value not in redacted_serialized
    assert first_value not in redacted_serialized
    assert second_value not in redacted_serialized
    assert "value_ciphertext" not in redacted_serialized
    assert "value_sha256" not in redacted_serialized

    async with database_session() as session:
        env_vars = (await session.execute(select(SubmissionEnvVar))).scalars().all()

    assert [env_var.key for env_var in env_vars] == ["API_TOKEN", "OPENROUTER_API_KEY"]
    assert env_vars[0].value_ciphertext != first_value
    assert env_vars[0].value_sha256 == hashlib.sha256(first_value.encode()).hexdigest()


async def test_put_env_on_waiting_submission_enqueues_once_and_locks_env(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
    monkeypatch,
):
    _ = env_key_file, miner_auth_override
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id = await create_waiting_submission(database_session)

    response = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": "put-sensitive-value"}},
    )
    duplicate = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": "duplicate-sensitive-value"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["locked"] is True
    assert payload["env_confirmed_empty"] is False
    assert duplicate.status_code == 409
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        env_var = await session.scalar(select(SubmissionEnvVar))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.env_locked_at is not None
    assert submission.latest_evaluation_job_id is not None
    assert job_count == 1
    assert env_var is not None
    assert env_var.locked_at is not None


async def test_wrong_hotkey_cannot_read_or_update_env(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
):
    _ = env_key_file
    submission_id = await create_waiting_submission(database_session)
    miner_auth_override.hotkey = "different-miner"

    update = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": "blocked-sensitive-value"}},
    )
    read = await client.get(f"/submissions/{submission_id}/env")

    assert update.status_code == 403
    assert update.json() == {"detail": "forbidden"}
    assert read.status_code == 403
    async with database_session() as session:
        count = await session.scalar(select(func.count(SubmissionEnvVar.id)))
    assert count == 0


async def test_invalid_key_is_rejected_without_value_echo(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
):
    _ = env_key_file, miner_auth_override
    submission_id = await create_waiting_submission(database_session)
    secret_value = "invalid-key-sensitive-value"

    response = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"1INVALID": secret_value}},
    )

    assert response.status_code == 422
    serialized = json.dumps(response.json(), sort_keys=True)
    assert secret_value not in serialized
    assert "1INVALID" not in serialized
    async with database_session() as session:
        count = await session.scalar(select(func.count(SubmissionEnvVar.id)))
    assert count == 0


async def test_locked_update_after_launch_returns_conflict_without_value_echo(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
    monkeypatch,
):
    _ = env_key_file, miner_auth_override
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id = await create_waiting_submission(database_session)
    put = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": "launch-sensitive-value"}},
    )
    launch = await client.post(f"/submissions/{submission_id}/launch")
    locked_value = "locked-update-sensitive-value"

    locked = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": locked_value}},
    )

    assert put.status_code == 200
    assert launch.status_code == 200
    assert launch.json()["status"] == "tb_queued"
    assert locked.status_code == 409
    assert locked_value not in json.dumps(locked.json(), sort_keys=True)
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        env_vars = (await session.execute(select(SubmissionEnvVar))).scalars().all()
    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.env_locked_at is not None
    assert env_vars[0].locked_at is not None


async def test_confirm_empty_records_confirmation_and_launches(
    client,
    database_session,
    miner_auth_override,
    monkeypatch,
):
    _ = miner_auth_override
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id = await create_waiting_submission(database_session)

    confirmation = await client.post(f"/submissions/{submission_id}/env/confirm-empty")
    launch = await client.post(f"/submissions/{submission_id}/launch")

    assert confirmation.status_code == 200
    confirmation_payload = confirmation.json()
    assert confirmation_payload["count"] == 0
    assert confirmation_payload["keys"] == []
    assert confirmation_payload["env_confirmed_empty"] is True
    assert confirmation_payload["env_confirmed_empty_at"] is not None
    assert confirmation_payload["locked"] is True
    assert launch.status_code == 200
    launch_payload = launch.json()
    assert launch_payload["status"] == "tb_queued"
    assert launch_payload["effective_status"] == "evaluation queued"
    assert launch_payload["job_id"] is not None
    assert launch_payload["env"]["locked"] is True
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job = await session.scalar(select(EvaluationJob))
    assert submission is not None
    assert submission.env_confirmed_empty is True
    assert submission.env_locked_at is not None
    assert job is not None
    assert job.triggered_by_hotkey == "miner-env-owner"
    assert job.trigger_reason == "miner_env_confirm_empty"


async def test_confirm_empty_on_waiting_submission_enqueues_once_and_locks_env(
    client,
    database_session,
    miner_auth_override,
    monkeypatch,
):
    _ = miner_auth_override
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id = await create_waiting_submission(database_session)

    confirmation = await client.post(f"/submissions/{submission_id}/env/confirm-empty")
    duplicate = await client.post(f"/submissions/{submission_id}/env/confirm-empty")
    launch = await client.post(f"/submissions/{submission_id}/launch")

    assert confirmation.status_code == 200
    payload = confirmation.json()
    assert payload["locked"] is True
    assert payload["env_confirmed_empty"] is True
    assert duplicate.status_code == 409
    assert launch.status_code == 200
    assert launch.json()["job_id"] is not None
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.env_confirmed_empty is True
    assert submission.env_locked_at is not None
    assert submission.latest_evaluation_job_id is not None
    assert job_count == 1


async def test_launch_requires_env_or_empty_confirmation(
    client,
    database_session,
    miner_auth_override,
):
    _ = miner_auth_override
    submission_id = await create_waiting_submission(database_session)

    response = await client.post(f"/submissions/{submission_id}/launch")

    assert response.status_code == 409
    assert response.json() == {"detail": "submission env confirmation is required"}
    async with database_session() as session:
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        submission = await session.get(AgentSubmission, submission_id)
    assert job_count == 0
    assert submission is not None
    assert submission.raw_status == "waiting_miner_env"


async def test_internal_launch_bridge_is_removed_and_creates_no_job(
    client,
    database_session,
    internal_headers,
):
    """The centralized internal launch bridge no longer exists (404)."""
    submission_id = await create_waiting_submission(database_session)

    response = await client.post(
        f"/internal/v1/submissions/{submission_id}/launch",
        headers=internal_headers,
    )

    assert response.status_code == 404
    async with database_session() as session:
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
    assert job_count == 0


async def test_env_limits_are_enforced_without_value_echo(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
):
    _ = env_key_file, miner_auth_override
    submission_id = await create_waiting_submission(database_session)
    oversize_value = "x" * (16 * 1024 + 1)

    response = await client.put(
        f"/submissions/{submission_id}/env",
        json={"env": {"API_TOKEN": oversize_value}},
    )

    assert response.status_code == 422
    serialized = json.dumps(response.json(), sort_keys=True)
    assert oversize_value not in serialized
    assert "API_TOKEN" not in serialized


async def test_expected_env_route_validation_states_never_return_service_unavailable(
    client,
    database_session,
    env_key_file,
    miner_auth_override,
):
    _ = env_key_file
    wrong_owner_submission = await create_waiting_submission(
        database_session, hotkey="env-owner-one"
    )
    miner_auth_override.hotkey = "different-miner"
    wrong_owner = await client.put(
        f"/submissions/{wrong_owner_submission}/env",
        json={"env": {"API_TOKEN": "wrong-owner-sensitive-value"}},
    )

    miner_auth_override.hotkey = "env-owner-two"
    invalid_key_submission = await create_waiting_submission(
        database_session, hotkey="env-owner-two"
    )
    invalid_key = await client.put(
        f"/submissions/{invalid_key_submission}/env",
        json={"env": {"1INVALID": "invalid-key-sensitive-value"}},
    )

    miner_auth_override.hotkey = "env-owner-three"
    missing_confirmation_submission = await create_waiting_submission(
        database_session, hotkey="env-owner-three"
    )
    missing_confirmation = await client.post(
        f"/submissions/{missing_confirmation_submission}/launch"
    )

    miner_auth_override.hotkey = "env-owner-four"
    locked_submission = await create_waiting_submission(database_session, hotkey="env-owner-four")
    async with database_session() as session:
        submission = await session.get(AgentSubmission, locked_submission)
        assert submission is not None
        submission.env_locked_at = NOW
        await session.commit()
    locked_update = await client.put(
        f"/submissions/{locked_submission}/env",
        json={"env": {"API_TOKEN": "locked-sensitive-value"}},
    )

    responses = [wrong_owner, invalid_key, missing_confirmation, locked_update]
    assert [response.status_code for response in responses] == [403, 422, 409, 409]
    for response in responses:
        assert response.status_code != 503
        serialized = json.dumps(response.json(), sort_keys=True)
        assert "wrong-owner-sensitive-value" not in serialized
        assert "invalid-key-sensitive-value" not in serialized
        assert "locked-sensitive-value" not in serialized
