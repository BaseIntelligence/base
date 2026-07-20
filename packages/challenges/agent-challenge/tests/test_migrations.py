from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from agent_challenge.db import Base
from agent_challenge.models import AgentSubmission, EvaluationJob, SubmissionEnvVar
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.db import Database

OLD_TABLES = {
    "agent_submissions",
    "evaluation_jobs",
    "task_results",
    "task_log_events",
    "request_nonces",
    "owner_action_audit",
    "rules_bundles",
    "analyzer_reports",
}

NEW_TABLES = {
    "submission_families",
    "submission_artifacts",
    "submission_status_events",
    "rate_limit_reservations",
    "analysis_runs",
    "python_ast_features",
    "similarity_matches",
    "llm_verdicts",
    "evaluation_attempts",
    "terminal_bench_trials",
    "external_execution_refs",
    "admin_review_decisions",
    "submission_env_vars",
    "review_sessions",
    "review_rules_snapshots",
    "review_assignments",
    "review_evidence_objects",
    "review_nonces",
    "review_operator_approvals",
}


async def test_database_init_creates_registered_schema(tmp_path):
    database_path = tmp_path / "fresh.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    await database.init()
    try:
        async with database.engine.begin() as connection:
            table_names = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            submission_columns = {
                column["name"]
                for column in await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_columns(
                        "agent_submissions"
                    )
                )
            }
            env_columns = {
                column["name"]
                for column in await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_columns(
                        "submission_env_vars"
                    )
                )
            }
            review_assignment_columns = {
                column["name"]
                for column in await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_columns(
                        "review_assignments"
                    )
                )
            }
            submission_indexes = {
                index["name"]
                for index in await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_indexes(
                        "agent_submissions"
                    )
                )
            }
    finally:
        await database.close()

    assert OLD_TABLES | NEW_TABLES <= table_names
    assert "ix_agent_submissions_created_at" in submission_indexes
    assert {
        "env_confirmed_empty",
        "env_confirmed_empty_at",
        "env_locked_at",
        "env_compatibility_reason",
    } <= submission_columns
    assert {
        "submission_id",
        "key",
        "value_ciphertext",
        "value_sha256",
        "created_at",
        "updated_at",
        "locked_at",
    } <= env_columns
    assert {
        "review_evidence_descriptor_json",
        "review_public_projection_json",
    } <= review_assignment_columns


async def test_create_all_bootstraps_new_tables_without_dropping_existing_rows(tmp_path):
    database_path = tmp_path / "upgrade.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection,
                    tables=[
                        AgentSubmission.__table__,
                        EvaluationJob.__table__,
                    ],
                )
            )
            await connection.execute(
                text(
                    "INSERT INTO agent_submissions "
                    "(miner_hotkey, name, agent_hash, artifact_uri, status, raw_status, "
                    "effective_status, created_at, submitted_at) "
                    "VALUES "
                    "('miner-hotkey', 'agent', 'hash-upgrade', '/tmp/agent.zip', "
                    "'pending', 'received', 'received', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )

        await database.init()

        async with database.engine.begin() as connection:
            table_names = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            row_count = (
                await connection.execute(text("SELECT COUNT(*) FROM agent_submissions"))
            ).scalar_one()
            existing_hash = (
                await connection.execute(text("SELECT agent_hash FROM agent_submissions"))
            ).scalar_one()
            version_row = (
                (
                    await connection.execute(
                        text(
                            "SELECT f.normalized_name, f.latest_submission_id, f.version_count, "
                            "s.submission_family_id, s.version_number, s.version_label, "
                            "s.canonical_artifact_hash, s.is_latest_version, "
                            "s.env_confirmed_empty, s.env_confirmed_empty_at, "
                            "s.env_locked_at, s.env_compatibility_reason "
                            "FROM agent_submissions s "
                            "JOIN submission_families f ON f.id = s.submission_family_id"
                        )
                    )
                )
                .mappings()
                .one()
            )
    finally:
        await database.close()

    assert OLD_TABLES | NEW_TABLES <= table_names
    assert row_count == 1
    assert existing_hash == "hash-upgrade"
    assert version_row["normalized_name"] == "agent"
    assert version_row["latest_submission_id"] == 1
    assert version_row["version_count"] == 1
    assert version_row["submission_family_id"] is not None
    assert version_row["version_number"] == 1
    assert version_row["version_label"] == "v1"
    assert version_row["canonical_artifact_hash"] == "legacy:1:hash-upgrade"
    assert version_row["is_latest_version"] == 1
    assert version_row["env_confirmed_empty"] == 0
    assert version_row["env_confirmed_empty_at"] is None
    assert version_row["env_locked_at"] is None
    assert version_row["env_compatibility_reason"] is None


async def test_database_init_backfills_legacy_rows_with_deterministic_families(tmp_path):
    database_path = tmp_path / "legacy-backfill.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(
                text(
                    "INSERT INTO agent_submissions "
                    "(miner_hotkey, name, agent_hash, artifact_uri, status, raw_status, "
                    "effective_status, zip_sha256, created_at, submitted_at) "
                    "VALUES "
                    "('owner-a', 'Alpha Agent', 'legacy-agent-a', '/tmp/a.zip', 'pending', "
                    "'received', 'received', 'zip-hash-a', "
                    "'2026-05-22 12:00:00', CURRENT_TIMESTAMP),"
                    "('owner-b', 'Alpha Agent', 'legacy-agent-b', '/tmp/b.zip', 'pending', "
                    "'received', 'received', 'zip-hash-b', "
                    "'2026-05-22 12:00:00', CURRENT_TIMESTAMP),"
                    "('owner-c', 'bad/name!', 'legacy-agent-c', '/tmp/c.zip', 'pending', "
                    "'received', 'received', 'zip-hash-a', "
                    "'2026-05-22 11:00:00', CURRENT_TIMESTAMP),"
                    "('owner-d', '   ', 'legacy-agent-d', '/tmp/d.zip', 'pending', "
                    "'received', 'received', NULL, "
                    "'2026-05-22 13:00:00', CURRENT_TIMESTAMP)"
                )
            )

        await database.init()
        await database.init()

        async with database.engine.begin() as connection:
            family_count = (
                await connection.execute(text("SELECT COUNT(*) FROM submission_families"))
            ).scalar_one()
            rows = (
                (
                    await connection.execute(
                        text(
                            "SELECT s.id, s.name, s.agent_hash, s.version_number, s.version_label, "
                            "s.canonical_artifact_hash, s.is_latest_version, "
                            "f.public_family_id, f.display_name, f.normalized_name, "
                            "f.latest_submission_id, f.version_count "
                            "FROM agent_submissions s "
                            "JOIN submission_families f ON f.id = s.submission_family_id "
                            "ORDER BY s.id"
                        )
                    )
                )
                .mappings()
                .all()
            )
    finally:
        await database.close()

    assert family_count == 4
    assert [row["name"] for row in rows] == ["Alpha Agent", "Alpha Agent", "bad/name!", "   "]
    assert [row["agent_hash"] for row in rows] == [
        "legacy-agent-a",
        "legacy-agent-b",
        "legacy-agent-c",
        "legacy-agent-d",
    ]
    assert [row["normalized_name"] for row in rows] == ["agent-1", "agent-2", "agent-3", "agent-4"]
    assert [row["canonical_artifact_hash"] for row in rows] == [
        "legacy-duplicate:1:zip-hash-a",
        "zip-hash-b",
        "zip-hash-a",
        "legacy:4:legacy-agent-d",
    ]
    for row in rows:
        assert row["public_family_id"]
        assert row["display_name"] == row["name"]
        assert row["latest_submission_id"] == row["id"]
        assert row["version_count"] == 1
        assert row["version_number"] == 1
        assert row["version_label"] == "v1"
        assert row["is_latest_version"] == 1


async def test_database_init_migrates_old_shape_submission_table(tmp_path):
    database_path = tmp_path / "old-shape.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    try:
        async with database.engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE agent_submissions ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "miner_hotkey VARCHAR(128) NOT NULL, "
                    "name VARCHAR(128) NOT NULL, "
                    "agent_hash VARCHAR(128) NOT NULL UNIQUE, "
                    "artifact_uri TEXT NOT NULL, "
                    "status VARCHAR(32) NOT NULL DEFAULT 'pending', "
                    "submitted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                    "raw_status VARCHAR(32) NOT NULL DEFAULT 'received', "
                    "effective_status VARCHAR(32) NOT NULL DEFAULT 'received', "
                    "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
            )
            await connection.execute(
                text(
                    "INSERT INTO agent_submissions "
                    "(miner_hotkey, name, agent_hash, artifact_uri) "
                    "VALUES ('old-owner', 'Old Agent', 'old-agent-hash', '/tmp/old.zip')"
                )
            )

        await database.init()
        await database.init()

        async with database.engine.begin() as connection:
            column_names = {
                row[1]
                for row in await connection.exec_driver_sql("PRAGMA table_info(agent_submissions)")
            }
            family_count = (
                await connection.execute(text("SELECT COUNT(*) FROM submission_families"))
            ).scalar_one()
            row = (
                (
                    await connection.execute(
                        text(
                            "SELECT s.name, s.agent_hash, s.version_number, s.version_label, "
                            "s.canonical_artifact_hash, s.is_latest_version, "
                            "s.env_confirmed_empty, s.env_confirmed_empty_at, "
                            "s.env_locked_at, s.env_compatibility_reason, "
                            "f.normalized_name, f.latest_submission_id, f.version_count "
                            "FROM agent_submissions s "
                            "JOIN submission_families f ON f.id = s.submission_family_id"
                        )
                    )
                )
                .mappings()
                .one()
            )
    finally:
        await database.close()

    assert {
        "submission_family_id",
        "version_number",
        "version_label",
        "canonical_artifact_hash",
        "is_latest_version",
        "env_confirmed_empty",
        "env_confirmed_empty_at",
        "env_locked_at",
        "env_compatibility_reason",
    } <= column_names
    assert family_count == 1
    assert row["name"] == "Old Agent"
    assert row["agent_hash"] == "old-agent-hash"
    assert row["normalized_name"] == "old agent"
    assert row["latest_submission_id"] == 1
    assert row["version_count"] == 1
    assert row["version_number"] == 1
    assert row["version_label"] == "v1"
    assert row["canonical_artifact_hash"] == "legacy:1:old-agent-hash"
    assert row["is_latest_version"] == 1
    assert row["env_confirmed_empty"] == 0
    assert row["env_confirmed_empty_at"] is None
    assert row["env_locked_at"] is None
    assert row["env_compatibility_reason"] is None


async def test_database_init_backfills_only_analysis_allowed_env_metadata(tmp_path):
    database_path = tmp_path / "legacy-env-policy.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    try:
        async with database.engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE agent_submissions ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "miner_hotkey VARCHAR(128) NOT NULL, "
                    "name VARCHAR(128) NOT NULL, "
                    "agent_hash VARCHAR(128) NOT NULL UNIQUE, "
                    "artifact_uri TEXT NOT NULL, "
                    "status VARCHAR(32) NOT NULL DEFAULT 'pending', "
                    "submitted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                    "raw_status VARCHAR(32) NOT NULL DEFAULT 'received', "
                    "effective_status VARCHAR(32) NOT NULL DEFAULT 'received', "
                    "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
            )
            await connection.execute(
                text(
                    "INSERT INTO agent_submissions "
                    "(miner_hotkey, name, agent_hash, artifact_uri, raw_status, effective_status) "
                    "VALUES "
                    "('owner-a', 'Allowed Agent', 'hash-allowed', '/tmp/a.zip', "
                    "'analysis_allowed', 'analysis_allowed'),"
                    "('owner-b', 'Queued Agent', 'hash-queued', '/tmp/b.zip', "
                    "'tb_queued', 'tb_queued'),"
                    "('owner-c', 'Complete Agent', 'hash-complete', '/tmp/c.zip', "
                    "'tb_completed', 'tb_completed')"
                )
            )

        await database.init()
        await database.init()

        async with database.engine.begin() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            "SELECT agent_hash, env_confirmed_empty, env_confirmed_empty_at, "
                            "env_locked_at, env_compatibility_reason "
                            "FROM agent_submissions ORDER BY id"
                        )
                    )
                )
                .mappings()
                .all()
            )
    finally:
        await database.close()

    allowed, queued, completed = rows
    assert allowed["agent_hash"] == "hash-allowed"
    assert allowed["env_confirmed_empty"] == 1
    assert allowed["env_confirmed_empty_at"] is not None
    assert allowed["env_locked_at"] is not None
    assert allowed["env_compatibility_reason"] == "pre_env_gate_analysis_allowed"
    for unchanged in (queued, completed):
        assert unchanged["env_confirmed_empty"] == 0
        assert unchanged["env_confirmed_empty_at"] is None
        assert unchanged["env_locked_at"] is None
        assert unchanged["env_compatibility_reason"] is None


async def test_submission_env_vars_are_unique_and_encrypted(tmp_path):
    from cryptography.fernet import Fernet

    database_path = tmp_path / "env-vars.sqlite3"
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    plaintext = "generated-sensitive-sentinel-value"

    try:
        await database.init()
        async with database.session() as session:
            submission = AgentSubmission(
                miner_hotkey="miner-env",
                name="env-agent",
                agent_hash="env-agent-hash",
                artifact_uri="/tmp/env-agent.zip",
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

        async with database.session() as session:
            duplicate = SubmissionEnvVar.encrypted(
                submission_id=submission_id,
                key="AGENT_API_KEY",
                value="other-value",
                settings=settings,
            )
            session.add(duplicate)
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

        async with database.engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        text("SELECT key, value_ciphertext, value_sha256 FROM submission_env_vars")
                    )
                )
                .mappings()
                .one()
            )
    finally:
        await database.close()

    assert row["key"] == "AGENT_API_KEY"
    assert row["value_ciphertext"] != plaintext
    assert plaintext not in row["value_ciphertext"]
    assert row["value_sha256"]
    assert row["value_sha256"] != plaintext

    env_var = SubmissionEnvVar(
        submission_id=submission_id,
        key=row["key"],
        value_ciphertext=row["value_ciphertext"],
        value_sha256=row["value_sha256"],
    )
    assert env_var.decrypt_value_for_launch(settings) == plaintext
    with pytest.raises(Exception, match="submission env encryption key"):
        env_var.decrypt_value_for_launch(ChallengeSettings(submission_env_encryption_key_file=None))


def test_submission_env_encryption_key_file_required_for_env_writes():
    with pytest.raises(Exception, match="submission env encryption key"):
        SubmissionEnvVar.encrypted(
            submission_id=1,
            key="AGENT_API_KEY",
            value="value",
            settings=ChallengeSettings(submission_env_encryption_key_file=None),
        )


async def test_database_init_runs_postgresql_submission_version_migration(monkeypatch):
    executed_sql: list[str] = []
    backfilled_connections = []
    run_sync_calls = []

    class _FakeResult:
        def scalar_one_or_none(self):
            # Report the running-total table as already seeded so the task-log
            # byte-total backfill short-circuits without further execute() calls.
            return 1

    class FakeConnection:
        async def exec_driver_sql(self, statement: str):
            executed_sql.append(statement)

        async def execute(self, statement, parameters=None):
            executed_sql.append(str(statement))
            return _FakeResult()

        async def run_sync(self, callback):
            run_sync_calls.append(callback)

    class FakeBegin:
        def __init__(self, connection: FakeConnection) -> None:
            self.connection = connection

        async def __aenter__(self) -> FakeConnection:
            return self.connection

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeEngine:
        url = SimpleNamespace(get_backend_name=lambda: "postgresql+asyncpg")

        def __init__(self) -> None:
            self.connection = FakeConnection()

        def begin(self) -> FakeBegin:
            return FakeBegin(self.connection)

    async def fake_backfill(self, connection):
        backfilled_connections.append(connection)

    monkeypatch.setattr(Database, "_backfill_legacy_submission_versions", fake_backfill)
    monkeypatch.setattr(Database, "_backfill_legacy_submission_env_metadata", fake_backfill)
    database = Database.__new__(Database)
    database.engine = FakeEngine()

    await database.init()

    required_columns = {
        "submission_family_id",
        "version_number",
        "version_label",
        "canonical_artifact_hash",
        "is_latest_version",
        "agent_name",
        "zip_sha256",
        "zip_size_bytes",
        "artifact_path",
        "latest_evaluation_job_id",
        "env_confirmed_empty",
        "env_confirmed_empty_at",
        "env_locked_at",
        "env_compatibility_reason",
        "signature",
        "signature_nonce",
        "signature_timestamp",
        "signature_payload_sha256",
        "signature_message",
    }

    assert run_sync_calls == [Base.metadata.create_all]
    assert backfilled_connections == [database.engine.connection, database.engine.connection]
    for column_name in required_columns:
        assert any(
            f"ALTER TABLE agent_submissions ADD COLUMN IF NOT EXISTS {column_name}" in statement
            for statement in executed_sql
        )
    assert any(
        "CREATE INDEX IF NOT EXISTS ix_agent_submissions_family_latest" in statement
        for statement in executed_sql
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS ix_agent_submissions_owner_created" in statement
        for statement in executed_sql
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS ix_agent_submissions_created_at" in statement
        for statement in executed_sql
    )
    assert any(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_submissions_family_version" in statement
        for statement in executed_sql
    )
