"""Async SQLAlchemy helpers for challenge-owned SQLite databases."""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import import_module
from uuid import uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from agent_challenge.submissions.versioning import normalize_submission_name, version_label

_AGENT_SUBMISSION_SQLITE_COLUMNS = {
    "submission_family_id": "INTEGER REFERENCES submission_families(id)",
    "version_number": "INTEGER",
    "version_label": "VARCHAR(32)",
    "canonical_artifact_hash": "VARCHAR(256)",
    "is_latest_version": "BOOLEAN NOT NULL DEFAULT 0",
    "agent_name": "VARCHAR(128)",
    "zip_sha256": "VARCHAR(64)",
    "package_tree_sha": "VARCHAR(64)",
    "zip_size_bytes": "INTEGER",
    "artifact_path": "TEXT",
    "latest_evaluation_job_id": "INTEGER REFERENCES evaluation_jobs(id)",
    "env_confirmed_empty": "BOOLEAN NOT NULL DEFAULT 0",
    "env_confirmed_empty_at": "DATETIME",
    "env_locked_at": "DATETIME",
    "env_compatibility_reason": "VARCHAR(128)",
    "signature": "TEXT",
    "signature_nonce": "VARCHAR(128)",
    "signature_timestamp": "VARCHAR(64)",
    "signature_payload_sha256": "VARCHAR(64)",
    "signature_message": "TEXT",
}

_AGENT_SUBMISSION_POSTGRESQL_COLUMNS = {
    "submission_family_id": "INTEGER REFERENCES submission_families(id)",
    "version_number": "INTEGER",
    "version_label": "VARCHAR(32)",
    "canonical_artifact_hash": "VARCHAR(256)",
    "is_latest_version": "BOOLEAN NOT NULL DEFAULT FALSE",
    "agent_name": "VARCHAR(128)",
    "zip_sha256": "VARCHAR(64)",
    "package_tree_sha": "VARCHAR(64)",
    "zip_size_bytes": "INTEGER",
    "artifact_path": "TEXT",
    "latest_evaluation_job_id": "INTEGER REFERENCES evaluation_jobs(id)",
    "env_confirmed_empty": "BOOLEAN NOT NULL DEFAULT FALSE",
    "env_confirmed_empty_at": "TIMESTAMP WITH TIME ZONE",
    "env_locked_at": "TIMESTAMP WITH TIME ZONE",
    "env_compatibility_reason": "VARCHAR(128)",
    "signature": "TEXT",
    "signature_nonce": "VARCHAR(128)",
    "signature_timestamp": "VARCHAR(64)",
    "signature_payload_sha256": "VARCHAR(64)",
    "signature_message": "TEXT",
}

_REVIEW_ASSIGNMENT_SQLITE_COLUMNS = {
    "token_delivered_at": "DATETIME",
    "model_call_started_json": "TEXT",
    "model_call_started_sha256": "VARCHAR(64)",
    "planned_request_sha256": "VARCHAR(64)",
    "request_body_sha256": "VARCHAR(64)",
    "request_body_length": "INTEGER",
    "infrastructure_failure_json": "TEXT",
    "infrastructure_failure_sha256": "VARCHAR(64)",
    "review_report_envelope_json": "TEXT",
    "review_report_sha256": "VARCHAR(64)",
    "review_digest": "VARCHAR(64)",
    "review_report_data_hex": "VARCHAR(128)",
    "review_report_received_at": "DATETIME",
    "review_verification_outcome_json": "TEXT",
    "review_evidence_descriptor_json": "TEXT",
    "review_public_projection_json": "TEXT",
}

_REVIEW_ASSIGNMENT_POSTGRESQL_COLUMNS = {
    "token_delivered_at": "TIMESTAMP WITH TIME ZONE",
    "model_call_started_json": "TEXT",
    "model_call_started_sha256": "VARCHAR(64)",
    "planned_request_sha256": "VARCHAR(64)",
    "request_body_sha256": "VARCHAR(64)",
    "request_body_length": "INTEGER",
    "infrastructure_failure_json": "TEXT",
    "infrastructure_failure_sha256": "VARCHAR(64)",
    "review_report_envelope_json": "TEXT",
    "review_report_sha256": "VARCHAR(64)",
    "review_digest": "VARCHAR(64)",
    "review_report_data_hex": "VARCHAR(128)",
    "review_report_received_at": "TIMESTAMP WITH TIME ZONE",
    "review_verification_outcome_json": "TEXT",
    "review_evidence_descriptor_json": "TEXT",
    "review_public_projection_json": "TEXT",
}

_REVIEW_SESSION_SQLITE_COLUMNS = {
    "harness_identity_json": "TEXT",
    "harness_identity_sha256": "VARCHAR(64)",
    "submission_received_at_ms": "INTEGER",
    "package_tree_sha": "VARCHAR(64)",
}

_REVIEW_SESSION_POSTGRESQL_COLUMNS = {
    "harness_identity_json": "TEXT",
    "harness_identity_sha256": "VARCHAR(64)",
    "submission_received_at_ms": "INTEGER",
    "package_tree_sha": "VARCHAR(64)",
}

_EVALUATION_JOB_SQLITE_COLUMNS = {
    "eval_plan_json": "TEXT",
    "canonical_score_record_json": "TEXT",
    "canonical_score_record_sha256": "VARCHAR(64)",
}

_EVALUATION_JOB_POSTGRESQL_COLUMNS = {
    "eval_plan_json": "TEXT",
    "canonical_score_record_json": "TEXT",
    "canonical_score_record_sha256": "VARCHAR(64)",
}

_AGENT_SUBMISSION_POSTGRESQL_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_agent_submissions_submission_family_id "
    "ON agent_submissions (submission_family_id)",
    "CREATE INDEX IF NOT EXISTS ix_agent_submissions_canonical_artifact_hash "
    "ON agent_submissions (canonical_artifact_hash)",
    "CREATE INDEX IF NOT EXISTS ix_agent_submissions_zip_sha256 ON agent_submissions (zip_sha256)",
    "CREATE INDEX IF NOT EXISTS ix_agent_submissions_created_at ON agent_submissions (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_agent_submissions_family_latest "
    "ON agent_submissions (submission_family_id, is_latest_version)",
    "CREATE INDEX IF NOT EXISTS ix_agent_submissions_owner_created "
    "ON agent_submissions (miner_hotkey, created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_submissions_family_version "
    "ON agent_submissions (submission_family_id, version_number) "
    "WHERE submission_family_id IS NOT NULL AND version_number IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_submissions_canonical_artifact_hash "
    "ON agent_submissions (canonical_artifact_hash) "
    "WHERE canonical_artifact_hash IS NOT NULL",
)


class Base(DeclarativeBase):
    """Base class for challenge models."""


class Database:
    """Small async database wrapper used by the BASE challenge app."""

    def __init__(self, database_url: str) -> None:
        # SQLite busy_timeout: worker and reconciler use separate connections, so a
        # contending writer must wait for the lock instead of raising immediately.
        connect_args: dict[str, object] = {}
        engine_kwargs: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False, "timeout": 30.0}
            # Modestly raise the pool (default 5+10) so concurrent SSE streams and
            # the combined worker cannot exhaust it (defense in depth; per-poll
            # SSE sessions are the primary fix). In-memory SQLite uses a StaticPool
            # that rejects pool sizing, so only size the file-backed QueuePool.
            if ":memory:" not in database_url:
                engine_kwargs["pool_size"] = 10
                engine_kwargs["max_overflow"] = 20
        else:
            # PostgreSQL (asyncpg): worker runs cross-node from Postgres, so idle
            # connections get silently dropped by NAT/firewall. pool_pre_ping +
            # pool_recycle rotate dead/old sockets; command_timeout bounds each
            # statement so a black-holed socket fails fast instead of hanging
            # (a missing command_timeout previously caused a ~16h analyzer hang).
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_recycle"] = 300
            engine_kwargs["pool_size"] = 10
            engine_kwargs["max_overflow"] = 20
            connect_args = {"command_timeout": 60}
        self.engine = create_async_engine(
            database_url,
            connect_args=connect_args,
            **engine_kwargs,
        )
        self._session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

    async def init(self) -> None:
        """Create all challenge-owned tables."""

        import_module("agent_challenge.core.models")
        async with self.engine.begin() as connection:
            backend_name = self.engine.url.get_backend_name()
            is_sqlite = backend_name.startswith("sqlite")
            is_postgresql = backend_name.startswith("postgresql")
            if is_sqlite:
                await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            await connection.run_sync(Base.metadata.create_all)
            if is_sqlite:
                await self._migrate_sqlite_submission_columns(connection)
                await self._migrate_sqlite_eval_ledger(connection)
                await self._migrate_sqlite_replay_audit(connection)
                await self._migrate_sqlite_evaluation_job_columns(connection)
                await self._migrate_sqlite_evaluation_attempt_columns(connection)
                await self._migrate_sqlite_task_log_columns(connection)
                await self._migrate_sqlite_review_columns(connection)
                await self._backfill_legacy_submission_versions(connection)
                await self._backfill_legacy_submission_env_metadata(connection)
            elif is_postgresql:
                await self._migrate_postgresql_submission_columns(connection)
                await self._migrate_postgresql_eval_ledger(connection)
                await self._migrate_postgresql_replay_audit(connection)
                await self._migrate_postgresql_evaluation_job_columns(connection)
                await self._migrate_postgresql_evaluation_attempt_columns(connection)
                await self._migrate_postgresql_task_log_columns(connection)
                await self._migrate_postgresql_review_columns(connection)
                await self._backfill_legacy_submission_versions(connection)
                await self._backfill_legacy_submission_env_metadata(connection)
                await self._migrate_postgresql_submission_indexes(connection)
            await self._seed_eval_resource_counters(connection)

    async def _seed_eval_resource_counters(self, connection: AsyncConnection) -> None:
        """Ensure global Eval capacity counters exist before admissions use them."""

        backend = self.engine.url.get_backend_name()
        for name in ("eval_result_outstanding", "eval_result_verifying"):
            if backend.startswith("sqlite"):
                await connection.execute(
                    text(
                        "INSERT OR IGNORE INTO eval_resource_counters "
                        "(name, value, updated_at) VALUES (:name, 0, CURRENT_TIMESTAMP)"
                    ),
                    {"name": name},
                )
            else:
                await connection.execute(
                    text(
                        "INSERT INTO eval_resource_counters (name, value, updated_at) "
                        "VALUES (:name, 0, NOW()) ON CONFLICT (name) DO NOTHING"
                    ),
                    {"name": name},
                )

    async def close(self) -> None:
        """Dispose database connections."""

        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an async SQLAlchemy session."""

        async with self._session_factory() as session:
            yield session

    async def session_dependency(self) -> AsyncIterator[AsyncSession]:
        """FastAPI dependency wrapper for request-scoped sessions."""

        async with self.session() as session:
            yield session

    async def _migrate_sqlite_submission_columns(self, connection: AsyncConnection) -> None:
        table_exists = (
            await connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'agent_submissions'"
                )
            )
        ).scalar_one_or_none()
        if table_exists is None:
            return

        columns = {
            row[1]
            for row in await connection.exec_driver_sql("PRAGMA table_info(agent_submissions)")
        }
        for column_name, column_definition in _AGENT_SUBMISSION_SQLITE_COLUMNS.items():
            if column_name not in columns:
                await connection.exec_driver_sql(
                    f"ALTER TABLE agent_submissions ADD COLUMN {column_name} {column_definition}"
                )
        await connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_submissions_family_version "
            "ON agent_submissions (submission_family_id, version_number)"
        )
        await connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_submissions_canonical_artifact_hash "
            "ON agent_submissions (canonical_artifact_hash)"
        )

    async def _migrate_postgresql_submission_columns(self, connection: AsyncConnection) -> None:
        for column_name, column_definition in _AGENT_SUBMISSION_POSTGRESQL_COLUMNS.items():
            await connection.exec_driver_sql(
                f"ALTER TABLE agent_submissions ADD COLUMN IF NOT EXISTS "
                f"{column_name} {column_definition}"
            )

    async def _migrate_sqlite_eval_ledger(self, connection: AsyncConnection) -> None:
        """Create Eval ledger tables for databases initialized before attested Eval."""

        await connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_run_id VARCHAR(128) NOT NULL UNIQUE,
                submission_id INTEGER NOT NULL REFERENCES agent_submissions(id),
                submission_version INTEGER NOT NULL,
                authorizing_review_digest VARCHAR(64) NOT NULL,
                plan_json TEXT NOT NULL,
                plan_sha256 VARCHAR(64) NOT NULL,
                token_sha256 VARCHAR(64) NOT NULL UNIQUE,
                token_delivered_at DATETIME,
                phase VARCHAR(32) NOT NULL DEFAULT 'eval_prepared',
                reason_code VARCHAR(128),
                failure_origin VARCHAR(64),
                retryable BOOLEAN NOT NULL DEFAULT 1,
                verified BOOLEAN NOT NULL DEFAULT 0,
                reward_eligible BOOLEAN NOT NULL DEFAULT 0,
                key_granted_at DATETIME,
                key_release_receipt_sha256 VARCHAR(64),
                key_release_receipt_received_at DATETIME,
                key_release_state VARCHAR(32),
                key_release_reason VARCHAR(128),
                key_release_completed_at DATETIME,
                key_release_grant_json TEXT,
                receipt_id VARCHAR(128),
                receipt_body_sha256 VARCHAR(64),
                receipt_body BLOB,
                receipt_received_at DATETIME,
                receipt_verification_claimed_at DATETIME,
                result_job_id INTEGER REFERENCES evaluation_jobs(id),
                score FLOAT,
                passed_tasks INTEGER,
                total_tasks INTEGER,
                canonical_score_record_json TEXT,
                canonical_score_record_sha256 VARCHAR(64),
                result_submission_count_window_start DATETIME,
                result_submission_count INTEGER NOT NULL DEFAULT 0,
                finalized_at DATETIME,
                issued_at DATETIME NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        await connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS eval_nonces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_run_id INTEGER NOT NULL REFERENCES eval_runs(id),
                nonce VARCHAR(128) NOT NULL UNIQUE,
                purpose VARCHAR(32) NOT NULL,
                state VARCHAR(32) NOT NULL DEFAULT 'outstanding',
                expires_at DATETIME NOT NULL,
                consumed_at DATETIME,
                created_at DATETIME NOT NULL,
                UNIQUE(eval_run_id, purpose)
            )
            """
        )
        columns = {
            row[1] for row in await connection.exec_driver_sql("PRAGMA table_info(eval_runs)")
        }
        for name, definition in {
            "attempt": "INTEGER NOT NULL DEFAULT 1",
            "prior_eval_run_id": "VARCHAR(128)",
            "result_available": "BOOLEAN NOT NULL DEFAULT 0",
            "key_release_receipt_sha256": "VARCHAR(64)",
            "key_release_receipt_received_at": "DATETIME",
            "key_release_state": "VARCHAR(32)",
            "key_release_reason": "VARCHAR(128)",
            "key_release_completed_at": "DATETIME",
            "key_release_grant_json": "TEXT",
            "receipt_body": "BLOB",
            "receipt_verification_claimed_at": "DATETIME",
            "result_job_id": "INTEGER REFERENCES evaluation_jobs(id)",
            "score": "FLOAT",
            "passed_tasks": "INTEGER",
            "total_tasks": "INTEGER",
            "canonical_score_record_json": "TEXT",
            "canonical_score_record_sha256": "VARCHAR(64)",
            "result_submission_count_window_start": "DATETIME",
            "result_submission_count": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if name not in columns:
                await connection.exec_driver_sql(
                    f"ALTER TABLE eval_runs ADD COLUMN {name} {definition}"
                )

    async def _migrate_postgresql_eval_ledger(self, connection: AsyncConnection) -> None:
        """Create Eval ledger tables for PostgreSQL deployments."""

        await connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id SERIAL PRIMARY KEY,
                eval_run_id VARCHAR(128) NOT NULL UNIQUE,
                submission_id INTEGER NOT NULL REFERENCES agent_submissions(id),
                submission_version INTEGER NOT NULL,
                authorizing_review_digest VARCHAR(64) NOT NULL,
                plan_json TEXT NOT NULL,
                plan_sha256 VARCHAR(64) NOT NULL,
                token_sha256 VARCHAR(64) NOT NULL UNIQUE,
                token_delivered_at TIMESTAMP WITH TIME ZONE,
                phase VARCHAR(32) NOT NULL DEFAULT 'eval_prepared',
                reason_code VARCHAR(128),
                failure_origin VARCHAR(64),
                retryable BOOLEAN NOT NULL DEFAULT TRUE,
                verified BOOLEAN NOT NULL DEFAULT FALSE,
                reward_eligible BOOLEAN NOT NULL DEFAULT FALSE,
                key_granted_at TIMESTAMP WITH TIME ZONE,
                key_release_receipt_sha256 VARCHAR(64),
                key_release_receipt_received_at TIMESTAMP WITH TIME ZONE,
                key_release_state VARCHAR(32),
                key_release_reason VARCHAR(128),
                key_release_completed_at TIMESTAMP WITH TIME ZONE,
                key_release_grant_json TEXT,
                receipt_id VARCHAR(128),
                receipt_body_sha256 VARCHAR(64),
                receipt_body BYTEA,
                receipt_received_at TIMESTAMP WITH TIME ZONE,
                receipt_verification_claimed_at TIMESTAMP WITH TIME ZONE,
                result_job_id INTEGER REFERENCES evaluation_jobs(id),
                score DOUBLE PRECISION,
                passed_tasks INTEGER,
                total_tasks INTEGER,
                canonical_score_record_json TEXT,
                canonical_score_record_sha256 VARCHAR(64),
                result_submission_count_window_start TIMESTAMP WITH TIME ZONE,
                result_submission_count INTEGER NOT NULL DEFAULT 0,
                finalized_at TIMESTAMP WITH TIME ZONE,
                issued_at TIMESTAMP WITH TIME ZONE NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
            """
        )
        await connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS eval_nonces (
                id SERIAL PRIMARY KEY,
                eval_run_id INTEGER NOT NULL REFERENCES eval_runs(id),
                nonce VARCHAR(128) NOT NULL UNIQUE,
                purpose VARCHAR(32) NOT NULL,
                state VARCHAR(32) NOT NULL DEFAULT 'outstanding',
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                consumed_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                UNIQUE(eval_run_id, purpose)
            )
            """
        )
        for name, definition in {
            "attempt": "INTEGER NOT NULL DEFAULT 1",
            "prior_eval_run_id": "VARCHAR(128)",
            "result_available": "BOOLEAN NOT NULL DEFAULT FALSE",
            "key_release_receipt_sha256": "VARCHAR(64)",
            "key_release_receipt_received_at": "TIMESTAMP WITH TIME ZONE",
            "key_release_state": "VARCHAR(32)",
            "key_release_reason": "VARCHAR(128)",
            "key_release_completed_at": "TIMESTAMP WITH TIME ZONE",
            "key_release_grant_json": "TEXT",
            "receipt_body": "BYTEA",
            "receipt_verification_claimed_at": "TIMESTAMP WITH TIME ZONE",
            "result_job_id": "INTEGER REFERENCES evaluation_jobs(id)",
            "score": "DOUBLE PRECISION",
            "passed_tasks": "INTEGER",
            "total_tasks": "INTEGER",
            "canonical_score_record_json": "TEXT",
            "canonical_score_record_sha256": "VARCHAR(64)",
            "result_submission_count_window_start": "TIMESTAMP WITH TIME ZONE",
            "result_submission_count": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            await connection.exec_driver_sql(
                f"ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS {name} {definition}"
            )

    async def _migrate_sqlite_replay_audit(self, connection: AsyncConnection) -> None:
        """Create replay dispute storage for databases initialized pre-audit."""

        await connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS replay_audit_disputes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id VARCHAR(160) NOT NULL UNIQUE,
                submission_id INTEGER NOT NULL REFERENCES agent_submissions(id),
                eval_run_id VARCHAR(128) NOT NULL,
                replay_attempt INTEGER NOT NULL DEFAULT 1,
                plan_sha256 VARCHAR(64) NOT NULL,
                scoring_policy_digest VARCHAR(64) NOT NULL,
                attested_score FLOAT NOT NULL,
                replay_score FLOAT NOT NULL,
                delta FLOAT NOT NULL,
                created_at DATETIME NOT NULL,
                UNIQUE(submission_id, eval_run_id, replay_attempt)
            )
            """
        )

    async def _migrate_postgresql_replay_audit(self, connection: AsyncConnection) -> None:
        """Create replay dispute storage for PostgreSQL deployments."""

        await connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS replay_audit_disputes (
                id SERIAL PRIMARY KEY,
                audit_id VARCHAR(160) NOT NULL UNIQUE,
                submission_id INTEGER NOT NULL REFERENCES agent_submissions(id),
                eval_run_id VARCHAR(128) NOT NULL,
                replay_attempt INTEGER NOT NULL DEFAULT 1,
                plan_sha256 VARCHAR(64) NOT NULL,
                scoring_policy_digest VARCHAR(64) NOT NULL,
                attested_score DOUBLE PRECISION NOT NULL,
                replay_score DOUBLE PRECISION NOT NULL,
                delta DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                UNIQUE(submission_id, eval_run_id, replay_attempt)
            )
            """
        )

    async def _migrate_sqlite_evaluation_job_columns(self, connection: AsyncConnection) -> None:
        table_exists = (
            await connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evaluation_jobs'"
                )
            )
        ).scalar_one_or_none()
        if table_exists is None:
            return
        columns = {
            row[1] for row in await connection.exec_driver_sql("PRAGMA table_info(evaluation_jobs)")
        }
        for column_name, column_definition in _EVALUATION_JOB_SQLITE_COLUMNS.items():
            if column_name not in columns:
                await connection.exec_driver_sql(
                    f"ALTER TABLE evaluation_jobs ADD COLUMN {column_name} {column_definition}"
                )

    async def _migrate_postgresql_evaluation_job_columns(self, connection: AsyncConnection) -> None:
        for column_name, column_definition in _EVALUATION_JOB_POSTGRESQL_COLUMNS.items():
            await connection.exec_driver_sql(
                "ALTER TABLE evaluation_jobs "
                f"ADD COLUMN IF NOT EXISTS {column_name} {column_definition}"
            )

    async def _migrate_postgresql_submission_indexes(self, connection: AsyncConnection) -> None:
        for statement in _AGENT_SUBMISSION_POSTGRESQL_INDEXES:
            await connection.exec_driver_sql(statement)

    async def _migrate_sqlite_evaluation_attempt_columns(self, connection: AsyncConnection) -> None:
        table_exists = (
            await connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'evaluation_attempts'"
                )
            )
        ).scalar_one_or_none()
        if table_exists is None:
            return
        columns = {
            row[1]
            for row in await connection.exec_driver_sql("PRAGMA table_info(evaluation_attempts)")
        }
        if "task_id" not in columns:
            await connection.exec_driver_sql(
                "ALTER TABLE evaluation_attempts ADD COLUMN task_id VARCHAR(256)"
            )
        await connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_task_id "
            "ON evaluation_attempts (task_id)"
        )
        await connection.exec_driver_sql(
            "UPDATE evaluation_attempts "
            "SET task_id = json_extract(metadata_json, '$.task_id') "
            "WHERE task_id IS NULL "
            "AND metadata_json IS NOT NULL "
            "AND json_valid(metadata_json) "
            "AND json_extract(metadata_json, '$.task_id') IS NOT NULL"
        )

    async def _migrate_postgresql_evaluation_attempt_columns(
        self, connection: AsyncConnection
    ) -> None:
        await connection.exec_driver_sql(
            "ALTER TABLE evaluation_attempts ADD COLUMN IF NOT EXISTS task_id VARCHAR(256)"
        )
        await connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_task_id "
            "ON evaluation_attempts (task_id)"
        )
        await connection.exec_driver_sql(
            "UPDATE evaluation_attempts "
            "SET task_id = metadata_json::json ->> 'task_id' "
            "WHERE task_id IS NULL "
            "AND metadata_json IS NOT NULL "
            "AND metadata_json::json ->> 'task_id' IS NOT NULL"
        )

    async def _migrate_sqlite_task_log_columns(self, connection: AsyncConnection) -> None:
        table_exists = (
            await connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_log_events'"
                )
            )
        ).scalar_one_or_none()
        if table_exists is None:
            return
        columns = {
            row[1] for row in await connection.exec_driver_sql("PRAGMA table_info(task_log_events)")
        }
        if "message_bytes" not in columns:
            await connection.exec_driver_sql(
                "ALTER TABLE task_log_events ADD COLUMN message_bytes INTEGER NOT NULL DEFAULT 0"
            )
        await connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_task_log_events_task_result_id "
            "ON task_log_events (task_result_id)"
        )
        await self._backfill_task_log_byte_totals(
            connection,
            message_bytes_expr="length(CAST(message AS BLOB))",
        )

    async def _migrate_postgresql_task_log_columns(self, connection: AsyncConnection) -> None:
        await connection.exec_driver_sql(
            "ALTER TABLE task_log_events "
            "ADD COLUMN IF NOT EXISTS message_bytes INTEGER NOT NULL DEFAULT 0"
        )
        await connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_task_log_events_task_result_id "
            "ON task_log_events (task_result_id)"
        )
        await self._backfill_task_log_byte_totals(
            connection,
            message_bytes_expr="octet_length(message)",
        )

    async def _migrate_sqlite_review_columns(self, connection: AsyncConnection) -> None:
        table_exists = (
            await connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'review_assignments'"
                )
            )
        ).scalar_one_or_none()
        if table_exists is None:
            return
        columns = {
            row[1]
            for row in await connection.exec_driver_sql("PRAGMA table_info(review_assignments)")
        }
        for column_name, column_definition in _REVIEW_ASSIGNMENT_SQLITE_COLUMNS.items():
            if column_name not in columns:
                await connection.exec_driver_sql(
                    f"ALTER TABLE review_assignments ADD COLUMN {column_name} {column_definition}"
                )
        await self._migrate_sqlite_review_session_columns(connection)

    async def _migrate_sqlite_review_session_columns(self, connection: AsyncConnection) -> None:
        table_exists = (
            await connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'review_sessions'"
                )
            )
        ).scalar_one_or_none()
        if table_exists is None:
            return
        columns = {
            row[1] for row in await connection.exec_driver_sql("PRAGMA table_info(review_sessions)")
        }
        for column_name, column_definition in _REVIEW_SESSION_SQLITE_COLUMNS.items():
            if column_name not in columns:
                await connection.exec_driver_sql(
                    f"ALTER TABLE review_sessions ADD COLUMN {column_name} {column_definition}"
                )

    async def _migrate_postgresql_review_columns(self, connection: AsyncConnection) -> None:
        for column_name, column_definition in _REVIEW_ASSIGNMENT_POSTGRESQL_COLUMNS.items():
            await connection.exec_driver_sql(
                "ALTER TABLE review_assignments "
                f"ADD COLUMN IF NOT EXISTS {column_name} {column_definition}"
            )
        for column_name, column_definition in _REVIEW_SESSION_POSTGRESQL_COLUMNS.items():
            await connection.exec_driver_sql(
                "ALTER TABLE review_sessions "
                f"ADD COLUMN IF NOT EXISTS {column_name} {column_definition}"
            )

    async def _backfill_task_log_byte_totals(
        self,
        connection: AsyncConnection,
        *,
        message_bytes_expr: str,
    ) -> None:
        """One-time seed of the running-total counters from existing log rows.

        Runs only while ``task_log_byte_totals`` is still empty: once seeded (or
        once live ingest has incremented it), later startups skip so the O(1)
        counters are never clobbered. The entire ``init()`` runs in one
        transaction, so a crash mid-backfill rolls back and retries cleanly.
        ``message_bytes_expr`` is the dialect's UTF-8 byte-length function so the
        seed is byte-exact with ``len(message.encode("utf-8"))``.
        """

        already_seeded = (
            await connection.execute(text("SELECT 1 FROM task_log_byte_totals LIMIT 1"))
        ).scalar_one_or_none()
        if already_seeded is not None:
            return

        from agent_challenge.evaluation.task_events import (
            _NON_COUNTED_EVENT_TYPES,
            LOG_BYTE_SCOPE_SUBMISSION,
            LOG_BYTE_SCOPE_TASK,
            LOG_BYTE_SCOPE_TASK_RESULT,
        )

        await connection.execute(
            text(
                f"UPDATE task_log_events SET message_bytes = {message_bytes_expr} "
                "WHERE message_bytes = 0 AND message <> ''"
            )
        )

        non_counted = sorted(_NON_COUNTED_EVENT_TYPES)
        await connection.execute(
            text(
                "INSERT INTO task_log_byte_totals "
                "(submission_id, scope, scope_key, total_bytes) "
                "SELECT submission_id, :scope, '', COALESCE(SUM(message_bytes), 0) "
                "FROM task_log_events "
                "WHERE event_type NOT IN :types "
                "GROUP BY submission_id"
            ).bindparams(bindparam("types", expanding=True)),
            {"scope": LOG_BYTE_SCOPE_SUBMISSION, "types": non_counted},
        )
        await connection.execute(
            text(
                "INSERT INTO task_log_byte_totals "
                "(submission_id, scope, scope_key, total_bytes) "
                "SELECT submission_id, :scope, CAST(task_result_id AS TEXT), "
                "COALESCE(SUM(message_bytes), 0) "
                "FROM task_log_events "
                "WHERE task_result_id IS NOT NULL AND event_type NOT IN :types "
                "GROUP BY submission_id, task_result_id"
            ).bindparams(bindparam("types", expanding=True)),
            {"scope": LOG_BYTE_SCOPE_TASK_RESULT, "types": non_counted},
        )
        await connection.execute(
            text(
                "INSERT INTO task_log_byte_totals "
                "(submission_id, scope, scope_key, total_bytes) "
                "SELECT submission_id, :scope, task_id, COALESCE(SUM(message_bytes), 0) "
                "FROM task_log_events "
                "WHERE task_id IS NOT NULL AND event_type NOT IN :types "
                "GROUP BY submission_id, task_id"
            ).bindparams(bindparam("types", expanding=True)),
            {"scope": LOG_BYTE_SCOPE_TASK, "types": non_counted},
        )

    async def _backfill_legacy_submission_versions(self, connection: AsyncConnection) -> None:
        legacy_rows = (
            (
                await connection.execute(
                    text(
                        "SELECT id, miner_hotkey, name, agent_hash, zip_sha256 "
                        ", created_at "
                        "FROM agent_submissions "
                        "WHERE submission_family_id IS NULL "
                        "ORDER BY id"
                    )
                )
            )
            .mappings()
            .all()
        )
        if not legacy_rows:
            return

        used_names = set(
            (
                await connection.execute(text("SELECT normalized_name FROM submission_families"))
            ).scalars()
        )
        used_zip_hashes: set[str] = set()
        existing_zip_hashes = (
            await connection.execute(
                text(
                    "SELECT zip_sha256 FROM agent_submissions "
                    "WHERE zip_sha256 IS NOT NULL AND submission_family_id IS NOT NULL "
                    "ORDER BY id"
                )
            )
        ).scalars()
        used_zip_hashes.update(hash_value for hash_value in existing_zip_hashes if hash_value)
        normalized_candidates = {
            int(row["id"]): self._safe_normalized_name(row["name"]) for row in legacy_rows
        }
        candidate_counts = Counter(
            candidate for candidate in normalized_candidates.values() if candidate is not None
        )
        raw_zip_owner_ids = self._legacy_raw_zip_owner_ids(legacy_rows, used_zip_hashes)

        for row in legacy_rows:
            submission_id = int(row["id"])
            display_name = row["name"]
            normalized_name = self._legacy_normalized_name(
                submission_id=submission_id,
                normalized_candidate=normalized_candidates[submission_id],
                candidate_counts=candidate_counts,
                used_names=used_names,
            )
            canonical_artifact_hash = self._legacy_canonical_artifact_hash(
                submission_id=submission_id,
                agent_hash=row["agent_hash"],
                zip_sha256=row["zip_sha256"],
                raw_zip_owner_ids=raw_zip_owner_ids,
            )
            public_family_id = uuid4().hex
            family_id = (
                await connection.execute(
                    text(
                        "INSERT INTO submission_families "
                        "(public_family_id, owner_hotkey, display_name, normalized_name, "
                        "version_count, created_at, updated_at) "
                        "VALUES (:public_family_id, :owner_hotkey, :display_name, "
                        ":normalized_name, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                        "RETURNING id"
                    ),
                    {
                        "public_family_id": public_family_id,
                        "owner_hotkey": row["miner_hotkey"],
                        "display_name": display_name,
                        "normalized_name": normalized_name,
                    },
                )
            ).scalar_one()
            await connection.execute(
                text(
                    "UPDATE agent_submissions "
                    "SET submission_family_id = :family_id, "
                    "version_number = 1, "
                    "version_label = :version_label, "
                    "canonical_artifact_hash = :canonical_artifact_hash, "
                    "is_latest_version = :is_latest_version "
                    "WHERE id = :submission_id"
                ),
                {
                    "family_id": family_id,
                    "version_label": version_label(1),
                    "canonical_artifact_hash": canonical_artifact_hash,
                    "is_latest_version": True,
                    "submission_id": submission_id,
                },
            )
            await connection.execute(
                text(
                    "UPDATE submission_families "
                    "SET latest_submission_id = :submission_id "
                    "WHERE id = :family_id"
                ),
                {"submission_id": submission_id, "family_id": family_id},
            )

    @staticmethod
    def _safe_normalized_name(display_name: str) -> str | None:
        try:
            return normalize_submission_name(display_name)
        except ValueError:
            return None

    @staticmethod
    def _legacy_normalized_name(
        *,
        submission_id: int,
        normalized_candidate: str | None,
        candidate_counts: Counter[str],
        used_names: set[str],
    ) -> str:
        normalized_name = normalized_candidate
        if (
            normalized_name is None
            or candidate_counts[normalized_name] > 1
            or normalized_name in used_names
        ):
            normalized_name = f"agent-{submission_id}"
        used_names.add(normalized_name)
        return normalized_name

    @staticmethod
    def _legacy_raw_zip_owner_ids(legacy_rows: list[dict], used_zip_hashes: set[str]) -> set[int]:
        owner_ids: set[int] = set()
        owned_zip_hashes = set(used_zip_hashes)
        sorted_rows = sorted(
            legacy_rows,
            key=lambda row: (row["created_at"], int(row["id"])),
        )
        for row in sorted_rows:
            zip_sha256 = row["zip_sha256"]
            if not zip_sha256 or zip_sha256 in owned_zip_hashes:
                continue
            owner_ids.add(int(row["id"]))
            owned_zip_hashes.add(zip_sha256)
        return owner_ids

    @staticmethod
    def _legacy_canonical_artifact_hash(
        *,
        submission_id: int,
        agent_hash: str,
        zip_sha256: str | None,
        raw_zip_owner_ids: set[int],
    ) -> str:
        if not zip_sha256:
            return f"legacy:{submission_id}:{agent_hash}"
        if submission_id in raw_zip_owner_ids:
            return zip_sha256
        return f"legacy-duplicate:{submission_id}:{zip_sha256}"

    async def _backfill_legacy_submission_env_metadata(self, connection: AsyncConnection) -> None:
        await connection.execute(
            text(
                "UPDATE agent_submissions "
                "SET env_confirmed_empty = :confirmed_empty, "
                "env_confirmed_empty_at = CURRENT_TIMESTAMP, "
                "env_locked_at = CURRENT_TIMESTAMP, "
                "env_compatibility_reason = :reason "
                "WHERE raw_status = 'analysis_allowed' "
                "AND env_confirmed_empty = :not_confirmed "
                "AND env_compatibility_reason IS NULL"
            ),
            {
                "confirmed_empty": True,
                "not_confirmed": False,
                "reason": "pre_env_gate_analysis_allowed",
            },
        )
