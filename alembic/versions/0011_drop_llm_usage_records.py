"""Drop removed LLM gateway usage metering table.

Revision ID: 0011_drop_llm_usage_records
Revises: 0010_create_worker_assignments
Create Date: 2026-07-12 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0011_drop_llm_usage_records"
down_revision: str | None = "0010_create_worker_assignments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop legacy gateway-only usage tables without converting them into auth state."""

    # Best-effort cleanup for databases that already applied 0004. Fresh installs
    # never create the table once the ORM model is removed; IF EXISTS keeps this
    # revision idempotent and restart-safe.
    op.execute("DROP INDEX IF EXISTS ix_llm_usage_records_created_at")
    op.execute("DROP INDEX IF EXISTS ix_llm_usage_records_validator_assignment")
    op.execute("DROP TABLE IF EXISTS llm_usage_records")


def downgrade() -> None:
    """Gateway removal is forward-only; do not recreate provider usage state."""

    raise NotImplementedError(
        "Cannot restore llm_usage_records: the LLM gateway has been removed."
    )
