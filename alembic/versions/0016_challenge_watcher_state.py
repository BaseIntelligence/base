"""Durable Compose challenge-watcher intent table.

Revision ID: 0016_watcher_state
Revises: 0015_submission_obs
Create Date: 2026-07-12 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0016_watcher_state"
down_revision: str | None = "0015_submission_obs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "challenge_watcher_state",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("desired_digest", sa.Text(), nullable=True),
        sa.Column("current_digest", sa.Text(), nullable=True),
        sa.Column("rollback_digest", sa.Text(), nullable=True),
        sa.Column("desired_image", sa.Text(), nullable=True),
        sa.Column("rollback_image", sa.Text(), nullable=True),
        sa.Column("phase", sa.Text(), nullable=False, server_default="idle"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_eligible_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_result", sa.Text(), nullable=True),
        sa.Column("last_health_ok", sa.Boolean(), nullable=True),
        sa.Column("last_version_ok", sa.Boolean(), nullable=True),
        sa.Column("alerted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("project_name", sa.Text(), nullable=True),
        sa.Column(
            "payload",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("slug", name="uq_challenge_watcher_state_slug"),
    )
    op.create_index(
        "ix_challenge_watcher_state_phase",
        "challenge_watcher_state",
        ["phase"],
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        "ix_challenge_watcher_state_phase", table_name="challenge_watcher_state"
    )
    op.drop_table("challenge_watcher_state")
