"""Create the work_results table for validator-reported results.

Revision ID: 0006_create_work_results
Revises: 0005_create_work_assignments
Create Date: 2026-06-27 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_create_work_results"
down_revision: str | None = "0005_create_work_assignments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "work_results",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("assignment_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("challenge_slug", sa.Text(), nullable=False),
        sa.Column("work_unit_id", sa.Text(), nullable=False),
        sa.Column("submission_ref", sa.Text(), nullable=False),
        sa.Column("validator_hotkey", sa.Text(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("payload", sa.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_work_results")),
    )
    op.create_index(
        "ix_work_results_assignment_id",
        "work_results",
        ["assignment_id"],
        unique=False,
    )
    op.create_index(
        "ix_work_results_challenge_slug",
        "work_results",
        ["challenge_slug"],
        unique=False,
    )
    op.create_index(
        "ix_work_results_validator_hotkey",
        "work_results",
        ["validator_hotkey"],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index("ix_work_results_validator_hotkey", table_name="work_results")
    op.drop_index("ix_work_results_challenge_slug", table_name="work_results")
    op.drop_index("ix_work_results_assignment_id", table_name="work_results")
    op.drop_table("work_results")
