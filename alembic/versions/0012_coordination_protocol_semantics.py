"""Durable coordination protocol fields for validators and results.

Adds monotonic heartbeat sequencing on ``validators``, durable payload digests
on ``work_assignments`` / ``work_results``, and unique-result ownership so
exact retries remain idempotent while conflicting terminal mutations fail
closed.

Revision ID: 0012_coordination_protocol_semantics
Revises: 0011_drop_llm_usage_records
Create Date: 2026-07-12 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0012_coord_protocol"
down_revision: str | None = "0011_drop_llm_usage_records"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.add_column(
        "validators",
        sa.Column(
            "last_heartbeat_sequence",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "validators",
        sa.Column(
            "last_heartbeat_payload_digest",
            sa.Text(),
            nullable=True,
        ),
    )

    op.add_column(
        "work_assignments",
        sa.Column(
            "revision",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
    )
    op.add_column(
        "work_assignments",
        sa.Column(
            "payload_digest",
            sa.Text(),
            nullable=True,
        ),
    )

    op.add_column(
        "work_results",
        sa.Column(
            "result_digest",
            sa.Text(),
            nullable=True,
        ),
    )
    op.add_column(
        "work_results",
        sa.Column(
            "checkpoint_ref",
            sa.Text(),
            nullable=True,
        ),
    )
    op.add_column(
        "work_results",
        sa.Column(
            "proof",
            sa.JSON(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )
    # One terminal result row per assignment (exact retries re-use it).
    # SQLite requires batch_alter_table to add constraints.
    with op.batch_alter_table("work_results") as batch_op:
        batch_op.create_unique_constraint(
            "uq_work_results_assignment_id",
            ["assignment_id"],
        )


def downgrade() -> None:
    """Revert the migration."""

    with op.batch_alter_table("work_results") as batch_op:
        batch_op.drop_constraint(
            "uq_work_results_assignment_id",
            type_="unique",
        )
    op.drop_column("work_results", "proof")
    op.drop_column("work_results", "checkpoint_ref")
    op.drop_column("work_results", "result_digest")

    op.drop_column("work_assignments", "payload_digest")
    op.drop_column("work_assignments", "revision")

    op.drop_column("validators", "last_heartbeat_payload_digest")
    op.drop_column("validators", "last_heartbeat_sequence")
