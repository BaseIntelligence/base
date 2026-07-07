"""Create the miner-funded worker assignment replica table.

Adds ``worker_assignments`` (architecture.md sec 3.3): one row PER (gpu work
unit, worker) so a unit can be replicated to multiple distinct-owner workers
(R=2). Carries the worker identity the pull/post routes authenticate against,
the owner hotkey used for anti-collusion accounting, and the reported result +
``ExecutionProof`` (with the extracted ``manifest_sha256`` used for
reconciliation). No pre-existing table is altered.

Revision ID: 0010_create_worker_assignments
Revises: 0009_create_worker_registry
Create Date: 2026-07-06 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010_create_worker_assignments"
down_revision: str | None = "0009_create_worker_registry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

work_assignment_status = sa.Enum(
    "pending",
    "assigned",
    "running",
    "completed",
    "failed",
    "disputed",
    name="work_assignment_status",
    native_enum=False,
)


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "worker_assignments",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("challenge_slug", sa.Text(), nullable=False),
        sa.Column("work_unit_id", sa.Text(), nullable=False),
        sa.Column("submission_ref", sa.Text(), nullable=False),
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column("worker_pubkey", sa.Text(), nullable=False),
        sa.Column("miner_hotkey", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "required_capability",
            sa.Text(),
            server_default="gpu",
            nullable=False,
        ),
        sa.Column(
            "status",
            work_assignment_status,
            server_default="pending",
            nullable=False,
        ),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_progress_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checkpoint_ref", sa.Text(), nullable=True),
        sa.Column("result_success", sa.Boolean(), nullable=True),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("manifest_sha256", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_worker_assignments")),
        sa.UniqueConstraint(
            "work_unit_id",
            "worker_id",
            name="uq_worker_assignments_work_unit_worker",
        ),
    )
    op.create_index(
        "ix_worker_assignments_work_unit_id",
        "worker_assignments",
        ["work_unit_id"],
        unique=False,
    )
    op.create_index(
        "ix_worker_assignments_status",
        "worker_assignments",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_worker_assignments_worker_pubkey",
        "worker_assignments",
        ["worker_pubkey"],
        unique=False,
    )
    op.create_index(
        "ix_worker_assignments_status_worker_pubkey",
        "worker_assignments",
        ["status", "worker_pubkey"],
        unique=False,
    )
    op.create_index(
        "ix_worker_assignments_status_deadline",
        "worker_assignments",
        ["status", "deadline_at"],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        "ix_worker_assignments_status_deadline",
        table_name="worker_assignments",
    )
    op.drop_index(
        "ix_worker_assignments_status_worker_pubkey",
        table_name="worker_assignments",
    )
    op.drop_index(
        "ix_worker_assignments_worker_pubkey",
        table_name="worker_assignments",
    )
    op.drop_index(
        "ix_worker_assignments_status",
        table_name="worker_assignments",
    )
    op.drop_index(
        "ix_worker_assignments_work_unit_id",
        table_name="worker_assignments",
    )
    op.drop_table("worker_assignments")
