"""Create miner-funded GPU worker registry tables.

Adds the worker-plane storage (architecture.md sec 3.3): ``worker_registrations``
(the enrolled workers + their ``pending`` -> ``active`` -> ``stale`` -> ``retired``
lifecycle), ``worker_faults`` (reconciliation/audit fault attribution surfaced in
the fleet view), and ``worker_request_nonces`` (replay protection for the miner
binding nonce and the signed request-envelope nonce). No pre-existing table is
altered.

Revision ID: 0009_create_worker_registry
Revises: 0008_validator_subscriptions
Create Date: 2026-07-06 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009_create_worker_registry"
down_revision: str | None = "0008_validator_subscriptions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

worker_status = sa.Enum(
    "pending",
    "active",
    "stale",
    "retired",
    name="worker_status",
    native_enum=False,
)


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "worker_registrations",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column("worker_pubkey", sa.Text(), nullable=False),
        sa.Column("miner_hotkey", sa.Text(), nullable=False),
        sa.Column("binding_signature", sa.Text(), nullable=False),
        sa.Column("binding_nonce", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_instance_ref", sa.Text(), nullable=True),
        sa.Column("capabilities", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("status", worker_status, server_default="pending", nullable=False),
        sa.Column("last_seen_meta", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_worker_registrations")),
        sa.UniqueConstraint("worker_id", name="uq_worker_registrations_worker_id"),
        sa.UniqueConstraint(
            "worker_pubkey", name="uq_worker_registrations_worker_pubkey"
        ),
    )
    op.create_index(
        "ix_worker_registrations_status",
        "worker_registrations",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_worker_registrations_miner_hotkey",
        "worker_registrations",
        ["miner_hotkey"],
        unique=False,
    )
    op.create_index(
        "ix_worker_registrations_last_heartbeat_at",
        "worker_registrations",
        ["last_heartbeat_at"],
        unique=False,
    )
    op.create_index(
        "ix_worker_registrations_status_miner_hotkey",
        "worker_registrations",
        ["status", "miner_hotkey"],
        unique=False,
    )

    op.create_table(
        "worker_faults",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column("work_unit_id", sa.Text(), nullable=False),
        sa.Column("challenge_slug", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_worker_faults")),
    )
    op.create_index(
        "ix_worker_faults_worker_id",
        "worker_faults",
        ["worker_id"],
        unique=False,
    )
    op.create_index(
        "ix_worker_faults_work_unit_id",
        "worker_faults",
        ["work_unit_id"],
        unique=False,
    )

    op.create_table(
        "worker_request_nonces",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("hotkey", sa.Text(), nullable=False),
        sa.Column("nonce", sa.Text(), nullable=False),
        sa.Column("body_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_worker_request_nonces")),
        sa.UniqueConstraint(
            "hotkey", "nonce", name="uq_worker_request_nonces_hotkey_nonce"
        ),
    )
    op.create_index(
        "ix_worker_request_nonces_created_at",
        "worker_request_nonces",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_worker_request_nonces_hotkey",
        "worker_request_nonces",
        ["hotkey"],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        "ix_worker_request_nonces_hotkey",
        table_name="worker_request_nonces",
    )
    op.drop_index(
        "ix_worker_request_nonces_created_at",
        table_name="worker_request_nonces",
    )
    op.drop_table("worker_request_nonces")

    op.drop_index("ix_worker_faults_work_unit_id", table_name="worker_faults")
    op.drop_index("ix_worker_faults_worker_id", table_name="worker_faults")
    op.drop_table("worker_faults")

    op.drop_index(
        "ix_worker_registrations_status_miner_hotkey",
        table_name="worker_registrations",
    )
    op.drop_index(
        "ix_worker_registrations_last_heartbeat_at",
        table_name="worker_registrations",
    )
    op.drop_index(
        "ix_worker_registrations_miner_hotkey",
        table_name="worker_registrations",
    )
    op.drop_index(
        "ix_worker_registrations_status",
        table_name="worker_registrations",
    )
    op.drop_table("worker_registrations")
