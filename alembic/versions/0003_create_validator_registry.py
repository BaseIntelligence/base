"""Create validator coordination-plane tables.

Revision ID: 0003_create_validator_registry
Revises: 0002_create_miner_request_nonces
Create Date: 2026-06-27 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_create_validator_registry"
down_revision: str | None = "0002_create_miner_request_nonces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

validator_status = sa.Enum(
    "online",
    "offline",
    "unknown",
    name="validator_status",
    native_enum=False,
)

validator_health_event_type = sa.Enum(
    "registered",
    "online",
    "offline",
    "crash_detected",
    name="validator_health_event_type",
    native_enum=False,
)


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "validators",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("hotkey", sa.Text(), nullable=False),
        sa.Column("uid", sa.Integer(), nullable=True),
        sa.Column("status", validator_status, server_default="unknown", nullable=False),
        sa.Column("capabilities", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_meta", sa.JSON(), server_default="{}", nullable=False),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_validators")),
        sa.UniqueConstraint("hotkey", name=op.f("uq_validators_hotkey")),
    )
    op.create_index("ix_validators_status", "validators", ["status"], unique=False)
    op.create_index(
        "ix_validators_last_heartbeat_at",
        "validators",
        ["last_heartbeat_at"],
        unique=False,
    )

    op.create_table(
        "validator_health_events",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("validator_hotkey", sa.Text(), nullable=False),
        sa.Column("event", validator_health_event_type, nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_validator_health_events")),
    )
    op.create_index(
        "ix_validator_health_events_hotkey_created",
        "validator_health_events",
        ["validator_hotkey", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_validator_health_events_event",
        "validator_health_events",
        ["event"],
        unique=False,
    )

    op.create_table(
        "validator_request_nonces",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("hotkey", sa.Text(), nullable=False),
        sa.Column("nonce", sa.Text(), nullable=False),
        sa.Column("body_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_validator_request_nonces")),
        sa.UniqueConstraint(
            "hotkey", "nonce", name="uq_validator_request_nonces_hotkey_nonce"
        ),
    )
    op.create_index(
        "ix_validator_request_nonces_created_at",
        "validator_request_nonces",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_validator_request_nonces_hotkey",
        "validator_request_nonces",
        ["hotkey"],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        "ix_validator_request_nonces_hotkey",
        table_name="validator_request_nonces",
    )
    op.drop_index(
        "ix_validator_request_nonces_created_at",
        table_name="validator_request_nonces",
    )
    op.drop_table("validator_request_nonces")
    op.drop_index(
        "ix_validator_health_events_event",
        table_name="validator_health_events",
    )
    op.drop_index(
        "ix_validator_health_events_hotkey_created",
        table_name="validator_health_events",
    )
    op.drop_table("validator_health_events")
    op.drop_index("ix_validators_last_heartbeat_at", table_name="validators")
    op.drop_index("ix_validators_status", table_name="validators")
    op.drop_table("validators")
