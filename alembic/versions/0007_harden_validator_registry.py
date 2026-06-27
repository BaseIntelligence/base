"""Harden the validator coordination registry.

Adds a monotonic ``seq`` ordering column to ``validator_health_events`` (and
folds it into the per-hotkey audit index), indexes ``validators.registered_at``,
and makes ``validators.version`` non-null with a server default.

Revision ID: 0007_harden_validator_registry
Revises: 0006_create_work_results
Create Date: 2026-06-27 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_harden_validator_registry"
down_revision: str | None = "0006_create_work_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEFAULT_VALIDATOR_VERSION = "unknown"


def upgrade() -> None:
    """Apply the migration."""

    # validator_health_events: monotonic ordering column + audit index.
    op.add_column(
        "validator_health_events",
        sa.Column("seq", sa.BigInteger(), server_default="0", nullable=False),
    )
    op.drop_index(
        "ix_validator_health_events_hotkey_created",
        table_name="validator_health_events",
    )
    op.create_index(
        "ix_validator_health_events_hotkey_created",
        "validator_health_events",
        ["validator_hotkey", "created_at", "seq"],
        unique=False,
    )

    # validators: index registered_at + make version non-null with a default.
    op.create_index(
        "ix_validators_registered_at",
        "validators",
        ["registered_at"],
        unique=False,
    )
    op.execute(
        sa.text(
            "UPDATE validators SET version = :default WHERE version IS NULL"
        ).bindparams(default=DEFAULT_VALIDATOR_VERSION)
    )
    with op.batch_alter_table("validators") as batch_op:
        batch_op.alter_column(
            "version",
            existing_type=sa.Text(),
            nullable=False,
            server_default=DEFAULT_VALIDATOR_VERSION,
        )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index("ix_validators_registered_at", table_name="validators")
    with op.batch_alter_table("validators") as batch_op:
        batch_op.alter_column(
            "version",
            existing_type=sa.Text(),
            nullable=True,
            server_default=None,
        )

    op.drop_index(
        "ix_validator_health_events_hotkey_created",
        table_name="validator_health_events",
    )
    with op.batch_alter_table("validator_health_events") as batch_op:
        batch_op.drop_column("seq")
    op.create_index(
        "ix_validator_health_events_hotkey_created",
        "validator_health_events",
        ["validator_hotkey", "created_at"],
        unique=False,
    )
