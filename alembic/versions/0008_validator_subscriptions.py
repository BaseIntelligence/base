"""Add per-validator challenge subscriptions.

Adds the ``validators.subscriptions`` JSON-list column (NOT NULL, server default
``[]``) so a validator can opt in to the challenges it validates. Mirrors the
``capabilities`` JSON-list column pattern; empty/absent means "all challenges"
(back-compat).

Revision ID: 0008_validator_subscriptions
Revises: 0007_harden_validator_registry
Create Date: 2026-06-28 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008_validator_subscriptions"
down_revision: str | None = "0007_harden_validator_registry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    with op.batch_alter_table("validators") as batch_op:
        batch_op.add_column(
            sa.Column(
                "subscriptions",
                sa.JSON(),
                server_default="[]",
                nullable=False,
            )
        )


def downgrade() -> None:
    """Revert the migration."""

    with op.batch_alter_table("validators") as batch_op:
        batch_op.drop_column("subscriptions")
