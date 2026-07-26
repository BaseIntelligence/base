"""BASE-issued single-use attestation nonces for prism-lium constation.

Adds durable storage for mechanism 1 (nonce-bound attestation). Issue/consume
rules live in ``base.compute.attestation_nonce``; this table only persists
issued nonces and their consume timestamps (BASE clocks only).

Revision ID: 0018_attestation_nonces
Revises: 0017_digest_allowlist
Create Date: 2026-07-26 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0018_attestation_nonces"
down_revision: str | None = "0017_digest_allowlist"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "attestation_nonces",
        sa.Column("nonce", sa.Text(), nullable=False),
        sa.Column("work_unit_id", sa.Text(), nullable=False),
        sa.Column("miner_hotkey", sa.Text(), nullable=False),
        sa.Column("pod_id", sa.Text(), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("nonce", name=op.f("pk_attestation_nonces")),
    )
    op.create_index(
        "ix_attestation_nonces_work_unit_id",
        "attestation_nonces",
        ["work_unit_id"],
        unique=False,
    )
    op.create_index(
        "ix_attestation_nonces_miner_hotkey",
        "attestation_nonces",
        ["miner_hotkey"],
        unique=False,
    )
    op.create_index(
        "ix_attestation_nonces_expires_at",
        "attestation_nonces",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        "ix_attestation_nonces_expires_at",
        table_name="attestation_nonces",
    )
    op.drop_index(
        "ix_attestation_nonces_miner_hotkey",
        table_name="attestation_nonces",
    )
    op.drop_index(
        "ix_attestation_nonces_work_unit_id",
        table_name="attestation_nonces",
    )
    op.drop_table("attestation_nonces")
