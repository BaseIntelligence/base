"""Durable raw-weight snapshot, nonce, and epoch sealing tables.

Revision ID: 0013_raw_weight_ingress
Revises: 0012_coordination_protocol_semantics
Create Date: 2026-07-12 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0013_raw_weight_ingress"
down_revision: str | None = "0012_coordination_protocol_semantics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "aggregation_epochs",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="open",
        ),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint("epoch", name="uq_aggregation_epochs_epoch"),
    )
    op.create_index(
        "ix_aggregation_epochs_status",
        "aggregation_epochs",
        ["status"],
    )

    op.create_table(
        "raw_weight_snapshots",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("challenge_slug", sa.Text(), nullable=False),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("protocol_version", sa.Text(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("nonce", sa.Text(), nullable=False),
        sa.Column("payload_digest", sa.Text(), nullable=False),
        sa.Column("canonical_payload", sa.Text(), nullable=False),
        sa.Column(
            "weights",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "is_selected_source",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "challenge_slug",
            "epoch",
            "revision",
            name="uq_raw_weight_snapshots_challenge_epoch_revision",
        ),
        sa.UniqueConstraint(
            "challenge_slug",
            "nonce",
            name="uq_raw_weight_snapshots_challenge_nonce",
        ),
    )
    op.create_index(
        "ix_raw_weight_snapshots_challenge_epoch",
        "raw_weight_snapshots",
        ["challenge_slug", "epoch"],
    )
    op.create_index(
        "ix_raw_weight_snapshots_selected",
        "raw_weight_snapshots",
        ["challenge_slug", "epoch", "is_selected_source"],
    )
    op.create_index(
        "ix_raw_weight_snapshots_payload_digest",
        "raw_weight_snapshots",
        ["payload_digest"],
    )
    op.create_index(
        "ix_raw_weight_snapshots_received_at",
        "raw_weight_snapshots",
        ["received_at"],
    )

    op.create_table(
        "raw_weight_nonces",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("challenge_slug", sa.Text(), nullable=False),
        sa.Column("nonce", sa.Text(), nullable=False),
        sa.Column("body_hash", sa.Text(), nullable=False),
        sa.Column("payload_digest", sa.Text(), nullable=False),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column(
            "snapshot_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("raw_weight_snapshots.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "challenge_slug",
            "nonce",
            name="uq_raw_weight_nonces_challenge_nonce",
        ),
    )
    op.create_index(
        "ix_raw_weight_nonces_created_at",
        "raw_weight_nonces",
        ["created_at"],
    )
    op.create_index(
        "ix_raw_weight_nonces_challenge",
        "raw_weight_nonces",
        ["challenge_slug"],
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index("ix_raw_weight_nonces_challenge", table_name="raw_weight_nonces")
    op.drop_index("ix_raw_weight_nonces_created_at", table_name="raw_weight_nonces")
    op.drop_table("raw_weight_nonces")

    op.drop_index(
        "ix_raw_weight_snapshots_received_at", table_name="raw_weight_snapshots"
    )
    op.drop_index(
        "ix_raw_weight_snapshots_payload_digest", table_name="raw_weight_snapshots"
    )
    op.drop_index("ix_raw_weight_snapshots_selected", table_name="raw_weight_snapshots")
    op.drop_index(
        "ix_raw_weight_snapshots_challenge_epoch", table_name="raw_weight_snapshots"
    )
    op.drop_table("raw_weight_snapshots")

    op.drop_index("ix_aggregation_epochs_status", table_name="aggregation_epochs")
    op.drop_table("aggregation_epochs")
