"""Immutable final vectors and aggregation epoch provenance.

Revision ID: 0014_final_weight_vectors
Revises: 0013_raw_weight_ingress
Create Date: 2026-07-12 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0014_final_weight_vectors"
down_revision: str | None = "0013_raw_weight_ingress"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "final_weight_vectors",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("protocol_version", sa.Text(), nullable=False),
        sa.Column("netuid", sa.Integer(), nullable=False),
        sa.Column("chain_endpoint", sa.Text(), nullable=False, server_default=""),
        sa.Column("vector_digest", sa.Text(), nullable=False),
        sa.Column(
            "uids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "weights",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "hotkey_weights",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("chain_domain_bytes", sa.Text(), nullable=False),
        sa.Column("canonical_payload", sa.Text(), nullable=False),
        sa.Column(
            "source_snapshot_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "source_snapshot_digests",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "source_outcomes",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("emission_policy_version", sa.Text(), nullable=False),
        sa.Column(
            "emission_shares",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("burn_policy_version", sa.Text(), nullable=False),
        sa.Column("mapping_policy_version", sa.Text(), nullable=False),
        sa.Column("metagraph_block", sa.BigInteger(), nullable=True),
        sa.Column("metagraph_hash", sa.Text(), nullable=True),
        sa.Column(
            "metagraph_identity",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "hotkey_to_uid",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metagraph_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("vector_digest", name="uq_final_weight_vectors_digest"),
        sa.UniqueConstraint("epoch", name="uq_final_weight_vectors_epoch"),
    )
    op.create_index(
        "ix_final_weight_vectors_computed_at",
        "final_weight_vectors",
        ["computed_at"],
    )
    op.create_index(
        "ix_final_weight_vectors_expires_at",
        "final_weight_vectors",
        ["expires_at"],
    )

    op.add_column(
        "aggregation_epochs",
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column(
            "expected_challenges",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column("emission_policy_version", sa.Text(), nullable=True),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column(
            "emission_shares",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column("source_outcome_policy_version", sa.Text(), nullable=True),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column("burn_policy_version", sa.Text(), nullable=True),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column("mapping_policy_version", sa.Text(), nullable=True),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column("outcome_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column(
            "vector_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("final_weight_vectors.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "aggregation_epochs",
        sa.Column(
            "source_outcomes",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_column("aggregation_epochs", "source_outcomes")
    op.drop_column("aggregation_epochs", "vector_id")
    op.drop_column("aggregation_epochs", "outcome_reason")
    op.drop_column("aggregation_epochs", "mapping_policy_version")
    op.drop_column("aggregation_epochs", "burn_policy_version")
    op.drop_column("aggregation_epochs", "source_outcome_policy_version")
    op.drop_column("aggregation_epochs", "emission_shares")
    op.drop_column("aggregation_epochs", "emission_policy_version")
    op.drop_column("aggregation_epochs", "expected_challenges")
    op.drop_column("aggregation_epochs", "deadline_at")

    op.drop_index(
        "ix_final_weight_vectors_expires_at", table_name="final_weight_vectors"
    )
    op.drop_index(
        "ix_final_weight_vectors_computed_at", table_name="final_weight_vectors"
    )
    op.drop_table("final_weight_vectors")
