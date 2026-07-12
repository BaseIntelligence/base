"""Non-authoritative validator chain-submission observations.

Revision ID: 0015_validator_submission_observations
Revises: 0014_final_weight_vectors
Create Date: 2026-07-12 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0015_submission_obs"
down_revision: str | None = "0014_final_weight_vectors"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "validator_submission_observations",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("validator_hotkey", sa.Text(), nullable=False),
        sa.Column("vector_id", sa.Text(), nullable=False),
        sa.Column("vector_digest", sa.Text(), nullable=False),
        sa.Column("netuid", sa.Integer(), nullable=False),
        sa.Column("chain_endpoint", sa.Text(), nullable=False, server_default=""),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "validator_hotkey",
            "vector_id",
            "vector_digest",
            "outcome",
            "attempt",
            name="uq_validator_submission_observation_identity",
        ),
    )
    op.create_index(
        "ix_validator_submission_observations_vector",
        "validator_submission_observations",
        ["vector_id"],
    )
    op.create_index(
        "ix_validator_submission_observations_hotkey",
        "validator_submission_observations",
        ["validator_hotkey"],
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        "ix_validator_submission_observations_hotkey",
        table_name="validator_submission_observations",
    )
    op.drop_index(
        "ix_validator_submission_observations_vector",
        table_name="validator_submission_observations",
    )
    op.drop_table("validator_submission_observations")
