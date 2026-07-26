"""BASE-produced image digest allowlist + revocation denylists.

Adds durable storage for mechanism 4 (digest allowlist) of prism-lium image
attestation. Lookup rules live in ``base.compute.digest_allowlist``; these
tables only persist registered bindings and denylist entries.

Revision ID: 0017_digest_allowlist
Revises: 0016_watcher_state
Create Date: 2026-07-26 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0017_digest_allowlist"
down_revision: str | None = "0016_watcher_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_table(
        "image_digest_allowlist",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("commit_sha", sa.Text(), nullable=False),
        sa.Column("tree_sha", sa.Text(), nullable=False),
        sa.Column("variant", sa.Text(), nullable=False),
        sa.Column("digest", sa.Text(), nullable=False),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_image_digest_allowlist")),
        sa.UniqueConstraint("digest", name="uq_image_digest_allowlist_digest"),
        sa.UniqueConstraint(
            "commit_sha",
            "tree_sha",
            "variant",
            name="uq_image_digest_allowlist_commit_tree_variant",
        ),
    )
    op.create_index(
        "ix_image_digest_allowlist_commit_sha",
        "image_digest_allowlist",
        ["commit_sha"],
        unique=False,
    )
    op.create_index(
        "ix_image_digest_allowlist_variant",
        "image_digest_allowlist",
        ["variant"],
        unique=False,
    )

    op.create_table(
        "denied_image_digests",
        sa.Column("digest", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("digest", name=op.f("pk_denied_image_digests")),
    )

    op.create_table(
        "denied_image_commits",
        sa.Column("commit_sha", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("commit_sha", name=op.f("pk_denied_image_commits")),
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_table("denied_image_commits")
    op.drop_table("denied_image_digests")
    op.drop_index(
        "ix_image_digest_allowlist_variant",
        table_name="image_digest_allowlist",
    )
    op.drop_index(
        "ix_image_digest_allowlist_commit_sha",
        table_name="image_digest_allowlist",
    )
    op.drop_table("image_digest_allowlist")
