"""Add sealed_manifest_hashes to image_digest_allowlist.

Stores the build-time sealed surface (path → 64-hex SHA-256) alongside each
BASE-produced digest so production can supply prism's non-empty
``expected_sealed_manifest_hashes``. Application registration rejects empty
maps; the column uses a non-null JSON server default of ``{}`` so existing
rows migrate without rewrite. Legacy empty maps cannot be re-registered and
fail closed if loaded into DigestRecord until backfilled.

Revision ID: 0019_allowlist_sealed_hashes
Revises: 0018_attestation_nonces
Create Date: 2026-07-27 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0019_allowlist_sealed_hashes"
down_revision: str | None = "0018_attestation_nonces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.add_column(
        "image_digest_allowlist",
        sa.Column(
            "sealed_manifest_hashes",
            sa.JSON(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_column("image_digest_allowlist", "sealed_manifest_hashes")
