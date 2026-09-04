"""Create knowledge entries table.

Revision ID: 20260904_0001
Revises:
Create Date: 2026-09-04 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260904_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

knowledge_type = postgresql.ENUM(
    "condition",
    "medication",
    "procedure",
    "symptom",
    name="knowledge_type",
    create_type=False,
)


def upgrade() -> None:
    """Create the PostgreSQL enum and knowledge entry table."""
    bind = op.get_bind()
    knowledge_type.create(bind, checkfirst=True)
    op.create_table(
        "knowledge_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("knowledge_type", knowledge_type, nullable=False),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.String(length=50)),
            server_default=sa.text("'{}'::varchar[]"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("title", name="uq_knowledge_entries_title"),
    )


def downgrade() -> None:
    """Drop the knowledge entry table and its PostgreSQL enum."""
    bind = op.get_bind()
    op.drop_table("knowledge_entries")
    knowledge_type.drop(bind, checkfirst=True)
