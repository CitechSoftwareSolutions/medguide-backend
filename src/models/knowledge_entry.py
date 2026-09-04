"""SQLAlchemy ORM model for a medical knowledge entry."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from src.enums import KnowledgeType
from src.models.base import Base


class KnowledgeEntry(Base):
    """A persisted medical knowledge item."""

    __tablename__ = "knowledge_entries"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    title: Mapped[str] = mapped_column(String(200), unique=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    knowledge_type: Mapped[KnowledgeType] = mapped_column(
        Enum(
            KnowledgeType,
            name="knowledge_type",
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(50)), nullable=False, server_default=text("'{}'::varchar[]")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
