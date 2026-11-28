from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from codeguard.db.base import Base
from codeguard.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from codeguard.db.models.analysis import Analysis


class Finding(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A deterministic finding produced by one analysis execution."""

    __tablename__ = "findings"

    __table_args__ = (
        UniqueConstraint(
            "analysis_id",
            "fingerprint",
            name="uq_findings_analysis_fingerprint",
        ),
        CheckConstraint(
            "fingerprint ~ '^[0-9a-f]{64}$'",
            name="fingerprint_sha256",
        ),
        CheckConstraint(
            "severity IN ('critical', 'high', 'medium', 'low', 'info')",
            name="severity",
        ),
        CheckConstraint(
            "confidence IN ('low', 'medium', 'high')",
            name="confidence",
        ),
        CheckConstraint(
            "line_number > 0",
            name="line_number_positive",
        ),
        CheckConstraint(
            """
            jsonb_typeof("references") = 'array'
            AND jsonb_array_length("references") > 0
            """,
            name="references_nonempty_array",
        ),
        Index(
            "ix_findings_analysis_id_severity",
            "analysis_id",
            "severity",
        ),
        Index(
            "ix_findings_analysis_id_file_path_line_number",
            "analysis_id",
            "file_path",
            "line_number",
        ),
    )

    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
    )

    fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    rule_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
    )

    confidence: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    line_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    cwe: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    evidence: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    remediation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    references: Mapped[list[str]] = mapped_column(
        "references",
        JSONB,
        nullable=False,
    )

    analysis: Mapped[Analysis] = relationship(
        "Analysis",
        back_populates="findings",
    )
