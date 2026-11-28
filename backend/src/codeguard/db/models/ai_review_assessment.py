from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from codeguard.db.base import Base
from codeguard.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from codeguard.db.models.analysis import Analysis


class AIReviewAssessment(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """An AI-generated candidate and its independent evidence assessment."""

    __tablename__ = "ai_review_assessments"

    __table_args__ = (
        UniqueConstraint(
            "analysis_id",
            "candidate_index",
            name="uq_analysis_candidate_index",
        ),
        CheckConstraint(
            "candidate_index BETWEEN 0 AND 49",
            name="candidate_index_range",
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
            "evidence_match_status IN ('matched', 'mismatched', 'unverifiable')",
            name="evidence_match_status",
        ),
        CheckConstraint(
            """
            evidence_match_reason IN (
                'exact_match',
                'content_mismatch',
                'location_invalid',
                'location_unverifiable'
            )
            """,
            name="evidence_match_reason",
        ),
        CheckConstraint(
            """
            (
                evidence_match_status = 'matched'
                AND evidence_match_reason = 'exact_match'
            )
            OR (
                evidence_match_status = 'mismatched'
                AND evidence_match_reason IN (
                    'content_mismatch',
                    'location_invalid'
                )
            )
            OR (
                evidence_match_status = 'unverifiable'
                AND evidence_match_reason = 'location_unverifiable'
            )
            """,
            name="evidence_match_consistency",
        ),
        CheckConstraint(
            """
            location_reason IS NULL
            OR location_reason IN (
                'location_valid',
                'file_not_found',
                'source_retrieval_failed',
                'source_omitted',
                'incomplete_snapshot',
                'unsupported_file_type',
                'line_out_of_range',
                'commit_mismatch'
            )
            """,
            name="location_reason",
        ),
        Index(
            "ix_ai_review_assessments_analysis_id_evidence_match_status",
            "analysis_id",
            "evidence_match_status",
        ),
    )

    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
    )

    candidate_index: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(200),
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

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    line_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    problem: Mapped[str] = mapped_column(
        String(3000),
        nullable=False,
    )

    evidence: Mapped[str] = mapped_column(
        String(4000),
        nullable=False,
    )

    why_it_matters: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
    )

    suggested_fix: Mapped[str] = mapped_column(
        String(3000),
        nullable=False,
    )

    recommended_validation: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
    )

    evidence_match_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    evidence_match_reason: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    location_reason: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    analysis: Mapped[Analysis] = relationship(
        "Analysis",
        back_populates="ai_review_assessments",
    )
