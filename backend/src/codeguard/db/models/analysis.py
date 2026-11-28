from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from codeguard.db.base import Base
from codeguard.db.mixins import (
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from codeguard.db.models.ai_review_assessment import AIReviewAssessment
    from codeguard.db.models.finding import Finding
    from codeguard.db.models.pull_request import PullRequest


class Analysis(
    UUIDPrimaryKeyMixin,
    CreatedAtMixin,
    UpdatedAtMixin,
    Base,
):
    """One historical execution of a pull-request analysis."""

    __tablename__ = "analyses"

    __table_args__ = (
        CheckConstraint(
            "status IN "
            "('queued', 'running', 'succeeded', 'partial', 'failed', 'cancelled')",
            name="status",
        ),
        CheckConstraint(
            "files_analyzed >= 0",
            name="files_analyzed_nonnegative",
        ),
        CheckConstraint(
            "length(head_sha) BETWEEN 1 AND 64",
            name="head_sha_length",
        ),
        CheckConstraint(
            "length(base_sha) BETWEEN 1 AND 64",
            name="base_sha_length",
        ),
        CheckConstraint(
            """
            (
                status = 'queued'
                AND started_at IS NULL
                AND completed_at IS NULL
            )
            OR (
                status = 'running'
                AND started_at IS NOT NULL
                AND completed_at IS NULL
            )
            OR (
                status IN ('succeeded', 'partial')
                AND started_at IS NOT NULL
                AND completed_at IS NOT NULL
            )
            OR (
                status IN ('failed', 'cancelled')
                AND completed_at IS NOT NULL
            )
            """,
            name="lifecycle_timestamps",
        ),
        CheckConstraint(
            """
            started_at IS NULL
            OR completed_at IS NULL
            OR completed_at >= started_at
            """,
            name="completed_after_started",
        ),
        CheckConstraint(
            "jsonb_typeof(files_with_unavailable_mapping) = 'array'",
            name="unavailable_mapping_array",
        ),
        CheckConstraint(
            """
            repository_context IS NULL
            OR jsonb_typeof(repository_context) = 'object'
            """,
            name="repository_context_object",
        ),
        CheckConstraint(
            """
            ai_execution_telemetry IS NULL
            OR jsonb_typeof(ai_execution_telemetry) = 'object'
            """,
            name="ai_execution_telemetry_object",
        ),
        Index(
            "ix_analyses_pull_request_id_created_at",
            "pull_request_id",
            text("created_at DESC"),
        ),
        Index(
            "ix_analyses_pull_request_id_head_sha",
            "pull_request_id",
            "head_sha",
        ),
    )

    pull_request_id: Mapped[UUID] = mapped_column(
        ForeignKey("pull_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )

    findings: Mapped[list[Finding]] = relationship(
        "Finding",
        back_populates="analysis",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    head_sha: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    base_sha: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    repository_owner_login: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    repository_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    pull_request_title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    ai_review_assessments: Mapped[list[AIReviewAssessment]] = relationship(
        "AIReviewAssessment",
        back_populates="analysis",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    pull_request_html_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    files_analyzed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    files_with_unavailable_mapping: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    repository_context: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    ai_review_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    ai_execution_telemetry: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    pull_request: Mapped[PullRequest] = relationship(
        "PullRequest",
        back_populates="analyses",
    )
