from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from codeguard.db.base import Base
from codeguard.db.mixins import (
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from codeguard.db.models.analysis import Analysis
    from codeguard.db.models.repository import Repository


class PullRequest(
    UUIDPrimaryKeyMixin,
    CreatedAtMixin,
    UpdatedAtMixin,
    Base,
):
    """Persist current GitHub pull-request metadata."""

    __tablename__ = "pull_requests"

    __table_args__ = (
        UniqueConstraint(
            "repository_id",
            "number",
            name="uq_pull_requests_repository_number",
        ),
        CheckConstraint(
            "number > 0",
            name="positive_number",
        ),
        CheckConstraint(
            "state IN ('open', 'closed')",
            name="valid_state",
        ),
        CheckConstraint(
            "length(head_sha) BETWEEN 1 AND 64",
            name="valid_head_sha",
        ),
        CheckConstraint(
            "length(base_sha) BETWEEN 1 AND 64",
            name="valid_base_sha",
        ),
        Index(
            "ix_pull_requests_repository_state_updated",
            "repository_id",
            "state",
            text("github_updated_at DESC"),
        ),
    )

    repository_id: Mapped[UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="RESTRICT"),
        nullable=False,
    )

    analyses: Mapped[list[Analysis]] = relationship(
        "Analysis",
        back_populates="pull_request",
        passive_deletes="all",
    )

    number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    body: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    html_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    is_draft: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )

    is_merged: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    head_ref: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    head_sha: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    base_ref: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    base_sha: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    github_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    github_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    repository: Mapped[Repository] = relationship(
        "Repository",
        back_populates="pull_requests",
    )
