from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from codeguard.db.base import Base
from codeguard.db.mixins import (
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from codeguard.db.models.pull_request import PullRequest


class Repository(
    UUIDPrimaryKeyMixin,
    CreatedAtMixin,
    UpdatedAtMixin,
    Base,
):
    """Persist GitHub repository identity and current metadata."""

    __tablename__ = "repositories"

    __table_args__ = (
        Index(
            "ix_repositories_owner_login_name",
            "owner_login",
            "name",
        ),
    )

    github_repository_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        unique=True,
    )

    owner_login: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    private: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    html_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    default_branch: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    language: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    github_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    github_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    pull_requests: Mapped[list[PullRequest]] = relationship(
        "PullRequest",
        back_populates="repository",
        passive_deletes="all",
    )
