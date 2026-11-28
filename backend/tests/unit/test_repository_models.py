from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import configure_mappers

from codeguard.db.base import Base
from codeguard.db.models import PullRequest, Repository


def test_repository_and_pull_request_tables_registered() -> None:
    assert "repositories" in Base.metadata.tables
    assert "pull_requests" in Base.metadata.tables


def test_repository_primary_key_and_github_identity() -> None:
    table = Repository.__table__

    assert isinstance(table.c.id.type, PostgreSQLUUID)
    assert table.c.id.primary_key
    assert table.c.id.default is not None

    assert table.c.github_repository_id.unique is True
    assert table.c.github_repository_id.nullable is False


def test_repository_owner_name_index_is_not_unique() -> None:
    indexes = {index.name: index for index in Repository.__table__.indexes}

    index = indexes["ix_repositories_owner_login_name"]

    assert [column.name for column in index.columns] == [
        "owner_login",
        "name",
    ]
    assert index.unique is False


def test_pull_request_repository_number_is_unique() -> None:
    constraints = [
        constraint
        for constraint in PullRequest.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    assert any(
        set(constraint.columns.keys()) == {"repository_id", "number"}
        for constraint in constraints
    )


def test_pull_request_check_constraints_exist() -> None:
    constraints = [
        constraint
        for constraint in PullRequest.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    ]

    names = {constraint.name for constraint in constraints}

    assert "ck_pull_requests_positive_number" in names
    assert "ck_pull_requests_valid_state" in names
    assert "ck_pull_requests_valid_head_sha" in names
    assert "ck_pull_requests_valid_base_sha" in names


def test_pull_request_foreign_key_restricts_repository_deletion() -> None:
    foreign_keys = [
        constraint
        for constraint in PullRequest.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]

    assert len(foreign_keys) == 1
    assert foreign_keys[0].ondelete == "RESTRICT"


def test_repository_relationships_are_configured() -> None:
    configure_mappers()

    assert Repository.pull_requests.property.back_populates == "repository"
    assert PullRequest.repository.property.back_populates == "pull_requests"
