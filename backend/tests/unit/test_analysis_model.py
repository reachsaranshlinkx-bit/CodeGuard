from sqlalchemy import CheckConstraint, DateTime, inspect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID

from codeguard.db.base import Base
from codeguard.db.models import Analysis, PullRequest


def test_analysis_table_is_registered():
    assert "analyses" in Base.metadata.tables
    assert Analysis.__tablename__ == "analyses"


def test_analysis_primary_key_and_columns():
    table = Analysis.__table__

    assert isinstance(table.c.id.type, PostgreSQLUUID)
    assert table.c.id.primary_key

    expected_columns = {
        "pull_request_id",
        "head_sha",
        "base_sha",
        "repository_owner_login",
        "repository_name",
        "pull_request_title",
        "pull_request_html_url",
        "status",
        "started_at",
        "completed_at",
        "files_analyzed",
        "files_with_unavailable_mapping",
        "repository_context",
        "ai_review_summary",
        "ai_execution_telemetry",
        "created_at",
        "updated_at",
    }

    assert expected_columns.issubset(set(table.columns.keys()))


def test_analysis_pull_request_foreign_key():
    column = Analysis.__table__.c.pull_request_id

    assert not column.nullable

    foreign_key = next(iter(column.foreign_keys))

    assert foreign_key.target_fullname == "pull_requests.id"
    assert foreign_key.ondelete == "RESTRICT"


def test_analysis_timestamp_columns_are_timezone_aware():
    table = Analysis.__table__

    assert isinstance(table.c.started_at.type, DateTime)
    assert isinstance(table.c.completed_at.type, DateTime)
    assert table.c.started_at.type.timezone
    assert table.c.completed_at.type.timezone


def test_analysis_jsonb_columns():
    table = Analysis.__table__

    assert isinstance(
        table.c.files_with_unavailable_mapping.type,
        JSONB,
    )
    assert isinstance(table.c.repository_context.type, JSONB)
    assert isinstance(table.c.ai_execution_telemetry.type, JSONB)

    assert not table.c.files_with_unavailable_mapping.nullable
    assert table.c.repository_context.nullable
    assert table.c.ai_execution_telemetry.nullable


def test_analysis_default_values():
    table = Analysis.__table__

    assert table.c.files_analyzed.default.arg == 0

    default_factory = table.c.files_with_unavailable_mapping.default.arg

    first_value = default_factory(None)
    second_value = default_factory(None)

    assert first_value == []
    assert second_value == []
    assert first_value is not second_value


def test_analysis_check_constraints_exist():
    constraints = {
        constraint.name
        for constraint in Analysis.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    expected = {
        "ck_analyses_status",
        "ck_analyses_files_analyzed_nonnegative",
        "ck_analyses_head_sha_length",
        "ck_analyses_base_sha_length",
        "ck_analyses_lifecycle_timestamps",
        "ck_analyses_completed_after_started",
        "ck_analyses_unavailable_mapping_array",
        "ck_analyses_repository_context_object",
        "ck_analyses_ai_execution_telemetry_object",
    }

    assert expected.issubset(constraints)


def test_analysis_indexes_exist():
    indexes = {index.name for index in Analysis.__table__.indexes}

    assert "ix_analyses_pull_request_id_created_at" in indexes
    assert "ix_analyses_pull_request_id_head_sha" in indexes


def test_analysis_relationships_are_configured():
    analysis_relationships = inspect(Analysis).relationships
    pull_request_relationships = inspect(PullRequest).relationships

    assert analysis_relationships["pull_request"].back_populates == "analyses"
    assert pull_request_relationships["analyses"].back_populates == "pull_request"


def test_repeated_analyses_for_same_head_sha_are_allowed():
    unique_constraints = [
        constraint
        for constraint in Analysis.__table__.constraints
        if getattr(constraint, "unique", False)
    ]

    assert not unique_constraints
