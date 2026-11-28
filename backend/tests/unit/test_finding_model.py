from sqlalchemy import CheckConstraint, UniqueConstraint, inspect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID

from codeguard.db.base import Base
from codeguard.db.models import Analysis, Finding


def test_finding_table_is_registered():
    assert "findings" in Base.metadata.tables
    assert Finding.__tablename__ == "findings"


def test_finding_primary_key_and_expected_columns():
    table = Finding.__table__

    assert isinstance(table.c.id.type, PostgreSQLUUID)
    assert table.c.id.primary_key

    expected_columns = {
        "analysis_id",
        "fingerprint",
        "rule_id",
        "title",
        "severity",
        "confidence",
        "message",
        "description",
        "file_path",
        "line_number",
        "cwe",
        "evidence",
        "remediation",
        "references",
        "created_at",
    }

    assert expected_columns.issubset(set(table.columns.keys()))


def test_finding_analysis_foreign_key_cascades_on_delete():
    column = Finding.__table__.c.analysis_id

    assert not column.nullable

    foreign_key = next(iter(column.foreign_keys))

    assert foreign_key.target_fullname == "analyses.id"
    assert foreign_key.ondelete == "CASCADE"


def test_finding_fingerprint_configuration():
    column = Finding.__table__.c.fingerprint

    assert column.type.length == 64
    assert not column.nullable


def test_finding_references_is_required_jsonb():
    column = Finding.__table__.c["references"]

    assert isinstance(column.type, JSONB)
    assert not column.nullable


def test_finding_has_expected_check_constraints():
    constraints = {
        constraint.name
        for constraint in Finding.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    expected = {
        "ck_findings_fingerprint_sha256",
        "ck_findings_severity",
        "ck_findings_confidence",
        "ck_findings_line_number_positive",
        "ck_findings_references_nonempty_array",
    }

    assert expected.issubset(constraints)


def test_finding_analysis_fingerprint_is_unique_together():
    unique_constraints = {
        tuple(column.name for column in constraint.columns)
        for constraint in Finding.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert ("analysis_id", "fingerprint") in unique_constraints


def test_finding_indexes_exist():
    indexes = {index.name for index in Finding.__table__.indexes}

    assert "ix_findings_analysis_id_severity" in indexes
    assert "ix_findings_analysis_id_file_path_line_number" in indexes


def test_finding_analysis_relationship_is_configured():
    finding_relationships = inspect(Finding).relationships
    analysis_relationships = inspect(Analysis).relationships

    assert finding_relationships["analysis"].back_populates == "findings"
    assert analysis_relationships["findings"].back_populates == "analysis"


def test_analysis_findings_relationship_supports_delete_cascade():
    relationship = inspect(Analysis).relationships["findings"]

    assert "delete" in relationship.cascade
    assert "delete-orphan" in relationship.cascade
    assert relationship.passive_deletes is True
