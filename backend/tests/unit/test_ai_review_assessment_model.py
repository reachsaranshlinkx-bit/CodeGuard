from sqlalchemy import CheckConstraint, SmallInteger, UniqueConstraint, inspect

from codeguard.db.base import Base
from codeguard.db.models import AIReviewAssessment, Analysis


def test_ai_review_assessment_table_is_registered():
    assert "ai_review_assessments" in Base.metadata.tables
    assert AIReviewAssessment.__tablename__ == "ai_review_assessments"


def test_ai_review_assessment_columns_and_lengths():
    table = AIReviewAssessment.__table__

    expected_columns = {
        "id",
        "analysis_id",
        "candidate_index",
        "title",
        "severity",
        "confidence",
        "file_path",
        "line_number",
        "problem",
        "evidence",
        "why_it_matters",
        "suggested_fix",
        "recommended_validation",
        "evidence_match_status",
        "evidence_match_reason",
        "location_reason",
        "created_at",
    }

    assert expected_columns == set(table.columns.keys())

    expected_lengths = {
        "title": 200,
        "severity": 8,
        "confidence": 8,
        "file_path": 500,
        "problem": 3000,
        "evidence": 4000,
        "why_it_matters": 2000,
        "suggested_fix": 3000,
        "recommended_validation": 2000,
        "evidence_match_status": 16,
        "evidence_match_reason": 32,
        "location_reason": 32,
    }

    for column_name, expected_length in expected_lengths.items():
        assert table.c[column_name].type.length == expected_length


def test_ai_review_assessment_analysis_foreign_key():
    column = AIReviewAssessment.__table__.c.analysis_id

    assert not column.nullable

    foreign_key = next(iter(column.foreign_keys))

    assert foreign_key.target_fullname == "analyses.id"
    assert foreign_key.ondelete == "CASCADE"


def test_ai_review_assessment_candidate_index_configuration():
    column = AIReviewAssessment.__table__.c.candidate_index

    assert isinstance(column.type, SmallInteger)
    assert not column.nullable


def test_ai_review_assessment_location_reason_is_optional():
    column = AIReviewAssessment.__table__.c.location_reason

    assert column.nullable


def test_ai_review_assessment_check_constraints():
    constraints = {
        constraint.name
        for constraint in AIReviewAssessment.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    expected = {
        "ck_ai_review_assessments_candidate_index_range",
        "ck_ai_review_assessments_severity",
        "ck_ai_review_assessments_confidence",
        "ck_ai_review_assessments_line_number_positive",
        "ck_ai_review_assessments_evidence_match_status",
        "ck_ai_review_assessments_evidence_match_reason",
        "ck_ai_review_assessments_evidence_match_consistency",
        "ck_ai_review_assessments_location_reason",
    }

    assert expected.issubset(constraints)


def test_ai_review_assessment_candidate_index_is_unique_per_analysis():
    unique_constraints = {
        tuple(column.name for column in constraint.columns)
        for constraint in AIReviewAssessment.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert ("analysis_id", "candidate_index") in unique_constraints


def test_ai_review_assessment_index_exists():
    indexes = {index.name for index in AIReviewAssessment.__table__.indexes}

    assert "ix_ai_review_assessments_analysis_id_evidence_match_status" in indexes


def test_ai_review_assessment_relationships_are_configured():
    assessment_relationships = inspect(AIReviewAssessment).relationships
    analysis_relationships = inspect(Analysis).relationships

    assert (
        assessment_relationships["analysis"].back_populates == "ai_review_assessments"
    )

    assert analysis_relationships["ai_review_assessments"].back_populates == "analysis"


def test_analysis_assessments_relationship_supports_delete_cascade():
    relationship = inspect(Analysis).relationships["ai_review_assessments"]

    assert "delete" in relationship.cascade
    assert "delete-orphan" in relationship.cascade
    assert relationship.passive_deletes is True


def test_ai_review_assessment_recommended_validation_is_required():
    column = AIReviewAssessment.__table__.c.recommended_validation

    assert column.type.length == 2000
    assert not column.nullable
