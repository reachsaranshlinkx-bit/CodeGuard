from sqlalchemy.orm import configure_mappers

from codeguard.db.base import Base
from codeguard.db.models import (
    AIReviewAssessment,
    Analysis,
    Finding,
    PullRequest,
    Repository,
)

EXPECTED_TABLES = {
    "repositories",
    "pull_requests",
    "analyses",
    "findings",
    "ai_review_assessments",
}


def test_all_five_models_are_registered():
    assert EXPECTED_TABLES.issubset(set(Base.metadata.tables))


def test_all_model_mappers_configure():
    configure_mappers()


def test_repository_to_pull_request_foreign_key():
    foreign_keys = {fk.target_fullname for fk in PullRequest.__table__.foreign_keys}

    assert "repositories.id" in foreign_keys


def test_pull_request_to_analysis_foreign_key():
    foreign_keys = {fk.target_fullname for fk in Analysis.__table__.foreign_keys}

    assert "pull_requests.id" in foreign_keys


def test_analysis_to_child_foreign_keys():
    finding_foreign_keys = {fk.target_fullname for fk in Finding.__table__.foreign_keys}

    assessment_foreign_keys = {
        fk.target_fullname for fk in AIReviewAssessment.__table__.foreign_keys
    }

    assert "analyses.id" in finding_foreign_keys
    assert "analyses.id" in assessment_foreign_keys


def test_all_models_share_the_same_metadata():
    models = (
        Repository,
        PullRequest,
        Analysis,
        Finding,
        AIReviewAssessment,
    )

    for model in models:
        assert model.metadata is Base.metadata
