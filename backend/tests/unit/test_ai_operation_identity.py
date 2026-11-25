from dataclasses import FrozenInstanceError, fields

import pytest

from codeguard.ai.operation_identity import (
    AIOperationIdentity,
    build_ai_operation_key,
)


def make_identity(**updates: object) -> AIOperationIdentity:
    values: dict[str, object] = {
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
        "head_sha": "ABCDEF123456",
        "provider_identity": "provider-openai-compatible",
        "model_identity": "model-review-v1",
        "prompt_contract_version": "prompt-v1",
        "context_contract_version": "context-v1",
        "evidence_contract_version": "evidence-v1",
        "review_contract_version": "review-v1",
    }
    values.update(updates)
    return AIOperationIdentity(**values)  # type: ignore[arg-type]


def test_identical_identities_produce_identical_keys() -> None:
    identity = make_identity()

    assert build_ai_operation_key(identity) == build_ai_operation_key(make_identity())


def test_owner_and_repository_casing_is_normalized() -> None:
    identity = make_identity(owner="  OctoCat  ", repo="  codeguard  ")

    assert identity.owner == "octocat"
    assert identity.repo == "codeguard"
    assert build_ai_operation_key(identity) == build_ai_operation_key(make_identity())


def test_equivalent_normalized_sha_values_produce_the_same_key() -> None:
    identity = make_identity(head_sha="  ABCDEF123456  ")

    assert identity.head_sha == "abcdef123456"
    assert build_ai_operation_key(identity) == build_ai_operation_key(make_identity())


def test_new_head_sha_produces_different_key() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(head_sha="FEDCBA654321")
    )


def test_different_pull_numbers_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity(pull_number=42)) != (
        build_ai_operation_key(make_identity(pull_number=43))
    )


def test_different_repositories_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity(repo="CodeGuard")) != (
        build_ai_operation_key(make_identity(repo="AnotherRepo"))
    )


def test_different_repository_owners_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity(owner="octocat")) != (
        build_ai_operation_key(make_identity(owner="another-owner"))
    )


def test_different_provider_identities_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(provider_identity="another-provider")
    )


def test_different_model_identities_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(model_identity="model-review-v2")
    )


def test_different_prompt_contract_versions_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(prompt_contract_version="prompt-v2")
    )


def test_different_context_contract_versions_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(context_contract_version="context-v2")
    )


def test_different_evidence_contract_versions_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(evidence_contract_version="evidence-v2")
    )


def test_different_review_contract_versions_produce_different_keys() -> None:
    assert build_ai_operation_key(make_identity()) != build_ai_operation_key(
        make_identity(review_contract_version="review-v2")
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("owner", "", "owner"),
        ("repo", "   ", "repo"),
        ("head_sha", "", "head_sha"),
        ("provider_identity", "", "provider_identity"),
        ("model_identity", "   ", "model_identity"),
        ("prompt_contract_version", "", "prompt_contract_version"),
        ("context_contract_version", " ", "context_contract_version"),
        ("evidence_contract_version", "", "evidence_contract_version"),
        ("review_contract_version", " ", "review_contract_version"),
    ],
)
def test_invalid_identity_fields_are_rejected(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_identity(**{field: value})


@pytest.mark.parametrize("pull_number", [0, -1])
def test_zero_and_negative_pull_numbers_are_rejected(pull_number: int) -> None:
    with pytest.raises(ValueError, match="pull_number"):
        make_identity(pull_number=pull_number)


def test_boolean_pull_number_is_rejected() -> None:
    with pytest.raises(ValueError, match="pull_number"):
        make_identity(pull_number=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner", None),
        ("owner", 123),
        ("repo", None),
        ("repo", 123),
        ("head_sha", None),
        ("head_sha", 123),
        ("provider_identity", None),
        ("provider_identity", 123),
        ("model_identity", None),
        ("model_identity", 123),
        ("prompt_contract_version", None),
        ("prompt_contract_version", 123),
        ("context_contract_version", None),
        ("context_contract_version", 123),
        ("evidence_contract_version", None),
        ("evidence_contract_version", 123),
        ("review_contract_version", None),
        ("review_contract_version", 123),
    ],
)
def test_string_fields_reject_invalid_runtime_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        make_identity(**{field: value})


@pytest.mark.parametrize("pull_number", ["42", 42.0, 42.5])
def test_pull_number_rejects_string_and_float_values(pull_number: object) -> None:
    with pytest.raises(ValueError, match="pull_number"):
        make_identity(pull_number=pull_number)


@pytest.mark.parametrize(
    "excluded_field",
    [
        "delivery_id",
        "job_id",
        "attempt",
        "attempt_number",
        "lease_token",
        "timestamp",
        "result_schema_version",
        "serialization_version",
    ],
)
def test_operation_metadata_fields_are_not_identity_fields(
    excluded_field: str,
) -> None:
    identity_field_names = {field.name for field in fields(AIOperationIdentity)}

    assert excluded_field not in identity_field_names


@pytest.mark.parametrize("field_name", ["owner", "repo"])
def test_identity_is_immutable(field_name: str) -> None:
    identity = make_identity()

    with pytest.raises(FrozenInstanceError):
        setattr(identity, field_name, "changed")
