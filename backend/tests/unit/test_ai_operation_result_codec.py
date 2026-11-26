import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest

from codeguard.ai.contracts import AIFindingCandidate, AIReviewResult
from codeguard.ai.engine import (
    AIReviewExecutionResult,
    AIReviewFindingAssessment,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
from codeguard.ai.evidence_validation import EvidenceLocationReason
from codeguard.ai.operation_result_codec import (
    AIReviewResultCodecError,
    EncodedAIReviewResult,
    decode,
    encode,
)


def make_candidate(index: int, *, evidence: str | None = None) -> AIFindingCandidate:
    return AIFindingCandidate(
        title=f"Finding {index}",
        severity="high",
        confidence="medium",
        file_path=f"src/module_{index}.py",
        line_number=index + 1,
        problem=f"Problem description {index}.",
        evidence=evidence if evidence is not None else f"    source line {index}  \n",
        why_it_matters=f"Impact description {index}.",
        suggested_fix=f"Fix description {index}.",
        recommended_validation=f"Validation description {index}.",
    )


def make_execution_result(
    matches: Sequence[
        tuple[
            EvidenceMatchStatus,
            EvidenceMatchReason,
            EvidenceLocationReason | None,
        ]
    ]
    | None = None,
) -> AIReviewExecutionResult:
    candidates = tuple(
        make_candidate(index, evidence=f"  exact source {index} \n")
        for index in range(len(matches) if matches is not None else 2)
    )
    assessments = tuple(
        AIReviewFindingAssessment(
            candidate=candidate,
            evidence_match=EvidenceMatchResult(
                status=(
                    matches[index][0]
                    if matches is not None
                    else EvidenceMatchStatus.MATCHED
                ),
                reason=(
                    matches[index][1]
                    if matches is not None
                    else EvidenceMatchReason.EXACT_MATCH
                ),
                file_path=candidate.file_path,
                line_number=candidate.line_number,
                location_reason=matches[index][2] if matches is not None else None,
            ),
        )
        for index, candidate in enumerate(candidates)
    )
    return AIReviewExecutionResult(
        provider_result=AIReviewResult(
            summary="Review summary with exact result content.",
            findings=candidates,
        ),
        assessments=assessments,
    )


def decode_document(document: object) -> AIReviewExecutionResult:
    return decode(
        json.dumps(document, ensure_ascii=False),
        result_schema_version=1,
        serialization_version=1,
    )


def document_from(encoded: EncodedAIReviewResult) -> dict[str, object]:
    document = json.loads(encoded.result_json)
    assert isinstance(document, dict)
    return document


def test_empty_execution_result_round_trips() -> None:
    result = AIReviewExecutionResult(
        provider_result=AIReviewResult(summary="No findings."),
        assessments=(),
    )

    encoded = encode(result)

    assert (encoded.result_schema_version, encoded.serialization_version) == (1, 1)
    assert (
        decode(
            encoded.result_json,
            encoded.result_schema_version,
            encoded.serialization_version,
        )
        == result
    )


def test_populated_execution_result_round_trips_with_exact_evidence_whitespace() -> (
    None
):
    candidate = make_candidate(0, evidence="\n\t  exact source text  \n\n")
    result = AIReviewExecutionResult(
        provider_result=AIReviewResult(
            summary="A complete provider result.",
            findings=(candidate,),
        ),
        assessments=(
            AIReviewFindingAssessment(
                candidate=candidate,
                evidence_match=EvidenceMatchResult(
                    status=EvidenceMatchStatus.MISMATCHED,
                    reason=EvidenceMatchReason.CONTENT_MISMATCH,
                    file_path=candidate.file_path,
                    line_number=candidate.line_number,
                    location_reason=EvidenceLocationReason.LOCATION_VALID,
                ),
            ),
        ),
    )

    encoded = encode(result)
    decoded = decode(
        encoded.result_json,
        encoded.result_schema_version,
        encoded.serialization_version,
    )

    assert decoded == result
    assert (
        decoded.provider_result.findings[0].evidence == "\n\t  exact source text  \n\n"
    )
    assert decoded.assessments[0].candidate is decoded.provider_result.findings[0]


def test_multiple_findings_and_assessments_preserve_order_and_indices() -> None:
    result = make_execution_result()

    encoded = encode(result)
    document = document_from(encoded)
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    assert [assessment["finding_index"] for assessment in assessments] == [0, 1]

    decoded = decode(
        encoded.result_json,
        encoded.result_schema_version,
        encoded.serialization_version,
    )
    assert decoded == result
    assert tuple(item.candidate for item in decoded.assessments) == (
        decoded.provider_result.findings
    )


def test_all_evidence_statuses_reasons_and_location_reasons_round_trip() -> None:
    statuses = tuple(EvidenceMatchStatus)
    reasons = tuple(EvidenceMatchReason)
    location_reasons: tuple[EvidenceLocationReason | None, ...] = (
        None,
        *tuple(EvidenceLocationReason),
    )
    matches = tuple(
        (
            statuses[index % len(statuses)],
            reasons[index % len(reasons)],
            location_reason,
        )
        for index, location_reason in enumerate(location_reasons)
    )
    result = make_execution_result(matches)

    decoded = decode(
        encode(result).result_json,
        result_schema_version=1,
        serialization_version=1,
    )

    assert decoded == result
    assert {item.evidence_match.status for item in decoded.assessments} == set(statuses)
    assert {item.evidence_match.reason for item in decoded.assessments} == set(reasons)
    assert {item.evidence_match.location_reason for item in decoded.assessments} == set(
        location_reasons
    )


def test_identical_results_have_deterministic_json() -> None:
    result = make_execution_result()

    assert encode(result).result_json == encode(result).result_json
    assert encode(result).result_json == json.dumps(
        json.loads(encode(result).result_json),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_encoding_rejects_mismatched_assessment_candidate() -> None:
    result = make_execution_result()
    mismatched_assessment = AIReviewFindingAssessment(
        candidate=result.provider_result.findings[1],
        evidence_match=result.assessments[0].evidence_match,
    )
    invalid_result = AIReviewExecutionResult(
        provider_result=result.provider_result,
        assessments=(mismatched_assessment, result.assessments[1]),
    )

    with pytest.raises(AIReviewResultCodecError):
        encode(invalid_result)


def test_encoding_rejects_assessment_with_mismatched_file_path() -> None:
    result = make_execution_result()
    assessment = result.assessments[0]
    invalid_assessment = replace(
        assessment,
        evidence_match=replace(
            assessment.evidence_match,
            file_path="src/unrelated.py",
        ),
    )
    invalid_result = replace(
        result,
        assessments=(invalid_assessment, *result.assessments[1:]),
    )

    with pytest.raises(AIReviewResultCodecError):
        encode(invalid_result)


def test_encoding_rejects_assessment_with_mismatched_line_number() -> None:
    result = make_execution_result()
    assessment = result.assessments[0]
    invalid_assessment = replace(
        assessment,
        evidence_match=replace(
            assessment.evidence_match,
            line_number=assessment.candidate.line_number + 1,
        ),
    )
    invalid_result = replace(
        result,
        assessments=(invalid_assessment, *result.assessments[1:]),
    )

    with pytest.raises(AIReviewResultCodecError):
        encode(invalid_result)


def test_encoding_rejects_missing_assessment() -> None:
    result = make_execution_result()
    invalid_result = AIReviewExecutionResult(
        provider_result=result.provider_result,
        assessments=result.assessments[:1],
    )

    with pytest.raises(AIReviewResultCodecError):
        encode(invalid_result)


def test_decoding_rejects_duplicate_finding_indices() -> None:
    document = document_from(encode(make_execution_result()))
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    assessments[1]["finding_index"] = 0

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


def test_decoding_rejects_reordered_assessments() -> None:
    document = document_from(encode(make_execution_result()))
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    assessments.reverse()

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


@pytest.mark.parametrize("index", [-1, 2, 100])
def test_decoding_rejects_out_of_range_finding_indices(index: int) -> None:
    document = document_from(encode(make_execution_result()))
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    assessments[0]["finding_index"] = index

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


@pytest.mark.parametrize(
    "remove_field",
    [
        lambda document: document.pop("payload"),
        lambda document: document["payload"].pop("provider_result"),
        lambda document: document["payload"]["provider_result"].pop("findings"),
        lambda document: document["payload"]["provider_result"]["findings"][0].pop(
            "evidence"
        ),
    ],
)
def test_decoding_rejects_missing_required_fields(
    remove_field: Callable[[dict[str, Any]], object],
) -> None:
    document = document_from(encode(make_execution_result()))
    remove_field(document)

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


def test_decoding_rejects_missing_assessment_or_finding_index() -> None:
    document = document_from(encode(make_execution_result()))
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    assessments.pop()

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)

    document = document_from(encode(make_execution_result()))
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    del assessments[0]["finding_index"]

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


def test_decoding_rejects_mismatched_finding_and_assessment_location() -> None:
    document = document_from(encode(make_execution_result()))
    payload = document["payload"]
    assert isinstance(payload, dict)
    assessments = payload["assessments"]
    assert isinstance(assessments, list)
    assessments[0]["evidence_match"]["file_path"] = "src/other.py"

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(AIReviewResultCodecError):
        decode("{", result_schema_version=1, serialization_version=1)


def test_duplicate_json_object_keys_are_rejected() -> None:
    duplicate_key_json = (
        '{"kind":"codeguard.ai-review-execution-result",'
        '"kind":"codeguard.ai-review-execution-result",'
        '"serialization_version":1,"result_schema_version":1,'
        '"payload":{"provider_result":{"summary":"x","findings":[]},'
        '"assessments":[]}}'
    )

    with pytest.raises(AIReviewResultCodecError):
        decode(duplicate_key_json, result_schema_version=1, serialization_version=1)


def test_unsupported_serialization_version_is_rejected() -> None:
    document = document_from(encode(make_execution_result()))
    document["serialization_version"] = 2
    document["result_schema_version"] = 1

    with pytest.raises(AIReviewResultCodecError):
        decode(
            json.dumps(document),
            result_schema_version=1,
            serialization_version=2,
        )


def test_unsupported_result_schema_version_is_rejected() -> None:
    document = document_from(encode(make_execution_result()))
    document["serialization_version"] = 1
    document["result_schema_version"] = 2

    with pytest.raises(AIReviewResultCodecError):
        decode(
            json.dumps(document),
            result_schema_version=2,
            serialization_version=1,
        )


@pytest.mark.parametrize(
    ("database_schema_version", "database_serialization_version"),
    [(2, 1), (1, 2)],
)
def test_envelope_database_version_mismatch_is_rejected(
    database_schema_version: int,
    database_serialization_version: int,
) -> None:
    encoded = encode(make_execution_result())

    with pytest.raises(AIReviewResultCodecError):
        decode(
            encoded.result_json,
            result_schema_version=database_schema_version,
            serialization_version=database_serialization_version,
        )


def test_incorrect_envelope_kind_is_rejected() -> None:
    document = document_from(encode(make_execution_result()))
    document["kind"] = "other.result"

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update({"unexpected": True}),
        lambda document: document["payload"].update({"unexpected": True}),
        lambda document: document["payload"]["provider_result"].update(
            {"unexpected": True}
        ),
        lambda document: document["payload"]["provider_result"]["findings"][0].update(
            {"unexpected": True}
        ),
        lambda document: document["payload"]["assessments"][0]["evidence_match"].update(
            {"unexpected": True}
        ),
    ],
)
def test_unknown_fields_are_rejected(
    mutation: Callable[[dict[str, Any]], object],
) -> None:
    document = document_from(encode(make_execution_result()))
    mutation(document)

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            ("payload", "assessments", 0, "evidence_match", "status"),
            "unknown_status",
        ),
        (
            ("payload", "assessments", 0, "evidence_match", "reason"),
            "unknown_reason",
        ),
        (
            ("payload", "assessments", 0, "evidence_match", "location_reason"),
            "unknown_location_reason",
        ),
        (
            ("payload", "provider_result", "findings", 0, "severity"),
            "unknown_severity",
        ),
    ],
)
def test_unknown_enum_values_are_rejected(
    path: tuple[str | int, ...], value: str
) -> None:
    document = document_from(encode(make_execution_result()))
    current: Any = document
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


def test_invalid_pydantic_data_is_rejected_without_coercion() -> None:
    document = document_from(encode(make_execution_result()))
    document["payload"]["provider_result"]["findings"][0]["line_number"] = True

    with pytest.raises(AIReviewResultCodecError):
        decode_document(document)


def test_serialized_result_excludes_raw_context_and_snapshot_content() -> None:
    result = make_execution_result()
    document = document_from(encode(result))
    payload = document["payload"]
    assert isinstance(payload, dict)
    provider_result = payload["provider_result"]
    assert isinstance(provider_result, dict)

    assert set(payload) == {"provider_result", "assessments"}
    assert "context" not in provider_result
    assert "source_snapshot" not in provider_result
    assert "exact source" in result.provider_result.findings[0].evidence
