import json
from dataclasses import dataclass
from typing import Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import PydanticSerializationError

from codeguard.ai.contracts import AIReviewResult
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
from codeguard.ai.telemetry import AIExecutionTelemetry

_RESULT_KIND = "codeguard.ai-review-execution-result"
_SERIALIZATION_VERSION = 2
_SUPPORTED_SERIALIZATION_VERSIONS = frozenset({1, 2})
_RESULT_SCHEMA_VERSION = 1


class AIReviewResultCodecError(ValueError):
    """Stored AI review results that cannot be safely encoded or decoded."""


@dataclass(frozen=True, slots=True)
class EncodedAIReviewResult:
    """Serialized AI execution result and its explicit format versions."""

    result_json: str
    result_schema_version: int
    serialization_version: int


class _SerializedEvidenceMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: EvidenceMatchStatus
    reason: EvidenceMatchReason
    file_path: str
    line_number: int = Field(gt=0)
    location_reason: EvidenceLocationReason | None


class _SerializedAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    finding_index: int = Field(ge=0)
    evidence_match: _SerializedEvidenceMatch


class _SerializedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider_result: dict[str, object]
    assessments: list[_SerializedAssessment]
    telemetry: AIExecutionTelemetry | None = None


class _SerializedEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["codeguard.ai-review-execution-result"]
    serialization_version: int
    result_schema_version: int
    payload: _SerializedPayload


class _DuplicateJSONKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON number {value} is not permitted.")


def _validate_provider_result(provider_result: AIReviewResult) -> AIReviewResult:
    try:
        return AIReviewResult.model_validate(
            provider_result.model_dump(mode="python"),
            strict=True,
        )
    except (ValidationError, PydanticSerializationError, TypeError, ValueError):
        raise AIReviewResultCodecError(
            "AI review provider result is invalid."
        ) from None


def _serialized_assessment(
    assessment: AIReviewFindingAssessment,
    *,
    finding_index: int,
) -> _SerializedAssessment:
    evidence_match = assessment.evidence_match
    if not isinstance(evidence_match, EvidenceMatchResult):
        raise AIReviewResultCodecError("AI review evidence assessment is invalid.")

    try:
        return _SerializedAssessment.model_validate(
            {
                "finding_index": finding_index,
                "evidence_match": {
                    "status": evidence_match.status,
                    "reason": evidence_match.reason,
                    "file_path": evidence_match.file_path,
                    "line_number": evidence_match.line_number,
                    "location_reason": evidence_match.location_reason,
                },
            },
            strict=True,
        )
    except ValidationError:
        raise AIReviewResultCodecError(
            "AI review evidence assessment is invalid."
        ) from None


def encode(result: AIReviewExecutionResult) -> EncodedAIReviewResult:
    """Validate and deterministically encode a complete AI execution result."""

    if not isinstance(result, AIReviewExecutionResult):
        raise AIReviewResultCodecError("AI review execution result is invalid.")
    if not isinstance(result.provider_result, AIReviewResult):
        raise AIReviewResultCodecError("AI review provider result is invalid.")
    if not isinstance(result.assessments, tuple):
        raise AIReviewResultCodecError("AI review assessments are invalid.")
    if result.telemetry is not None and not isinstance(
        result.telemetry,
        AIExecutionTelemetry,
    ):
        raise AIReviewResultCodecError("AI review execution telemetry is invalid.")

    provider_result = _validate_provider_result(result.provider_result)
    findings = provider_result.findings

    if len(result.assessments) != len(findings):
        raise AIReviewResultCodecError(
            "AI review result must have one assessment per finding."
        )

    serialized_assessments: list[_SerializedAssessment] = []
    for finding_index, assessment in enumerate(result.assessments):
        if not isinstance(assessment, AIReviewFindingAssessment):
            raise AIReviewResultCodecError("AI review assessment is invalid.")
        if not isinstance(assessment.evidence_match, EvidenceMatchResult):
            raise AIReviewResultCodecError("AI review evidence assessment is invalid.")
        if assessment.candidate != findings[finding_index]:
            raise AIReviewResultCodecError(
                "AI review assessment does not match its finding."
            )
        if (
            assessment.evidence_match.file_path != findings[finding_index].file_path
            or assessment.evidence_match.line_number
            != findings[finding_index].line_number
        ):
            raise AIReviewResultCodecError(
                "AI review assessment location does not match its finding."
            )
        serialized_assessments.append(
            _serialized_assessment(assessment, finding_index=finding_index)
        )

    try:
        envelope = _SerializedEnvelope.model_validate(
            {
                "kind": _RESULT_KIND,
                "serialization_version": _SERIALIZATION_VERSION,
                "result_schema_version": _RESULT_SCHEMA_VERSION,
                "payload": {
                    "provider_result": provider_result.model_dump(mode="json"),
                    "assessments": serialized_assessments,
                    "telemetry": (
                        result.telemetry.model_dump(mode="json")
                        if result.telemetry is not None
                        else None
                    ),
                },
            },
            strict=True,
        )
        result_json = json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (
        ValidationError,
        PydanticSerializationError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise AIReviewResultCodecError(
            "AI review execution result could not be encoded."
        ) from None

    return EncodedAIReviewResult(
        result_json=result_json,
        result_schema_version=_RESULT_SCHEMA_VERSION,
        serialization_version=_SERIALIZATION_VERSION,
    )


def _parse_json(result_json: str) -> dict[str, object]:
    if not isinstance(result_json, str):
        raise AIReviewResultCodecError("Stored AI review result is invalid.")
    try:
        parsed = json.loads(
            result_json,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (
        json.JSONDecodeError,
        _DuplicateJSONKeyError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise AIReviewResultCodecError("Stored AI review result is invalid.") from None

    if not isinstance(parsed, dict):
        raise AIReviewResultCodecError("Stored AI review result is invalid.")
    return cast(dict[str, object], parsed)


def _check_database_versions(
    result_schema_version: int,
    serialization_version: int,
    envelope: dict[str, object],
) -> None:
    if type(result_schema_version) is not int or type(serialization_version) is not int:
        raise AIReviewResultCodecError("Stored AI review result version is invalid.")

    envelope_serialization_version = envelope.get("serialization_version")
    envelope_schema_version = envelope.get("result_schema_version")
    if (
        type(envelope_serialization_version) is not int
        or type(envelope_schema_version) is not int
        or envelope_serialization_version != serialization_version
        or envelope_schema_version != result_schema_version
    ):
        raise AIReviewResultCodecError("Stored AI review result versions do not match.")

    if serialization_version not in _SUPPORTED_SERIALIZATION_VERSIONS:
        raise AIReviewResultCodecError(
            "Stored AI review serialization version is unsupported."
        )
    if result_schema_version != _RESULT_SCHEMA_VERSION:
        raise AIReviewResultCodecError(
            "Stored AI review result schema version is unsupported."
        )


def _decode_provider_result(data: dict[str, object]) -> AIReviewResult:
    if set(data) != {"summary", "findings"}:
        raise AIReviewResultCodecError(
            "Stored AI review provider result is incomplete or invalid."
        )
    try:
        return AIReviewResult.model_validate_json(
            json.dumps(data, ensure_ascii=False, allow_nan=False),
            strict=True,
        )
    except (
        ValidationError,
        PydanticSerializationError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise AIReviewResultCodecError(
            "Stored AI review provider result is invalid."
        ) from None


def _decode_assessments(
    serialized_assessments: list[_SerializedAssessment],
    provider_result: AIReviewResult,
) -> tuple[AIReviewFindingAssessment, ...]:
    findings = provider_result.findings
    if len(serialized_assessments) != len(findings):
        raise AIReviewResultCodecError(
            "Stored AI review result must have one assessment per finding."
        )

    indices = [item.finding_index for item in serialized_assessments]
    if len(set(indices)) != len(indices):
        raise AIReviewResultCodecError(
            "Stored AI review result has duplicate finding indices."
        )
    if any(index >= len(findings) for index in indices):
        raise AIReviewResultCodecError(
            "Stored AI review result has an out-of-range finding index."
        )
    if indices != list(range(len(findings))):
        raise AIReviewResultCodecError(
            "Stored AI review result has missing or unordered finding indices."
        )

    assessments: list[AIReviewFindingAssessment] = []
    for item in serialized_assessments:
        candidate = findings[item.finding_index]
        evidence_match = item.evidence_match
        if (
            evidence_match.file_path != candidate.file_path
            or evidence_match.line_number != candidate.line_number
        ):
            raise AIReviewResultCodecError(
                "Stored AI review assessment does not match its finding."
            )
        assessments.append(
            AIReviewFindingAssessment(
                candidate=candidate,
                evidence_match=EvidenceMatchResult(
                    status=evidence_match.status,
                    reason=evidence_match.reason,
                    file_path=evidence_match.file_path,
                    line_number=evidence_match.line_number,
                    location_reason=evidence_match.location_reason,
                ),
            )
        )
    return tuple(assessments)


def decode(
    result_json: str,
    result_schema_version: int,
    serialization_version: int,
) -> AIReviewExecutionResult:
    """Strictly decode a stored AI execution result using its recorded versions."""

    document = _parse_json(result_json)
    expected_envelope_fields = {
        "kind",
        "serialization_version",
        "result_schema_version",
        "payload",
    }
    if set(document) != expected_envelope_fields:
        raise AIReviewResultCodecError(
            "Stored AI review result envelope is incomplete or invalid."
        )
    if document["kind"] != _RESULT_KIND:
        raise AIReviewResultCodecError("Stored AI review result kind is invalid.")
    _check_database_versions(
        result_schema_version,
        serialization_version,
        document,
    )
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise AIReviewResultCodecError("Stored AI review result payload is invalid.")
    expected_payload_fields = (
        {"provider_result", "assessments"}
        if serialization_version == 1
        else {"provider_result", "assessments", "telemetry"}
    )
    if set(payload) != expected_payload_fields:
        raise AIReviewResultCodecError(
            "Stored AI review result payload is incomplete or invalid."
        )

    try:
        envelope = _SerializedEnvelope.model_validate_json(
            result_json,
            strict=True,
        )
    except (RecursionError, ValidationError, ValueError):
        raise AIReviewResultCodecError(
            "Stored AI review result envelope is invalid."
        ) from None

    provider_result = _decode_provider_result(envelope.payload.provider_result)
    assessments = _decode_assessments(
        envelope.payload.assessments,
        provider_result,
    )
    return AIReviewExecutionResult(
        provider_result=provider_result,
        assessments=assessments,
        telemetry=envelope.payload.telemetry,
    )
