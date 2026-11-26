from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NoReturn, Protocol, TypeVar

from codeguard.ai import operation_result_codec
from codeguard.ai.contracts import AIReviewInput
from codeguard.ai.engine import AIReviewEngine, AIReviewExecutionResult
from codeguard.ai.exceptions import (
    AIProviderError,
    AIProviderFailureDisposition,
    AIProviderRequestError,
    AIRequestTransmissionState,
    classify_ai_provider_error,
)
from codeguard.ai.execution_fingerprint import build_ai_execution_input_fingerprint
from codeguard.ai.execution_metadata import AIExecutionMetadata
from codeguard.ai.factory import AIReviewProviderSelection
from codeguard.ai.operation_identity import (
    AIOperationIdentity,
    build_ai_operation_key,
)
from codeguard.ai.operation_repository import (
    AIOperationClaimResult,
    AIOperationClaimStatus,
    AIOperationRecord,
    AIOperationStatus,
)
from codeguard.ai.operation_result_codec import AIReviewResultCodecError
from codeguard.ai.provider import AIReviewProvider
from codeguard.context.source_snapshot import RepositorySourceSnapshot

_ResultT = TypeVar("_ResultT")
_LEASE_REJECTION_MESSAGE = (
    "AI operation transition rejected: ownership or state changed."
)


class AIExecutionCoordinatorError(RuntimeError):
    """Base class for durable AI execution coordination failures."""

    def __init__(self, operation_key: str, message: str) -> None:
        super().__init__(message)
        self.operation_key = operation_key


class AIExecutionInProgressError(AIExecutionCoordinatorError):
    """Raised when another active lease owns the AI operation."""


class AIExecutionRetryableError(AIExecutionCoordinatorError):
    """Raised after a confirmed NOT_SENT failure has returned to PENDING."""


class AIExecutionPermanentFailureError(AIExecutionCoordinatorError):
    """Raised when the operation has a permanent failure disposition."""

    def __init__(self, record: AIOperationRecord) -> None:
        super().__init__(
            record.operation_key,
            "AI operation has a permanent failure disposition.",
        )
        self.disposition = record.disposition
        self.safe_error = record.safe_error


class AIExecutionOutcomeUnknownError(AIExecutionCoordinatorError):
    """Raised when the operation requires provider or manual reconciliation."""


class AIExecutionLeaseLostError(AIExecutionCoordinatorError):
    """Raised when the operation lease token is no longer current."""


class AIExecutionPersistenceError(AIExecutionCoordinatorError):
    """Raised when durable operation state or result persistence fails."""


class AIExecutionInvalidStoredResultError(AIExecutionCoordinatorError):
    """Raised when a successful operation has no usable stored result."""


@dataclass(frozen=True, slots=True)
class AIExecutionOutcome:
    """A complete AI execution result, either newly generated or reused."""

    operation_key: str
    execution_result: AIReviewExecutionResult
    reused: bool


class AIOperationRepositoryProtocol(Protocol):
    """Repository operations required by the execution coordinator."""

    def create_or_get(self, identity: AIOperationIdentity) -> AIOperationRecord: ...

    def claim(
        self,
        operation_key: str,
        lease_duration_seconds: float,
    ) -> AIOperationClaimResult: ...

    def get(self, operation_key: str) -> AIOperationRecord | None: ...

    def mark_request_started(
        self,
        operation_key: str,
        lease_token: str,
    ) -> AIOperationRecord: ...

    def mark_succeeded(
        self,
        operation_key: str,
        lease_token: str,
        *,
        result_json: str,
        result_schema_version: int,
        serialization_version: int,
        provider_request_id: str | None = None,
    ) -> AIOperationRecord: ...

    def mark_safe_to_retry(
        self,
        operation_key: str,
        lease_token: str,
        *,
        transmission_state: AIRequestTransmissionState,
    ) -> AIOperationRecord: ...

    def mark_failed_permanent(
        self,
        operation_key: str,
        lease_token: str,
        *,
        disposition: str,
        safe_error: str,
    ) -> AIOperationRecord: ...

    def mark_outcome_unknown(
        self,
        operation_key: str,
        lease_token: str,
        *,
        disposition: str,
        safe_error: str,
    ) -> AIOperationRecord: ...


class RepositoryCallAdapter(Protocol):
    """Run synchronous repository calls without blocking the async event loop."""

    async def run(self, operation: Callable[[], _ResultT]) -> _ResultT: ...


class ThreadedRepositoryCallAdapter:
    """Offload SQLite calls and wait for an in-flight call before propagating cancel."""

    async def run(self, operation: Callable[[], _ResultT]) -> _ResultT:
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            try:
                result = await asyncio.shield(task)
            except Exception as operation_error:
                raise _RepositoryCallCancelled(
                    operation_error=operation_error,
                ) from cancellation
            raise _RepositoryCallCancelled(result=result) from cancellation


class _RepositoryCallCancelled(asyncio.CancelledError):
    """Cancellation received after a thread-offloaded repository call has settled."""

    def __init__(
        self,
        *,
        result: object = None,
        operation_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.operation_error = operation_error


class AIExecutionCoordinator:
    """Coordinate durable identity, lease ownership, AI execution, and result reuse."""

    def __init__(
        self,
        *,
        provider_selection: AIReviewProviderSelection,
        repository: AIOperationRepositoryProtocol,
        lease_duration_seconds: float = 300.0,
        engine_factory: Callable[[AIReviewProvider], AIReviewEngine] = AIReviewEngine,
        repository_call_adapter: RepositoryCallAdapter | None = None,
    ) -> None:
        if not isinstance(provider_selection, AIReviewProviderSelection):
            raise TypeError("provider_selection must be an AIReviewProviderSelection.")
        if not isinstance(provider_selection.metadata, AIExecutionMetadata):
            raise TypeError("provider_selection metadata must be AIExecutionMetadata.")
        if isinstance(lease_duration_seconds, bool) or not isinstance(
            lease_duration_seconds,
            (int, float),
        ):
            raise ValueError("lease_duration_seconds must be finite and positive.")
        try:
            finite_lease_duration = isfinite(lease_duration_seconds)
        except OverflowError:
            finite_lease_duration = False
        if not finite_lease_duration or lease_duration_seconds <= 0:
            raise ValueError("lease_duration_seconds must be finite and positive.")

        self._provider_selection = provider_selection
        self._repository = repository
        self._lease_duration_seconds = lease_duration_seconds
        self._repository_call_adapter = (
            repository_call_adapter or ThreadedRepositoryCallAdapter()
        )

        # The engine is always constructed from the provider paired with this metadata.
        self._engine = engine_factory(provider_selection.provider)
        if not isinstance(self._engine, AIReviewEngine):
            raise TypeError("engine_factory must return an AIReviewEngine.")

    async def execute(
        self,
        review_input: AIReviewInput,
        source_snapshot: RepositorySourceSnapshot,
        *,
        base_sha: str,
    ) -> AIExecutionOutcome:
        """Return a durable result, reusing a matching v2 success when available."""

        validated_input = self._validate_review_input(review_input)
        fingerprint = build_ai_execution_input_fingerprint(
            base_sha,
            validated_input,
            source_snapshot,
        )
        identity = AIOperationIdentity(
            owner=validated_input.owner,
            repo=validated_input.repo,
            pull_number=validated_input.pull_number,
            head_sha=validated_input.head_sha,
            input_fingerprint=fingerprint,
            **self._provider_selection.metadata.to_identity_components(),
        )
        operation_key = build_ai_operation_key(identity)

        lease_token: str | None = None
        try:
            record = await self._repository_call(
                self._repository.create_or_get,
                identity,
            )
            self._validate_record_identity(record, identity, operation_key)

            if record.status is AIOperationStatus.SUCCEEDED:
                return self._reuse_succeeded(record, identity, operation_key)
            if record.status is AIOperationStatus.REQUEST_STARTED:
                raise self._unknown_operation(record)
            if record.status is AIOperationStatus.FAILED_PERMANENT:
                raise AIExecutionPermanentFailureError(record)
            if record.status is AIOperationStatus.OUTCOME_UNKNOWN:
                raise self._unknown_operation(record)
            if record.status not in (
                AIOperationStatus.PENDING,
                AIOperationStatus.CLAIMED,
            ):
                raise AIExecutionPersistenceError(
                    operation_key,
                    "AI operation has an unsupported persisted state.",
                )

            claim = await self._repository_call(
                self._repository.claim,
                operation_key,
                self._lease_duration_seconds,
            )
            self._validate_record_identity(claim.record, identity, operation_key)
            if claim.status is not AIOperationClaimStatus.CLAIMED:
                return self._handle_unclaimed_operation(
                    claim,
                    identity,
                    operation_key,
                )
            if (
                claim.record.status is not AIOperationStatus.CLAIMED
                or not isinstance(claim.lease_token, str)
                or not claim.lease_token
            ):
                raise AIExecutionPersistenceError(
                    operation_key,
                    "Repository returned an invalid AI operation lease.",
                )

            lease_token = claim.lease_token
            try:
                started_record = await self._repository_call(
                    self._repository.mark_request_started,
                    operation_key,
                    lease_token,
                    transition=True,
                )
            except _RepositoryCallCancelled as cancellation:
                if (
                    isinstance(cancellation.result, AIOperationRecord)
                    and cancellation.result.status is AIOperationStatus.REQUEST_STARTED
                ):
                    lease_token = claim.lease_token
                raise
            self._validate_record_identity(started_record, identity, operation_key)
            if started_record.status is not AIOperationStatus.REQUEST_STARTED:
                raise AIExecutionPersistenceError(
                    operation_key,
                    "Repository did not record the AI request start.",
                )

            try:
                execution_result = await self._engine.review(
                    validated_input,
                    source_snapshot,
                )
            except asyncio.CancelledError:
                raise
            except AIProviderError as error:
                await self._handle_provider_error(
                    error,
                    operation_key=operation_key,
                    lease_token=lease_token,
                    identity=identity,
                )
            except Exception as error:
                await self._mark_unknown(
                    operation_key=operation_key,
                    lease_token=lease_token,
                    disposition="execution_error",
                    safe_error=type(error).__name__,
                )
                raise self._unknown_error(
                    operation_key,
                    "AI execution failed after request start.",
                ) from error

            try:
                encoded = operation_result_codec.encode(execution_result)
            except AIReviewResultCodecError as error:
                await self._mark_unknown(
                    operation_key=operation_key,
                    lease_token=lease_token,
                    disposition="result_encoding_failed",
                    safe_error=type(error).__name__,
                )
                raise AIExecutionPersistenceError(
                    operation_key,
                    "AI execution result could not be durably encoded.",
                ) from error
            except Exception as error:
                await self._mark_unknown(
                    operation_key=operation_key,
                    lease_token=lease_token,
                    disposition="result_encoding_failed",
                    safe_error=type(error).__name__,
                )
                raise AIExecutionPersistenceError(
                    operation_key,
                    "AI execution result could not be durably encoded.",
                ) from error

            try:
                persisted = await self._repository_call(
                    self._repository.mark_succeeded,
                    operation_key,
                    lease_token,
                    transition=True,
                    result_json=encoded.result_json,
                    result_schema_version=encoded.result_schema_version,
                    serialization_version=encoded.serialization_version,
                )
            except AIExecutionLeaseLostError:
                raise
            except asyncio.CancelledError:
                raise
            except AIExecutionPersistenceError as error:
                try:
                    await self._mark_unknown(
                        operation_key=operation_key,
                        lease_token=lease_token,
                        disposition="result_persistence_failed",
                        safe_error=type(error).__name__,
                    )
                except AIExecutionCoordinatorError as recovery_error:
                    error.add_note(
                        "Could not persist recovery state: "
                        f"{type(recovery_error).__name__}."
                    )
                raise

            self._validate_record_identity(persisted, identity, operation_key)
            if persisted.status is not AIOperationStatus.SUCCEEDED:
                raise AIExecutionPersistenceError(
                    operation_key,
                    "Repository did not persist the successful AI result.",
                )
            return AIExecutionOutcome(
                operation_key=operation_key,
                execution_result=execution_result,
                reused=False,
            )
        except asyncio.CancelledError as cancellation:
            if (
                isinstance(cancellation, _RepositoryCallCancelled)
                and cancellation.operation_error is not None
            ):
                cancellation.add_note(
                    "The cancelled repository call failed with "
                    f"{type(cancellation.operation_error).__name__}."
                )
            if lease_token is not None:
                try:
                    await self._resolve_cancellation(
                        operation_key=operation_key,
                        lease_token=lease_token,
                    )
                except Exception as cleanup_error:
                    cancellation.add_note(
                        "Could not persist cancellation recovery state: "
                        f"{type(cleanup_error).__name__}."
                    )
            raise

    @staticmethod
    def _validate_review_input(review_input: AIReviewInput) -> AIReviewInput:
        if not isinstance(review_input, AIReviewInput):
            raise TypeError("review_input must be an AIReviewInput instance.")
        return AIReviewInput.model_validate(
            review_input.model_dump(mode="python"),
            strict=True,
        )

    @staticmethod
    def _validate_record_identity(
        record: AIOperationRecord,
        identity: AIOperationIdentity,
        operation_key: str,
    ) -> None:
        if (
            not isinstance(record.identity, AIOperationIdentity)
            or record.identity != identity
            or record.operation_key != operation_key
            or build_ai_operation_key(record.identity) != operation_key
        ):
            raise AIExecutionPersistenceError(
                operation_key,
                "Persisted AI operation identity does not match the request.",
            )

    def _reuse_succeeded(
        self,
        record: AIOperationRecord,
        identity: AIOperationIdentity,
        operation_key: str,
    ) -> AIExecutionOutcome:
        self._validate_record_identity(record, identity, operation_key)
        if record.result_json is None:
            raise AIExecutionInvalidStoredResultError(
                operation_key,
                "Succeeded AI operation is missing its stored result.",
            )
        if record.result_schema_version is None or record.serialization_version is None:
            raise AIExecutionInvalidStoredResultError(
                operation_key,
                "Succeeded AI operation is missing result codec versions.",
            )
        try:
            execution_result = operation_result_codec.decode(
                record.result_json,
                record.result_schema_version,
                record.serialization_version,
            )
        except AIReviewResultCodecError as error:
            raise AIExecutionInvalidStoredResultError(
                operation_key,
                "Succeeded AI operation has an invalid or unsupported result.",
            ) from error
        return AIExecutionOutcome(
            operation_key=operation_key,
            execution_result=execution_result,
            reused=True,
        )

    def _handle_unclaimed_operation(
        self,
        claim: AIOperationClaimResult,
        identity: AIOperationIdentity,
        operation_key: str,
    ) -> AIExecutionOutcome:
        record = claim.record
        self._validate_record_identity(record, identity, operation_key)

        if claim.status is AIOperationClaimStatus.ALREADY_CLAIMED:
            raise AIExecutionInProgressError(
                operation_key,
                "Another worker currently owns the AI operation lease.",
            )
        if claim.status is AIOperationClaimStatus.SUCCEEDED:
            return self._reuse_succeeded(record, identity, operation_key)
        if claim.status is AIOperationClaimStatus.FAILED_PERMANENT:
            raise AIExecutionPermanentFailureError(record)
        if claim.status is AIOperationClaimStatus.OUTCOME_UNKNOWN:
            raise self._unknown_operation(record)
        if claim.status is AIOperationClaimStatus.REQUEST_STARTED:
            raise self._unknown_operation(record)
        raise AIExecutionPersistenceError(
            operation_key,
            "Repository returned an unsupported AI operation claim status.",
        )

    async def _handle_provider_error(
        self,
        error: AIProviderError,
        *,
        operation_key: str,
        lease_token: str,
        identity: AIOperationIdentity,
    ) -> NoReturn:
        disposition = classify_ai_provider_error(error)
        if disposition is AIProviderFailureDisposition.SAFE_TO_RETRY:
            if not (
                isinstance(error, AIProviderRequestError)
                and error.transmission_state is AIRequestTransmissionState.NOT_SENT
            ):
                await self._mark_unknown(
                    operation_key=operation_key,
                    lease_token=lease_token,
                    disposition=AIProviderFailureDisposition.OUTCOME_UNKNOWN.value,
                    safe_error=type(error).__name__,
                )
                raise self._unknown_error(
                    operation_key,
                    "AI provider transmission state is not confirmed safe to retry.",
                ) from error
            retry_record = await self._repository_call(
                self._repository.mark_safe_to_retry,
                operation_key,
                lease_token,
                transition=True,
                transmission_state=AIRequestTransmissionState.NOT_SENT,
            )
            self._validate_record_identity(retry_record, identity, operation_key)
            if retry_record.status is not AIOperationStatus.PENDING:
                raise AIExecutionPersistenceError(
                    operation_key,
                    "Repository did not return confirmed NOT_SENT to PENDING.",
                )
            raise AIExecutionRetryableError(
                operation_key,
                "AI provider confirmed that the request was not sent.",
            ) from error

        if disposition is AIProviderFailureDisposition.PERMANENT:
            failed_record = await self._repository_call(
                self._repository.mark_failed_permanent,
                operation_key,
                lease_token,
                transition=True,
                disposition=disposition.value,
                safe_error=type(error).__name__,
            )
            self._validate_record_identity(failed_record, identity, operation_key)
            if failed_record.status is not AIOperationStatus.FAILED_PERMANENT:
                raise AIExecutionPersistenceError(
                    operation_key,
                    "Repository did not persist the permanent AI failure.",
                )
            raise AIExecutionPermanentFailureError(failed_record) from error

        unknown_record = await self._mark_unknown(
            operation_key=operation_key,
            lease_token=lease_token,
            disposition=disposition.value,
            safe_error=type(error).__name__,
        )
        self._validate_record_identity(unknown_record, identity, operation_key)
        raise self._unknown_operation(unknown_record) from error

    async def _repository_call(
        self,
        method: Callable[..., _ResultT],
        *args: object,
        transition: bool = False,
        **kwargs: object,
    ) -> _ResultT:
        operation = partial(method, *args, **kwargs)
        try:
            return await self._repository_call_adapter.run(operation)
        except _RepositoryCallCancelled:
            raise
        except RuntimeError as error:
            if transition and str(error).startswith(_LEASE_REJECTION_MESSAGE):
                operation_key = str(args[0]) if args else "unknown"
                raise AIExecutionLeaseLostError(
                    operation_key,
                    "AI operation lease ownership was lost.",
                ) from error
            operation_key = str(args[0]) if args else "unknown"
            raise AIExecutionPersistenceError(
                operation_key,
                "AI operation repository access failed.",
            ) from error
        except Exception as error:
            operation_key = str(args[0]) if args else "unknown"
            raise AIExecutionPersistenceError(
                operation_key,
                "AI operation repository access failed.",
            ) from error

    async def _mark_unknown(
        self,
        *,
        operation_key: str,
        lease_token: str,
        disposition: str,
        safe_error: str,
    ) -> AIOperationRecord:
        record = await self._repository_call(
            self._repository.mark_outcome_unknown,
            operation_key,
            lease_token,
            transition=True,
            disposition=disposition,
            safe_error=safe_error,
        )
        if (
            record.operation_key != operation_key
            or record.status is not AIOperationStatus.OUTCOME_UNKNOWN
        ):
            raise AIExecutionPersistenceError(
                operation_key,
                "Repository did not persist the unknown AI operation state.",
            )
        return record

    async def _resolve_cancellation(
        self,
        *,
        operation_key: str,
        lease_token: str,
    ) -> None:
        record = await self._repository_call(self._repository.get, operation_key)
        if record is None or record.status is not AIOperationStatus.REQUEST_STARTED:
            return
        await self._mark_unknown(
            operation_key=operation_key,
            lease_token=lease_token,
            disposition="cancelled_after_request_started",
            safe_error="CancelledError",
        )

    @staticmethod
    def _unknown_operation(record: AIOperationRecord) -> AIExecutionOutcomeUnknownError:
        return AIExecutionOutcomeUnknownError(
            record.operation_key,
            "AI operation outcome is unknown and requires reconciliation.",
        )

    @staticmethod
    def _unknown_error(
        operation_key: str,
        message: str,
    ) -> AIExecutionOutcomeUnknownError:
        return AIExecutionOutcomeUnknownError(operation_key, message)
