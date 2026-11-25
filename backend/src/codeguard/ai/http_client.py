from __future__ import annotations

import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from codeguard.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderConfigurationError,
    AIProviderHTTPError,
    AIProviderPermissionError,
    AIProviderRateLimitError,
    AIProviderRequestError,
    AIProviderResponseError,
    AIProviderServerError,
    AIProviderTimeoutError,
)


class AIProviderHTTPClient:
    """Asynchronous HTTP transport shared by AI provider integrations."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        validated_base_url = self._validate_base_url(base_url)

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise AIProviderConfigurationError(
                "AI provider timeout must be a finite positive number."
            )

        try:
            is_finite_timeout = math.isfinite(timeout)
        except OverflowError:
            is_finite_timeout = False

        if not is_finite_timeout or timeout <= 0:
            raise AIProviderConfigurationError(
                "AI provider timeout must be a finite positive number."
            )

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if api_key is not None:
            if (
                not isinstance(api_key, str)
                or not api_key.strip()
                or "\r" in api_key
                or "\n" in api_key
            ):
                raise AIProviderConfigurationError(
                    "AI provider API key must be a non-empty valid string."
                )

            headers["Authorization"] = f"Bearer {api_key.strip()}"

        self._client = httpx.AsyncClient(
            base_url=validated_base_url,
            headers=headers,
            timeout=httpx.Timeout(float(timeout)),
            transport=transport,
            follow_redirects=False,
        )

    @staticmethod
    def _validate_base_url(base_url: str) -> httpx.URL:
        if (
            not isinstance(base_url, str)
            or not base_url
            or any(character.isspace() or character == "\\" for character in base_url)
        ):
            raise AIProviderConfigurationError(
                "AI provider base URL must be a valid HTTPS URL."
            )

        try:
            parsed_url = urlsplit(base_url)
            port = parsed_url.port
            validated_url = httpx.URL(base_url)
        except (ValueError, httpx.InvalidURL, UnicodeError):
            raise AIProviderConfigurationError(
                "AI provider base URL must be a valid HTTPS URL."
            ) from None

        if (
            parsed_url.scheme.lower() != "https"
            or parsed_url.hostname is None
            or parsed_url.username is not None
            or parsed_url.password is not None
            or (port is not None and not 0 <= port <= 65535)
            or parsed_url.query
            or parsed_url.fragment
            or "?" in base_url
            or "#" in base_url
        ):
            raise AIProviderConfigurationError(
                "AI provider base URL must be a valid HTTPS URL without credentials, "
                "query strings, or fragments."
            )

        return validated_url

    @staticmethod
    def _validate_path(path: str) -> None:
        if (
            not isinstance(path, str)
            or not path
            or any(character.isspace() or ord(character) < 32 for character in path)
            or "\\" in path
            or path.startswith("//")
            or "?" in path
            or "#" in path
        ):
            raise ValueError("AI provider API path must be a safe relative path.")

        try:
            parsed_path = urlsplit(path)
        except ValueError:
            raise ValueError(
                "AI provider API path must be a safe relative path."
            ) from None

        decoded_path = unquote(parsed_path.path)
        if (
            parsed_path.scheme
            or parsed_path.netloc
            or parsed_path.query
            or parsed_path.fragment
            or "\\" in decoded_path
            or any(segment in {".", ".."} for segment in decoded_path.split("/"))
        ):
            raise ValueError("AI provider API path must be a safe relative path.")

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON object and return the validated JSON object response."""

        self._validate_path(path)

        try:
            response = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise AIProviderTimeoutError("AI provider request timed out.") from exc
        except httpx.RequestError as exc:
            raise AIProviderRequestError("AI provider request failed.") from exc

        self._raise_for_status(response)

        try:
            response_payload = response.json()
        except (UnicodeError, ValueError) as exc:
            raise AIProviderResponseError(
                "AI provider returned invalid JSON.",
                status_code=response.status_code,
                request_id=self._request_id(response),
            ) from exc

        if not isinstance(response_payload, dict):
            raise AIProviderResponseError(
                "AI provider response must be a JSON object.",
                status_code=response.status_code,
                request_id=self._request_id(response),
            )

        return response_payload

    @staticmethod
    def _request_id(response: httpx.Response) -> str | None:
        for header_name in ("x-request-id", "request-id"):
            request_id = response.headers.get(header_name)
            if request_id is not None and request_id.strip():
                return request_id.strip()

        return None

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None

        try:
            delay = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None

        if not math.isfinite(delay) or delay < 0:
            return None

        return delay

    @classmethod
    def _raise_for_status(cls, response: httpx.Response) -> None:
        if response.is_success:
            return

        status_code = response.status_code
        request_id = cls._request_id(response)
        message = f"AI provider returned HTTP {status_code}."

        if status_code == 401:
            raise AIProviderAuthenticationError(
                message,
                status_code=status_code,
                request_id=request_id,
            )

        if status_code == 403:
            raise AIProviderPermissionError(
                message,
                status_code=status_code,
                request_id=request_id,
            )

        if status_code == 429:
            raise AIProviderRateLimitError(
                message,
                status_code=status_code,
                request_id=request_id,
                retry_after=cls._retry_after(response),
            )

        if 500 <= status_code <= 599:
            raise AIProviderServerError(
                message,
                status_code=status_code,
                request_id=request_id,
            )

        raise AIProviderHTTPError(
            message,
            status_code=status_code,
            request_id=request_id,
        )

    async def __aenter__(self) -> AIProviderHTTPClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTPX client."""

        await self._client.aclose()
