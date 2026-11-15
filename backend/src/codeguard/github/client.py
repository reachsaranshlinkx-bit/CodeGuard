from __future__ import annotations

import asyncio
import base64
import binascii
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from pydantic import ValidationError

from codeguard.github.exceptions import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubRequestError,
    GitHubResponseError,
    GitHubServerError,
)
from codeguard.github.schemas import (
    GitHubFileContent,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubRepository,
)

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
MAX_GITHUB_FILE_SIZE_BYTES = 1_000_000


class GitHubClient:
    """Asynchronous HTTP client for GitHub REST API communication."""

    def __init__(
        self,
        token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 2,
        max_retry_delay: float = 120.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 5
        ):
            raise ValueError("max_retries must be an integer between 0 and 5.")

        if (
            isinstance(max_retry_delay, bool)
            or not isinstance(max_retry_delay, (int, float))
            or not math.isfinite(max_retry_delay)
            or max_retry_delay <= 0
        ):
            raise ValueError("max_retry_delay must be a finite positive number.")

        self._max_retries = max_retries
        self._max_retry_delay = float(max_retry_delay)
        self._sleep = sleep if sleep is not None else asyncio.sleep

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "CodeGuard/0.1.0",
        }

        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"

        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            follow_redirects=False,
        )

    async def __aenter__(self) -> GitHubClient:
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

    def _get_retry_delay(
        self,
        error: GitHubRequestError | GitHubServerError | GitHubRateLimitError,
        retry_index: int,
    ) -> float | None:
        """Calculate a safe delay before the next retry."""

        if isinstance(error, GitHubRateLimitError):
            if error.retry_after is not None:
                delay = error.retry_after
            else:
                delay = 60.0 * (2**retry_index)

        else:
            delay = 0.5 * (2**retry_index)

        if not math.isfinite(delay) or delay < 0:
            return None

        if delay > self._max_retry_delay:
            return None

        return delay

    async def _get_response(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> httpx.Response:
        """Send a GET request with bounded transient-error retries."""

        if not path.startswith("/") or path.startswith("//") or "://" in path:
            raise ValueError(
                "GitHub API path must start with '/' and "
                "must not contain an absolute URL."
            )
        total_retry_delay = 0.0
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.get(
                    path,
                    params=params,
                )

            except httpx.RequestError as exc:
                error = GitHubRequestError(
                    f"GitHub request failed: {type(exc).__name__}."
                )

                if attempt >= self._max_retries:
                    raise error from exc

                delay = self._get_retry_delay(
                    error,
                    retry_index=attempt,
                )

                if delay is None or total_retry_delay + delay > self._max_retry_delay:
                    raise error from exc

                total_retry_delay += delay
                await self._sleep(delay)
                continue

            try:
                self._raise_for_status(response)

            except (GitHubRateLimitError, GitHubServerError) as exc:
                if attempt >= self._max_retries:
                    raise

                delay = self._get_retry_delay(
                    exc,
                    retry_index=attempt,
                )

                if delay is None or total_retry_delay + delay > self._max_retry_delay:
                    raise

                total_retry_delay += delay
                await self._sleep(delay)
                continue

            return response

        raise RuntimeError("GitHub request retry loop exited unexpectedly.")

    @staticmethod
    def _decode_json_response(
        response: httpx.Response,
    ) -> dict[str, Any] | list[Any]:
        """Decode and validate the top-level JSON structure."""

        try:
            payload = response.json()
        except ValueError as exc:
            raise GitHubResponseError(
                "GitHub returned an invalid JSON response.",
                status_code=response.status_code,
                request_url=str(response.request.url),
            ) from exc

        if not isinstance(payload, (dict, list)):
            raise GitHubResponseError(
                "GitHub returned an unexpected JSON structure.",
                status_code=response.status_code,
                request_url=str(response.request.url),
            )

        return payload

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Send a GET request and return its decoded JSON response."""

        response = await self._get_response(
            path,
            params=params,
        )

        return self._decode_json_response(response)

    @staticmethod
    def _get_next_page(
        link_header: str | None,
        *,
        expected_path: str,
        pull_number: int,
        current_page: int,
    ) -> int | None:
        """Safely extract the next page number from GitHub's Link header."""

        if not link_header:
            return None

        links = re.finditer(
            r'<([^<>]+)>\s*;\s*rel="?([^";,\s]+)"?',
            link_header,
        )

        for match in links:
            next_url, relation = match.groups()

            if relation != "next":
                continue

            parsed_url = urlsplit(next_url)

            is_expected_path = (
                parsed_url.path == expected_path
                or re.fullmatch(
                    rf"/repositories/\d+/pulls/{pull_number}/files",
                    parsed_url.path,
                )
                is not None
            )

            try:
                valid_origin = (
                    parsed_url.scheme == "https"
                    and parsed_url.hostname == "api.github.com"
                    and parsed_url.port in (None, 443)
                    and parsed_url.username is None
                    and parsed_url.password is None
                )
            except ValueError:
                valid_origin = False

            if not valid_origin or not is_expected_path:
                raise GitHubResponseError(
                    "GitHub returned an untrusted pagination URL.",
                    status_code=200,
                    request_url=next_url,
                )

            try:
                query = parse_qs(parsed_url.query)

                next_page = int(query["page"][0])
                per_page = int(query["per_page"][0])

            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise GitHubResponseError(
                    "GitHub returned an invalid pagination link.",
                    status_code=200,
                    request_url=next_url,
                ) from exc

            if per_page != 100 or next_page != current_page + 1:
                raise GitHubResponseError(
                    "GitHub returned unexpected pagination parameters.",
                    status_code=200,
                    request_url=next_url,
                )

            return next_page

        return None

    async def get_all_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        *,
        max_pages: int = 30,
    ) -> list[GitHubPullRequestFile]:
        """Retrieve all available changed PR files using pagination."""

        owner = owner.strip()
        repo = repo.strip()

        if not owner or not repo:
            raise ValueError("Repository owner and name must not be empty.")

        if (
            any(char in owner + repo for char in ("/", "\\", "?", "#"))
            or owner in {".", ".."}
            or repo in {".", ".."}
        ):
            raise ValueError("Repository owner and name must be valid path segments.")

        if (
            isinstance(pull_number, bool)
            or not isinstance(pull_number, int)
            or pull_number < 1
        ):
            raise ValueError("Pull request number must be a positive integer.")

        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or not 1 <= max_pages <= 30
        ):
            raise ValueError("max_pages must be an integer between 1 and 30.")

        path = (
            f"/repos/{quote(owner, safe='')}"
            f"/{quote(repo, safe='')}"
            f"/pulls/{pull_number}/files"
        )

        files: list[GitHubPullRequestFile] = []
        current_page = 1

        for _ in range(max_pages):
            request_url = (
                f"{GITHUB_API_BASE_URL}{path}?per_page=100&page={current_page}"
            )

            response = await self._get_response(
                path,
                params={
                    "per_page": 100,
                    "page": current_page,
                },
            )

            payload = self._decode_json_response(response)

            if not isinstance(payload, list):
                raise GitHubResponseError(
                    "GitHub pull request files response must be a JSON array.",
                    status_code=200,
                    request_url=request_url,
                )

            try:
                page_files = [
                    GitHubPullRequestFile.model_validate(item) for item in payload
                ]
            except ValidationError as exc:
                raise GitHubResponseError(
                    "GitHub pull request file response did not "
                    "match the expected schema.",
                    status_code=200,
                    request_url=request_url,
                ) from exc

            files.extend(page_files)

            next_page = self._get_next_page(
                response.headers.get("Link"),
                expected_path=path,
                pull_number=pull_number,
                current_page=current_page,
            )

            if next_page is None:
                return files

            current_page = next_page

        raise GitHubResponseError(
            "Maximum pagination page limit reached. "
            "The returned file list may be incomplete.",
            status_code=200,
            request_url=request_url,
        )

    async def get_repository(
        self,
        owner: str,
        repo: str,
    ) -> GitHubRepository:
        """Retrieve and validate GitHub repository metadata."""

        owner = owner.strip()
        repo = repo.strip()

        if not owner or not repo:
            raise ValueError("Repository owner and name must not be empty.")

        if any(char in owner + repo for char in ("/", "\\", "?", "#")):
            raise ValueError("Repository owner and name must be valid path segments.")

        path = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"

        request_url = f"{GITHUB_API_BASE_URL}{path}"

        payload = await self.get_json(path)

        try:
            return GitHubRepository.model_validate(payload)
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub repository response did not match the expected schema.",
                status_code=200,
                request_url=request_url,
            ) from exc

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
        """Retrieve and validate GitHub pull request metadata."""

        owner = owner.strip()
        repo = repo.strip()

        if not owner or not repo:
            raise ValueError("Repository owner and name must not be empty.")

        if (
            any(char in owner + repo for char in ("/", "\\", "?", "#"))
            or owner in {".", ".."}
            or repo in {".", ".."}
        ):
            raise ValueError("Repository owner and name must be valid path segments.")

        if (
            isinstance(pull_number, bool)
            or not isinstance(pull_number, int)
            or pull_number < 1
        ):
            raise ValueError("Pull request number must be a positive integer.")

        path = (
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/pulls/{pull_number}"
        )

        request_url = f"{GITHUB_API_BASE_URL}{path}"

        payload = await self.get_json(path)

        try:
            return GitHubPullRequest.model_validate(payload)
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request response did not match the expected schema.",
                status_code=200,
                request_url=request_url,
            ) from exc

    async def get_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        *,
        per_page: int = 100,
        page: int = 1,
    ) -> list[GitHubPullRequestFile]:
        """Retrieve and validate one page of changed PR files."""

        owner = owner.strip()
        repo = repo.strip()

        if not owner or not repo:
            raise ValueError("Repository owner and name must not be empty.")

        if (
            any(char in owner + repo for char in ("/", "\\", "?", "#"))
            or owner in {".", ".."}
            or repo in {".", ".."}
        ):
            raise ValueError("Repository owner and name must be valid path segments.")

        if (
            isinstance(pull_number, bool)
            or not isinstance(pull_number, int)
            or pull_number < 1
        ):
            raise ValueError("Pull request number must be a positive integer.")

        if (
            isinstance(per_page, bool)
            or not isinstance(per_page, int)
            or not 1 <= per_page <= 100
        ):
            raise ValueError("per_page must be an integer between 1 and 100.")

        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer.")

        path = (
            f"/repos/{quote(owner, safe='')}"
            f"/{quote(repo, safe='')}"
            f"/pulls/{pull_number}/files"
        )

        request_url = f"{GITHUB_API_BASE_URL}{path}?per_page={per_page}&page={page}"

        payload = await self.get_json(
            path,
            params={
                "per_page": per_page,
                "page": page,
            },
        )

        if not isinstance(payload, list):
            raise GitHubResponseError(
                "GitHub pull request files response must be a JSON array.",
                status_code=200,
                request_url=request_url,
            )

        files: list[GitHubPullRequestFile] = []

        try:
            for item in payload:
                files.append(GitHubPullRequestFile.model_validate(item))
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request file response did not match the expected schema.",
                status_code=200,
                request_url=request_url,
            ) from exc

        return files

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        """Retrieve and decode a text file at a specific Git reference."""

        owner = owner.strip()
        repo = repo.strip()
        ref = ref.strip()

        if not owner or not repo:
            raise ValueError("Repository owner and name must not be empty.")

        if (
            any(char in owner + repo for char in ("/", "\\", "?", "#"))
            or owner in {".", ".."}
            or repo in {".", ".."}
        ):
            raise ValueError("Repository owner and name must be valid path segments.")

        if not path or not path.strip():
            raise ValueError("File path must not be empty.")

        if (
            path.startswith("/")
            or "\\" in path
            or any(segment in {"", ".", ".."} for segment in path.split("/"))
        ):
            raise ValueError("File path must be a safe relative path.")

        if not ref:
            raise ValueError("Git reference must not be empty.")

        encoded_path = "/".join(quote(segment, safe="") for segment in path.split("/"))

        api_path = (
            f"/repos/{quote(owner, safe='')}"
            f"/{quote(repo, safe='')}"
            f"/contents/{encoded_path}"
        )

        request_url = f"{GITHUB_API_BASE_URL}{api_path}?ref={quote(ref, safe='')}"

        payload = await self.get_json(
            api_path,
            params={"ref": ref},
        )

        if not isinstance(payload, dict):
            raise GitHubResponseError(
                "GitHub file content response must be a JSON object.",
                status_code=200,
                request_url=request_url,
            )

        raw_size = payload.get("size")

        if (
            isinstance(raw_size, int)
            and not isinstance(raw_size, bool)
            and raw_size > MAX_GITHUB_FILE_SIZE_BYTES
        ):
            raise GitHubResponseError(
                "GitHub file exceeds CodeGuard's supported size limit.",
                status_code=200,
                request_url=request_url,
            )

        try:
            file_content = GitHubFileContent.model_validate(payload)
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub file content response did not match the expected schema.",
                status_code=200,
                request_url=request_url,
            ) from exc

        encoded_content = "".join(file_content.content.split())

        try:
            decoded_content = base64.b64decode(
                encoded_content,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise GitHubResponseError(
                "GitHub returned invalid Base64 file content.",
                status_code=200,
                request_url=request_url,
            ) from exc

        if len(decoded_content) != file_content.size:
            raise GitHubResponseError(
                "GitHub file content size does not match its metadata.",
                status_code=200,
                request_url=request_url,
            )

        try:
            return decoded_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitHubResponseError(
                "GitHub file content is not valid UTF-8 text.",
                status_code=200,
                request_url=request_url,
            ) from exc

    @staticmethod
    def _get_error_message(response: httpx.Response) -> str:
        """Extract an error message from GitHub's response."""

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if isinstance(payload, dict):
            message = payload.get("message")

            if isinstance(message, str) and message.strip():
                return message

        return f"GitHub API returned HTTP {response.status_code}."

    @staticmethod
    def _get_retry_after(
        response: httpx.Response,
    ) -> float | None:
        """Calculate the remaining rate-limit wait time."""

        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)

                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)

                    return max(
                        (retry_at - datetime.now(UTC)).total_seconds(),
                        0.0,
                    )
                except (TypeError, ValueError, OverflowError):
                    pass

        reset_time = response.headers.get("X-RateLimit-Reset")

        if reset_time:
            try:
                return max(float(reset_time) - time.time(), 0.0)
            except ValueError:
                pass

        return None

    @classmethod
    def _raise_for_status(cls, response: httpx.Response) -> None:
        """Convert unsuccessful HTTP responses into GitHub exceptions."""

        if response.is_success:
            return

        status_code = response.status_code
        request_url = str(response.request.url)
        message = cls._get_error_message(response)

        if status_code == 401:
            raise GitHubAuthenticationError(
                message,
                status_code=status_code,
                request_url=request_url,
            )

        if status_code in (403, 429):
            is_rate_limited = (
                status_code == 429
                or response.headers.get("X-RateLimit-Remaining") == "0"
                or "rate limit" in message.lower()
                or "retry-after" in response.headers
            )

            if is_rate_limited:
                raise GitHubRateLimitError(
                    message,
                    retry_after=cls._get_retry_after(response),
                    status_code=status_code,
                    request_url=request_url,
                )

            raise GitHubPermissionError(
                message,
                status_code=status_code,
                request_url=request_url,
            )

        if status_code == 404:
            raise GitHubNotFoundError(
                message,
                status_code=status_code,
                request_url=request_url,
            )

        if 500 <= status_code <= 599:
            raise GitHubServerError(
                message,
                status_code=status_code,
                request_url=request_url,
            )

        raise GitHubAPIError(
            message,
            status_code=status_code,
            request_url=request_url,
        )
