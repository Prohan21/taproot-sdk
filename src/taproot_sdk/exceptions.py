"""SDK-level exceptions for the Taproot platform.

These wrap raw HTTP errors with contextual information (prompt name,
project ID, status code) so callers can handle failures without
parsing httpx internals.
"""

from __future__ import annotations

from typing import Any


class TaprootError(Exception):
    """Base exception for all Taproot SDK errors."""


class TaprootAPIError(TaprootError):
    """An HTTP error response from a Taproot service.

    Attributes:
        status_code: HTTP status code from the response.
        detail: Parsed error detail from the JSON body (if available).
        service: Which Taproot service returned the error (e.g. "prompts", "evals").
        request_url: The URL that was requested.
        body: Raw response body text.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        service: str = "",
        request_url: str = "",
        body: str = "",
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.service = service
        self.request_url = request_url
        self.body = body

        parts: list[str] = [f"HTTP {status_code}"]
        if service:
            parts[0] += f" [{service}]"
        parts.append(detail)
        if request_url:
            parts.append(f"(URL: {request_url})")
        super().__init__(" — ".join(parts) if len(parts) > 1 else parts[0])


class PromptNotFoundError(TaprootAPIError):
    """Raised when a prompt, version, or label is not found (404)."""

    def __init__(
        self,
        name: str,
        *,
        project_id: str = "",
        version: int | None = None,
        label: str | None = None,
        request_url: str = "",
        body: str = "",
    ) -> None:
        self.prompt_name = name
        self.prompt_project_id = project_id
        self.prompt_version = version
        self.prompt_label = label

        parts = [f"Prompt '{name}'"]
        if project_id:
            parts.append(f"in project '{project_id}'")
        if version is not None:
            parts.append(f"version {version}")
        if label is not None:
            parts.append(f"label '{label}'")
        detail = " ".join(parts) + " not found"

        super().__init__(
            404, detail, service="prompts", request_url=request_url, body=body,
        )


class AuthenticationError(TaprootAPIError):
    """Raised on 401/403 responses.

    Attributes:
        project_id: The project context (when available) to aid debugging.
        hint: An optional user-friendly suggestion for fixing the issue.
    """

    def __init__(
        self,
        status_code: int = 401,
        detail: str = "",
        *,
        service: str = "",
        request_url: str = "",
        body: str = "",
        project_id: str = "",
    ) -> None:
        self.auth_project_id = project_id

        # Build an actionable message based on status code
        if not detail or detail in ("Forbidden", "Unauthorized"):
            if status_code == 403:
                msg = "Access denied"
                if project_id:
                    msg += f" for project '{project_id}'"
                msg += (
                    ". Verify your API key has access to this project "
                    "and the correct permissions."
                )
                detail = msg
            elif status_code == 401:
                detail = (
                    "Authentication failed. Verify your API key is valid "
                    "and not expired."
                )

        super().__init__(
            status_code, detail, service=service, request_url=request_url, body=body,
        )


class ConflictError(TaprootAPIError):
    """Raised on 409 Conflict (e.g. duplicate prompt name)."""

    def __init__(
        self,
        detail: str = "Resource conflict",
        *,
        service: str = "",
        request_url: str = "",
        body: str = "",
    ) -> None:
        super().__init__(
            409, detail, service=service, request_url=request_url, body=body,
        )


class RateLimitError(TaprootAPIError):
    """Raised when rate limit is exceeded after all retries (429).

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header).
    """

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        *,
        service: str = "",
        request_url: str = "",
        retry_after: float | None = None,
        body: str = "",
    ) -> None:
        self.retry_after = retry_after
        if retry_after is not None and "retry" not in detail.lower():
            detail += f". Retry after {retry_after}s."
        super().__init__(
            429, detail, service=service, request_url=request_url, body=body,
        )


class ServerError(TaprootAPIError):
    """Raised on 5xx responses after all retries are exhausted.

    Attributes:
        attempts: Total number of attempts made (including retries).
        total_wait_seconds: Approximate total time spent waiting between retries.
    """

    def __init__(
        self,
        status_code: int,
        detail: str = "",
        *,
        service: str = "",
        request_url: str = "",
        body: str = "",
        attempts: int = 1,
        total_wait_seconds: float = 0,
    ) -> None:
        self.attempts = attempts
        self.total_wait_seconds = total_wait_seconds
        if not detail:
            detail = "Internal server error"
        if attempts > 1:
            detail += (
                f" (after {attempts} attempts over ~{total_wait_seconds:.0f}s)"
            )
            detail += ". This is a server-side issue — please try again later."
        super().__init__(
            status_code, detail, service=service, request_url=request_url, body=body,
        )


class ValidationError(TaprootAPIError):
    """Raised on 422 Unprocessable Entity (server-side validation failure).

    Attributes:
        errors: List of validation error details from the server.
    """

    def __init__(
        self,
        detail: str = "Validation error",
        *,
        errors: list[dict[str, Any]] | None = None,
        service: str = "",
        request_url: str = "",
        body: str = "",
    ) -> None:
        self.errors = errors or []

        # Build a more informative detail message when errors are available
        if self.errors and detail == "Validation error":
            field_msgs: list[str] = []
            for err in self.errors[:5]:  # Cap at 5 to avoid huge messages
                loc = err.get("loc", [])
                msg = err.get("msg", str(err))
                if loc:
                    # loc is typically ["body", "field_name"]
                    field = ".".join(str(part) for part in loc if part != "body")
                    if field:
                        field_msgs.append(f"  - {field}: {msg}")
                    else:
                        field_msgs.append(f"  - {msg}")
                else:
                    field_msgs.append(f"  - {msg}")
            if field_msgs:
                detail = "Validation failed:\n" + "\n".join(field_msgs)
                if len(self.errors) > 5:
                    detail += f"\n  ... and {len(self.errors) - 5} more"

        super().__init__(
            422, detail, service=service, request_url=request_url, body=body,
        )
