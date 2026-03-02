"""Tests for taproot_sdk.exceptions — error verbosity and hierarchy."""

from __future__ import annotations

import pytest

from taproot_sdk.exceptions import (
    AuthenticationError,
    ConflictError,
    PromptNotFoundError,
    RateLimitError,
    ServerError,
    TaprootAPIError,
    TaprootError,
    ValidationError,
)
from taproot_sdk.prompts.exceptions import MissingVariableError


class TestExceptionHierarchy:
    """All SDK exceptions should be catchable via TaprootError."""

    def test_taproot_api_error_is_taproot_error(self) -> None:
        assert issubclass(TaprootAPIError, TaprootError)

    def test_prompt_not_found_is_taproot_error(self) -> None:
        assert issubclass(PromptNotFoundError, TaprootError)

    def test_authentication_error_is_taproot_error(self) -> None:
        assert issubclass(AuthenticationError, TaprootError)

    def test_conflict_error_is_taproot_error(self) -> None:
        assert issubclass(ConflictError, TaprootError)

    def test_rate_limit_error_is_taproot_error(self) -> None:
        assert issubclass(RateLimitError, TaprootError)

    def test_server_error_is_taproot_error(self) -> None:
        assert issubclass(ServerError, TaprootError)

    def test_validation_error_is_taproot_error(self) -> None:
        assert issubclass(ValidationError, TaprootError)

    def test_missing_variable_error_is_taproot_error(self) -> None:
        """MissingVariableError must inherit from TaprootError (gap #9)."""
        assert issubclass(MissingVariableError, TaprootError)

    def test_catch_all_catches_missing_variable(self) -> None:
        """except TaprootError should catch MissingVariableError."""
        with pytest.raises(TaprootError):
            raise MissingVariableError("user", ("user", "name"))


class TestTaprootAPIErrorMessage:
    """TaprootAPIError.__str__ includes request_url (gap #7)."""

    def test_includes_url_in_str(self) -> None:
        err = TaprootAPIError(
            500, "Internal server error",
            service="prompts",
            request_url="https://api.test/serve/test/welcome",
        )
        msg = str(err)
        assert "https://api.test/serve/test/welcome" in msg
        assert "HTTP 500 [prompts]" in msg

    def test_omits_url_when_empty(self) -> None:
        err = TaprootAPIError(400, "Bad request")
        msg = str(err)
        assert "URL:" not in msg
        assert "Bad request" in msg

    def test_service_in_brackets(self) -> None:
        err = TaprootAPIError(500, "fail", service="evals")
        assert "[evals]" in str(err)


class TestPromptNotFoundError:
    """PromptNotFoundError includes request_url (gap #2)."""

    def test_includes_name_and_project(self) -> None:
        err = PromptNotFoundError("welcome", project_id="test")
        assert "Prompt 'welcome'" in str(err)
        assert "project 'test'" in str(err)
        assert "not found" in str(err)

    def test_includes_version(self) -> None:
        err = PromptNotFoundError("welcome", project_id="test", version=3)
        assert "version 3" in str(err)

    def test_includes_label(self) -> None:
        err = PromptNotFoundError("welcome", project_id="test", label="production")
        assert "label 'production'" in str(err)

    def test_includes_request_url(self) -> None:
        err = PromptNotFoundError(
            "welcome", project_id="test",
            request_url="https://api.test/serve/test/welcome",
        )
        assert "https://api.test/serve/test/welcome" in str(err)

    def test_status_code_is_404(self) -> None:
        err = PromptNotFoundError("welcome")
        assert err.status_code == 404


class TestAuthenticationError:
    """AuthenticationError includes project_id and actionable hints (gaps #3, #5)."""

    def test_403_includes_project_id(self) -> None:
        err = AuthenticationError(403, "Forbidden", project_id="my-project")
        msg = str(err)
        assert "my-project" in msg
        assert "Verify your API key" in msg

    def test_403_bare_forbidden_becomes_actionable(self) -> None:
        err = AuthenticationError(403, "Forbidden")
        msg = str(err)
        assert "Access denied" in msg
        assert "Verify your API key" in msg

    def test_401_bare_becomes_actionable(self) -> None:
        err = AuthenticationError(401, "Unauthorized")
        msg = str(err)
        assert "Authentication failed" in msg
        assert "valid" in msg

    def test_401_empty_detail_becomes_actionable(self) -> None:
        err = AuthenticationError(401)
        msg = str(err)
        assert "Authentication failed" in msg

    def test_custom_detail_preserved(self) -> None:
        err = AuthenticationError(403, "Custom auth failure message")
        assert "Custom auth failure message" in str(err)


class TestServerError:
    """ServerError includes retry context (gap #4)."""

    def test_single_attempt_no_retry_context(self) -> None:
        err = ServerError(500, "Internal server error", attempts=1)
        msg = str(err)
        assert "attempts" not in msg.lower()
        assert "Internal server error" in msg

    def test_multiple_attempts_includes_context(self) -> None:
        err = ServerError(500, "Internal server error", attempts=6, total_wait_seconds=31)
        msg = str(err)
        assert "6 attempts" in msg
        assert "31s" in msg
        assert "try again later" in msg

    def test_status_code_preserved(self) -> None:
        err = ServerError(502, "Bad Gateway", attempts=3, total_wait_seconds=7)
        assert err.status_code == 502
        assert err.attempts == 3

    def test_is_taproot_api_error(self) -> None:
        err = ServerError(500, "fail")
        assert isinstance(err, TaprootAPIError)


class TestValidationError:
    """ValidationError formats field errors readably (gap #1)."""

    def test_formats_fastapi_errors(self) -> None:
        errors = [
            {"loc": ["body", "name"], "msg": "field required", "type": "value_error.missing"},
            {"loc": ["body", "prompt_type"], "msg": "value is not a valid enumeration member", "type": "value_error"},
        ]
        err = ValidationError(errors=errors)
        msg = str(err)
        assert "name: field required" in msg
        assert "prompt_type: value is not a valid enumeration member" in msg

    def test_empty_errors_uses_default_detail(self) -> None:
        err = ValidationError()
        assert "Validation error" in str(err)

    def test_custom_detail_preserved_when_not_default(self) -> None:
        err = ValidationError("Custom validation message", errors=[{"loc": [], "msg": "test"}])
        assert "Custom validation message" in str(err)

    def test_errors_attribute_populated(self) -> None:
        errors = [{"loc": ["body", "name"], "msg": "required"}]
        err = ValidationError(errors=errors)
        assert len(err.errors) == 1
        assert err.errors[0]["msg"] == "required"

    def test_caps_at_five_errors(self) -> None:
        errors = [{"loc": ["body", f"field{i}"], "msg": f"error {i}"} for i in range(10)]
        err = ValidationError(errors=errors)
        msg = str(err)
        assert "5 more" in msg


class TestRateLimitError:
    """RateLimitError includes retry_after hint."""

    def test_includes_retry_after(self) -> None:
        err = RateLimitError(retry_after=30.0)
        msg = str(err)
        assert "30.0s" in msg

    def test_no_retry_after(self) -> None:
        err = RateLimitError()
        assert err.retry_after is None
