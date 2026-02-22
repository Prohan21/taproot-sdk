"""Tests for taproot_evals.prompts.models module."""

from __future__ import annotations

import hashlib
import warnings

import pytest

from taproot_evals.prompts.exceptions import MissingVariableError
from taproot_evals.prompts.models import PromptResponse


def _make_prompt(
    content: str = "Hello {{user_name}}, welcome to {{service}}!",
    required_variables: tuple[str, ...] = ("user_name", "service"),
    content_hash: str | None = None,
    **overrides: object,
) -> PromptResponse:
    """Factory helper to build a PromptResponse with sensible defaults."""
    if content_hash is None:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    defaults = dict(
        schema_version=1,
        name="test-prompt",
        version=3,
        content=content,
        content_hash=content_hash,
        config={"model": "gpt-4o", "temperature": 0.7},
        required_variables=required_variables,
        label=None,
        cached_at=None,
    )
    defaults.update(overrides)
    return PromptResponse(**defaults)  # type: ignore[arg-type]


class TestPromptResponseRender:
    """Tests for PromptResponse.render()."""

    def test_render_replaces_all_variables(self) -> None:
        prompt = _make_prompt()

        result = prompt.render(user_name="Alice", service="Taproot")

        assert result == "Hello Alice, welcome to Taproot!"

    def test_render_returns_new_string(self) -> None:
        """render() must not mutate the original content."""
        prompt = _make_prompt()

        result = prompt.render(user_name="Alice", service="Taproot")

        assert result != prompt.content
        assert prompt.content == "Hello {{user_name}}, welcome to {{service}}!"

    def test_render_with_no_variables(self) -> None:
        prompt = _make_prompt(
            content="No variables here.",
            required_variables=(),
        )

        result = prompt.render()

        assert result == "No variables here."

    def test_render_missing_required_variable_raises(self) -> None:
        prompt = _make_prompt()

        with pytest.raises(MissingVariableError) as exc_info:
            prompt.render(user_name="Alice")

        assert exc_info.value.variable_name == "service"
        assert "service" in str(exc_info.value)

    def test_render_missing_multiple_required_variables_raises_on_first(self) -> None:
        prompt = _make_prompt()

        with pytest.raises(MissingVariableError) as exc_info:
            prompt.render()

        # Should raise on the first missing required variable
        assert exc_info.value.variable_name == "user_name"
        assert exc_info.value.required_variables == ("user_name", "service")

    def test_render_extra_variable_warns(self) -> None:
        prompt = _make_prompt()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = prompt.render(user_name="Alice", service="Taproot", extra_var="ignored")

        assert result == "Hello Alice, welcome to Taproot!"
        assert len(caught) == 1
        assert "extra_var" in str(caught[0].message)

    def test_render_extra_variable_substituted_if_placeholder_exists(self) -> None:
        """Extra variables should still be substituted if a matching placeholder exists."""
        prompt = _make_prompt(
            content="Hi {{user_name}} on {{service}}. Note: {{extra}}.",
            required_variables=("user_name", "service"),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = prompt.render(
                user_name="Bob", service="Taproot", extra="bonus content"
            )

        assert result == "Hi Bob on Taproot. Note: bonus content."

    def test_render_unmatched_placeholder_left_intact(self) -> None:
        """Placeholders without a matching variable arg should remain as-is."""
        prompt = _make_prompt(
            content="Hi {{user_name}}, your id is {{internal_id}}.",
            required_variables=("user_name",),
        )

        result = prompt.render(user_name="Alice")

        assert result == "Hi Alice, your id is {{internal_id}}."

    def test_render_repeated_placeholder(self) -> None:
        prompt = _make_prompt(
            content="{{name}} said: hello {{name}}!",
            required_variables=("name",),
        )

        result = prompt.render(name="Eve")

        assert result == "Eve said: hello Eve!"

    def test_render_special_characters_in_value(self) -> None:
        """Variable values containing regex-special chars should not break rendering."""
        prompt = _make_prompt(
            content="Query: {{query}}",
            required_variables=("query",),
        )

        result = prompt.render(query="price >= $100 && (category == 'A')")

        assert result == "Query: price >= $100 && (category == 'A')"


class TestPromptResponseVerifyHash:
    """Tests for PromptResponse.verify_hash()."""

    def test_verify_hash_success(self) -> None:
        content = "Hello world"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        prompt = _make_prompt(content=content, content_hash=content_hash)

        assert prompt.verify_hash() is True

    def test_verify_hash_failure(self) -> None:
        prompt = _make_prompt(
            content="Hello world",
            content_hash="0000000000000000000000000000000000000000000000000000000000000000",
        )

        assert prompt.verify_hash() is False

    def test_verify_hash_empty_content(self) -> None:
        content = ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        prompt = _make_prompt(
            content=content,
            content_hash=content_hash,
            required_variables=(),
        )

        assert prompt.verify_hash() is True

    def test_verify_hash_unicode_content(self) -> None:
        content = "Bonjour le monde! \u2603 \U0001f680"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        prompt = _make_prompt(
            content=content,
            content_hash=content_hash,
            required_variables=(),
        )

        assert prompt.verify_hash() is True


class TestPromptResponseImmutability:
    """Tests for PromptResponse being a frozen (immutable) dataclass."""

    def test_cannot_set_attribute(self) -> None:
        prompt = _make_prompt()

        with pytest.raises(AttributeError):
            prompt.name = "modified"  # type: ignore[misc]

    def test_cannot_set_content(self) -> None:
        prompt = _make_prompt()

        with pytest.raises(AttributeError):
            prompt.content = "modified"  # type: ignore[misc]

    def test_cannot_set_version(self) -> None:
        prompt = _make_prompt()

        with pytest.raises(AttributeError):
            prompt.version = 999  # type: ignore[misc]


class TestMissingVariableError:
    """Tests for the MissingVariableError exception."""

    def test_attributes(self) -> None:
        err = MissingVariableError("foo", ("foo", "bar"))

        assert err.variable_name == "foo"
        assert err.required_variables == ("foo", "bar")

    def test_message_format(self) -> None:
        err = MissingVariableError("foo", ("foo", "bar"))

        assert "foo" in str(err)
        assert "Required variables" in str(err)

    def test_is_exception(self) -> None:
        err = MissingVariableError("x", ("x",))
        assert isinstance(err, Exception)
