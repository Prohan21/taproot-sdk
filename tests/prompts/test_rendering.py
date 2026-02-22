"""C3 – SDK Variable Rendering tests for PromptResponse.render().

Dedicated test suite verifying that client-side variable rendering works
correctly: placeholder substitution, required-variable validation,
extra-variable warnings, and immutability of the original content.
"""

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

    defaults: dict[str, object] = dict(
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


class TestVariableRendering:
    """C3 – Variable rendering via PromptResponse.render()."""

    def test_basic_variable_substitution(self) -> None:
        """Placeholders like {{var}} are replaced with the corresponding kwarg values."""
        prompt = _make_prompt()

        result = prompt.render(user_name="Alice", service="Taproot")

        assert result == "Hello Alice, welcome to Taproot!"

    def test_missing_required_variable_raises(self) -> None:
        """Omitting a required variable must raise MissingVariableError."""
        prompt = _make_prompt()

        with pytest.raises(MissingVariableError) as exc_info:
            prompt.render(user_name="Alice")  # missing 'service'

        assert exc_info.value.variable_name == "service"

    def test_extra_variable_warns(self) -> None:
        """Providing variables not listed in required_variables emits a warning."""
        prompt = _make_prompt()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prompt.render(user_name="Alice", service="Taproot", colour="blue")

        assert len(caught) == 1
        assert "colour" in str(caught[0].message)

    def test_required_variables_from_server_used(self) -> None:
        """Validation uses the server-provided required_variables, not placeholders
        extracted from the template content at render time.

        Here the template has a {{bonus}} placeholder that is NOT in
        required_variables.  Rendering should succeed without providing 'bonus'
        (the placeholder is left intact), but omitting a truly required variable
        still raises.
        """
        prompt = _make_prompt(
            content="{{greeting}} {{bonus}}",
            required_variables=("greeting",),
        )

        # 'bonus' is NOT required — render should succeed and leave it as-is
        result = prompt.render(greeting="Hi")
        assert result == "Hi {{bonus}}"

        # 'greeting' IS required — omitting it must raise
        with pytest.raises(MissingVariableError) as exc_info:
            prompt.render()

        assert exc_info.value.variable_name == "greeting"
        assert exc_info.value.required_variables == ("greeting",)

    def test_rendering_preserves_original_content(self) -> None:
        """Rendering returns a new string; the original content field is unchanged."""
        prompt = _make_prompt()
        original_content = prompt.content

        result = prompt.render(user_name="Bob", service="Taproot")

        # The rendered string differs from the template
        assert result != original_content
        # The original content is untouched
        assert prompt.content == original_content
        assert prompt.content == "Hello {{user_name}}, welcome to {{service}}!"

    def test_empty_content_renders(self) -> None:
        """An empty-string template with no required variables renders to ''."""
        prompt = _make_prompt(content="", required_variables=())

        result = prompt.render()

        assert result == ""

    def test_no_variables_content_unchanged(self) -> None:
        """A template with no placeholders and no required variables returns content as-is."""
        prompt = _make_prompt(
            content="Plain text with no placeholders.",
            required_variables=(),
        )

        result = prompt.render()

        assert result == "Plain text with no placeholders."

    def test_special_chars_in_values(self) -> None:
        """Replacement values containing regex-special characters render correctly."""
        prompt = _make_prompt(
            content="Filter: {{expr}}",
            required_variables=("expr",),
        )

        result = prompt.render(expr="price >= $100 && (x|y) \\d+")

        assert result == "Filter: price >= $100 && (x|y) \\d+"

    def test_multiple_occurrences_of_same_variable(self) -> None:
        """A variable appearing multiple times in the template is replaced everywhere."""
        prompt = _make_prompt(
            content="{{name}} says: Hello, I am {{name}}. Nice to meet you, from {{name}}.",
            required_variables=("name",),
        )

        result = prompt.render(name="Eve")

        assert result == "Eve says: Hello, I am Eve. Nice to meet you, from Eve."

    def test_missing_variable_error_attributes(self) -> None:
        """MissingVariableError exposes variable_name, required_variables, and a
        human-readable message string."""
        prompt = _make_prompt(
            content="{{a}} {{b}} {{c}}",
            required_variables=("a", "b", "c"),
        )

        with pytest.raises(MissingVariableError) as exc_info:
            prompt.render(a="1")  # missing b and c

        err = exc_info.value

        # Should raise on first missing required variable in order
        assert err.variable_name == "b"
        assert err.required_variables == ("a", "b", "c")
        # Human-readable message includes the variable name and the full list
        assert "b" in str(err)
        assert "Required variables" in str(err)
        # It is a proper Exception subclass
        assert isinstance(err, Exception)
