"""Tests for A/B testing fields on PromptResponse (N2)."""

from __future__ import annotations

import hashlib

import pytest

from taproot_sdk.prompts.models import PromptResponse, PromptType


def _make_prompt(
    content: str = "Hello {{user_name}}!",
    required_variables: tuple[str, ...] = ("user_name",),
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


class TestAbTestFields:
    """Tests for ab_test and selected_variant fields on PromptResponse."""

    def test_default_ab_test_is_false(self) -> None:
        """By default, ab_test should be False."""
        prompt = _make_prompt()
        assert prompt.ab_test is False

    def test_default_selected_variant_is_none(self) -> None:
        """By default, selected_variant should be None."""
        prompt = _make_prompt()
        assert prompt.selected_variant is None

    def test_ab_test_true(self) -> None:
        """Can create PromptResponse with ab_test=True."""
        prompt = _make_prompt(ab_test=True, selected_variant=3)
        assert prompt.ab_test is True
        assert prompt.selected_variant == 3

    def test_ab_test_with_selected_variant(self) -> None:
        """selected_variant should match the version when ab_test is True."""
        prompt = _make_prompt(
            ab_test=True,
            selected_variant=5,
            version=5,
        )
        assert prompt.ab_test is True
        assert prompt.selected_variant == 5
        assert prompt.version == 5

    def test_ab_test_fields_are_frozen(self) -> None:
        """ab_test and selected_variant should be immutable."""
        prompt = _make_prompt(ab_test=True, selected_variant=3)

        with pytest.raises(AttributeError):
            prompt.ab_test = False  # type: ignore[misc]

        with pytest.raises(AttributeError):
            prompt.selected_variant = 5  # type: ignore[misc]

    def test_backward_compatible_without_ab_fields(self) -> None:
        """Existing code creating PromptResponse without ab fields should work."""
        prompt = PromptResponse(
            schema_version=1,
            name="legacy-prompt",
            version=1,
            content="Hello world",
            content_hash=hashlib.sha256(b"Hello world").hexdigest(),
            config={},
            required_variables=(),
        )
        assert prompt.ab_test is False
        assert prompt.selected_variant is None
        assert prompt.render() == "Hello world"

    def test_render_works_with_ab_test(self) -> None:
        """render() should work normally when ab_test is True."""
        prompt = _make_prompt(ab_test=True, selected_variant=3)
        result = prompt.render(user_name="Alice")
        assert result == "Hello Alice!"

    def test_verify_hash_works_with_ab_test(self) -> None:
        """verify_hash() should work normally when ab_test is True."""
        prompt = _make_prompt(ab_test=True, selected_variant=3)
        assert prompt.verify_hash() is True

    def test_ab_test_false_with_selected_variant_none(self) -> None:
        """Non-A/B test prompts should have selected_variant=None."""
        prompt = _make_prompt(ab_test=False, selected_variant=None)
        assert prompt.ab_test is False
        assert prompt.selected_variant is None

    def test_prompt_type_text_with_ab_test(self) -> None:
        """TEXT prompts with ab_test should work."""
        prompt = _make_prompt(
            ab_test=True,
            selected_variant=3,
            prompt_type=PromptType.TEXT,
        )
        assert prompt.prompt_type == PromptType.TEXT
        assert prompt.ab_test is True
