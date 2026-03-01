"""Tests for taproot_sdk.prompts.models module."""

from __future__ import annotations

import hashlib
import json
import warnings

import pytest

from taproot_sdk.prompts.exceptions import MissingVariableError
from taproot_sdk.prompts.models import (
    ChatMessage,
    PromptResponse,
    PromptType,
    ToolDefinition,
)


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


# -----------------------------------------------------------------------
# Chat prompt helpers and tests
# -----------------------------------------------------------------------


def _make_chat_prompt(
    messages: tuple[ChatMessage, ...] | None = None,
    required_variables: tuple[str, ...] = ("user_name",),
    content: str | None = None,
    content_hash: str | None = None,
    **overrides: object,
) -> PromptResponse:
    """Factory helper to build a CHAT PromptResponse with sensible defaults."""
    if messages is None:
        messages = (
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello {{user_name}}, how are you?"),
        )

    # For chat prompts, content is a canonical JSON representation of messages
    if content is None:
        content = json.dumps(
            [
                {"role": m.role, "content": m.content, "name": m.name}
                for m in messages
            ],
            sort_keys=True,
        )

    if content_hash is None:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    defaults = dict(
        schema_version=1,
        name="test-chat-prompt",
        version=1,
        content=content,
        content_hash=content_hash,
        config={"model": "gpt-4o"},
        required_variables=required_variables,
        label=None,
        cached_at=None,
        prompt_type=PromptType.CHAT,
        messages=messages,
    )
    defaults.update(overrides)
    return PromptResponse(**defaults)  # type: ignore[arg-type]


class TestChatPromptResponse:
    """Tests for chat prompt support on PromptResponse."""

    def test_default_prompt_type_is_text(self) -> None:
        """Existing prompts should default to TEXT."""
        prompt = _make_prompt()
        assert prompt.prompt_type == PromptType.TEXT

    def test_chat_prompt_type(self) -> None:
        """Create with prompt_type=PromptType.CHAT and messages."""
        prompt = _make_chat_prompt()
        assert prompt.prompt_type == PromptType.CHAT
        assert prompt.messages is not None
        assert len(prompt.messages) == 2
        assert prompt.messages[0].role == "system"
        assert prompt.messages[1].role == "user"

    def test_render_messages_substitutes_variables(self) -> None:
        """render_messages() should replace {{vars}} in each message content."""
        prompt = _make_chat_prompt()

        rendered = prompt.render_messages(user_name="Alice")

        assert rendered[0].content == "You are a helpful assistant."
        assert rendered[1].content == "Hello Alice, how are you?"

    def test_render_messages_preserves_roles(self) -> None:
        """Roles and names should be unchanged after rendering."""
        messages = (
            ChatMessage(role="system", content="Be helpful.", name="sys"),
            ChatMessage(role="user", content="Hi {{user_name}}!", name="usr"),
            ChatMessage(role="assistant", content="Hello!", name="bot"),
        )
        prompt = _make_chat_prompt(
            messages=messages,
            required_variables=("user_name",),
        )

        rendered = prompt.render_messages(user_name="Bob")

        assert rendered[0].role == "system"
        assert rendered[0].name == "sys"
        assert rendered[1].role == "user"
        assert rendered[1].name == "usr"
        assert rendered[2].role == "assistant"
        assert rendered[2].name == "bot"

    def test_render_messages_returns_new_objects(self) -> None:
        """Original messages must not be mutated (frozen dataclass)."""
        prompt = _make_chat_prompt()
        original_messages = prompt.messages

        rendered = prompt.render_messages(user_name="Alice")

        # Rendered messages are different objects
        assert rendered is not original_messages
        assert rendered[1] is not original_messages[1]  # type: ignore[index]
        # Original content unchanged
        assert original_messages[1].content == "Hello {{user_name}}, how are you?"  # type: ignore[index]
        assert rendered[1].content == "Hello Alice, how are you?"

    def test_render_messages_missing_variable_raises(self) -> None:
        """Should raise MissingVariableError."""
        prompt = _make_chat_prompt()

        with pytest.raises(MissingVariableError) as exc_info:
            prompt.render_messages()

        assert exc_info.value.variable_name == "user_name"

    def test_render_on_chat_prompt_raises(self) -> None:
        """Calling render() on a CHAT prompt should raise ValueError."""
        prompt = _make_chat_prompt()

        with pytest.raises(ValueError, match="render_messages"):
            prompt.render(user_name="Alice")

    def test_render_messages_on_text_prompt_raises(self) -> None:
        """Calling render_messages() on a TEXT prompt should raise ValueError."""
        prompt = _make_prompt()

        with pytest.raises(ValueError, match="render\\(\\)"):
            prompt.render_messages(user_name="Alice", service="Taproot")

    def test_verify_hash_works_for_chat(self) -> None:
        """content field for chat is canonical JSON, verify_hash should work."""
        prompt = _make_chat_prompt()

        assert prompt.verify_hash() is True

    def test_verify_hash_fails_for_wrong_chat_hash(self) -> None:
        """verify_hash should fail when the hash does not match."""
        prompt = _make_chat_prompt(
            content_hash="0" * 64,
        )

        assert prompt.verify_hash() is False

    def test_backward_compatible_no_messages(self) -> None:
        """Existing code creating PromptResponse without messages/prompt_type should work."""
        prompt = PromptResponse(
            schema_version=1,
            name="legacy-prompt",
            version=1,
            content="Hello world",
            content_hash=hashlib.sha256(b"Hello world").hexdigest(),
            config={},
            required_variables=(),
        )

        assert prompt.prompt_type == PromptType.TEXT
        assert prompt.messages is None
        assert prompt.render() == "Hello world"

    def test_render_messages_multiple_variables(self) -> None:
        """render_messages handles multiple variables across multiple messages."""
        messages = (
            ChatMessage(role="system", content="You serve {{company}}."),
            ChatMessage(role="user", content="I'm {{user_name}} from {{company}}."),
        )
        prompt = _make_chat_prompt(
            messages=messages,
            required_variables=("user_name", "company"),
        )

        rendered = prompt.render_messages(user_name="Alice", company="Acme")

        assert rendered[0].content == "You serve Acme."
        assert rendered[1].content == "I'm Alice from Acme."

    def test_render_messages_empty_messages_tuple(self) -> None:
        """render_messages on a chat prompt with no messages returns empty tuple."""
        prompt = _make_chat_prompt(
            messages=(),
            required_variables=(),
            content="[]",
        )

        rendered = prompt.render_messages()

        assert rendered == ()

    def test_chat_message_is_frozen(self) -> None:
        """ChatMessage should be immutable."""
        msg = ChatMessage(role="user", content="hi")

        with pytest.raises(AttributeError):
            msg.content = "modified"  # type: ignore[misc]


class TestChatMessageDataclass:
    """Tests for the ChatMessage frozen dataclass."""

    def test_default_name_is_none(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        assert msg.name is None

    def test_name_can_be_set(self) -> None:
        msg = ChatMessage(role="user", content="hello", name="alice")
        assert msg.name == "alice"

    def test_equality(self) -> None:
        a = ChatMessage(role="user", content="hi")
        b = ChatMessage(role="user", content="hi")
        assert a == b

    def test_inequality_on_role(self) -> None:
        a = ChatMessage(role="user", content="hi")
        b = ChatMessage(role="system", content="hi")
        assert a != b


class TestPromptTypeEnum:
    """Tests for the PromptType enum."""

    def test_text_value(self) -> None:
        assert PromptType.TEXT.value == "text"

    def test_chat_value(self) -> None:
        assert PromptType.CHAT.value == "chat"

    def test_from_string(self) -> None:
        assert PromptType("text") == PromptType.TEXT
        assert PromptType("chat") == PromptType.CHAT

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            PromptType("invalid")


# -----------------------------------------------------------------------
# ToolDefinition dataclass tests (N1)
# -----------------------------------------------------------------------


class TestToolDefinition:
    """Tests for the ToolDefinition frozen dataclass."""

    def test_create_with_defaults(self) -> None:
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get the current weather"
        assert tool.type == "function"
        assert tool.parameters == {"type": "object", "properties": {}}

    def test_create_with_explicit_type(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={},
            type="function",
        )
        assert tool.type == "function"

    def test_is_frozen(self) -> None:
        tool = ToolDefinition(
            name="test",
            description="A test tool",
            parameters={},
        )
        with pytest.raises(AttributeError):
            tool.name = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        params = {"type": "object", "properties": {"q": {"type": "string"}}}
        a = ToolDefinition(name="search", description="Search", parameters=params)
        b = ToolDefinition(name="search", description="Search", parameters=params)
        assert a == b

    def test_inequality(self) -> None:
        a = ToolDefinition(name="a", description="A", parameters={})
        b = ToolDefinition(name="b", description="B", parameters={})
        assert a != b


# -----------------------------------------------------------------------
# PromptResponse with tools tests (N1)
# -----------------------------------------------------------------------


class TestPromptResponseWithTools:
    """Tests for PromptResponse with tool definitions."""

    def test_default_tools_is_none(self) -> None:
        prompt = _make_prompt()
        assert prompt.tools is None

    def test_with_tools(self) -> None:
        tools = (
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        )
        prompt = _make_prompt(tools=tools)
        assert prompt.tools is not None
        assert len(prompt.tools) == 2
        assert prompt.tools[0].name == "get_weather"
        assert prompt.tools[1].name == "search"

    def test_tools_with_chat_prompt(self) -> None:
        tools = (
            ToolDefinition(
                name="calculate",
                description="Perform math",
                parameters={"type": "object", "properties": {}},
            ),
        )
        prompt = _make_chat_prompt(tools=tools)
        assert prompt.tools is not None
        assert prompt.prompt_type == PromptType.CHAT
        assert prompt.messages is not None
        assert len(prompt.tools) == 1

    def test_tools_do_not_affect_verify_hash(self) -> None:
        """Tools are metadata, not content. Hash should not change."""
        content = "Hello world"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        prompt_without = _make_prompt(
            content=content,
            content_hash=content_hash,
            required_variables=(),
        )
        tools = (
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={},
            ),
        )
        prompt_with = _make_prompt(
            content=content,
            content_hash=content_hash,
            required_variables=(),
            tools=tools,
        )

        assert prompt_without.verify_hash() is True
        assert prompt_with.verify_hash() is True

    def test_backward_compatible_no_tools(self) -> None:
        """Existing code creating PromptResponse without tools should work."""
        prompt = PromptResponse(
            schema_version=1,
            name="legacy-prompt",
            version=1,
            content="Hello world",
            content_hash=hashlib.sha256(b"Hello world").hexdigest(),
            config={},
            required_variables=(),
        )
        assert prompt.tools is None
        assert prompt.render() == "Hello world"
