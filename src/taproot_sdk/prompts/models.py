"""Data models for prompt responses.

PromptResponse is a frozen dataclass representing a prompt template
fetched from the Taproot serving layer. It supports client-side variable
rendering and content integrity verification.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any

from taproot_sdk.prompts.exceptions import MissingVariableError

logger = logging.getLogger(__name__)

# Pattern for {{variable_name}} placeholders
_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


class PromptType(enum.Enum):
    """Discriminator for text vs chat prompts."""

    TEXT = "text"
    CHAT = "chat"


@dataclass(frozen=True)
class ToolDefinition:
    """A tool/function definition attached to a prompt version (N1).

    Attributes:
        name: The tool function name (e.g. "get_weather").
        description: Human-readable description of the tool.
        parameters: JSON Schema object describing the tool's parameters.
        type: The tool type. Currently always "function".
    """

    name: str
    description: str
    parameters: dict[str, Any]
    type: str = "function"


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a chat prompt template."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None


@dataclass(frozen=True)
class PromptResponse:
    """Immutable representation of a prompt fetched from the serving layer.

    Attributes:
        schema_version: API schema version (currently 1).
        name: The prompt name identifier.
        version: The resolved version number.
        content: The raw template content with {{variable}} placeholders.
            For chat prompts this is a canonical JSON representation of the
            messages array (used for hash verification).
        content_hash: SHA-256 hex digest of the content for integrity verification.
        config: Arbitrary configuration metadata associated with the prompt.
        required_variables: List of variable names that must be provided for rendering.
        label: Optional label (e.g. "production", "staging") if the prompt was
            resolved by label.
        cached_at: Optional ISO-8601 timestamp indicating when the response was cached.
        prompt_type: Whether this is a TEXT or CHAT prompt (default TEXT).
        messages: Tuple of chat messages for CHAT prompts (None for TEXT prompts).
        tools: Tuple of tool definitions attached to this prompt version (N1).
    """

    schema_version: int
    name: str
    version: int
    content: str
    content_hash: str
    config: dict[str, Any]
    required_variables: tuple[str, ...]
    label: str | None = None
    cached_at: str | None = None
    prompt_type: PromptType = PromptType.TEXT
    messages: tuple[ChatMessage, ...] | None = None
    tools: tuple[ToolDefinition, ...] | None = None
    ab_test: bool = False
    selected_variant: int | None = None

    def render(self, **variables: str) -> str:
        """Replace ``{{variable_name}}`` placeholders with provided values.

        This performs client-side rendering only (D21). The original ``content``
        field is never modified; a new string is returned.

        Required variables are taken from the server response, **not**
        re-extracted from the template.

        Args:
            **variables: Keyword arguments mapping variable names to their
                replacement values.

        Returns:
            A new string with all ``{{variable}}`` placeholders replaced.

        Raises:
            MissingVariableError: If any variable listed in
                ``required_variables`` is not supplied.
            ValueError: If this is a CHAT prompt. Use ``render_messages()`` instead.
        """
        if self.prompt_type == PromptType.CHAT:
            raise ValueError(
                "render() is not supported for chat prompts. "
                "Use render_messages() instead."
            )

        return self._render_content(self.content, **variables)

    def render_messages(self, **variables: str) -> tuple[ChatMessage, ...]:
        """Render all chat messages with variable substitution.

        Returns a new tuple of ChatMessage objects with ``{{variable}}``
        placeholders replaced in each message's content field. The original
        messages are not modified.

        Args:
            **variables: Keyword arguments mapping variable names to their
                replacement values.

        Returns:
            A new tuple of ``ChatMessage`` objects with placeholders replaced.

        Raises:
            MissingVariableError: If any required variable is not provided.
            ValueError: If this is not a chat prompt (prompt_type != CHAT).
        """
        if self.prompt_type != PromptType.CHAT:
            raise ValueError(
                "render_messages() is only supported for chat prompts "
                "(prompt_type=PromptType.CHAT). Use render() for text prompts."
            )

        if self.messages is None:
            return ()

        # Validate required variables up front (before rendering any messages)
        for var_name in self.required_variables:
            if var_name not in variables:
                raise MissingVariableError(var_name, self.required_variables)

        # Warn about extra variables
        extra = set(variables.keys()) - set(self.required_variables)
        if extra:
            warnings.warn(
                f"Extra variables provided that are not in required_variables: "
                f"{sorted(extra)}. They will still be substituted if matching "
                f"placeholders exist in the template.",
                stacklevel=2,
            )

        rendered: list[ChatMessage] = []
        for msg in self.messages:
            rendered_content = self._substitute(msg.content, variables)
            rendered.append(
                ChatMessage(
                    role=msg.role,
                    content=rendered_content,
                    name=msg.name,
                )
            )
        return tuple(rendered)

    def _render_content(self, content: str, **variables: str) -> str:
        """Shared rendering logic for text content.

        Validates required variables, warns about extras, and performs
        substitution on the given content string.
        """
        # Validate that all required variables are present
        for var_name in self.required_variables:
            if var_name not in variables:
                raise MissingVariableError(var_name, self.required_variables)

        # Warn about extra variables that are not in required_variables
        extra = set(variables.keys()) - set(self.required_variables)
        if extra:
            warnings.warn(
                f"Extra variables provided that are not in required_variables: "
                f"{sorted(extra)}. They will still be substituted if matching "
                f"placeholders exist in the template.",
                stacklevel=2,
            )

        return self._substitute(content, variables)

    @staticmethod
    def _substitute(text: str, variables: dict[str, str]) -> str:
        """Replace {{variable}} placeholders in *text* using *variables*."""

        def _replacer(match: re.Match[str]) -> str:
            var = match.group(1)
            if var in variables:
                return variables[var]
            return match.group(0)  # Leave unmatched placeholders intact

        return _VARIABLE_PATTERN.sub(_replacer, text)

    def verify_hash(self) -> bool:
        """Verify the content integrity against the stored SHA-256 hash.

        Returns:
            True if the SHA-256 of ``content`` matches ``content_hash``,
            False otherwise.
        """
        actual_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return hmac.compare_digest(actual_hash, self.content_hash)

    # ------------------------------------------------------------------
    # LLM provider convenience methods
    # ------------------------------------------------------------------

    def to_openai_messages(self, **variables: str) -> list[dict[str, str]]:
        """Render and format as OpenAI chat completion messages.

        For TEXT prompts, wraps the rendered content in a single
        ``{"role": "user", "content": ...}`` message.

        For CHAT prompts, renders all messages and returns them in
        OpenAI format (``role`` + ``content``).

        Args:
            **variables: Template variables for rendering.

        Returns:
            A list of dicts compatible with ``openai.chat.completions.create(messages=...)``.
        """
        if self.prompt_type == PromptType.CHAT:
            rendered = self.render_messages(**variables)
            return [{"role": m.role, "content": m.content} for m in rendered]
        rendered_text = self.render(**variables)
        return [{"role": "user", "content": rendered_text}]

    def to_anthropic_messages(
        self, **variables: str
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Render and format for Anthropic's Messages API.

        Returns a ``(system_prompt, messages)`` tuple. For CHAT prompts,
        the first ``system`` role message is extracted as the system
        parameter; remaining messages are returned as the messages list.

        For TEXT prompts, returns ``(None, [{"role": "user", "content": ...}])``.

        Args:
            **variables: Template variables for rendering.

        Returns:
            A tuple of ``(system_prompt_or_none, messages_list)``.
        """
        if self.prompt_type == PromptType.CHAT:
            rendered = self.render_messages(**variables)
            system_prompt: str | None = None
            messages: list[dict[str, str]] = []
            for m in rendered:
                if m.role == "system" and system_prompt is None and not messages:
                    system_prompt = m.content
                else:
                    messages.append({"role": m.role, "content": m.content})
            return system_prompt, messages
        rendered_text = self.render(**variables)
        return None, [{"role": "user", "content": rendered_text}]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prompt response to a plain dict.

        Useful for logging, debugging, or passing to APIs that expect
        raw dictionaries.
        """
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "content": self.content,
            "content_hash": self.content_hash,
            "config": self.config,
            "required_variables": list(self.required_variables),
            "prompt_type": self.prompt_type.value,
        }
        if self.label is not None:
            result["label"] = self.label
        if self.cached_at is not None:
            result["cached_at"] = self.cached_at
        if self.messages is not None:
            result["messages"] = [
                {"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})}
                for m in self.messages
            ]
        if self.tools is not None:
            result["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "type": t.type,
                }
                for t in self.tools
            ]
        if self.ab_test:
            result["ab_test"] = True
            result["selected_variant"] = self.selected_variant
        return result
