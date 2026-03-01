"""Prompt fetching and rendering for the Taproot SDK.

Usage:
    from taproot_sdk.prompts import PromptClient, PromptResponse

    client = PromptClient(
        serving_url="https://prompts.taproot.dev",
        api_key="your-api-key-id",
    )

    prompt = await client.get("my-project", "welcome-email")
    rendered = prompt.render(user_name="Alice", plan="Pro")
"""

from taproot_sdk.prompts.client import PromptClient
from taproot_sdk.prompts.exceptions import MissingVariableError
from taproot_sdk.prompts.models import (
    ChatMessage,
    PromptResponse,
    PromptType,
    ToolDefinition,
)

__all__ = [
    "ChatMessage",
    "PromptClient",
    "PromptResponse",
    "PromptType",
    "ToolDefinition",
    "MissingVariableError",
]
