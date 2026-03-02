"""Prompt fetching and rendering for the Taproot SDK.

Usage (via unified client):
    client = TaprootClient(
        base_url="https://gateway.taproot.dev",
        api_key="your-api-key",
        project_id="my-project",
    )

    prompt = await client.get_prompt("welcome-email")
    rendered = prompt.render(user_name="Alice", plan="Pro")
"""

from taproot_sdk.prompts.exceptions import MissingVariableError
from taproot_sdk.prompts.models import (
    ChatMessage,
    PromptResponse,
    PromptType,
    ToolDefinition,
)

__all__ = [
    "ChatMessage",
    "PromptResponse",
    "PromptType",
    "ToolDefinition",
    "MissingVariableError",
]
