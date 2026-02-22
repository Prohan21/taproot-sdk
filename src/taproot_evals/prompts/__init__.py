"""Prompt fetching and rendering for the Taproot SDK.

Usage:
    from taproot_evals.prompts import PromptClient, PromptResponse

    client = PromptClient(
        serving_url="https://prompts.taproot.dev",
        api_key="your-api-key-id",
    )

    prompt = await client.get("my-project", "welcome-email")
    rendered = prompt.render(user_name="Alice", plan="Pro")
"""

from taproot_evals.prompts.client import PromptClient
from taproot_evals.prompts.exceptions import MissingVariableError
from taproot_evals.prompts.models import PromptResponse

__all__ = [
    "PromptClient",
    "PromptResponse",
    "MissingVariableError",
]
