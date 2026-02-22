"""
Taproot SDK - Instrumentation for LLM observability.

Usage:
    import taproot_sdk as ev

    ev.init(
        project_id="my-project",
        api_url="https://your-backend.com",
        api_key="sk-...",
        auto_instrument=["openai", "anthropic"],
    )

    @ev.instrument(spankind="tool")
    def my_function():
        pass

Prompt fetching:
    from taproot_sdk.prompts import PromptClient

    client = PromptClient(serving_url="https://prompts.taproot.dev", api_key="key-id")
    prompt = await client.get("my-project", "welcome-email")
    rendered = prompt.render(user_name="Alice")
"""

from taproot_sdk.client import TaprootClient
from taproot_sdk.core import get_tracer, init, is_initialized, shutdown
from taproot_sdk.decorators import instrument
from taproot_sdk.prompts import MissingVariableError, PromptClient, PromptResponse

__version__ = "0.1.0"

__all__ = [
    # Client
    "TaprootClient",
    # Core functions
    "init",
    "shutdown",
    "get_tracer",
    "is_initialized",
    # Decorators
    "instrument",
    # Prompts
    "PromptClient",
    "PromptResponse",
    "MissingVariableError",
    # Version
    "__version__",
]
