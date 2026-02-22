"""
Taproot Evals SDK - Instrumentation for LLM observability.

Usage:
    import taproot_evals as ev

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
    from taproot_evals.prompts import PromptClient

    client = PromptClient(serving_url="https://prompts.taproot.dev", api_key="key-id")
    prompt = await client.get("my-project", "welcome-email")
    rendered = prompt.render(user_name="Alice")
"""

from taproot_evals.core import get_tracer, init, is_initialized, shutdown
from taproot_evals.decorators import instrument
from taproot_evals.prompts import MissingVariableError, PromptClient, PromptResponse

__version__ = "0.1.0"

__all__ = [
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
