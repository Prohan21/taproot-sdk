"""
Auto-instrumentation for LLM libraries using OpenLLMetry.

This module provides automatic instrumentation for popular LLM client libraries.
When enabled, all LLM API calls are automatically traced without code changes.

Supported libraries:
- openai: ChatCompletion, Completion, Embeddings
- anthropic: Messages, Completions
- google-generativeai: GenerativeModel
- cohere: Chat, Generate, Embed
- vertexai: GenerativeModel
- bedrock: InvokeModel (via boto3)
- mistralai: Chat
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mapping of library names to their instrumentor module paths
# These follow the opentelemetry-instrumentation-* package naming
INSTRUMENTORS: dict[str, str] = {
    "openai": "opentelemetry.instrumentation.openai.OpenAIInstrumentor",
    "anthropic": "opentelemetry.instrumentation.anthropic.AnthropicInstrumentor",
    "google": "opentelemetry.instrumentation.google_generativeai.GoogleGenerativeAiInstrumentor",
    "cohere": "opentelemetry.instrumentation.cohere.CohereInstrumentor",
    "vertexai": "opentelemetry.instrumentation.vertexai.VertexAIInstrumentor",
    "bedrock": "opentelemetry.instrumentation.bedrock.BedrockInstrumentor",
    "mistral": "opentelemetry.instrumentation.mistralai.MistralAiInstrumentor",
}

# Track which instrumentors have been initialized to avoid double-instrumentation
_initialized_instrumentors: set[str] = set()


def setup_auto_instrumentation(libraries: list[str] | None = None) -> list[str]:
    """
    Enable auto-instrumentation for specified LLM libraries.

    This function dynamically loads and initializes OpenTelemetry instrumentors
    for the specified libraries. Once instrumented, all API calls to those
    libraries will automatically create spans.

    Args:
        libraries: List of library names to instrument. If None, attempts to
            instrument all available libraries.
            Supported values: "openai", "anthropic", "google", "cohere",
            "vertexai", "bedrock", "mistral"

    Returns:
        List of library names that were successfully instrumented.

    Example:
        >>> setup_auto_instrumentation(["openai", "anthropic"])
        ['openai', 'anthropic']

        >>> # Now all OpenAI and Anthropic calls are automatically traced
        >>> import openai
        >>> client = openai.OpenAI()
        >>> response = client.chat.completions.create(...)  # Automatically traced!
    """
    if libraries is None:
        libraries = list(INSTRUMENTORS.keys())

    successfully_instrumented: list[str] = []

    for lib in libraries:
        if lib in _initialized_instrumentors:
            logger.debug(f"Library '{lib}' is already instrumented, skipping")
            successfully_instrumented.append(lib)
            continue

        instrumentor_path = INSTRUMENTORS.get(lib)
        if instrumentor_path is None:
            logger.warning(
                f"Unknown library '{lib}'. "
                f"Supported libraries: {', '.join(INSTRUMENTORS.keys())}"
            )
            continue

        try:
            instrumentor = _load_instrumentor(instrumentor_path)
            if instrumentor is not None:
                instrumentor.instrument()
                _initialized_instrumentors.add(lib)
                successfully_instrumented.append(lib)
                logger.info(f"Auto-instrumented: {lib}")

        except ImportError as e:
            # Library or instrumentor not installed - this is expected if the
            # user hasn't installed the optional dependency
            logger.debug(
                f"Could not instrument '{lib}': {e}. "
                f"Install with: pip install taproot-sdk[{lib}]"
            )
        except Exception as e:
            logger.warning(f"Failed to instrument '{lib}': {e}")

    return successfully_instrumented


def uninstrument(libraries: list[str] | None = None) -> list[str]:
    """
    Disable auto-instrumentation for specified libraries.

    Args:
        libraries: List of library names to uninstrument. If None, uninstruments
            all currently instrumented libraries.

    Returns:
        List of library names that were successfully uninstrumented.
    """
    if libraries is None:
        libraries = list(_initialized_instrumentors)

    successfully_uninstrumented: list[str] = []

    for lib in libraries:
        if lib not in _initialized_instrumentors:
            logger.debug(f"Library '{lib}' is not instrumented, skipping")
            continue

        instrumentor_path = INSTRUMENTORS.get(lib)
        if instrumentor_path is None:
            continue

        try:
            instrumentor = _load_instrumentor(instrumentor_path)
            if instrumentor is not None:
                instrumentor.uninstrument()
                _initialized_instrumentors.discard(lib)
                successfully_uninstrumented.append(lib)
                logger.debug(f"Uninstrumented: {lib}")

        except Exception as e:
            logger.warning(f"Failed to uninstrument '{lib}': {e}")

    return successfully_uninstrumented


def uninstrument_all() -> None:
    """Disable all auto-instrumentation."""
    uninstrument(None)


def get_instrumented_libraries() -> list[str]:
    """
    Get list of currently instrumented libraries.

    Returns:
        List of library names that are currently instrumented.
    """
    return list(_initialized_instrumentors)


def is_instrumented(library: str) -> bool:
    """
    Check if a specific library is currently instrumented.

    Args:
        library: Name of the library to check.

    Returns:
        True if the library is instrumented, False otherwise.
    """
    return library in _initialized_instrumentors


def _load_instrumentor(instrumentor_path: str) -> Any | None:
    """
    Dynamically load an instrumentor class.

    Args:
        instrumentor_path: Full module path to the instrumentor class,
            e.g., "opentelemetry.instrumentation.openai.OpenAIInstrumentor"

    Returns:
        An instance of the instrumentor class, or None if loading failed.
    """
    try:
        module_path, class_name = instrumentor_path.rsplit(".", 1)

        # Import the module
        import importlib
        module = importlib.import_module(module_path)

        # Get the instrumentor class
        instrumentor_class = getattr(module, class_name)

        # Return an instance
        return instrumentor_class()

    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not load instrumentor from '{instrumentor_path}': {e}")
        raise ImportError(f"Instrumentor not available: {instrumentor_path}") from e
