"""
Decorator-based instrumentation for custom functions.

Provides the @instrument() decorator for tracing custom functions and methods.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    ParamSpec,
    TypeVar,
    overload,
)

from opentelemetry import trace
from opentelemetry.trace import StatusCode

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Valid span kinds
SpanKind = Literal[
    "workflow",
    "agent",
    "chain",
    "tool",
    "retrieval",
    "embedding",
    "completion",
    "chat",
    "rerank",
]

# Default limits
DEFAULT_MAX_ATTRIBUTE_SIZE = 65536  # 64KB
DEFAULT_MAX_ERROR_LENGTH = 1000


def instrument(
    spankind: SpanKind = "workflow",
    name: str | None = None,
    ignore_inputs: bool | list[str] = False,
    ignore_outputs: bool = False,
    max_attribute_size: int = DEFAULT_MAX_ATTRIBUTE_SIZE,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for automatic span instrumentation.

    Wraps a function to automatically create an OpenTelemetry span that tracks
    execution time, inputs, outputs, and errors.

    Args:
        spankind: Type of operation. One of:
            - "workflow": End-to-end pipeline (default)
            - "agent": Autonomous agent operations
            - "chain": Sequential processing
            - "tool": External tool/function calls
            - "retrieval": Knowledge base/RAG operations
            - "embedding": Vector embedding generation
            - "completion": Text generation (non-chat)
            - "chat": Conversational LLM calls
            - "rerank": Result reordering
        name: Custom span name. Defaults to function name.
        ignore_inputs: If True, don't capture any inputs. If a list of strings,
            ignore only those parameter names.
        ignore_outputs: If True, don't capture the return value.
        max_attribute_size: Maximum size in bytes for input/output attributes.
            Larger values are truncated.

    Returns:
        A decorator that wraps the function with tracing.

    Example:
        >>> @instrument(spankind="tool")
        ... def search_database(query: str) -> list:
        ...     return db.search(query)

        >>> @instrument(spankind="retrieval", ignore_inputs=["api_key"])
        ... def fetch_documents(query: str, api_key: str) -> list:
        ...     return api.search(query, api_key=api_key)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Pre-compute at decoration time for efficiency
        span_name = name or func.__name__
        is_async = asyncio.iscoroutinefunction(func)
        func_signature = inspect.signature(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = trace.get_tracer("taproot-sdk")
            start_time = time.perf_counter()

            with tracer.start_as_current_span(span_name) as span:
                # Set span attributes
                span.set_attribute("ev.type.node", spankind)
                span.set_attribute("ev.meta.function", func.__name__)
                span.set_attribute("ev.meta.module", func.__module__)

                # Capture inputs
                if not ignore_inputs:
                    try:
                        inputs_str = _serialize_inputs(
                            args, kwargs, func_signature, ignore_inputs
                        )
                        _set_attribute_safe(
                            span, "ev.data.inputs", inputs_str, max_attribute_size
                        )
                    except Exception as e:
                        logger.debug(f"Failed to serialize inputs: {e}")

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Record duration
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("ev.metrics.duration_ms", duration_ms)

                    # Capture outputs
                    if not ignore_outputs:
                        try:
                            output_str = _serialize(result)
                            _set_attribute_safe(
                                span, "ev.data.outputs", output_str, max_attribute_size
                            )
                        except Exception as e:
                            logger.debug(f"Failed to serialize output: {e}")

                    span.set_status(StatusCode.OK)
                    return result

                except Exception as e:
                    # Record duration even on error
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("ev.metrics.duration_ms", duration_ms)

                    # Record exception
                    span.record_exception(e)
                    span.set_status(
                        StatusCode.ERROR,
                        str(e)[:DEFAULT_MAX_ERROR_LENGTH],
                    )
                    raise

        @functools.wraps(func)
        async def async_wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> Coroutine[Any, Any, R]:
            tracer = trace.get_tracer("taproot-sdk")
            start_time = time.perf_counter()

            with tracer.start_as_current_span(span_name) as span:
                # Set span attributes
                span.set_attribute("ev.type.node", spankind)
                span.set_attribute("ev.meta.function", func.__name__)
                span.set_attribute("ev.meta.module", func.__module__)

                # Capture inputs
                if not ignore_inputs:
                    try:
                        inputs_str = _serialize_inputs(
                            args, kwargs, func_signature, ignore_inputs
                        )
                        _set_attribute_safe(
                            span, "ev.data.inputs", inputs_str, max_attribute_size
                        )
                    except Exception as e:
                        logger.debug(f"Failed to serialize inputs: {e}")

                try:
                    # Execute async function
                    result = await func(*args, **kwargs)

                    # Record duration
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("ev.metrics.duration_ms", duration_ms)

                    # Capture outputs
                    if not ignore_outputs:
                        try:
                            output_str = _serialize(result)
                            _set_attribute_safe(
                                span, "ev.data.outputs", output_str, max_attribute_size
                            )
                        except Exception as e:
                            logger.debug(f"Failed to serialize output: {e}")

                    span.set_status(StatusCode.OK)
                    return result

                except Exception as e:
                    # Record duration even on error
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("ev.metrics.duration_ms", duration_ms)

                    # Record exception
                    span.record_exception(e)
                    span.set_status(
                        StatusCode.ERROR,
                        str(e)[:DEFAULT_MAX_ERROR_LENGTH],
                    )
                    raise

        return async_wrapper if is_async else sync_wrapper  # type: ignore[return-value]

    return decorator


def _serialize_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    signature: inspect.Signature,
    ignore: bool | list[str],
) -> str:
    """Serialize function inputs to JSON string."""
    if ignore is True:
        return "{}"

    ignore_params = set(ignore) if isinstance(ignore, list) else set()

    # Bind arguments to parameter names
    try:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = {
            k: v for k, v in bound.arguments.items() if k not in ignore_params
        }
    except TypeError:
        # Fallback if binding fails
        inputs = {
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ignore_params},
        }

    return _serialize(inputs)


def _serialize(value: Any) -> str:
    """Serialize a value to JSON string with fallback for non-serializable types."""
    try:
        return json.dumps(value, default=_json_default, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"__repr__": repr(value)}, ensure_ascii=False)


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for objects not serializable by default."""
    # Handle common types
    if hasattr(obj, "model_dump"):
        # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):
        # Pydantic v1
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if hasattr(obj, "tolist"):
        # numpy arrays
        return obj.tolist()
    # Fallback to string representation
    return repr(obj)


def _set_attribute_safe(
    span: trace.Span,
    key: str,
    value: str,
    max_size: int,
) -> None:
    """Set span attribute, truncating if necessary."""
    if len(value) <= max_size:
        span.set_attribute(key, value)
    else:
        # Truncate and add metadata
        span.set_attribute(key, value[:max_size] + "...[TRUNCATED]")
        span.set_attribute(f"{key}_size", len(value))
        span.set_attribute(f"{key}_truncated", True)
