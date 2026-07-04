"""
Core SDK initialization and configuration.

This module provides the main entry point for initializing the Taproot SDK
with OpenTelemetry-based tracing.
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)

# Global state
_tracer: Tracer | None = None
_provider: TracerProvider | None = None
_config: dict[str, Any] = {}
_initialized: bool = False


def init(
    project_id: str,
    api_url: str,
    api_key: str | None = None,
    auto_instrument: Sequence[str] | None = None,
    redact_by_default: bool = True,
    sampling_rate: float = 1.0,
    batch_size: int = 512,
    flush_interval_ms: int = 5000,
    service_name: str | None = None,
    service_version: str | None = None,
) -> Tracer:
    """
    Initialize the Taproot SDK.

    This must be called once at application startup before any tracing occurs.

    Args:
        project_id: Unique identifier for your project in Taproot.
        api_url: URL of your Taproot backend (e.g., "https://api.taproot.dev").
        api_key: Optional API key for authentication.
        auto_instrument: List of LLM libraries to auto-instrument.
            Supported: "openai", "anthropic", "google", "cohere", "bedrock", "vertexai", "mistral"
        redact_by_default: Scrub PII/secrets from ``@instrument`` span
            inputs/outputs before export, replacing them with stable
            non-reversible ``redacted:`` tokens (default: True). Correlation
            and interaction ids and ``ev.meta.*`` attributes are never
            scrubbed. CAVEAT: spans emitted by LLM auto-instrumentors
            (OpenLLMetry) are NOT covered — set the environment variable
            ``TRACELOOP_TRACE_CONTENT=false`` to suppress prompt/completion
            content in those spans.
        sampling_rate: Fraction of traces to sample (0.0 to 1.0, default: 1.0).
        batch_size: Maximum spans per export batch (default: 512).
        flush_interval_ms: Maximum time between flushes in milliseconds (default: 5000).
        service_name: Optional service name for resource attributes.
        service_version: Optional service version for resource attributes.

    Returns:
        The configured OpenTelemetry Tracer instance.

    Raises:
        RuntimeError: If the SDK is already initialized.

    Example:
        >>> import taproot_sdk as ev
        >>> ev.init(
        ...     project_id="my-project",
        ...     api_url="https://api.taproot.dev",
        ...     api_key="sk-...",
        ...     auto_instrument=["openai", "anthropic"],
        ... )
    """
    global _tracer, _provider, _config, _initialized

    if _initialized:
        raise RuntimeError(
            "Taproot SDK is already initialized. "
            "Call ev.shutdown() first if you need to reinitialize."
        )

    resolved_service_name = service_name or f"taproot-{project_id}"
    logger.info(
        "sdk.init.start",
        extra={"project_id": project_id, "service_name": resolved_service_name},
    )

    # Store configuration
    _config = {
        "project_id": project_id,
        "api_url": api_url.rstrip("/"),
        "api_key": api_key,
        "redact_by_default": redact_by_default,
        "sampling_rate": sampling_rate,
    }

    # Build resource attributes
    resource_attrs = {
        "service.name": resolved_service_name,
        "taproot.project_id": project_id,
    }
    if service_version:
        resource_attrs["service.version"] = service_version

    resource = Resource.create(resource_attrs)

    # Configure sampler
    # ParentBased respects parent's sampling decision, falls back to rate-based for root spans
    sampler = ParentBased(root=TraceIdRatioBased(sampling_rate))
    logger.info("sdk.init.sampler", extra={"sampling_rate": sampling_rate})

    # Create TracerProvider
    _provider = TracerProvider(resource=resource, sampler=sampler)

    # Configure OTLP exporter
    exporter = _create_otlp_exporter(
        endpoint=f"{_config['api_url']}/api/v1/evals/v1/traces",
        api_key=api_key,
    )
    logger.info(
        "sdk.init.exporter",
        extra={"api_url": _config["api_url"], "compression": "gzip"},
    )

    # Configure batch processor for efficient export
    processor = BatchSpanProcessor(
        exporter,
        max_queue_size=batch_size * 4,
        max_export_batch_size=batch_size,
        schedule_delay_millis=flush_interval_ms,
        export_timeout_millis=30000,
    )
    _provider.add_span_processor(processor)
    logger.info(
        "sdk.init.batch_processor",
        extra={"batch_size": batch_size, "max_queue": batch_size * 4},
    )

    # Set as global provider
    trace.set_tracer_provider(_provider)

    # Get tracer
    _tracer = trace.get_tracer(
        instrumenting_module_name="taproot-sdk",
        instrumenting_library_version="0.1.0",
    )

    # Register shutdown handler
    atexit.register(shutdown)

    # Auto-instrument LLM libraries if requested
    if auto_instrument:
        _setup_auto_instrumentation(auto_instrument)

    _initialized = True
    logger.info(
        "sdk.init.complete",
        extra={"project_id": project_id},
    )

    return _tracer


def _create_otlp_exporter(endpoint: str, api_key: str | None) -> Any:
    """Create OTLP HTTP exporter with JSON serialization.

    Uses JSON (not protobuf) because AWS API Gateway REST API v1 does not
    support binary media types by default and will corrupt protobuf payloads.
    """
    from taproot_sdk.otlp_json_exporter import JsonOtlpSpanExporter

    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    return JsonOtlpSpanExporter(
        endpoint=endpoint,
        headers=headers,
        compress=True,
    )


def _setup_auto_instrumentation(libraries: Sequence[str]) -> None:
    """Set up auto-instrumentation for specified LLM libraries."""
    from taproot_sdk.auto_instrument import setup_auto_instrumentation

    setup_auto_instrumentation(list(libraries))


def shutdown() -> None:
    """
    Shutdown the SDK and flush any pending spans.

    This is automatically called at program exit, but can be called manually
    if you need to ensure all spans are exported before a specific point.
    """
    global _tracer, _provider, _config, _initialized

    logger.info("sdk.shutdown.start")

    if _provider is not None:
        timeout_ms = 10000
        try:
            _provider.force_flush(timeout_millis=timeout_ms)
            _provider.shutdown()
            logger.info("sdk.shutdown.complete")
        except Exception as e:
            logger.warning(
                "sdk.shutdown.timeout",
                extra={"timeout_ms": timeout_ms, "error": str(e)},
            )

    _tracer = None
    _provider = None
    _config = {}
    _initialized = False


def get_tracer() -> Tracer:
    """
    Get the configured tracer instance.

    Returns:
        The OpenTelemetry Tracer instance.

    Raises:
        RuntimeError: If the SDK has not been initialized.
    """
    if _tracer is None:
        raise RuntimeError("Taproot SDK not initialized. Call ev.init() first.")
    return _tracer


def is_initialized() -> bool:
    """Check if the SDK has been initialized."""
    return _initialized


def get_config() -> dict[str, Any]:
    """
    Get the current SDK configuration.

    Returns:
        A copy of the configuration dictionary.
    """
    return _config.copy()
