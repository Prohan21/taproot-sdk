"""App-level instrumentation for Taproot SDK.

Provides ``instrument_app(app)`` — a single call that instruments an ASGI
application (FastAPI / Starlette) for the Taproot platform.

Currently bundles:
- Correlation ID middleware (reads ``X-Correlation-ID`` from incoming
  requests, stores in ``ContextVar`` for ``TaprootClient`` auto-propagation)

Future additions (transparent to the developer):
- Structlog request context binding
- OTLP request/response spans
- Latency metrics export

Usage::

    from google.adk.cli.fast_api import get_fast_api_app
    from taproot_sdk import instrument_app

    app = get_fast_api_app(agents_dir=AGENTS_DIR)
    instrument_app(app)
"""

from __future__ import annotations

import uuid
from typing import Any

from taproot_sdk._context import correlation_id_var


class _CorrelationMiddleware:
    """ASGI middleware that extracts ``X-Correlation-ID`` from incoming
    request headers and stores it in a ``ContextVar``.

    If the header is absent, a new UUID is generated so that downstream
    calls and OTLP traces are always correlated.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract correlation ID from headers
        headers = dict(scope.get("headers", []))
        cid_header = headers.get(b"x-correlation-id", b"").decode("utf-8", errors="ignore")
        cid = cid_header if cid_header else str(uuid.uuid4())

        token = correlation_id_var.set(cid)
        try:
            await self.app(scope, receive, send)
        finally:
            correlation_id_var.reset(token)


def instrument_app(app: Any) -> None:
    """Instrument an ASGI application for the Taproot platform.

    One call adds all necessary middleware for correlation propagation,
    tracing, and logging.  The developer never needs to know the details.

    Args:
        app: A FastAPI or Starlette application instance.

    Example::

        from taproot_sdk import instrument_app

        app = FastAPI()
        instrument_app(app)
    """
    # Starlette/FastAPI expose add_middleware for class-based middleware
    if hasattr(app, "add_middleware"):
        app.add_middleware(_CorrelationMiddleware)
    else:
        raise TypeError(
            "instrument_app() requires a Starlette or FastAPI application. "
            f"Got {type(app).__name__}."
        )
