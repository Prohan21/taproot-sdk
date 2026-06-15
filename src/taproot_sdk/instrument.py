"""App-level instrumentation for Taproot SDK.

Provides ``instrument_app(app)`` — a single call that instruments an ASGI
application (FastAPI / Starlette) for the Taproot platform.

Currently bundles:
- TAP-38 middleware (reads interaction and correlation headers from incoming
  requests, stores them in ``ContextVar`` for ``TaprootClient`` propagation)

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

from taproot_sdk._context import (
    HEADER_CALLER_ID,
    HEADER_CALLER_TYPE,
    HEADER_CORRELATION_ID,
    HEADER_INTERACTION_ID,
    HEADER_INTERACTION_TYPE,
    HEADER_PARENT_ACTIVITY_ID,
    HEADER_ROOT_AGENT_ID,
    HEADER_SOURCE_AGENT_ID,
    HEADER_TRACEPARENT,
    TaprootActorRef,
    TaprootInteractionContext,
    correlation_id_var,
    interaction_context_var,
)


class _TaprootContextMiddleware:
    """ASGI middleware that binds inbound TAP-38 headers for one request.

    If the correlation header is absent, a new UUID is generated so that
    downstream calls and OTLP traces are always correlated.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin1").lower(): value.decode("utf-8", errors="ignore")
            for key, value in scope.get("headers", [])
        }

        cid_header = headers.get(HEADER_CORRELATION_ID.lower(), "")
        cid = cid_header if cid_header else str(uuid.uuid4())

        correlation_token = correlation_id_var.set(cid)
        interaction_token = interaction_context_var.set(_context_from_headers(headers, cid))
        try:
            await self.app(scope, receive, send)
        finally:
            interaction_context_var.reset(interaction_token)
            correlation_id_var.reset(correlation_token)


def _context_from_headers(
    headers: dict[str, str],
    correlation_id: str,
) -> TaprootInteractionContext | None:
    interaction_id = headers.get(HEADER_INTERACTION_ID.lower())
    if not interaction_id:
        return None

    caller_id = headers.get(HEADER_CALLER_ID.lower())
    caller_type = headers.get(HEADER_CALLER_TYPE.lower())
    caller = (
        TaprootActorRef(actor_type=caller_type, actor_id=caller_id)
        if caller_id and caller_type
        else None
    )

    return TaprootInteractionContext(
        interaction_id=interaction_id,
        interaction_type=headers.get(HEADER_INTERACTION_TYPE.lower(), "sdk_operation"),
        caller=caller,
        source_agent_id=headers.get(HEADER_SOURCE_AGENT_ID.lower()),
        root_agent_id=headers.get(HEADER_ROOT_AGENT_ID.lower()),
        correlation_id=correlation_id,
        trace_id=headers.get(HEADER_TRACEPARENT.lower()),
        parent_activity_id=headers.get(HEADER_PARENT_ACTIVITY_ID.lower()),
    )


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
        app.add_middleware(_TaprootContextMiddleware)
    else:
        raise TypeError(
            "instrument_app() requires a Starlette or FastAPI application. "
            f"Got {type(app).__name__}."
        )
