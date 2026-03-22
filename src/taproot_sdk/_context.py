"""Request-scoped context variables for Taproot SDK.

The ``instrument_app()`` ASGI middleware sets these per-request.
``TaprootClient`` reads them automatically on every outgoing call.
For non-ASGI contexts (batch jobs), set them manually via
``correlation_id_var.set(value)``.
"""

from contextvars import ContextVar

correlation_id_var: ContextVar[str | None] = ContextVar(
    "taproot_correlation_id", default=None
)
