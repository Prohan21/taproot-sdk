"""Request-scoped context variables for Taproot SDK.

The ``instrument_app()`` ASGI middleware sets correlation IDs per-request.
``TaprootClient`` reads the current context automatically on every outgoing call.
For non-ASGI contexts (batch jobs), set context manually with the helpers here.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping

ACTIVITY_HEADER_VERSION = 1

HEADER_ACTIVITY_VERSION = "X-Taproot-Activity-Version"
HEADER_INTERACTION_ID = "X-Taproot-Interaction-Id"
HEADER_INTERACTION_TYPE = "X-Taproot-Interaction-Type"
HEADER_CALLER_ID = "X-Taproot-Caller-Id"
HEADER_CALLER_TYPE = "X-Taproot-Caller-Type"
HEADER_SOURCE_AGENT_ID = "X-Taproot-Source-Agent-Id"
HEADER_ROOT_AGENT_ID = "X-Taproot-Root-Agent-Id"
HEADER_PARENT_ACTIVITY_ID = "X-Taproot-Parent-Activity-Id"
HEADER_PARENT_INTERACTION_ID = "X-Taproot-Parent-Interaction-Id"
HEADER_CORRELATION_ID = "X-Correlation-ID"
HEADER_TRACEPARENT = "traceparent"


@dataclass(frozen=True)
class TaprootActorRef:
    """Actor identity propagated with a TAP-38 interaction."""

    actor_type: str
    actor_id: str


@dataclass(frozen=True)
class TaprootInteractionContext:
    """SDK-local TAP-38 interaction context for outbound header propagation.

    ``parent_activity_id`` is the v1 wire name for upstream parent interaction.
    ``parent_interaction_id`` is accepted for clients using the newer name.
    """

    interaction_id: str
    interaction_type: str = "sdk_operation"
    caller: TaprootActorRef | None = None
    source_agent_id: str | None = None
    root_agent_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    parent_activity_id: str | None = None
    parent_interaction_id: str | None = None

    def __post_init__(self) -> None:
        if self.parent_interaction_id is None and self.parent_activity_id is not None:
            object.__setattr__(self, "parent_interaction_id", self.parent_activity_id)
        if self.parent_activity_id is None and self.parent_interaction_id is not None:
            object.__setattr__(self, "parent_activity_id", self.parent_interaction_id)


correlation_id_var: ContextVar[str | None] = ContextVar("taproot_correlation_id", default=None)

interaction_context_var: ContextVar[TaprootInteractionContext | None] = ContextVar(
    "taproot_activity_interaction_context",
    default=None,
)


def get_interaction_context() -> TaprootInteractionContext | None:
    """Return the current TAP-38 interaction context, if one is bound."""

    return interaction_context_var.get()


def set_interaction_context(
    context: TaprootInteractionContext,
) -> Token[TaprootInteractionContext | None]:
    """Bind a TAP-38 interaction context to the current execution context."""

    return interaction_context_var.set(context)


def reset_interaction_context(token: Token[TaprootInteractionContext | None]) -> None:
    """Reset a token returned by ``set_interaction_context``."""

    interaction_context_var.reset(token)


def clear_interaction_context() -> None:
    """Clear the current TAP-38 interaction context."""

    interaction_context_var.set(None)


def propagation_headers(
    context: TaprootInteractionContext | None = None,
) -> dict[str, str]:
    """Build outbound TAP-38 propagation headers for a context.

    ``HEADER_PARENT_ACTIVITY_ID`` carries parent-interaction semantics in v1.
    """

    current = context or get_interaction_context()
    if current is None:
        return {}

    headers = {
        HEADER_ACTIVITY_VERSION: str(ACTIVITY_HEADER_VERSION),
        HEADER_INTERACTION_ID: current.interaction_id,
        HEADER_INTERACTION_TYPE: current.interaction_type,
    }
    # Deprecated X-Taproot-Caller-* are never emitted: caller identity is
    # computed downstream from trusted context (TAP-38), not propagated.
    if current.source_agent_id:
        headers[HEADER_SOURCE_AGENT_ID] = current.source_agent_id
    if current.root_agent_id:
        headers[HEADER_ROOT_AGENT_ID] = current.root_agent_id
    upstream_parent = current.parent_activity_id or current.parent_interaction_id
    if upstream_parent:
        headers[HEADER_PARENT_ACTIVITY_ID] = upstream_parent
    # ponytail: this caller's local interaction becomes the downstream parent.
    headers[HEADER_PARENT_INTERACTION_ID] = current.interaction_id
    if current.correlation_id:
        headers[HEADER_CORRELATION_ID] = current.correlation_id
    if current.trace_id:
        headers[HEADER_TRACEPARENT] = current.trace_id

    return headers


def create_interaction_id() -> str:
    """Create an SDK-local interaction identity."""

    return str(uuid4())


def merge_propagation_headers(
    headers: Mapping[str, str] | None = None,
    *,
    context: TaprootInteractionContext | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Merge TAP-38 propagation headers while preserving explicit headers."""

    merged = dict(headers or {})
    existing = {key.lower() for key in merged}
    for key, value in propagation_headers(context).items():
        if overwrite or key.lower() not in existing:
            merged[key] = value
            existing.add(key.lower())
    return merged
