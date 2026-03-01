"""HTTP client for fetching prompts from the Taproot serving layer.

The client is async-first (D22) with a synchronous convenience wrapper.
An L1 in-memory cache with stale-while-revalidate semantics sits in
front of every HTTP call (C2).

Optional OpenTelemetry instrumentation: when ``opentelemetry-api`` is
installed, HTTP fetches are wrapped in a ``taproot.prompt.fetch`` span
with prompt metadata attributes.  If the library is absent, the client
behaves identically to an uninstrumented build.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from taproot_sdk.prompts.cache import PromptCache
from taproot_sdk.prompts.models import (
    ChatMessage,
    PromptResponse,
    PromptType,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Optional OpenTelemetry tracing
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace

    _tracer = trace.get_tracer("taproot-sdk.prompts")
except ImportError:  # pragma: no cover
    _tracer = None  # type: ignore[assignment]


class PromptClient:
    """Async HTTP client for the Taproot prompt serving endpoint.

    Every instance owns its own :class:`PromptCache`.  Responses are
    served from the cache when fresh, returned stale with a background
    revalidation when within *max_stale_seconds*, and block on a fresh
    fetch when the cached value is too old.

    Args:
        serving_url: Base URL of the prompt serving layer
            (e.g. ``"https://prompts.taproot.dev"``). Trailing slashes are stripped.
        api_key: The API key ID sent as ``X-Api-Key-Id`` header on every request.
        max_stale_seconds: Maximum acceptable age of a cached response in seconds
            beyond the TTL before the caller must block on a fresh fetch (default 60.0).
        cache_ttl_seconds: How long a cached entry is considered *fresh*
            (default 30.0).  After this period but before
            ``cache_ttl_seconds + max_stale_seconds`` the entry is served
            stale while a background revalidation runs.

    Example:
        >>> client = PromptClient(
        ...     serving_url="https://prompts.taproot.dev",
        ...     api_key="my-key-id",
        ... )
        >>> prompt = await client.get("my-project", "welcome-email")
        >>> print(prompt.render(user_name="Alice"))
    """

    def __init__(
        self,
        serving_url: str,
        api_key: str,
        max_stale_seconds: float = 60.0,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self._serving_url = serving_url.rstrip("/")
        self._api_key = api_key
        self._max_stale_seconds = max_stale_seconds
        self._cache = PromptCache(
            ttl_seconds=cache_ttl_seconds,
            max_stale_seconds=max_stale_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> PromptResponse:
        """Fetch a prompt template, served from cache when possible.

        Args:
            project_id: The project identifier.
            name: The prompt name.
            version: Optional specific version number to fetch.
            label: Optional label (e.g. ``"production"``) to resolve.

        Returns:
            A frozen ``PromptResponse`` containing the template and metadata.

        Raises:
            ValueError: If both ``version`` and ``label`` are specified, or
                if the server returns an unsupported ``schema_version``.
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        if version is not None and label is not None:
            raise ValueError("Cannot specify both 'version' and 'label'")

        # Snapshot cache size before the fetch so we can detect a cache hit
        cache_size_before = len(self._cache._store)

        result = await self._cache.get_or_fetch(
            project_id,
            name,
            version=version,
            label=label,
            fetch_fn=self._fetch,
        )

        # If the cache did NOT grow, the response came from cache.
        # Record a lightweight span attribute on the current active span.
        if _tracer is not None and len(self._cache._store) == cache_size_before:
            span = trace.get_current_span()
            if span.is_recording():
                span.set_attribute("prompt.cached", True)

        return result

    def get_sync(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> PromptResponse:
        """Synchronous wrapper around :meth:`get`.

        Convenience method for use in non-async contexts. Uses
        ``asyncio.run()`` when no event loop is running, or
        ``loop.run_until_complete()`` when called from a context
        with an existing (but not running) loop.

        Args:
            project_id: The project identifier.
            name: The prompt name.
            version: Optional specific version number to fetch.
            label: Optional label to resolve.

        Returns:
            A frozen ``PromptResponse`` containing the template and metadata.
        """
        if version is not None and label is not None:
            raise ValueError("Cannot specify both 'version' and 'label'")

        coro = self._cache.get_or_fetch(
            project_id,
            name,
            version=version,
            label=label,
            fetch_fn=self._fetch,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside a running event loop (e.g. Jupyter).
            # Cannot use asyncio.run() here. Create a new thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()

        return asyncio.run(coro)

    @property
    def cache(self) -> PromptCache:
        """Access the underlying cache for manual invalidation or inspection."""
        return self._cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> PromptResponse:
        """Perform the actual HTTP GET to the serving layer.

        This is the raw fetch with no caching; the cache calls it as a
        callback when it needs a fresh value.

        When OpenTelemetry is available, the HTTP call is wrapped in a
        ``taproot.prompt.fetch`` span with prompt metadata attributes.
        """
        if _tracer is not None:
            return await self._fetch_with_span(
                project_id, name, version=version, label=label
            )
        return await self._do_fetch(project_id, name, version=version, label=label)

    async def _fetch_with_span(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> PromptResponse:
        """Execute the HTTP fetch inside an OpenTelemetry span."""
        assert _tracer is not None  # noqa: S101 — guarded by caller

        with _tracer.start_as_current_span("taproot.prompt.fetch") as span:
            result = await self._do_fetch(
                project_id, name, version=version, label=label
            )
            span.set_attribute("prompt.name", name)
            span.set_attribute("prompt.version", result.version)
            span.set_attribute("prompt.project_id", project_id)
            span.set_attribute("prompt.hash", result.content_hash)
            span.set_attribute("prompt.type", result.prompt_type.value)
            if label is not None:
                span.set_attribute("prompt.label", label)
            span.set_attribute("prompt.cached", False)
            return result

    async def _do_fetch(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> PromptResponse:
        """Low-level HTTP GET to the serving layer (no tracing, no caching)."""
        url = f"{self._serving_url}/serve/{project_id}/{name}"

        params: dict[str, str] = {}
        if version is not None:
            params["version"] = str(version)
        if label is not None:
            params["label"] = label

        headers = {
            "X-Api-Key-Id": self._api_key,
        }

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                url,
                params=params,
                headers=headers,
                timeout=self._max_stale_seconds,
            )
            response.raise_for_status()

        data: dict[str, Any] = response.json()

        schema_version = data.get("schema_version")
        if schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version: {schema_version}. "
                f"This SDK supports schema_version={_SUPPORTED_SCHEMA_VERSION}. "
                f"Please upgrade taproot-sdk."
            )

        required_variables = data.get("required_variables", [])

        # Parse prompt_type (default to TEXT for backward compat)
        raw_prompt_type = data.get("prompt_type", "text")
        try:
            prompt_type = PromptType(raw_prompt_type)
        except ValueError:
            logger.warning(
                "Unknown prompt_type '%s', defaulting to TEXT", raw_prompt_type
            )
            prompt_type = PromptType.TEXT

        # Parse messages for chat prompts
        raw_messages = data.get("messages")
        messages: tuple[ChatMessage, ...] | None = None
        if raw_messages is not None:
            messages = tuple(
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    name=msg.get("name"),
                )
                for msg in raw_messages
            )

        # Parse tool definitions (N1)
        raw_tools = data.get("tools")
        tools: tuple[ToolDefinition, ...] | None = None
        if raw_tools is not None:
            tools = tuple(
                ToolDefinition(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool.get("parameters", {}),
                    type=tool.get("type", "function"),
                )
                for tool in raw_tools
            )

        # Parse A/B test metadata (N2)
        ab_test = bool(data.get("ab_test", False))
        raw_variant = data.get("selected_variant")
        selected_variant = int(raw_variant) if raw_variant is not None else None

        return PromptResponse(
            schema_version=schema_version,
            name=data["name"],
            version=data["version"],
            content=data["content"],
            content_hash=data["content_hash"],
            config=data.get("config", {}),
            required_variables=tuple(required_variables),
            label=data.get("label"),
            cached_at=data.get("cached_at"),
            prompt_type=prompt_type,
            messages=messages,
            tools=tools,
            ab_test=ab_test,
            selected_variant=selected_variant,
        )
