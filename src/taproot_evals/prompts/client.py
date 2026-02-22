"""HTTP client for fetching prompts from the Taproot serving layer.

The client is async-first (D22) with a synchronous convenience wrapper.
An L1 in-memory cache with stale-while-revalidate semantics sits in
front of every HTTP call (C2).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from taproot_evals.prompts.cache import PromptCache
from taproot_evals.prompts.models import PromptResponse

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMA_VERSION = 1


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

        return await self._cache.get_or_fetch(
            project_id,
            name,
            version=version,
            label=label,
            fetch_fn=self._fetch,
        )

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
        """
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
                f"Please upgrade taproot-evals."
            )

        required_variables = data.get("required_variables", [])

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
        )
