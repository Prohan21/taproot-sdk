"""In-memory TTL cache with stale-while-revalidate for prompt responses.

Provides an L1 cache that sits in front of the HTTP serving layer.
Entries have a TTL (derived from ``max_stale_seconds``); once stale
but still within the staleness window the cache returns the old value
immediately and triggers a background revalidation.  Beyond the
staleness window the caller blocks on a fresh fetch.

Thread safety:
    * Async callers are serialised via ``asyncio.Lock``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from taproot_sdk.prompts.models import PromptResponse

logger = logging.getLogger(__name__)


def _make_cache_key(
    project_id: str,
    name: str,
    *,
    version: int | None = None,
    label: str | None = None,
) -> str:
    """Build a deterministic cache key from lookup parameters."""
    version_part = str(version) if version is not None else ""
    label_part = label if label is not None else ""
    return f"{project_id}:{name}:v={version_part}:l={label_part}"


@dataclass
class _CacheEntry:
    """A single cached prompt response with monotonic timing metadata."""

    response: PromptResponse
    fetched_at: float  # time.monotonic() when the value was stored


@dataclass
class PromptCache:
    """In-memory TTL cache with stale-while-revalidate semantics.

    Args:
        ttl_seconds: How long a cached entry is considered *fresh*.
            After this period the entry is *stale* but may still be
            served while a background refresh is in flight.
        max_stale_seconds: Maximum additional staleness beyond ``ttl_seconds``
            before the caller must block on a fresh fetch.  The total
            acceptable age of an entry is ``ttl_seconds + max_stale_seconds``.
    """

    ttl_seconds: float = 30.0
    max_stale_seconds: float = 60.0

    _store: dict[str, _CacheEntry] = field(default_factory=dict, repr=False)
    _async_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _inflight: set[str] = field(default_factory=set, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_fetch(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        fetch_fn: Callable[..., Awaitable[PromptResponse]],
    ) -> PromptResponse:
        """Return a cached response or fetch a fresh one.

        Decision matrix (all times relative to ``time.monotonic()``):

        * **Cache miss** -- call *fetch_fn*, populate cache, return.
        * **Fresh hit** (age < ``ttl_seconds``) -- return immediately.
        * **Stale hit** (age < ``ttl_seconds + max_stale_seconds``) --
          return stale entry, schedule background revalidation.
        * **Expired hit** (age >= ``ttl_seconds + max_stale_seconds``) --
          block on *fetch_fn*, replace cache entry, return.

        Args:
            project_id: Project identifier (part of cache key).
            name: Prompt name (part of cache key).
            version: Optional pinned version.
            label: Optional label.
            fetch_fn: Async callable that performs the real HTTP fetch.
                Signature must match
                ``fetch_fn(project_id, name, *, version, label)``.

        Returns:
            A ``PromptResponse``.
        """
        key = _make_cache_key(project_id, name, version=version, label=label)

        async with self._async_lock:
            entry = self._store.get(key)
            now = time.monotonic()

            if entry is None:
                # Cache miss -- fetch, store, return
                response = await fetch_fn(
                    project_id, name, version=version, label=label
                )
                self._store[key] = _CacheEntry(
                    response=response, fetched_at=now
                )
                return response

            age = now - entry.fetched_at

            if age < self.ttl_seconds:
                # Fresh -- return immediately
                return entry.response

            if age < self.ttl_seconds + self.max_stale_seconds:
                # Stale but acceptable -- return stale and kick off
                # background revalidation (if not already in-flight).
                stale_response = entry.response
                if key not in self._inflight:
                    self._inflight.add(key)
                    asyncio.ensure_future(
                        self._revalidate(
                            key,
                            fetch_fn,
                            project_id,
                            name,
                            version=version,
                            label=label,
                        )
                    )
                return stale_response

            # Expired beyond max_stale -- must block on a fresh fetch
            response = await fetch_fn(
                project_id, name, version=version, label=label
            )
            self._store[key] = _CacheEntry(
                response=response, fetched_at=time.monotonic()
            )
            return response

    def invalidate(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> bool:
        """Remove a specific entry from the cache.

        Returns:
            True if an entry was removed, False if the key was not present.
        """
        key = _make_cache_key(project_id, name, version=version, label=label)
        removed = self._store.pop(key, None)
        return removed is not None

    def clear(self) -> None:
        """Drop all cached entries."""
        self._store.clear()
        self._inflight.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _revalidate(
        self,
        key: str,
        fetch_fn: Callable[..., Awaitable[PromptResponse]],
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> None:
        """Background task that refreshes a stale cache entry."""
        try:
            response = await fetch_fn(
                project_id, name, version=version, label=label
            )
            async with self._async_lock:
                self._store[key] = _CacheEntry(
                    response=response, fetched_at=time.monotonic()
                )
        except Exception:
            logger.warning(
                "Background revalidation failed for cache key %r", key, exc_info=True
            )
        finally:
            async with self._async_lock:
                self._inflight.discard(key)
