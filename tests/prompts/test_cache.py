"""Tests for taproot_sdk.prompts.cache module.

Covers the five core behaviours of the L1 in-memory cache:
    1. Fresh cache hit -- returns cached, no HTTP call.
    2. Stale within max_stale -- returns stale + background refresh.
    3. Stale beyond max_stale -- blocks on a fresh fetch.
    4. Concurrent requests -- only one revalidation in-flight.
    5. Cache miss -- fetches and populates cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from taproot_sdk.prompts.cache import PromptCache, _CacheEntry, _make_cache_key
from taproot_sdk.prompts.models import PromptResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt_response(
    *,
    name: str = "welcome-email",
    version: int = 3,
    content: str = "Hello {{user_name}}!",
    label: str | None = None,
) -> PromptResponse:
    """Build a minimal ``PromptResponse`` for testing."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return PromptResponse(
        schema_version=1,
        name=name,
        version=version,
        content=content,
        content_hash=content_hash,
        config={"model": "gpt-4o"},
        required_variables=("user_name",),
        label=label,
        cached_at=None,
    )


def _make_async_fetch(
    response: PromptResponse | None = None,
) -> AsyncMock:
    """Return an ``AsyncMock`` that behaves like ``PromptClient._fetch``."""
    mock = AsyncMock()
    mock.return_value = response or _make_prompt_response()
    return mock


# ---------------------------------------------------------------------------
# _make_cache_key
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    """Unit tests for the key derivation function."""

    def test_basic_key(self) -> None:
        key = _make_cache_key("proj", "my-prompt")
        assert key == "proj:my-prompt:v=:l="

    def test_key_with_version(self) -> None:
        key = _make_cache_key("proj", "my-prompt", version=5)
        assert key == "proj:my-prompt:v=5:l="

    def test_key_with_label(self) -> None:
        key = _make_cache_key("proj", "my-prompt", label="production")
        assert key == "proj:my-prompt:v=:l=production"

    def test_key_with_version_and_label(self) -> None:
        key = _make_cache_key("proj", "my-prompt", version=2, label="staging")
        assert key == "proj:my-prompt:v=2:l=staging"

    def test_different_projects_produce_different_keys(self) -> None:
        k1 = _make_cache_key("proj-a", "prompt")
        k2 = _make_cache_key("proj-b", "prompt")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------

class TestCacheMiss:
    """When the cache has no entry, it should call fetch_fn and populate."""

    async def test_miss_calls_fetch_fn(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        expected = _make_prompt_response()
        fetch = _make_async_fetch(expected)

        result = await cache.get_or_fetch(
            "proj", "prompt", fetch_fn=fetch,
        )

        fetch.assert_awaited_once_with("proj", "prompt", version=None, label=None)
        assert result is expected

    async def test_miss_populates_cache(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        expected = _make_prompt_response()
        fetch = _make_async_fetch(expected)

        await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)

        # A second call should NOT invoke fetch_fn again (it's now cached).
        result2 = await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)
        assert fetch.await_count == 1
        assert result2 is expected

    async def test_miss_with_version(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        expected = _make_prompt_response(version=7)
        fetch = _make_async_fetch(expected)

        result = await cache.get_or_fetch(
            "proj", "prompt", version=7, fetch_fn=fetch,
        )

        fetch.assert_awaited_once_with("proj", "prompt", version=7, label=None)
        assert result.version == 7

    async def test_miss_with_label(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        expected = _make_prompt_response(label="staging")
        fetch = _make_async_fetch(expected)

        result = await cache.get_or_fetch(
            "proj", "prompt", label="staging", fetch_fn=fetch,
        )

        fetch.assert_awaited_once_with("proj", "prompt", version=None, label="staging")
        assert result.label == "staging"


# ---------------------------------------------------------------------------
# Fresh cache hit
# ---------------------------------------------------------------------------

class TestFreshCacheHit:
    """A fresh entry (age < ttl_seconds) should be returned without an HTTP call."""

    async def test_fresh_hit_skips_fetch(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        expected = _make_prompt_response()
        fetch = _make_async_fetch(expected)

        # Populate the cache.
        await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)
        assert fetch.await_count == 1

        # Second call -- should come from cache.
        result = await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)

        assert fetch.await_count == 1  # No additional fetch
        assert result is expected

    async def test_different_keys_are_independent(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        resp_a = _make_prompt_response(name="prompt-a")
        resp_b = _make_prompt_response(name="prompt-b")

        fetch_a = _make_async_fetch(resp_a)
        fetch_b = _make_async_fetch(resp_b)

        await cache.get_or_fetch("proj", "prompt-a", fetch_fn=fetch_a)
        await cache.get_or_fetch("proj", "prompt-b", fetch_fn=fetch_b)

        # Both should have been fetched once
        assert fetch_a.await_count == 1
        assert fetch_b.await_count == 1

        # Now both should be fresh hits
        result_a = await cache.get_or_fetch("proj", "prompt-a", fetch_fn=fetch_a)
        result_b = await cache.get_or_fetch("proj", "prompt-b", fetch_fn=fetch_b)

        assert fetch_a.await_count == 1
        assert fetch_b.await_count == 1
        assert result_a.name == "prompt-a"
        assert result_b.name == "prompt-b"


# ---------------------------------------------------------------------------
# Stale within max_stale_seconds
# ---------------------------------------------------------------------------

class TestStaleWithinMaxStale:
    """A stale entry within max_stale should return immediately and trigger
    background revalidation."""

    async def test_returns_stale_and_triggers_revalidation(self) -> None:
        """Stale but acceptable entry should be returned immediately.
        A background task should refresh it."""
        cache = PromptCache(ttl_seconds=1.0, max_stale_seconds=60.0)
        stale_response = _make_prompt_response(version=1)
        fresh_response = _make_prompt_response(version=2)

        call_count = 0

        async def fake_fetch(
            project_id: str,
            name: str,
            *,
            version: int | None = None,
            label: str | None = None,
        ) -> PromptResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stale_response
            return fresh_response

        # Populate the cache.
        result1 = await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)
        assert result1.version == 1
        assert call_count == 1

        # Make the entry stale by faking the timestamp.
        key = _make_cache_key("proj", "prompt")
        cache._store[key] = _CacheEntry(
            response=stale_response,
            fetched_at=time.monotonic() - 2.0,  # older than ttl_seconds=1.0
        )

        # This call should return the stale value immediately...
        result2 = await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)
        assert result2.version == 1  # stale value

        # ...and the background revalidation should eventually update the cache.
        await asyncio.sleep(0.1)  # Give the background task time to finish.
        result3 = await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)
        assert result3.version == 2  # refreshed value
        assert call_count == 2  # Only one revalidation, not two.

    async def test_stale_entry_returns_old_response_object(self) -> None:
        """The stale response object should be returned, not a copy."""
        cache = PromptCache(ttl_seconds=1.0, max_stale_seconds=60.0)
        original = _make_prompt_response()
        fetch = _make_async_fetch(original)

        await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)

        # Make entry stale.
        key = _make_cache_key("proj", "prompt")
        cache._store[key] = _CacheEntry(
            response=original,
            fetched_at=time.monotonic() - 2.0,
        )

        result = await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)
        assert result is original


# ---------------------------------------------------------------------------
# Stale beyond max_stale_seconds
# ---------------------------------------------------------------------------

class TestStaleBeyondMaxStale:
    """When an entry is stale beyond max_stale_seconds, the caller must
    block on a fresh fetch."""

    async def test_expired_entry_blocks_on_fetch(self) -> None:
        cache = PromptCache(ttl_seconds=1.0, max_stale_seconds=2.0)
        old_response = _make_prompt_response(version=1)
        new_response = _make_prompt_response(version=99)

        call_count = 0

        async def fake_fetch(
            project_id: str,
            name: str,
            *,
            version: int | None = None,
            label: str | None = None,
        ) -> PromptResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return old_response
            return new_response

        # Populate.
        await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)

        # Make it fully expired (age > ttl + max_stale = 3.0).
        key = _make_cache_key("proj", "prompt")
        cache._store[key] = _CacheEntry(
            response=old_response,
            fetched_at=time.monotonic() - 5.0,
        )

        result = await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)
        assert result.version == 99
        assert call_count == 2

    async def test_expired_entry_replaces_cache(self) -> None:
        """After blocking on a fresh fetch, the cache should contain the
        new value."""
        cache = PromptCache(ttl_seconds=1.0, max_stale_seconds=2.0)
        old_response = _make_prompt_response(version=1)
        new_response = _make_prompt_response(version=42)

        call_count = 0

        async def fake_fetch(
            project_id: str,
            name: str,
            *,
            version: int | None = None,
            label: str | None = None,
        ) -> PromptResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return old_response
            return new_response

        await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)

        key = _make_cache_key("proj", "prompt")
        cache._store[key] = _CacheEntry(
            response=old_response,
            fetched_at=time.monotonic() - 5.0,
        )

        await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)

        # Subsequent call should be a fresh hit (no further fetch).
        result = await cache.get_or_fetch("proj", "prompt", fetch_fn=fake_fetch)
        assert result.version == 42
        assert call_count == 2


# ---------------------------------------------------------------------------
# Concurrent requests / single revalidation
# ---------------------------------------------------------------------------

class TestConcurrentRevalidation:
    """Multiple stale requests should produce only a single background
    revalidation task."""

    async def test_only_one_revalidation_inflight(self) -> None:
        cache = PromptCache(ttl_seconds=1.0, max_stale_seconds=60.0)
        stale_response = _make_prompt_response(version=1)
        fresh_response = _make_prompt_response(version=2)

        fetch_started = asyncio.Event()
        fetch_proceed = asyncio.Event()
        call_count = 0

        async def slow_fetch(
            project_id: str,
            name: str,
            *,
            version: int | None = None,
            label: str | None = None,
        ) -> PromptResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial population -- return immediately.
                return stale_response
            # Background revalidation -- wait for signal.
            fetch_started.set()
            await fetch_proceed.wait()
            return fresh_response

        # Populate.
        await cache.get_or_fetch("proj", "prompt", fetch_fn=slow_fetch)

        # Make stale.
        key = _make_cache_key("proj", "prompt")
        cache._store[key] = _CacheEntry(
            response=stale_response,
            fetched_at=time.monotonic() - 2.0,
        )

        # Fire multiple stale requests.
        r1 = await cache.get_or_fetch("proj", "prompt", fetch_fn=slow_fetch)
        # Wait for the background task to actually start.
        await asyncio.wait_for(fetch_started.wait(), timeout=1.0)

        r2 = await cache.get_or_fetch("proj", "prompt", fetch_fn=slow_fetch)
        r3 = await cache.get_or_fetch("proj", "prompt", fetch_fn=slow_fetch)

        # All should return the stale value.
        assert r1.version == 1
        assert r2.version == 1
        assert r3.version == 1

        # Only one revalidation should have been launched
        # (1 initial + 1 background = 2 total calls).
        assert call_count == 2

        # Let the background finish.
        fetch_proceed.set()
        await asyncio.sleep(0.1)

        # Now the cache should have the fresh value.
        r4 = await cache.get_or_fetch("proj", "prompt", fetch_fn=slow_fetch)
        assert r4.version == 2
        assert call_count == 2  # No additional fetch.


# ---------------------------------------------------------------------------
# Cache invalidation & clear
# ---------------------------------------------------------------------------

class TestCacheManagement:
    """Tests for invalidate() and clear()."""

    async def test_invalidate_removes_entry(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        fetch = _make_async_fetch()

        await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)
        assert fetch.await_count == 1

        removed = cache.invalidate("proj", "prompt")
        assert removed is True

        # Next call should miss and trigger a new fetch.
        await cache.get_or_fetch("proj", "prompt", fetch_fn=fetch)
        assert fetch.await_count == 2

    async def test_invalidate_nonexistent_returns_false(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        assert cache.invalidate("proj", "nope") is False

    async def test_clear_drops_all(self) -> None:
        cache = PromptCache(ttl_seconds=30.0, max_stale_seconds=60.0)
        fetch = _make_async_fetch()

        await cache.get_or_fetch("proj", "a", fetch_fn=fetch)
        await cache.get_or_fetch("proj", "b", fetch_fn=fetch)
        assert fetch.await_count == 2

        cache.clear()

        await cache.get_or_fetch("proj", "a", fetch_fn=fetch)
        await cache.get_or_fetch("proj", "b", fetch_fn=fetch)
        assert fetch.await_count == 4


# ---------------------------------------------------------------------------
# Background revalidation failure
# ---------------------------------------------------------------------------

class TestRevalidationFailure:
    """If the background revalidation fails, the stale entry should remain
    and the in-flight flag should be cleared."""

    async def test_failed_revalidation_keeps_stale_entry(self) -> None:
        cache = PromptCache(ttl_seconds=1.0, max_stale_seconds=60.0)
        stale_response = _make_prompt_response(version=1)

        call_count = 0

        async def failing_fetch(
            project_id: str,
            name: str,
            *,
            version: int | None = None,
            label: str | None = None,
        ) -> PromptResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stale_response
            raise RuntimeError("network error")

        # Populate.
        await cache.get_or_fetch("proj", "prompt", fetch_fn=failing_fetch)

        # Make stale.
        key = _make_cache_key("proj", "prompt")
        cache._store[key] = _CacheEntry(
            response=stale_response,
            fetched_at=time.monotonic() - 2.0,
        )

        # Should return stale and trigger (failing) revalidation.
        result = await cache.get_or_fetch("proj", "prompt", fetch_fn=failing_fetch)
        assert result.version == 1

        await asyncio.sleep(0.1)

        # The stale entry should still be there.
        assert key in cache._store
        assert cache._store[key].response.version == 1

        # The in-flight flag should be cleared so a retry is possible.
        assert key not in cache._inflight
