"""Tests for taproot_sdk.prompts.client module."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from taproot_sdk.prompts.client import PromptClient
from taproot_sdk.prompts.models import PromptResponse


def _serving_response(
    *,
    name: str = "welcome-email",
    version: int = 3,
    content: str = "Hello {{user_name}}!",
    config: dict[str, Any] | None = None,
    required_variables: list[str] | None = None,
    schema_version: int = 1,
    label: str | None = None,
    cached_at: str | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Build a mock serving-layer JSON response body."""
    if content_hash is None:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "schema_version": schema_version,
        "name": name,
        "version": version,
        "content": content,
        "content_hash": content_hash,
        "config": config if config is not None else {"model": "gpt-4o"},
        "required_variables": required_variables if required_variables is not None else ["user_name"],
        "label": label,
        "cached_at": cached_at,
    }


def _mock_httpx_response(
    data: dict[str, Any],
    status_code: int = 200,
) -> httpx.Response:
    """Build a fake httpx.Response from a dict body."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://fake.test/serve/proj/prompt"),
    )


class TestPromptClientGet:
    """Tests for PromptClient.get()."""

    async def test_get_calls_correct_url(self) -> None:
        """get() should call GET /serve/{project_id}/{name}."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="test-key-id",
        )
        body = _serving_response()
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            result = await client.get("my-project", "welcome-email")

        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        assert call_args[0][0] == "https://prompts.taproot.dev/serve/my-project/welcome-email"
        assert isinstance(result, PromptResponse)

    async def test_get_sends_api_key_header(self) -> None:
        """get() should send X-Api-Key-Id header via shared client."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="my-secret-key-id",
        )
        body = _serving_response()
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.is_closed = False
            mock_cls.return_value = mock_http

            await client.get("proj", "prompt")

        # Header is now set on the shared httpx.AsyncClient constructor
        constructor_kwargs = mock_cls.call_args[1]
        assert constructor_kwargs["headers"]["X-Api-Key-Id"] == "my-secret-key-id"

    async def test_get_with_version_param(self) -> None:
        """get(version=5) should add ?version=5 to the request."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(version=5)
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            await client.get("proj", "prompt", version=5)

        call_kwargs = mock_http.get.call_args[1]
        assert call_kwargs["params"] == {"version": "5"}

    async def test_get_with_label_param(self) -> None:
        """get(label='production') should add ?label=production to the request."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(label="production")
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            await client.get("proj", "prompt", label="production")

        call_kwargs = mock_http.get.call_args[1]
        assert call_kwargs["params"] == {"label": "production"}

    async def test_get_no_version_or_label_sends_empty_params(self) -> None:
        """get() without version/label should send empty params."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response()
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            await client.get("proj", "prompt")

        call_kwargs = mock_http.get.call_args[1]
        assert call_kwargs["params"] == {}

    async def test_get_both_version_and_label_raises(self) -> None:
        """get() with both version and label should raise ValueError."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            await client.get("proj", "prompt", version=1, label="prod")

    async def test_get_invalid_schema_version_raises(self) -> None:
        """get() with unsupported schema_version should raise ValueError."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(schema_version=99)
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            with pytest.raises(ValueError, match="Unsupported schema_version: 99"):
                await client.get("proj", "prompt")

    async def test_get_maps_response_fields_correctly(self) -> None:
        """get() should map all JSON fields to PromptResponse attributes."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        content = "Hello {{name}} at {{place}}!"
        body = _serving_response(
            name="greeting",
            version=7,
            content=content,
            config={"model": "claude-3", "max_tokens": 100},
            required_variables=["name", "place"],
            label="staging",
            cached_at="2026-01-15T10:30:00Z",
        )
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            result = await client.get("proj", "greeting")

        assert result.schema_version == 1
        assert result.name == "greeting"
        assert result.version == 7
        assert result.content == content
        assert result.config == {"model": "claude-3", "max_tokens": 100}
        assert result.required_variables == ("name", "place")
        assert result.label == "staging"
        assert result.cached_at == "2026-01-15T10:30:00Z"
        assert result.verify_hash() is True

    async def test_get_strips_trailing_slash_from_serving_url(self) -> None:
        """PromptClient should strip trailing slashes from serving_url."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev/",
            api_key="key",
        )
        body = _serving_response()
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            await client.get("proj", "prompt")

        call_url = mock_http.get.call_args[0][0]
        assert call_url.startswith("https://prompts.taproot.dev/serve/")
        assert "//serve" not in call_url


class TestPromptClientGetSync:
    """Tests for PromptClient.get_sync()."""

    def test_get_sync_returns_prompt_response(self) -> None:
        """get_sync() should return a PromptResponse synchronously."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response()
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            result = client.get_sync("proj", "prompt")

        assert isinstance(result, PromptResponse)
        assert result.name == "welcome-email"

    def test_get_sync_with_version(self) -> None:
        """get_sync(version=2) should pass version through to get()."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(version=2)
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            result = client.get_sync("proj", "prompt", version=2)

        assert result.version == 2
        call_kwargs = mock_http.get.call_args[1]
        assert call_kwargs["params"] == {"version": "2"}

    def test_get_sync_with_label(self) -> None:
        """get_sync(label='prod') should pass label through to get()."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(label="prod")
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            result = client.get_sync("proj", "prompt", label="prod")

        assert result.label == "prod"
        call_kwargs = mock_http.get.call_args[1]
        assert call_kwargs["params"] == {"label": "prod"}


class TestPromptClientTracing:
    """Tests for optional OpenTelemetry instrumentation in PromptClient."""

    @pytest.fixture(autouse=True)
    def _setup_otel(self) -> Any:  # noqa: ANN401
        """Set up an isolated TracerProvider and patch the module _tracer."""
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        import taproot_sdk.prompts.client as client_mod

        self._original_tracer = client_mod._tracer
        # Create a tracer from our test-only provider (not the global one)
        client_mod._tracer = self.provider.get_tracer("taproot-sdk.prompts")
        yield
        client_mod._tracer = self._original_tracer
        self.provider.shutdown()

    async def test_fetch_creates_span_with_attributes(self) -> None:
        """_fetch() should create a taproot.prompt.fetch span with prompt metadata."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(name="greeting", version=5, label="production")
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            await client.get("my-project", "greeting", label="production")

        spans = self.exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "taproot.prompt.fetch"

        attrs = dict(span.attributes or {})
        assert attrs["prompt.name"] == "greeting"
        assert attrs["prompt.version"] == 5
        assert attrs["prompt.project_id"] == "my-project"
        assert attrs["prompt.type"] == "text"
        assert attrs["prompt.label"] == "production"
        assert attrs["prompt.cached"] is False
        assert "prompt.hash" in attrs

    async def test_fetch_span_omits_label_when_not_specified(self) -> None:
        """Span should not include prompt.label attribute when label is None."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response(name="greeting", version=1)
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            await client.get("proj", "greeting")

        spans = self.exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert "prompt.label" not in attrs

    async def test_client_works_without_otel(self) -> None:
        """Client should work normally when _tracer is None (no opentelemetry)."""
        import taproot_sdk.prompts.client as client_mod

        # Temporarily disable tracing
        original = client_mod._tracer
        client_mod._tracer = None  # type: ignore[assignment]
        try:
            client = PromptClient(
                serving_url="https://prompts.taproot.dev",
                api_key="key",
            )
            body = _serving_response()
            mock_response = _mock_httpx_response(body)

            with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
                mock_http = AsyncMock()
                mock_http.get.return_value = mock_response
                mock_http.__aenter__ = AsyncMock(return_value=mock_http)
                mock_http.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_http

                result = await client.get("proj", "prompt")

            assert isinstance(result, PromptResponse)
            assert result.name == "welcome-email"
            # No spans should be emitted
            assert len(self.exporter.get_finished_spans()) == 0
        finally:
            client_mod._tracer = original

    async def test_no_span_on_cache_hit(self) -> None:
        """Second get() on cached prompt should not create a new span."""
        client = PromptClient(
            serving_url="https://prompts.taproot.dev",
            api_key="key",
        )
        body = _serving_response()
        mock_response = _mock_httpx_response(body)

        with patch("taproot_sdk.prompts.client.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_response
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_http

            # First call -- fetches, creates span
            await client.get("proj", "prompt")
            # Second call -- cache hit, no new span
            await client.get("proj", "prompt")

        spans = self.exporter.get_finished_spans()
        # Only one span from the initial fetch
        assert len(spans) == 1
