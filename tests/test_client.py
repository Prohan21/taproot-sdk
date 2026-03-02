"""Tests for taproot_sdk.client (TaprootClient)."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from taproot_sdk.client import TaprootClient
from taproot_sdk.prompts.models import PromptResponse, PromptType


def _serving_response(
    *,
    name: str = "welcome",
    version: int = 1,
    content: str = "Hello {{user}}!",
    prompt_type: str = "text",
    messages: list[dict[str, str]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    ab_test: bool = False,
    selected_variant: int | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    if content_hash is None:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "version": version,
        "content": content,
        "content_hash": content_hash,
        "config": {},
        "required_variables": ["user"],
        "prompt_type": prompt_type,
    }
    if messages is not None:
        data["messages"] = messages
    if tools is not None:
        data["tools"] = tools
    if ab_test:
        data["ab_test"] = True
    if selected_variant is not None:
        data["selected_variant"] = selected_variant
    return data


def _mock_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("GET", "https://fake.test/serve/proj/prompt"),
    )


class TestTaprootClientGetPrompt:
    """Tests for TaprootClient.get_prompt() full field parsing."""

    async def test_parses_text_prompt(self) -> None:
        body = _serving_response(prompt_type="text")
        client = TaprootClient(base_url="https://api.test", api_key="k", project_id="p")
        client._http = AsyncMock()
        client._http.request.return_value = _mock_response(body)

        result = await client.get_prompt("welcome")

        assert isinstance(result, PromptResponse)
        assert result.prompt_type == PromptType.TEXT
        assert result.messages is None
        assert result.tools is None
        assert result.ab_test is False
        assert result.selected_variant is None

    async def test_parses_chat_prompt_with_messages(self) -> None:
        body = _serving_response(
            prompt_type="chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello {{user}}"},
            ],
        )
        client = TaprootClient(base_url="https://api.test", api_key="k", project_id="p")
        client._http = AsyncMock()
        client._http.request.return_value = _mock_response(body)

        result = await client.get_prompt("welcome")

        assert result.prompt_type == PromptType.CHAT
        assert result.messages is not None
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[1].content == "Hello {{user}}"

    async def test_parses_tools(self) -> None:
        body = _serving_response(
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            ],
        )
        client = TaprootClient(base_url="https://api.test", api_key="k", project_id="p")
        client._http = AsyncMock()
        client._http.request.return_value = _mock_response(body)

        result = await client.get_prompt("welcome")

        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == "get_weather"
        assert result.tools[0].type == "function"

    async def test_parses_ab_test_metadata(self) -> None:
        body = _serving_response(ab_test=True, selected_variant=2)
        client = TaprootClient(base_url="https://api.test", api_key="k", project_id="p")
        client._http = AsyncMock()
        client._http.request.return_value = _mock_response(body)

        result = await client.get_prompt("welcome")

        assert result.ab_test is True
        assert result.selected_variant == 2

    async def test_unknown_prompt_type_defaults_to_text(self) -> None:
        body = _serving_response(prompt_type="unknown_future_type")
        client = TaprootClient(base_url="https://api.test", api_key="k", project_id="p")
        client._http = AsyncMock()
        client._http.request.return_value = _mock_response(body)

        result = await client.get_prompt("welcome")

        assert result.prompt_type == PromptType.TEXT


class TestTaprootClientLifecycle:
    """Tests for TaprootClient async context manager and close."""

    async def test_async_context_manager(self) -> None:
        async with TaprootClient(
            base_url="https://api.test", api_key="k", project_id="p"
        ) as client:
            assert isinstance(client, TaprootClient)
        # After exiting, the http client should be closed
        assert client._http.is_closed

    async def test_close(self) -> None:
        client = TaprootClient(
            base_url="https://api.test", api_key="k", project_id="p"
        )
        assert not client._http.is_closed
        await client.close()
        assert client._http.is_closed

    def test_missing_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url is required"):
            TaprootClient(api_key="k", project_id="p")


class TestPromptClientLifecycle:
    """Tests for PromptClient lifecycle (close, async with, shared http)."""

    async def test_async_context_manager(self) -> None:
        from taproot_sdk.prompts.client import PromptClient

        async with PromptClient(
            serving_url="https://prompts.test", api_key="k"
        ) as client:
            # Trigger lazy creation
            http = client._get_http()
            assert http is not None
        # After exit, http should be closed
        assert client._http is None

    async def test_shared_http_client_reused(self) -> None:
        from taproot_sdk.prompts.client import PromptClient

        client = PromptClient(serving_url="https://prompts.test", api_key="k")
        http1 = client._get_http()
        http2 = client._get_http()
        assert http1 is http2
        await client.close()

    async def test_timeout_parameter(self) -> None:
        from taproot_sdk.prompts.client import PromptClient

        client = PromptClient(
            serving_url="https://prompts.test", api_key="k", timeout=5.0,
        )
        assert client._timeout == 5.0
        await client.close()

    async def test_default_timeout_is_10s(self) -> None:
        from taproot_sdk.prompts.client import PromptClient

        client = PromptClient(
            serving_url="https://prompts.test", api_key="k",
        )
        assert client._timeout == 10.0
        await client.close()
