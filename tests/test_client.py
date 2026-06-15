"""Tests for taproot_sdk.client (TaprootClient)."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from taproot_sdk._context import (
    TaprootActorRef,
    TaprootInteractionContext,
    clear_interaction_context,
    correlation_id_var,
    get_interaction_context,
    merge_propagation_headers,
    propagation_headers,
    reset_interaction_context,
    set_interaction_context,
)
from taproot_sdk.client import TaprootClient
from taproot_sdk.instrument import _TaprootContextMiddleware
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


class TestTaprootInteractionPropagation:
    def test_propagation_headers_include_tap38_context_fields(self) -> None:
        headers = propagation_headers(
            TaprootInteractionContext(
                interaction_id="int-1",
                interaction_type="agent_run",
                caller=TaprootActorRef(actor_type="user", actor_id="user-1"),
                source_agent_id="agent-1",
                root_agent_id="root-agent",
                correlation_id="corr-1",
                trace_id="00-trace-span-01",
                parent_activity_id="act-parent",
            )
        )

        assert headers == {
            "X-Taproot-Activity-Version": "1",
            "X-Taproot-Interaction-Id": "int-1",
            "X-Taproot-Interaction-Type": "agent_run",
            "X-Taproot-Caller-Id": "user-1",
            "X-Taproot-Caller-Type": "user",
            "X-Taproot-Source-Agent-Id": "agent-1",
            "X-Taproot-Root-Agent-Id": "root-agent",
            "X-Taproot-Parent-Activity-Id": "act-parent",
            "X-Correlation-ID": "corr-1",
            "traceparent": "00-trace-span-01",
        }

    def test_merge_propagation_headers_preserves_explicit_values(self) -> None:
        context = TaprootInteractionContext(
            interaction_id="int-new",
            interaction_type="agent_run",
            correlation_id="corr-new",
        )

        merged = merge_propagation_headers(
            {"x-taproot-interaction-id": "int-existing", "Authorization": "Bearer token"},
            context=context,
        )

        assert merged["x-taproot-interaction-id"] == "int-existing"
        assert merged["Authorization"] == "Bearer token"
        assert merged["X-Taproot-Interaction-Type"] == "agent_run"
        assert merged["X-Correlation-ID"] == "corr-new"

    async def test_taproot_client_propagates_current_context_without_overwriting_headers(self) -> None:
        client = TaprootClient(
            base_url="https://api.test",
            api_key="k",
            project_id="p",
            agent_id="agent-sdk",
        )
        client._http = AsyncMock()
        client._http.request.return_value = httpx.Response(status_code=200)

        token = set_interaction_context(
            TaprootInteractionContext(
                interaction_id="int-1",
                interaction_type="agent_run",
                caller=TaprootActorRef(actor_type="user", actor_id="user-1"),
                correlation_id="corr-context",
            )
        )
        try:
            await client._request(
                "GET",
                "/v1/test",
                headers={"X-Correlation-ID": "corr-explicit", "Idempotency-Key": "idem-1"},
            )
        finally:
            reset_interaction_context(token)
            clear_interaction_context()

        _, _, kwargs = client._http.request.mock_calls[0]
        headers = kwargs["headers"]
        assert headers["X-Taproot-Interaction-Id"] == "int-1"
        assert headers["X-Taproot-Interaction-Type"] == "agent_run"
        assert headers["X-Taproot-Caller-Id"] == "user-1"
        assert headers["X-Agent-Id"] == "agent-sdk"
        assert headers["X-Correlation-ID"] == "corr-explicit"
        assert headers["Idempotency-Key"] == "idem-1"

    async def test_asgi_instrumentation_binds_inbound_tap38_headers_and_resets(self) -> None:
        observed: dict[str, Any] = {}

        async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            observed["correlation_id"] = correlation_id_var.get()
            observed["context"] = get_interaction_context()

        middleware = _TaprootContextMiddleware(app)

        await middleware(
            {
                "type": "http",
                "headers": [
                    (b"x-taproot-interaction-id", b"int-1"),
                    (b"x-taproot-interaction-type", b"agent_run"),
                    (b"x-taproot-caller-id", b"user-1"),
                    (b"x-taproot-caller-type", b"user"),
                    (b"x-taproot-source-agent-id", b"agent-1"),
                    (b"x-taproot-root-agent-id", b"root-agent"),
                    (b"x-taproot-parent-activity-id", b"act-parent"),
                    (b"x-correlation-id", b"corr-1"),
                    (b"traceparent", b"00-trace-span-01"),
                ],
            },
            None,
            None,
        )

        context = observed["context"]
        assert observed["correlation_id"] == "corr-1"
        assert context == TaprootInteractionContext(
            interaction_id="int-1",
            interaction_type="agent_run",
            caller=TaprootActorRef(actor_type="user", actor_id="user-1"),
            source_agent_id="agent-1",
            root_agent_id="root-agent",
            correlation_id="corr-1",
            trace_id="00-trace-span-01",
            parent_activity_id="act-parent",
        )
        assert correlation_id_var.get() is None
        assert get_interaction_context() is None

    async def test_retry_headers_are_stable_across_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = TaprootClient(
            base_url="https://api.test",
            api_key="k",
            project_id="p",
            agent_id="agent-sdk",
        )
        client._http = AsyncMock()
        monkeypatch.setattr("taproot_sdk.client.asyncio.sleep", AsyncMock())

        call_count = 0

        async def request(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                clear_interaction_context()
                set_interaction_context(
                    TaprootInteractionContext(
                        interaction_id="int-mutated",
                        interaction_type="tool_call",
                        correlation_id="corr-mutated",
                    )
                )
                return httpx.Response(status_code=503)
            return httpx.Response(status_code=200)

        client._http.request.side_effect = request
        token = set_interaction_context(
            TaprootInteractionContext(
                interaction_id="int-1",
                interaction_type="agent_run",
                caller=TaprootActorRef(actor_type="user", actor_id="user-1"),
                correlation_id="corr-context",
            )
        )
        try:
            await client._request(
                "POST",
                "/v1/test",
                json={"value": 1},
                headers={
                    "Authorization": "Bearer token",
                    "x-api-key": "auth-explicit",
                    "X-Correlation-ID": "corr-explicit",
                    "Idempotency-Key": "idem-1",
                    "X-Taproot-Caller-Id": "caller-explicit",
                },
            )
        finally:
            reset_interaction_context(token)
            clear_interaction_context()

        first_headers = client._http.request.mock_calls[0].kwargs["headers"]
        second_headers = client._http.request.mock_calls[1].kwargs["headers"]
        assert first_headers == second_headers
        assert first_headers["X-Taproot-Interaction-Id"] == "int-1"
        assert first_headers["X-Taproot-Interaction-Type"] == "agent_run"
        assert first_headers["X-Taproot-Caller-Id"] == "caller-explicit"
        assert first_headers["X-Taproot-Caller-Type"] == "user"
        assert first_headers["X-Agent-Id"] == "agent-sdk"
        assert first_headers["X-Correlation-ID"] == "corr-explicit"
        assert first_headers["Idempotency-Key"] == "idem-1"
        assert first_headers["Authorization"] == "Bearer token"
        assert first_headers["x-api-key"] == "auth-explicit"


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
