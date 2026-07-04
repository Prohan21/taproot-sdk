"""WO-013 T2: client routing/parse tests for Prompt-S serving (get_prompt).

Rendering/caching are covered elsewhere; this exercises the client
fetch/route path and typed PromptResponse parsing.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.exceptions import PromptNotFoundError
from taproot_sdk.prompts.models import PromptResponse, PromptType

BASE = "https://gateway.test"


def _client(direct_mode: bool = False) -> TaprootClient:
    return TaprootClient(
        base_url=BASE,
        api_key="test-key",
        project_id="proj-1",
        direct_mode=direct_mode,
    )


def _json_resp(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


SERVED_PROMPT = {
    "schema_version": 1,
    "name": "welcome-email",
    "version": 4,
    "content": "Hello {{user_name}}, welcome to {{plan}}!",
    "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "config": {"model": "gpt-4o"},
    "required_variables": ["user_name", "plan"],
    "label": "production",
    "prompt_type": "text",
}


class TestGetPromptServing:
    @respx.mock
    async def test_serving_route_and_typed_parse(self):
        route = respx.get(f"{BASE}/serve/proj-1/welcome-email").mock(
            return_value=_json_resp(SERVED_PROMPT)
        )
        prompt = await _client().get_prompt("welcome-email", label="production")
        assert isinstance(prompt, PromptResponse)
        assert prompt.version == 4
        assert prompt.prompt_type is PromptType.TEXT
        assert prompt.required_variables == ("user_name", "plan")
        assert prompt.config == {"model": "gpt-4o"}
        req = route.calls[0].request
        assert req.method == "GET"
        assert req.url.params["label"] == "production"
        assert req.headers["x-api-key"] == "test-key"

    @respx.mock
    async def test_direct_mode_uses_same_serving_path(self):
        route = respx.get(f"{BASE}/serve/proj-1/welcome-email").mock(
            return_value=_json_resp(SERVED_PROMPT)
        )
        prompt = await _client(direct_mode=True).get_prompt("welcome-email", version=4)
        assert prompt.name == "welcome-email"
        req = route.calls[0].request
        assert req.url.params["version"] == "4"
        assert req.headers["X-Api-Key-Id"] == "test-key"

    @respx.mock
    async def test_project_id_override_routes_to_project(self):
        respx.get(f"{BASE}/serve/other-proj/welcome-email").mock(
            return_value=_json_resp(SERVED_PROMPT)
        )
        prompt = await _client().get_prompt("welcome-email", project_id="other-proj")
        assert prompt.version == 4

    @respx.mock
    async def test_chat_prompt_parses_messages(self):
        respx.get(f"{BASE}/serve/proj-1/chat-template").mock(
            return_value=_json_resp(
                {
                    **SERVED_PROMPT,
                    "name": "chat-template",
                    "prompt_type": "chat",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Tell me about {{topic}}."},
                    ],
                }
            )
        )
        prompt = await _client().get_prompt("chat-template")
        assert prompt.prompt_type is PromptType.CHAT
        assert prompt.messages is not None
        assert prompt.messages[0].role == "system"

    @respx.mock
    async def test_404_maps_to_prompt_not_found(self):
        respx.get(f"{BASE}/serve/proj-1/missing").mock(
            return_value=httpx.Response(404, json={"detail": "Prompt not found"})
        )
        with pytest.raises(PromptNotFoundError) as exc_info:
            await _client().get_prompt("missing")
        assert exc_info.value.prompt_name == "missing"

    @respx.mock
    async def test_unsupported_schema_version_rejected(self):
        respx.get(f"{BASE}/serve/proj-1/welcome-email").mock(
            return_value=_json_resp({**SERVED_PROMPT, "schema_version": 99})
        )
        with pytest.raises(ValueError, match="schema_version"):
            await _client().get_prompt("welcome-email")

    @respx.mock
    async def test_version_and_label_mutually_exclusive(self):
        with pytest.raises(ValueError, match="version"):
            await _client().get_prompt("welcome-email", version=1, label="production")
