"""Tests for TaprootClient OAuth flow methods (mocked HTTP)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.toolbox.models import OAuthConnectionInfo, OAuthFlowResponse

BASE = "https://gateway.test"


def _client() -> TaprootClient:
    return TaprootClient(
        base_url=BASE,
        api_key="test-key",
        project_id="proj-1",
    )


def _json_resp(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


class TestStartOAuthFlow:
    @respx.mock
    @pytest.mark.asyncio
    async def test_returns_typed_response(self):
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/oauth/authorize").mock(
            return_value=_json_resp(
                {
                    "authorize_url": "https://auth.example.com/authorize?response_type=code&client_id=cid&state=s1",
                    "state": "s1",
                },
                200,
            )
        )

        async with _client() as c:
            result = await c.start_oauth_flow(
                tool_definition_id="tool-1",
                provider_url="https://auth.example.com/authorize",
                token_url="https://auth.example.com/token",
                client_id="cid",
                redirect_uri="https://app.example.com/callback",
                scopes=["read"],
            )

        assert isinstance(result, OAuthFlowResponse)
        assert result.state == "s1"
        assert "authorize" in result.authorize_url

    @respx.mock
    @pytest.mark.asyncio
    async def test_project_override(self):
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/other-proj/oauth/authorize").mock(
            return_value=_json_resp({"authorize_url": "u", "state": "s"}, 200)
        )

        async with _client() as c:
            result = await c.start_oauth_flow(
                tool_definition_id="tool-1",
                provider_url="https://auth.example.com/authorize",
                token_url="https://auth.example.com/token",
                client_id="cid",
                redirect_uri="https://app.example.com/callback",
                project_id="other-proj",
            )

        assert isinstance(result, OAuthFlowResponse)


class TestCompleteOAuthFlow:
    @respx.mock
    @pytest.mark.asyncio
    async def test_returns_connection_info(self):
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/oauth/callback").mock(
            return_value=_json_resp(
                {
                    "id": "conn-1",
                    "project_id": "proj-1",
                    "tool_definition_id": "tool-1",
                    "user_id": "user-1",
                    "token_type": "Bearer",
                    "scopes": ["read", "write"],
                    "expires_at": "2026-03-06T00:00:00Z",
                    "created_at": "2026-03-05T12:00:00Z",
                },
                201,
            )
        )

        async with _client() as c:
            result = await c.complete_oauth_flow(
                tool_definition_id="tool-1",
                code="auth-code",
                state="state-1",
                token_url="https://auth.example.com/token",
                client_id="cid",
                client_secret="csecret",
                redirect_uri="https://app.example.com/callback",
                user_id="user-1",
            )

        assert isinstance(result, OAuthConnectionInfo)
        assert result.id == "conn-1"
        assert result.user_id == "user-1"
        assert result.scopes == ("read", "write")
        assert result.token_type == "Bearer"


class TestClientCredentialsGrant:
    @respx.mock
    @pytest.mark.asyncio
    async def test_returns_connection_info(self):
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/oauth/client-credentials",
        ).mock(
            return_value=_json_resp(
                {
                    "id": "conn-2",
                    "project_id": "proj-1",
                    "tool_definition_id": "tool-1",
                    "user_id": "service",
                    "token_type": "Bearer",
                    "scopes": ["api.read"],
                    "expires_at": "2026-03-05T14:00:00Z",
                    "created_at": "2026-03-05T12:00:00Z",
                },
                201,
            )
        )

        async with _client() as c:
            result = await c.client_credentials_grant(
                tool_definition_id="tool-1",
                token_url="https://auth.example.com/token",
                client_id="cid",
                client_secret="csecret",
                scopes=["api.read"],
            )

        assert isinstance(result, OAuthConnectionInfo)
        assert result.id == "conn-2"
        assert result.user_id == "service"
        assert result.scopes == ("api.read",)

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_user_id(self):
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/oauth/client-credentials",
        ).mock(
            return_value=_json_resp(
                {
                    "id": "conn-3",
                    "project_id": "proj-1",
                    "tool_definition_id": "tool-1",
                    "user_id": "my-service",
                    "token_type": "Bearer",
                    "scopes": [],
                },
                201,
            )
        )

        async with _client() as c:
            result = await c.client_credentials_grant(
                tool_definition_id="tool-1",
                token_url="https://auth.example.com/token",
                client_id="cid",
                client_secret="csecret",
                user_id="my-service",
            )

        assert result.user_id == "my-service"
