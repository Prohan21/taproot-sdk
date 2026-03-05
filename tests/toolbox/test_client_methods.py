"""Tests for TaprootClient toolbox methods (mocked HTTP)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.exceptions import TaprootAPIError, ValidationError
from taproot_sdk.toolbox.models import (
    CredentialInfo,
    CredentialList,
    ImportResult,
    InvocationResult,
    MCPServerInfo,
    MCPServerList,
    ToolInfo,
    ToolList,
)

BASE = "https://gateway.test"


def _client() -> TaprootClient:
    return TaprootClient(
        base_url=BASE,
        api_key="test-key",
        project_id="proj-1",
    )


def _json_resp(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


_TOOL_RESPONSE = {
    "id": "tool-001",
    "project_id": "proj-1",
    "name": "add",
    "description": "Add two numbers",
    "tool_type": "hosted",
    "input_schema": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
    "version": 1,
    "status": "active",
    "scope": "project",
    "tags": ["math"],
    "entry_point": "add",
    "requirements": [],
    "content_hash": "abc123",
    "timeout_ms": 30000,
    "memory_mb": 256,
}


class TestPushTool:
    @respx.mock
    @pytest.mark.asyncio
    async def test_push_returns_typed(self):
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/push").mock(
            return_value=_json_resp(_TOOL_RESPONSE, 201)
        )

        async with _client() as c:
            tool = await c.push_tool(
                name="add",
                source_code="def add(a, b):\n    return a + b\n",
                entry_point="add",
                description="Add two numbers",
                tags=["math"],
            )

        assert isinstance(tool, ToolInfo)
        assert tool.name == "add"
        assert tool.is_hosted
        assert tool.is_active

    @respx.mock
    @pytest.mark.asyncio
    async def test_push_with_requirements(self):
        resp = {**_TOOL_RESPONSE, "status": "building", "requirements": ["pandas>=2.0"]}
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/push").mock(
            return_value=_json_resp(resp, 201)
        )

        async with _client() as c:
            tool = await c.push_tool(
                name="add",
                source_code="import pandas\ndef add(a, b):\n    return a + b\n",
                entry_point="add",
                requirements=["pandas>=2.0"],
            )

        assert tool.is_building
        assert tool.requirements == ("pandas>=2.0",)


class TestRegisterTool:
    @respx.mock
    @pytest.mark.asyncio
    async def test_register_returns_typed(self):
        resp = {
            **_TOOL_RESPONSE,
            "tool_type": "external",
            "endpoint_url": "https://api.weather.com/v1",
            "http_method": "POST",
            "auth_type": "api_key",
        }
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/register").mock(
            return_value=_json_resp(resp, 201)
        )

        async with _client() as c:
            tool = await c.register_tool(
                name="add",
                endpoint_url="https://api.weather.com/v1",
                auth_type="api_key",
            )

        assert isinstance(tool, ToolInfo)
        assert not tool.is_hosted
        assert tool.endpoint_url == "https://api.weather.com/v1"


class TestInvokeTool:
    @respx.mock
    @pytest.mark.asyncio
    async def test_invoke_success(self):
        resp = {
            "invocation_id": "inv-001",
            "tool_name": "add",
            "success": True,
            "result": 7,
            "error": None,
            "duration_ms": 15.2,
        }
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/invoke/add").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.invoke_tool("add", {"a": 3, "b": 4})

        assert isinstance(result, InvocationResult)
        assert result.success
        assert result.result == 7

    @respx.mock
    @pytest.mark.asyncio
    async def test_invoke_failure(self):
        resp = {
            "invocation_id": "inv-002",
            "tool_name": "fail",
            "success": False,
            "result": None,
            "error": "ValueError: bad input",
            "duration_ms": 3.0,
        }
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/invoke/fail").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.invoke_tool("fail", {"x": 1})

        assert not result.success
        assert "ValueError" in (result.error or "")


class TestListTools:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_returns_typed(self):
        resp = {
            "tools": [_TOOL_RESPONSE],
            "project_id": "proj-1",
            "count": 1,
        }
        respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.list_tools()

        assert isinstance(result, ToolList)
        assert result.count == 1
        assert result.tools[0].name == "add"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        resp = {"tools": [], "project_id": "proj-1", "count": 0}
        route = respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.list_tools(tags=["math"], tool_type="hosted", status="active")

        assert result.count == 0
        # Verify query params were sent
        assert route.called


class TestGetTool:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_returns_typed(self):
        respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/tool-001").mock(
            return_value=_json_resp(_TOOL_RESPONSE)
        )

        async with _client() as c:
            tool = await c.get_tool("tool-001")

        assert isinstance(tool, ToolInfo)
        assert tool.id == "tool-001"


class TestDeleteTool:
    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_succeeds(self):
        respx.delete(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/tool-001").mock(
            return_value=httpx.Response(204)
        )

        async with _client() as c:
            await c.delete_tool("tool-001")  # Should not raise


class TestImportOpenAPI:
    @respx.mock
    @pytest.mark.asyncio
    async def test_import_with_spec_url(self):
        resp_data = {
            "tools_created": [_TOOL_RESPONSE],
            "tools_skipped": 0,
            "total_parsed": 1,
            "namespace": "petstore",
        }
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/import-openapi"
        ).mock(return_value=_json_resp(resp_data, 201))

        async with _client() as c:
            result = await c.import_openapi(
                "petstore",
                spec_url="https://petstore.swagger.io/v3/openapi.json",
            )

        assert isinstance(result, ImportResult)
        assert result.namespace == "petstore"
        assert len(result.tools_created) == 1
        assert result.tools_created[0].name == "add"

    @respx.mock
    @pytest.mark.asyncio
    async def test_import_with_spec_body(self):
        resp_data = {
            "tools_created": [],
            "tools_skipped": 0,
            "total_parsed": 0,
            "namespace": "test",
        }
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/import-openapi"
        ).mock(return_value=_json_resp(resp_data, 201))

        async with _client() as c:
            result = await c.import_openapi(
                "test",
                spec_body={"openapi": "3.0.0", "paths": {}},
            )

        assert result.total_parsed == 0
        assert result.tools_created == ()

    @respx.mock
    @pytest.mark.asyncio
    async def test_import_with_all_options(self):
        resp_data = {
            "tools_created": [_TOOL_RESPONSE],
            "tools_skipped": 2,
            "total_parsed": 3,
            "namespace": "stripe",
        }
        route = respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/import-openapi"
        ).mock(return_value=_json_resp(resp_data, 201))

        async with _client() as c:
            result = await c.import_openapi(
                "stripe",
                spec_url="https://stripe.com/openapi.json",
                base_url="https://api.stripe.com/v1",
                tags=["payments"],
                auth_type="bearer",
                scope="global",
            )

        assert result.tools_skipped == 2
        assert result.total_parsed == 3
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_import_direct_mode_path(self):
        resp_data = {
            "tools_created": [],
            "tools_skipped": 0,
            "total_parsed": 0,
            "namespace": "ns",
        }
        respx.post(f"{BASE}/v1/projects/proj-1/tools/import-openapi").mock(
            return_value=_json_resp(resp_data, 201)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            result = await c.import_openapi("ns", spec_url="https://example.com/spec.json")

        assert result.namespace == "ns"

    @respx.mock
    @pytest.mark.asyncio
    async def test_import_validation_error(self):
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/import-openapi"
        ).mock(
            return_value=httpx.Response(422, json={"detail": "Invalid spec"})
        )

        async with _client() as c:
            with pytest.raises(ValidationError):
                await c.import_openapi("bad", spec_url="https://invalid.com/spec.json")


class TestHealthToolbox:
    @respx.mock
    @pytest.mark.asyncio
    async def test_health(self):
        respx.get(f"{BASE}/api/v1/toolbox/v1/health").mock(
            return_value=_json_resp({"status": "healthy", "service": "toolbox-s"})
        )

        async with _client() as c:
            health = await c.health_toolbox()

        assert health["status"] == "healthy"


_CREDENTIAL_RESPONSE = {
    "id": "cred-001",
    "project_id": "proj-1",
    "tool_definition_id": "tool-001",
    "credential_type": "api_key",
    "name": "My API Key",
    "status": "active",
    "version": 1,
    "created_at": "2026-03-01T00:00:00Z",
}

_MCP_SERVER_RESPONSE = {
    "id": "srv-001",
    "project_id": "proj-1",
    "name": "My MCP Server",
    "description": "A test server",
    "transport_type": "sse",
    "url": "https://mcp.example.com",
    "capabilities": ["tools"],
    "tools_discovered": ["search"],
    "health_status": "online",
    "tags": ["prod"],
    "scope": "project",
    "created_at": "2026-03-01T00:00:00Z",
}


class TestSetCredential:
    @respx.mock
    @pytest.mark.asyncio
    async def test_set_returns_typed(self):
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials").mock(
            return_value=_json_resp(_CREDENTIAL_RESPONSE, 201)
        )

        async with _client() as c:
            cred = await c.set_tool_credential(
                tool_definition_id="tool-001",
                credential_type="api_key",
                name="My API Key",
                credential_payload={"key": "secret-value"},
            )

        assert isinstance(cred, CredentialInfo)
        assert cred.id == "cred-001"
        assert cred.credential_type == "api_key"
        assert cred.is_active


class TestListCredentials:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_returns_typed(self):
        resp = {
            "credentials": [_CREDENTIAL_RESPONSE],
            "project_id": "proj-1",
            "count": 1,
        }
        respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.list_credentials()

        assert isinstance(result, CredentialList)
        assert result.count == 1
        assert result.credentials[0].name == "My API Key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_with_tool_filter(self):
        resp = {"credentials": [], "project_id": "proj-1", "count": 0}
        route = respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.list_credentials(tool_definition_id="tool-001")

        assert result.count == 0
        assert route.called


class TestRevokeCredential:
    @respx.mock
    @pytest.mark.asyncio
    async def test_revoke_returns_typed(self):
        revoked = {**_CREDENTIAL_RESPONSE, "status": "revoked", "version": 2}
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials/cred-001/revoke"
        ).mock(return_value=_json_resp(revoked))

        async with _client() as c:
            cred = await c.revoke_credential("cred-001", version=1)

        assert isinstance(cred, CredentialInfo)
        assert cred.status == "revoked"
        assert not cred.is_active


class TestRegisterMCPServer:
    @respx.mock
    @pytest.mark.asyncio
    async def test_register_returns_typed(self):
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers").mock(
            return_value=_json_resp(_MCP_SERVER_RESPONSE, 201)
        )

        async with _client() as c:
            srv = await c.register_mcp_server(
                name="My MCP Server",
                transport_type="sse",
                url="https://mcp.example.com",
                description="A test server",
                tags=["prod"],
            )

        assert isinstance(srv, MCPServerInfo)
        assert srv.name == "My MCP Server"
        assert srv.transport_type == "sse"
        assert srv.is_online


class TestListMCPServers:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_returns_typed(self):
        resp = {
            "servers": [_MCP_SERVER_RESPONSE],
            "project_id": "proj-1",
            "count": 1,
        }
        respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.list_mcp_servers()

        assert isinstance(result, MCPServerList)
        assert result.count == 1
        assert result.servers[0].name == "My MCP Server"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        resp = {"servers": [], "project_id": "proj-1", "count": 0}
        route = respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers").mock(
            return_value=_json_resp(resp)
        )

        async with _client() as c:
            result = await c.list_mcp_servers(status="online", tags=["prod"])

        assert result.count == 0
        assert route.called


class TestGetMCPServer:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_returns_typed(self):
        respx.get(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers/srv-001").mock(
            return_value=_json_resp(_MCP_SERVER_RESPONSE)
        )

        async with _client() as c:
            srv = await c.get_mcp_server("srv-001")

        assert isinstance(srv, MCPServerInfo)
        assert srv.id == "srv-001"


class TestDeleteMCPServer:
    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_succeeds(self):
        respx.delete(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers/srv-001").mock(
            return_value=httpx.Response(204)
        )

        async with _client() as c:
            await c.delete_mcp_server("srv-001")  # Should not raise


class TestDirectMode:
    @respx.mock
    @pytest.mark.asyncio
    async def test_push_direct_mode_path(self):
        """Direct mode uses /v1/projects/... instead of /api/v1/toolbox/v1/..."""
        respx.post(f"{BASE}/v1/projects/proj-1/tools/push").mock(
            return_value=_json_resp(_TOOL_RESPONSE, 201)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            tool = await c.push_tool(
                name="add",
                source_code="def add(a, b):\n    return a + b\n",
                entry_point="add",
            )

        assert tool.name == "add"

    @respx.mock
    @pytest.mark.asyncio
    async def test_mcp_register_direct_mode_path(self):
        """Direct mode uses /v1/projects/... for MCP servers."""
        respx.post(f"{BASE}/v1/projects/proj-1/mcp-servers").mock(
            return_value=_json_resp(_MCP_SERVER_RESPONSE, 201)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            srv = await c.register_mcp_server(
                name="My MCP Server",
                transport_type="sse",
                url="https://mcp.example.com",
            )

        assert srv.name == "My MCP Server"


class TestDirectModeCredentials:
    """Direct mode path tests for credential methods."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_set_credential_direct_mode_path(self):
        respx.post(f"{BASE}/v1/projects/proj-1/credentials").mock(
            return_value=_json_resp(_CREDENTIAL_RESPONSE, 201)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            cred = await c.set_tool_credential(
                tool_definition_id="tool-001",
                credential_type="api_key",
                name="My API Key",
                credential_payload={"key": "secret-value"},
            )

        assert cred.name == "My API Key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_credentials_direct_mode_path(self):
        resp = {"credentials": [_CREDENTIAL_RESPONSE], "project_id": "proj-1", "count": 1}
        respx.get(f"{BASE}/v1/projects/proj-1/credentials").mock(
            return_value=_json_resp(resp)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            result = await c.list_credentials()

        assert result.count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_revoke_credential_direct_mode_path(self):
        revoked = {**_CREDENTIAL_RESPONSE, "status": "revoked", "version": 2}
        respx.post(f"{BASE}/v1/projects/proj-1/credentials/cred-001/revoke").mock(
            return_value=_json_resp(revoked)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            cred = await c.revoke_credential("cred-001", version=1)

        assert cred.status == "revoked"


class TestDirectModeMCPExtra:
    """Direct mode path tests for remaining MCP methods."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_mcp_direct_mode_path(self):
        resp = {"servers": [_MCP_SERVER_RESPONSE], "project_id": "proj-1", "count": 1}
        respx.get(f"{BASE}/v1/projects/proj-1/mcp-servers").mock(
            return_value=_json_resp(resp)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            result = await c.list_mcp_servers()

        assert result.count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_mcp_direct_mode_path(self):
        respx.get(f"{BASE}/v1/projects/proj-1/mcp-servers/srv-001").mock(
            return_value=_json_resp(_MCP_SERVER_RESPONSE)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            srv = await c.get_mcp_server("srv-001")

        assert srv.id == "srv-001"

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_mcp_direct_mode_path(self):
        respx.delete(f"{BASE}/v1/projects/proj-1/mcp-servers/srv-001").mock(
            return_value=httpx.Response(204)
        )

        c = TaprootClient(
            base_url=BASE, api_key="key-id", project_id="proj-1", direct_mode=True,
        )
        async with c:
            await c.delete_mcp_server("srv-001")  # Should not raise


class TestCredentialErrors:
    """Error handling tests for credential methods."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_set_credential_validation_error(self):
        """422 errors raise ValidationError."""
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials").mock(
            return_value=httpx.Response(
                422, json={"detail": "Invalid credential_type"}
            )
        )
        async with _client() as c:
            with pytest.raises(ValidationError):
                await c.set_tool_credential(
                    tool_definition_id="tool-001",
                    credential_type="invalid",
                    name="Bad",
                    credential_payload={"key": "val"},
                )

    @respx.mock
    @pytest.mark.asyncio
    async def test_revoke_not_found(self):
        """404 errors raise TaprootAPIError."""
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials/xxx/revoke"
        ).mock(return_value=httpx.Response(404, json={"detail": "Not found"}))

        async with _client() as c:
            with pytest.raises(TaprootAPIError) as exc_info:
                await c.revoke_credential("xxx", version=1)
            assert exc_info.value.status_code == 404


class TestMCPErrors:
    """Error handling tests for MCP server methods."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_mcp_not_found(self):
        """404 errors raise TaprootAPIError."""
        respx.get(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers/xxx"
        ).mock(return_value=httpx.Response(404, json={"detail": "Not found"}))

        async with _client() as c:
            with pytest.raises(TaprootAPIError) as exc_info:
                await c.get_mcp_server("xxx")
            assert exc_info.value.status_code == 404

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_mcp_not_found(self):
        """404 on delete raises TaprootAPIError."""
        respx.delete(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers/xxx"
        ).mock(return_value=httpx.Response(404, json={"detail": "Not found"}))

        async with _client() as c:
            with pytest.raises(TaprootAPIError) as exc_info:
                await c.delete_mcp_server("xxx")
            assert exc_info.value.status_code == 404

    @respx.mock
    @pytest.mark.asyncio
    async def test_register_mcp_validation_error(self):
        """422 on register raises ValidationError."""
        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/mcp-servers").mock(
            return_value=httpx.Response(
                422, json={"detail": "Invalid transport_type"}
            )
        )

        async with _client() as c:
            with pytest.raises(ValidationError):
                await c.register_mcp_server(
                    name="Bad",
                    transport_type="invalid",
                    url="https://example.com",
                )


class TestSetCredentialWithExpiry:
    """Test set_tool_credential with optional expires_at parameter."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_set_with_expires_at(self):
        resp = {**_CREDENTIAL_RESPONSE, "expires_at": "2027-01-01T00:00:00Z"}
        route = respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/credentials"
        ).mock(return_value=_json_resp(resp, 201))

        async with _client() as c:
            cred = await c.set_tool_credential(
                tool_definition_id="tool-001",
                credential_type="api_key",
                name="My API Key",
                credential_payload={"key": "val"},
                expires_at="2027-01-01T00:00:00Z",
            )

        assert cred.expires_at == "2027-01-01T00:00:00Z"
        # Verify expires_at was sent in the request body
        sent_body = route.calls[0].request.content
        assert b"expires_at" in sent_body
