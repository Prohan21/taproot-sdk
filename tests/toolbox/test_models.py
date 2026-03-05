"""Tests for ToolBox-S SDK models."""

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


class TestToolInfo:
    def test_from_api_response_hosted(self):
        data = {
            "id": "abc-123",
            "project_id": "test-project",
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
            "tags": ["math", "utils"],
            "entry_point": "add",
            "requirements": ["numpy"],
            "requirements_hash": "abc123",
            "runtime": "python3.11",
            "timeout_ms": 30000,
            "memory_mb": 256,
            "content_hash": "def456",
        }

        tool = ToolInfo.from_api_response(data)

        assert tool.id == "abc-123"
        assert tool.name == "add"
        assert tool.is_hosted
        assert tool.is_active
        assert tool.is_invocable
        assert not tool.is_building
        assert tool.tags == ("math", "utils")
        assert tool.requirements == ("numpy",)
        assert tool.entry_point == "add"
        assert tool.input_schema["required"] == ["a", "b"]

    def test_from_api_response_external(self):
        data = {
            "id": "ext-456",
            "project_id": "test-project",
            "name": "weather",
            "description": "Get weather",
            "tool_type": "external",
            "input_schema": {"type": "object"},
            "version": 1,
            "status": "active",
            "scope": "project",
            "tags": [],
            "endpoint_url": "https://api.weather.com/v1",
            "http_method": "POST",
            "auth_type": "api_key",
        }

        tool = ToolInfo.from_api_response(data)

        assert not tool.is_hosted
        assert tool.endpoint_url == "https://api.weather.com/v1"
        assert tool.auth_type == "api_key"

    def test_from_api_response_building(self):
        data = {
            "id": "bld-789",
            "project_id": "test-project",
            "name": "slow",
            "description": "Needs deps",
            "tool_type": "hosted",
            "input_schema": {"type": "object"},
            "version": 1,
            "status": "building",
            "scope": "project",
            "tags": [],
        }

        tool = ToolInfo.from_api_response(data)

        assert tool.is_building
        assert not tool.is_invocable

    def test_frozen(self):
        data = {
            "id": "frz-000",
            "project_id": "p",
            "name": "t",
            "description": "",
            "tool_type": "hosted",
            "input_schema": {},
            "version": 1,
            "status": "active",
            "scope": "project",
            "tags": [],
        }
        tool = ToolInfo.from_api_response(data)

        import pytest
        with pytest.raises(AttributeError):
            tool.name = "changed"  # type: ignore[misc]

    def test_defaults_for_missing_fields(self):
        """Minimal response should use sensible defaults."""
        data = {
            "id": "min-111",
            "project_id": "p",
            "name": "minimal",
            "description": "Minimal tool",
            "tool_type": "hosted",
        }
        tool = ToolInfo.from_api_response(data)

        assert tool.input_schema == {}
        assert tool.version == 1
        assert tool.status == "active"
        assert tool.scope == "project"
        assert tool.tags == ()
        assert tool.requirements == ()
        assert tool.auth_type == "none"


class TestToolList:
    def test_from_api_response(self):
        data = {
            "tools": [
                {
                    "id": "t1",
                    "project_id": "p",
                    "name": "tool_a",
                    "description": "A",
                    "tool_type": "hosted",
                    "input_schema": {},
                    "version": 1,
                    "status": "active",
                    "scope": "project",
                    "tags": [],
                },
                {
                    "id": "t2",
                    "project_id": "p",
                    "name": "tool_b",
                    "description": "B",
                    "tool_type": "external",
                    "input_schema": {},
                    "version": 1,
                    "status": "active",
                    "scope": "global",
                    "tags": ["io"],
                },
            ],
            "project_id": "p",
            "count": 2,
        }

        result = ToolList.from_api_response(data)

        assert result.count == 2
        assert len(result.tools) == 2
        assert result.tools[0].name == "tool_a"
        assert result.tools[1].scope == "global"

    def test_empty_list(self):
        data = {"tools": [], "project_id": "p", "count": 0}
        result = ToolList.from_api_response(data)
        assert result.count == 0
        assert result.tools == ()


class TestInvocationResult:
    def test_success(self):
        data = {
            "invocation_id": "inv-001",
            "tool_name": "add",
            "success": True,
            "result": 42,
            "error": None,
            "duration_ms": 12.5,
        }

        result = InvocationResult.from_api_response(data)

        assert result.success
        assert result.result == 42
        assert result.error is None
        assert result.duration_ms == 12.5

    def test_failure(self):
        data = {
            "invocation_id": "inv-002",
            "tool_name": "fail",
            "success": False,
            "result": None,
            "error": "ValueError: bad input",
            "duration_ms": 3.1,
        }

        result = InvocationResult.from_api_response(data)

        assert not result.success
        assert result.result is None
        assert "ValueError" in (result.error or "")

    def test_frozen(self):
        data = {
            "invocation_id": "inv-003",
            "tool_name": "t",
            "success": True,
        }
        result = InvocationResult.from_api_response(data)

        import pytest
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


class TestImportResult:
    def test_from_api_response(self):
        data = {
            "tools_created": [
                {
                    "id": "t1",
                    "project_id": "p",
                    "name": "stripe_listCustomers",
                    "description": "List customers",
                    "tool_type": "external",
                    "input_schema": {},
                    "version": 1,
                    "status": "active",
                    "scope": "project",
                    "tags": ["payments"],
                },
            ],
            "tools_skipped": 2,
            "total_parsed": 3,
            "namespace": "stripe",
        }

        result = ImportResult.from_api_response(data)

        assert len(result.tools_created) == 1
        assert result.tools_created[0].name == "stripe_listCustomers"
        assert result.tools_skipped == 2
        assert result.total_parsed == 3
        assert result.namespace == "stripe"

    def test_empty_import(self):
        data = {
            "tools_created": [],
            "tools_skipped": 0,
            "total_parsed": 0,
            "namespace": "empty",
        }
        result = ImportResult.from_api_response(data)
        assert result.tools_created == ()
        assert result.tools_skipped == 0
        assert result.total_parsed == 0

    def test_frozen(self):
        data = {
            "tools_created": [],
            "tools_skipped": 0,
            "total_parsed": 0,
            "namespace": "ns",
        }
        result = ImportResult.from_api_response(data)

        import pytest
        with pytest.raises(AttributeError):
            result.namespace = "changed"  # type: ignore[misc]

    def test_tools_created_is_tuple(self):
        data = {
            "tools_created": [
                {
                    "id": "t1",
                    "project_id": "p",
                    "name": "a",
                    "description": "A",
                    "tool_type": "external",
                },
            ],
            "tools_skipped": 0,
            "total_parsed": 1,
            "namespace": "ns",
        }
        result = ImportResult.from_api_response(data)
        assert isinstance(result.tools_created, tuple)

    def test_defaults_for_missing_fields(self):
        """Minimal response should use sensible defaults."""
        data: dict = {}
        result = ImportResult.from_api_response(data)
        assert result.tools_created == ()
        assert result.tools_skipped == 0
        assert result.total_parsed == 0
        assert result.namespace == ""


class TestCredentialInfo:
    def test_from_api_response(self):
        data = {
            "id": "cred-001",
            "project_id": "proj-1",
            "tool_definition_id": "tool-001",
            "credential_type": "api_key",
            "name": "My API Key",
            "status": "active",
            "expires_at": "2027-01-01T00:00:00Z",
            "created_at": "2026-03-01T00:00:00Z",
            "updated_at": "2026-03-01T00:00:00Z",
            "created_by": "user-1",
            "version": 2,
        }

        cred = CredentialInfo.from_api_response(data)

        assert cred.id == "cred-001"
        assert cred.project_id == "proj-1"
        assert cred.tool_definition_id == "tool-001"
        assert cred.credential_type == "api_key"
        assert cred.name == "My API Key"
        assert cred.status == "active"
        assert cred.version == 2
        assert cred.expires_at == "2027-01-01T00:00:00Z"
        assert cred.created_by == "user-1"

    def test_defaults(self):
        data = {
            "id": "cred-002",
            "project_id": "proj-1",
            "tool_definition_id": "tool-001",
            "credential_type": "oauth2",
            "name": "OAuth Token",
        }

        cred = CredentialInfo.from_api_response(data)

        assert cred.status == "active"
        assert cred.version == 1
        assert cred.expires_at is None
        assert cred.created_at is None
        assert cred.created_by is None

    def test_frozen(self):
        data = {
            "id": "cred-003",
            "project_id": "p",
            "tool_definition_id": "t",
            "credential_type": "api_key",
            "name": "k",
        }
        cred = CredentialInfo.from_api_response(data)

        import pytest
        with pytest.raises(AttributeError):
            cred.name = "changed"  # type: ignore[misc]

    def test_is_active(self):
        active = CredentialInfo.from_api_response({
            "id": "c1", "project_id": "p", "tool_definition_id": "t",
            "credential_type": "api_key", "name": "k", "status": "active",
        })
        revoked = CredentialInfo.from_api_response({
            "id": "c2", "project_id": "p", "tool_definition_id": "t",
            "credential_type": "api_key", "name": "k", "status": "revoked",
        })

        assert active.is_active
        assert not revoked.is_active


class TestCredentialList:
    def test_from_api_response(self):
        data = {
            "credentials": [
                {
                    "id": "cred-001",
                    "project_id": "proj-1",
                    "tool_definition_id": "tool-001",
                    "credential_type": "api_key",
                    "name": "Key A",
                    "status": "active",
                    "version": 1,
                },
                {
                    "id": "cred-002",
                    "project_id": "proj-1",
                    "tool_definition_id": "tool-002",
                    "credential_type": "oauth2",
                    "name": "Token B",
                    "status": "revoked",
                    "version": 2,
                },
            ],
            "project_id": "proj-1",
            "count": 2,
        }

        result = CredentialList.from_api_response(data)

        assert result.count == 2
        assert len(result.credentials) == 2
        assert result.credentials[0].name == "Key A"
        assert result.credentials[1].status == "revoked"

    def test_empty(self):
        data = {"credentials": [], "project_id": "proj-1", "count": 0}
        result = CredentialList.from_api_response(data)
        assert result.count == 0
        assert result.credentials == ()


class TestMCPServerInfo:
    def test_from_api_response(self):
        data = {
            "id": "srv-001",
            "project_id": "proj-1",
            "name": "My MCP Server",
            "description": "A test server",
            "transport_type": "sse",
            "url": "https://mcp.example.com",
            "capabilities": ["tools", "resources"],
            "tools_discovered": ["search", "calc"],
            "health_status": "online",
            "last_health_check": "2026-03-01T12:00:00Z",
            "tags": ["prod", "search"],
            "scope": "global",
            "created_by": "user-1",
            "created_at": "2026-03-01T00:00:00Z",
            "updated_at": "2026-03-01T00:00:00Z",
        }

        srv = MCPServerInfo.from_api_response(data)

        assert srv.id == "srv-001"
        assert srv.name == "My MCP Server"
        assert srv.transport_type == "sse"
        assert srv.url == "https://mcp.example.com"
        assert srv.capabilities == ("tools", "resources")
        assert srv.tools_discovered == ("search", "calc")
        assert srv.health_status == "online"
        assert srv.tags == ("prod", "search")
        assert srv.scope == "global"
        assert srv.is_online

    def test_defaults(self):
        data = {
            "id": "srv-002",
            "project_id": "proj-1",
            "name": "Minimal",
            "transport_type": "stdio",
            "url": "/path/to/server",
        }

        srv = MCPServerInfo.from_api_response(data)

        assert srv.description == ""
        assert srv.capabilities == ()
        assert srv.tools_discovered == ()
        assert srv.health_status == "unknown"
        assert srv.last_health_check is None
        assert srv.tags == ()
        assert srv.scope == "project"
        assert srv.created_by is None
        assert not srv.is_online

    def test_frozen(self):
        data = {
            "id": "srv-003",
            "project_id": "p",
            "name": "n",
            "transport_type": "sse",
            "url": "http://localhost",
        }
        srv = MCPServerInfo.from_api_response(data)

        import pytest
        with pytest.raises(AttributeError):
            srv.name = "changed"  # type: ignore[misc]

    def test_tuples_for_list_fields(self):
        """Lists from API response should be stored as tuples."""
        data = {
            "id": "srv-004",
            "project_id": "p",
            "name": "n",
            "transport_type": "sse",
            "url": "http://localhost",
            "capabilities": ["a"],
            "tools_discovered": ["b"],
            "tags": ["c"],
        }
        srv = MCPServerInfo.from_api_response(data)

        assert isinstance(srv.capabilities, tuple)
        assert isinstance(srv.tools_discovered, tuple)
        assert isinstance(srv.tags, tuple)


class TestMCPServerList:
    def test_from_api_response(self):
        data = {
            "servers": [
                {
                    "id": "srv-001",
                    "project_id": "proj-1",
                    "name": "Server A",
                    "description": "First",
                    "transport_type": "sse",
                    "url": "https://a.example.com",
                    "health_status": "online",
                },
                {
                    "id": "srv-002",
                    "project_id": "proj-1",
                    "name": "Server B",
                    "description": "Second",
                    "transport_type": "stdio",
                    "url": "/path/b",
                    "health_status": "offline",
                },
            ],
            "project_id": "proj-1",
            "count": 2,
        }

        result = MCPServerList.from_api_response(data)

        assert result.count == 2
        assert len(result.servers) == 2
        assert result.servers[0].name == "Server A"
        assert result.servers[1].health_status == "offline"

    def test_empty(self):
        data = {"servers": [], "project_id": "proj-1", "count": 0}
        result = MCPServerList.from_api_response(data)
        assert result.count == 0
        assert result.servers == ()


class TestExports:
    """Verify all toolbox models are exported from package and top-level."""

    def test_toolbox_package_exports(self):
        import taproot_sdk.toolbox as toolbox

        for name in [
            "CredentialInfo", "CredentialList", "ImportResult",
            "MCPServerInfo", "MCPServerList",
            "ToolInfo", "ToolList", "InvocationResult",
        ]:
            assert hasattr(toolbox, name), f"{name} missing from taproot_sdk.toolbox"
            assert name in toolbox.__all__, f"{name} missing from toolbox.__all__"

    def test_top_level_exports(self):
        import taproot_sdk

        for name in [
            "CredentialInfo", "CredentialList", "ImportResult",
            "MCPServerInfo", "MCPServerList",
            "ToolInfo", "ToolList", "InvocationResult",
        ]:
            assert hasattr(taproot_sdk, name), f"{name} missing from taproot_sdk"
            assert name in taproot_sdk.__all__, f"{name} missing from taproot_sdk.__all__"
