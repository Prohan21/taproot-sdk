"""
Typed response models for ToolBox-S API.

Field names match the ToolBox-S Pydantic response models exactly
(src/toolbox_service/api/schemas.py in ToolBox-S).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ToolInfo:
    """Typed representation of a ToolBox-S tool definition."""

    id: str
    project_id: str
    name: str
    description: str
    tool_type: str
    input_schema: dict[str, Any]
    version: int
    status: str
    scope: str
    tags: tuple[str, ...]

    # Hosted fields
    entry_point: str | None = None
    requirements: tuple[str, ...] = ()
    requirements_hash: str | None = None
    runtime: str = "python3.11"
    timeout_ms: int = 30000
    memory_mb: int = 256
    content_hash: str | None = None
    build_error: str | None = None

    # External fields
    endpoint_url: str | None = None
    http_method: str | None = None
    auth_type: str = "none"

    # Output
    output_schema: dict[str, Any] | None = None

    # Audit
    created_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def is_hosted(self) -> bool:
        return self.tool_type == "hosted"

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_building(self) -> bool:
        return self.status == "building"

    @property
    def is_invocable(self) -> bool:
        return self.status == "active"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ToolInfo:
        """Parse a raw JSON dict into a typed ToolInfo."""
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            description=data["description"],
            tool_type=data["tool_type"],
            input_schema=data.get("input_schema", {}),
            version=data.get("version", 1),
            status=data.get("status", "active"),
            scope=data.get("scope", "project"),
            tags=tuple(data.get("tags", [])),
            entry_point=data.get("entry_point"),
            requirements=tuple(data.get("requirements", [])),
            requirements_hash=data.get("requirements_hash"),
            runtime=data.get("runtime", "python3.11"),
            timeout_ms=data.get("timeout_ms", 30000),
            memory_mb=data.get("memory_mb", 256),
            content_hash=data.get("content_hash"),
            build_error=data.get("build_error"),
            endpoint_url=data.get("endpoint_url"),
            http_method=data.get("http_method"),
            auth_type=data.get("auth_type", "none"),
            output_schema=data.get("output_schema"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class ToolList:
    """Paginated list of tools."""

    tools: tuple[ToolInfo, ...]
    project_id: str
    count: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ToolList:
        return cls(
            tools=tuple(ToolInfo.from_api_response(t) for t in data.get("tools", [])),
            project_id=data.get("project_id", ""),
            count=data.get("count", 0),
        )


@dataclass(frozen=True)
class InvocationResult:
    """Typed representation of a tool invocation response."""

    invocation_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> InvocationResult:
        return cls(
            invocation_id=data["invocation_id"],
            tool_name=data["tool_name"],
            success=data["success"],
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass(frozen=True)
class CredentialInfo:
    """Credential metadata (never contains plaintext values)."""

    id: str
    project_id: str
    tool_definition_id: str
    credential_type: str
    name: str
    status: str
    expires_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    created_by: str | None = None
    version: int = 1

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> CredentialInfo:
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            tool_definition_id=data["tool_definition_id"],
            credential_type=data["credential_type"],
            name=data["name"],
            status=data.get("status", "active"),
            expires_at=data.get("expires_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            created_by=data.get("created_by"),
            version=data.get("version", 1),
        )

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class CredentialList:
    """List of credentials for a project."""

    credentials: tuple[CredentialInfo, ...]
    project_id: str
    count: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> CredentialList:
        return cls(
            credentials=tuple(
                CredentialInfo.from_api_response(c)
                for c in data.get("credentials", [])
            ),
            project_id=data["project_id"],
            count=data.get("count", 0),
        )


@dataclass(frozen=True)
class MCPServerInfo:
    """MCP server registration info."""

    id: str
    project_id: str
    name: str
    description: str
    transport_type: str
    url: str
    capabilities: tuple[str, ...] = ()
    tools_discovered: tuple[str, ...] = ()
    health_status: str = "unknown"
    last_health_check: str | None = None
    tags: tuple[str, ...] = ()
    scope: str = "project"
    created_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> MCPServerInfo:
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            description=data.get("description", ""),
            transport_type=data["transport_type"],
            url=data["url"],
            capabilities=tuple(data.get("capabilities", [])),
            tools_discovered=tuple(data.get("tools_discovered", [])),
            health_status=data.get("health_status", "unknown"),
            last_health_check=data.get("last_health_check"),
            tags=tuple(data.get("tags", [])),
            scope=data.get("scope", "project"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    @property
    def is_online(self) -> bool:
        return self.health_status == "online"


@dataclass(frozen=True)
class ImportResult:
    """Result of an OpenAPI spec import."""

    tools_created: tuple[ToolInfo, ...]
    tools_skipped: int
    total_parsed: int
    namespace: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ImportResult:
        return cls(
            tools_created=tuple(
                ToolInfo.from_api_response(t) for t in data.get("tools_created", [])
            ),
            tools_skipped=data.get("tools_skipped", 0),
            total_parsed=data.get("total_parsed", 0),
            namespace=data.get("namespace", ""),
        )


@dataclass(frozen=True)
class ToolUsageStats:
    """Usage statistics for a single tool."""

    tool_id: str
    tool_name: str
    tool_type: str
    status: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ToolUsageStats:
        return cls(
            tool_id=data["tool_id"],
            tool_name=data["tool_name"],
            tool_type=data["tool_type"],
            status=data["status"],
        )


@dataclass(frozen=True)
class UsageReport:
    """Aggregate usage report for a project's tools."""

    project_id: str
    tools: tuple[ToolUsageStats, ...]
    count: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> UsageReport:
        return cls(
            project_id=data["project_id"],
            tools=tuple(
                ToolUsageStats.from_api_response(t) for t in data.get("tools", [])
            ),
            count=data.get("count", 0),
        )


@dataclass(frozen=True)
class MCPRegistryImportResult:
    """Result of an MCP registry import."""

    servers_created: tuple[MCPServerInfo, ...]
    tools_created: tuple[ToolInfo, ...]
    total_servers_parsed: int
    total_tools_parsed: int
    servers_skipped: int
    tools_skipped: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> MCPRegistryImportResult:
        return cls(
            servers_created=tuple(
                MCPServerInfo.from_api_response(s)
                for s in data.get("servers_created", [])
            ),
            tools_created=tuple(
                ToolInfo.from_api_response(t)
                for t in data.get("tools_created", [])
            ),
            total_servers_parsed=data.get("total_servers_parsed", 0),
            total_tools_parsed=data.get("total_tools_parsed", 0),
            servers_skipped=data.get("servers_skipped", 0),
            tools_skipped=data.get("tools_skipped", 0),
        )


@dataclass(frozen=True)
class OAuthFlowResponse:
    """Response from initiating an OAuth authorization code flow."""

    authorize_url: str
    state: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> OAuthFlowResponse:
        return cls(
            authorize_url=data["authorize_url"],
            state=data["state"],
        )


@dataclass(frozen=True)
class OAuthConnectionInfo:
    """OAuth connection metadata (never contains plaintext tokens)."""

    id: str
    project_id: str
    tool_definition_id: str
    user_id: str
    token_type: str
    scopes: tuple[str, ...]
    expires_at: str | None = None
    created_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> OAuthConnectionInfo:
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            tool_definition_id=data["tool_definition_id"],
            user_id=data["user_id"],
            token_type=data.get("token_type", "Bearer"),
            scopes=tuple(data.get("scopes", [])),
            expires_at=data.get("expires_at"),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True)
class MCPServerList:
    """List of MCP servers for a project."""

    servers: tuple[MCPServerInfo, ...]
    project_id: str
    count: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> MCPServerList:
        return cls(
            servers=tuple(
                MCPServerInfo.from_api_response(s)
                for s in data.get("servers", [])
            ),
            project_id=data["project_id"],
            count=data.get("count", 0),
        )
