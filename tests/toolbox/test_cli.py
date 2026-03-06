"""Tests for taproot_sdk.toolbox.cli."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from taproot_sdk.toolbox.cli import main
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENV = {
    "TAPROOT_BASE_URL": "https://api.example.com",
    "TAPROOT_API_KEY": "test-key-123",
    "TAPROOT_PROJECT_ID": "proj-1",
}


def _fake_tool(**overrides: Any) -> ToolInfo:
    defaults: dict[str, Any] = {
        "id": "tool-abc-123",
        "project_id": "proj-1",
        "name": "my_tool",
        "description": "A test tool",
        "tool_type": "hosted",
        "input_schema": {},
        "version": 1,
        "status": "active",
        "scope": "project",
        "tags": ("utils",),
        "entry_point": "run",
        "created_at": "2026-03-01T00:00:00Z",
    }
    defaults.update(overrides)
    return ToolInfo(**defaults)


def _fake_invocation(**overrides: Any) -> InvocationResult:
    defaults: dict[str, Any] = {
        "invocation_id": "inv-001",
        "tool_name": "my_tool",
        "success": True,
        "result": {"answer": 42},
        "duration_ms": 120.5,
    }
    defaults.update(overrides)
    return InvocationResult(**defaults)


def _fake_tool_list(tools: list[ToolInfo] | None = None) -> ToolList:
    items = tuple(tools) if tools else (_fake_tool(),)
    return ToolList(tools=items, project_id="proj-1", count=len(items))


# ---------------------------------------------------------------------------
# push command
# ---------------------------------------------------------------------------

class TestPushCommand:
    """Tests for ``taproot-tools push``."""

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_push_reads_file_and_calls_push_tool(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.push_tool = AsyncMock(return_value=_fake_tool())
        mock_make_client.return_value = client

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as f:
            f.write("def run(x): return x + 1\n")
            tmp_path = f.name

        try:
            main([
                "push", tmp_path,
                "--name", "my_tool",
                "--entry-point", "run",
                "--description", "adds one",
                "--requirements", "numpy,pandas>=2",
                "--tags", "math,utils",
                "--timeout-ms", "5000",
                "--memory-mb", "512",
                "--scope", "global",
            ])
        finally:
            os.unlink(tmp_path)

        client.push_tool.assert_awaited_once()
        call_kwargs = client.push_tool.call_args
        assert call_kwargs.kwargs["name"] == "my_tool"
        assert call_kwargs.kwargs["entry_point"] == "run"
        assert call_kwargs.kwargs["description"] == "adds one"
        assert call_kwargs.kwargs["requirements"] == ["numpy", "pandas>=2"]
        assert call_kwargs.kwargs["tags"] == ["math", "utils"]
        assert call_kwargs.kwargs["timeout_ms"] == 5000
        assert call_kwargs.kwargs["memory_mb"] == 512
        assert call_kwargs.kwargs["scope"] == "global"
        assert "def run(x)" in call_kwargs.kwargs["source_code"]

        out = capsys.readouterr().out
        assert "tool-abc-123" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_push_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.push_tool = AsyncMock(return_value=_fake_tool())
        mock_make_client.return_value = client

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as f:
            f.write("def run(x): return x\n")
            tmp_path = f.name

        try:
            main([
                "--json", "push", tmp_path,
                "--name", "my_tool", "--entry-point", "run",
            ])
        finally:
            os.unlink(tmp_path)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "tool-abc-123"
        assert data["name"] == "my_tool"

    @patch.dict(os.environ, _ENV, clear=False)
    def test_push_missing_file(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["push", "/nonexistent/tool.py", "--name", "x", "--entry-point", "y"])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# register command
# ---------------------------------------------------------------------------

class TestRegisterCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_register_calls_register_tool(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        tool = _fake_tool(tool_type="external", endpoint_url="https://ext.example.com/run")
        client = MagicMock()
        client.register_tool = AsyncMock(return_value=tool)
        mock_make_client.return_value = client

        main([
            "register", "ext_tool",
            "--endpoint-url", "https://ext.example.com/run",
            "--auth-type", "bearer",
            "--tags", "api",
        ])

        client.register_tool.assert_awaited_once()
        kw = client.register_tool.call_args.kwargs
        assert kw["name"] == "ext_tool"
        assert kw["endpoint_url"] == "https://ext.example.com/run"
        assert kw["auth_type"] == "bearer"


# ---------------------------------------------------------------------------
# invoke command
# ---------------------------------------------------------------------------

class TestInvokeCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_invoke_with_inline_json(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.invoke_tool = AsyncMock(return_value=_fake_invocation())
        mock_make_client.return_value = client

        main(["invoke", "my_tool", "--input", '{"x": 10}'])

        client.invoke_tool.assert_awaited_once_with("my_tool", {"x": 10})
        out = capsys.readouterr().out
        assert "SUCCESS" in out
        assert "120.5" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_invoke_with_input_file(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.invoke_tool = AsyncMock(return_value=_fake_invocation())
        mock_make_client.return_value = client

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump({"x": 99}, f)
            tmp_path = f.name

        try:
            main(["invoke", "my_tool", "--input-file", tmp_path])
        finally:
            os.unlink(tmp_path)

        client.invoke_tool.assert_awaited_once_with("my_tool", {"x": 99})

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_invoke_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.invoke_tool = AsyncMock(return_value=_fake_invocation())
        mock_make_client.return_value = client

        main(["--json", "invoke", "my_tool", "--input", '{"x": 1}'])

        data = json.loads(capsys.readouterr().out)
        assert data["success"] is True
        assert data["invocation_id"] == "inv-001"

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_invoke_failed_result(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        inv = _fake_invocation(success=False, error="timeout", result=None)
        client = MagicMock()
        client.invoke_tool = AsyncMock(return_value=inv)
        mock_make_client.return_value = client

        main(["invoke", "my_tool"])

        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "timeout" in out

    @patch.dict(os.environ, _ENV, clear=False)
    def test_invoke_invalid_json(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["invoke", "my_tool", "--input", "not json"])
        assert exc_info.value.code == 1

    @patch.dict(os.environ, _ENV, clear=False)
    def test_invoke_both_input_flags(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["invoke", "my_tool", "--input", "{}", "--input-file", "f.json"])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

class TestListCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_table_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        tools = [
            _fake_tool(id="aaa", name="tool_a"),
            _fake_tool(id="bbb", name="tool_b", status="building"),
        ]
        client = MagicMock()
        client.list_tools = AsyncMock(return_value=_fake_tool_list(tools))
        mock_make_client.return_value = client

        main(["list", "--tags", "utils", "--type", "hosted"])

        out = capsys.readouterr().out
        assert "tool_a" in out
        assert "tool_b" in out
        assert "building" in out
        assert "2 tool(s)" in out

        kw = client.list_tools.call_args.kwargs
        assert kw["tags"] == ["utils"]
        assert kw["tool_type"] == "hosted"

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_empty(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_tools = AsyncMock(
            return_value=ToolList(tools=(), project_id="proj-1", count=0),
        )
        mock_make_client.return_value = client

        main(["list"])

        assert "No tools found" in capsys.readouterr().out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_tools = AsyncMock(return_value=_fake_tool_list())
        mock_make_client.return_value = client

        main(["--json", "list"])

        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1
        assert data["tools"][0]["name"] == "my_tool"


# ---------------------------------------------------------------------------
# get command
# ---------------------------------------------------------------------------

class TestGetCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_get_prints_tool(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.get_tool = AsyncMock(return_value=_fake_tool())
        mock_make_client.return_value = client

        main(["get", "tool-abc-123"])

        client.get_tool.assert_awaited_once_with("tool-abc-123")
        out = capsys.readouterr().out
        assert "tool-abc-123" in out
        assert "my_tool" in out


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------

class TestDeleteCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_delete_prints_confirmation(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.delete_tool = AsyncMock(return_value=None)
        mock_make_client.return_value = client

        main(["delete", "tool-abc-123"])

        client.delete_tool.assert_awaited_once_with("tool-abc-123")
        assert "Deleted" in capsys.readouterr().out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_delete_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.delete_tool = AsyncMock(return_value=None)
        mock_make_client.return_value = client

        main(["--json", "delete", "tool-abc-123"])

        data = json.loads(capsys.readouterr().out)
        assert data["deleted"] is True


# ---------------------------------------------------------------------------
# import-openapi command
# ---------------------------------------------------------------------------

def _fake_import_result(**overrides: Any) -> ImportResult:
    tools = overrides.pop("tools_created", (_fake_tool(
        name="petstore_listPets",
        tool_type="external",
        endpoint_url="https://petstore.example.com/pets",
    ),))
    defaults: dict[str, Any] = {
        "tools_created": tools,
        "tools_skipped": 0,
        "total_parsed": 1,
        "namespace": "petstore",
    }
    defaults.update(overrides)
    return ImportResult(**defaults)


class TestImportOpenAPICommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_with_spec_url(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_openapi = AsyncMock(return_value=_fake_import_result())
        mock_make_client.return_value = client

        main([
            "import-openapi",
            "--namespace", "petstore",
            "--spec-url", "https://petstore.swagger.io/v3/openapi.json",
        ])

        client.import_openapi.assert_awaited_once()
        kw = client.import_openapi.call_args.kwargs
        assert kw["namespace"] == "petstore"
        assert kw["spec_url"] == "https://petstore.swagger.io/v3/openapi.json"

        out = capsys.readouterr().out
        assert "Imported 1 tools" in out
        assert "petstore" in out
        assert "petstore_listPets" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_with_spec_file(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_openapi = AsyncMock(return_value=_fake_import_result())
        mock_make_client.return_value = client

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump({"openapi": "3.0.0", "paths": {}}, f)
            tmp_path = f.name

        try:
            main([
                "import-openapi",
                "--namespace", "petstore",
                "--spec-file", tmp_path,
            ])
        finally:
            os.unlink(tmp_path)

        kw = client.import_openapi.call_args.kwargs
        assert kw["spec_body"] == {"openapi": "3.0.0", "paths": {}}
        assert kw["spec_url"] is None

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_openapi = AsyncMock(return_value=_fake_import_result(
            tools_skipped=2,
            total_parsed=3,
        ))
        mock_make_client.return_value = client

        main([
            "--json", "import-openapi",
            "--namespace", "petstore",
            "--spec-url", "https://example.com/spec.json",
        ])

        data = json.loads(capsys.readouterr().out)
        assert data["tools_created"] == 1
        assert data["tools_skipped"] == 2
        assert data["total_parsed"] == 3
        assert data["namespace"] == "petstore"
        assert data["tool_names"] == ["petstore_listPets"]

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_with_all_options(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_openapi = AsyncMock(return_value=_fake_import_result())
        mock_make_client.return_value = client

        main([
            "import-openapi",
            "--namespace", "stripe",
            "--spec-url", "https://stripe.com/spec.json",
            "--base-url", "https://api.stripe.com/v1",
            "--tags", "payments,billing",
            "--auth-type", "bearer",
            "--scope", "global",
        ])

        kw = client.import_openapi.call_args.kwargs
        assert kw["namespace"] == "stripe"
        assert kw["base_url"] == "https://api.stripe.com/v1"
        assert kw["tags"] == ["payments", "billing"]
        assert kw["auth_type"] == "bearer"
        assert kw["scope"] == "global"

    @patch.dict(os.environ, _ENV, clear=False)
    def test_import_missing_spec(self) -> None:
        """Must provide either --spec-url or --spec-file."""
        with pytest.raises(SystemExit) as exc_info:
            main(["import-openapi", "--namespace", "test"])
        assert exc_info.value.code == 1

    @patch.dict(os.environ, _ENV, clear=False)
    def test_import_spec_file_not_found(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "import-openapi",
                "--namespace", "test",
                "--spec-file", "/nonexistent/spec.json",
            ])
        assert exc_info.value.code == 1

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_empty_result(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        result = ImportResult(
            tools_created=(),
            tools_skipped=0,
            total_parsed=0,
            namespace="empty",
        )
        client = MagicMock()
        client.import_openapi = AsyncMock(return_value=result)
        mock_make_client.return_value = client

        main([
            "import-openapi",
            "--namespace", "empty",
            "--spec-url", "https://example.com/spec.json",
        ])

        out = capsys.readouterr().out
        assert "Imported 0 tools" in out
        assert "Created tools:" not in out


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_env_var_base_url(self) -> None:
        env = {**_ENV}
        del env["TAPROOT_BASE_URL"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                main(["list"])
            assert exc_info.value.code == 1

    def test_missing_env_var_api_key(self) -> None:
        env = {**_ENV}
        del env["TAPROOT_API_KEY"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                main(["list"])
            assert exc_info.value.code == 1

    def test_missing_env_var_project_id(self) -> None:
        env = {**_ENV}
        del env["TAPROOT_PROJECT_ID"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                main(["list"])
            assert exc_info.value.code == 1

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_connection_error(
        self, mock_make_client: MagicMock,
    ) -> None:
        client = MagicMock()
        client.list_tools = AsyncMock(side_effect=ConnectionError("refused"))
        mock_make_client.return_value = client

        with pytest.raises(SystemExit) as exc_info:
            main(["list"])
        assert exc_info.value.code == 1

    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _fake_credential(**overrides: Any) -> CredentialInfo:
    defaults: dict[str, Any] = {
        "id": "cred-001",
        "project_id": "proj-1",
        "tool_definition_id": "tool-001",
        "credential_type": "api_key",
        "name": "My Key",
        "status": "active",
        "version": 1,
        "created_at": "2026-03-01T00:00:00Z",
    }
    defaults.update(overrides)
    return CredentialInfo(**defaults)


def _fake_credential_list(
    creds: list[CredentialInfo] | None = None,
) -> CredentialList:
    items = tuple(creds) if creds else (_fake_credential(),)
    return CredentialList(credentials=items, project_id="proj-1", count=len(items))


def _fake_mcp_server(**overrides: Any) -> MCPServerInfo:
    defaults: dict[str, Any] = {
        "id": "srv-001",
        "project_id": "proj-1",
        "name": "My Server",
        "description": "Test",
        "transport_type": "sse",
        "url": "https://mcp.example.com",
        "health_status": "online",
        "tags": ("prod",),
        "created_at": "2026-03-01T00:00:00Z",
    }
    defaults.update(overrides)
    return MCPServerInfo(**defaults)


def _fake_mcp_server_list(
    servers: list[MCPServerInfo] | None = None,
) -> MCPServerList:
    items = tuple(servers) if servers else (_fake_mcp_server(),)
    return MCPServerList(servers=items, project_id="proj-1", count=len(items))


# ---------------------------------------------------------------------------
# set-credential command
# ---------------------------------------------------------------------------

class TestSetCredentialCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_set_credential(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.set_tool_credential = AsyncMock(return_value=_fake_credential())
        mock_make_client.return_value = client

        main([
            "set-credential",
            "--tool-id", "tool-001",
            "--type", "api_key",
            "--name", "My Key",
            "--payload", '{"key": "secret-value"}',
        ])

        client.set_tool_credential.assert_awaited_once()
        kw = client.set_tool_credential.call_args.kwargs
        assert kw["tool_definition_id"] == "tool-001"
        assert kw["credential_type"] == "api_key"
        assert kw["name"] == "My Key"
        assert kw["credential_payload"] == {"key": "secret-value"}

        out = capsys.readouterr().out
        assert "cred-001" in out
        assert "My Key" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_set_credential_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.set_tool_credential = AsyncMock(return_value=_fake_credential())
        mock_make_client.return_value = client

        main([
            "--json", "set-credential",
            "--tool-id", "tool-001",
            "--type", "api_key",
            "--name", "My Key",
            "--payload", '{"key": "val"}',
        ])

        data = json.loads(capsys.readouterr().out)
        assert data["id"] == "cred-001"
        assert data["credential_type"] == "api_key"

    @patch.dict(os.environ, _ENV, clear=False)
    def test_set_credential_invalid_payload(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "set-credential",
                "--tool-id", "t",
                "--type", "api_key",
                "--name", "k",
                "--payload", "not-json",
            ])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# list-credentials command
# ---------------------------------------------------------------------------

class TestListCredentialsCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_credentials_table(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_credentials = AsyncMock(return_value=_fake_credential_list())
        mock_make_client.return_value = client

        main(["list-credentials"])

        out = capsys.readouterr().out
        assert "My Key" in out
        assert "1 credential(s)" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_credentials_with_tool_filter(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_credentials = AsyncMock(return_value=_fake_credential_list())
        mock_make_client.return_value = client

        main(["list-credentials", "--tool-id", "tool-001"])

        client.list_credentials.assert_awaited_once_with(tool_definition_id="tool-001")

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_credentials_empty(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        empty = CredentialList(credentials=(), project_id="proj-1", count=0)
        client = MagicMock()
        client.list_credentials = AsyncMock(return_value=empty)
        mock_make_client.return_value = client

        main(["list-credentials"])

        assert "No credentials found" in capsys.readouterr().out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_credentials_json(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_credentials = AsyncMock(return_value=_fake_credential_list())
        mock_make_client.return_value = client

        main(["--json", "list-credentials"])

        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1


# ---------------------------------------------------------------------------
# revoke-credential command
# ---------------------------------------------------------------------------

class TestRevokeCredentialCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_revoke_credential(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        revoked = _fake_credential(status="revoked", version=2)
        client = MagicMock()
        client.revoke_credential = AsyncMock(return_value=revoked)
        mock_make_client.return_value = client

        main(["revoke-credential", "--credential-id", "cred-001", "--version", "1"])

        client.revoke_credential.assert_awaited_once_with("cred-001", 1)
        out = capsys.readouterr().out
        assert "revoked" in out


# ---------------------------------------------------------------------------
# mcp-register command
# ---------------------------------------------------------------------------

class TestMCPRegisterCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_register(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.register_mcp_server = AsyncMock(return_value=_fake_mcp_server())
        mock_make_client.return_value = client

        main([
            "mcp-register",
            "--name", "My Server",
            "--transport", "sse",
            "--url", "https://mcp.example.com",
            "--description", "Test",
            "--tags", "prod,search",
        ])

        client.register_mcp_server.assert_awaited_once()
        kw = client.register_mcp_server.call_args.kwargs
        assert kw["name"] == "My Server"
        assert kw["transport_type"] == "sse"
        assert kw["url"] == "https://mcp.example.com"
        assert kw["tags"] == ["prod", "search"]

        out = capsys.readouterr().out
        assert "srv-001" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_register_json(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.register_mcp_server = AsyncMock(return_value=_fake_mcp_server())
        mock_make_client.return_value = client

        main([
            "--json", "mcp-register",
            "--name", "My Server",
            "--transport", "sse",
            "--url", "https://mcp.example.com",
        ])

        data = json.loads(capsys.readouterr().out)
        assert data["id"] == "srv-001"


# ---------------------------------------------------------------------------
# mcp-list command
# ---------------------------------------------------------------------------

class TestMCPListCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_table(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_mcp_servers = AsyncMock(return_value=_fake_mcp_server_list())
        mock_make_client.return_value = client

        main(["mcp-list"])

        out = capsys.readouterr().out
        assert "My Server" in out
        assert "1 server(s)" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_empty(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        empty = MCPServerList(servers=(), project_id="proj-1", count=0)
        client = MagicMock()
        client.list_mcp_servers = AsyncMock(return_value=empty)
        mock_make_client.return_value = client

        main(["mcp-list"])

        assert "No MCP servers found" in capsys.readouterr().out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_json(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_mcp_servers = AsyncMock(return_value=_fake_mcp_server_list())
        mock_make_client.return_value = client

        main(["--json", "mcp-list"])

        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_list_with_filters(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_mcp_servers = AsyncMock(return_value=_fake_mcp_server_list())
        mock_make_client.return_value = client

        main(["mcp-list", "--status", "online", "--tags", "prod"])

        kw = client.list_mcp_servers.call_args.kwargs
        assert kw["status"] == "online"
        assert kw["tags"] == ["prod"]


# ---------------------------------------------------------------------------
# mcp-get command
# ---------------------------------------------------------------------------

class TestMCPGetCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_get(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.get_mcp_server = AsyncMock(return_value=_fake_mcp_server())
        mock_make_client.return_value = client

        main(["mcp-get", "--id", "srv-001"])

        client.get_mcp_server.assert_awaited_once_with("srv-001")
        out = capsys.readouterr().out
        assert "srv-001" in out
        assert "My Server" in out


# ---------------------------------------------------------------------------
# mcp-delete command
# ---------------------------------------------------------------------------

class TestMCPDeleteCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_delete(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.delete_mcp_server = AsyncMock(return_value=None)
        mock_make_client.return_value = client

        main(["mcp-delete", "--id", "srv-001"])

        client.delete_mcp_server.assert_awaited_once_with("srv-001")
        assert "Deleted MCP server srv-001" in capsys.readouterr().out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_delete_json(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.delete_mcp_server = AsyncMock(return_value=None)
        mock_make_client.return_value = client

        main(["--json", "mcp-delete", "--id", "srv-001"])

        data = json.loads(capsys.readouterr().out)
        assert data["deleted"] is True
        assert data["server_id"] == "srv-001"


# ---------------------------------------------------------------------------
# MCP Registry Import/Export helpers
# ---------------------------------------------------------------------------

def _fake_mcp_registry_import_result(**overrides: Any) -> Any:
    from taproot_sdk.toolbox.models import MCPRegistryImportResult

    defaults: dict[str, Any] = {
        "servers_created": (_fake_mcp_server(),),
        "tools_created": (
            _fake_tool(name="search_server_web_search", tool_type="mcp"),
        ),
        "total_servers_parsed": 1,
        "total_tools_parsed": 1,
        "servers_skipped": 0,
        "tools_skipped": 0,
    }
    defaults.update(overrides)
    return MCPRegistryImportResult(**defaults)


# ---------------------------------------------------------------------------
# import-mcp-registry command
# ---------------------------------------------------------------------------

class TestImportMCPRegistryCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_with_registry_url(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_mcp_registry = AsyncMock(
            return_value=_fake_mcp_registry_import_result()
        )
        mock_make_client.return_value = client

        main([
            "import-mcp-registry",
            "--registry-url", "https://registry.example.com/mcp.json",
        ])

        client.import_mcp_registry.assert_awaited_once()
        kw = client.import_mcp_registry.call_args.kwargs
        assert kw["registry_url"] == "https://registry.example.com/mcp.json"

        out = capsys.readouterr().out
        assert "Imported 1 servers" in out
        assert "1 tools" in out

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_with_registry_file(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_mcp_registry = AsyncMock(
            return_value=_fake_mcp_registry_import_result()
        )
        mock_make_client.return_value = client

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump({"servers": []}, f)
            tmp_path = f.name

        try:
            main([
                "import-mcp-registry",
                "--registry-file", tmp_path,
            ])
        finally:
            os.unlink(tmp_path)

        kw = client.import_mcp_registry.call_args.kwargs
        assert kw["registry_body"] == {"servers": []}
        assert kw["registry_url"] is None

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_json_output(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_mcp_registry = AsyncMock(
            return_value=_fake_mcp_registry_import_result(
                servers_skipped=1,
                tools_skipped=2,
                total_servers_parsed=2,
                total_tools_parsed=3,
            )
        )
        mock_make_client.return_value = client

        main([
            "--json", "import-mcp-registry",
            "--registry-url", "https://example.com/mcp.json",
        ])

        data = json.loads(capsys.readouterr().out)
        assert data["servers_created"] == 1
        assert data["tools_created"] == 1
        assert data["servers_skipped"] == 1
        assert data["tools_skipped"] == 2
        assert data["total_servers_parsed"] == 2
        assert data["total_tools_parsed"] == 3

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_import_with_all_options(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.import_mcp_registry = AsyncMock(
            return_value=_fake_mcp_registry_import_result()
        )
        mock_make_client.return_value = client

        main([
            "import-mcp-registry",
            "--registry-url", "https://registry.example.com/mcp.json",
            "--namespace", "acme",
            "--tags", "imported,test",
            "--scope", "global",
        ])

        kw = client.import_mcp_registry.call_args.kwargs
        assert kw["namespace"] == "acme"
        assert kw["tags"] == ["imported", "test"]
        assert kw["scope"] == "global"

    @patch.dict(os.environ, _ENV, clear=False)
    def test_import_missing_source(self) -> None:
        """Must provide either --registry-url or --registry-file."""
        with pytest.raises(SystemExit) as exc_info:
            main(["import-mcp-registry"])
        assert exc_info.value.code == 1

    @patch.dict(os.environ, _ENV, clear=False)
    def test_import_registry_file_not_found(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "import-mcp-registry",
                "--registry-file", "/nonexistent/registry.json",
            ])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# export-mcp-registry command
# ---------------------------------------------------------------------------

class TestExportMCPRegistryCommand:
    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_export_to_stdout(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        exported = {"servers": [{"name": "test", "url": "https://test.com"}]}
        client = MagicMock()
        client.export_mcp_registry = AsyncMock(return_value=exported)
        mock_make_client.return_value = client

        main(["export-mcp-registry"])

        client.export_mcp_registry.assert_awaited_once()
        data = json.loads(capsys.readouterr().out)
        assert data["servers"][0]["name"] == "test"

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_export_to_file(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        exported = {"servers": [{"name": "test", "url": "https://test.com"}]}
        client = MagicMock()
        client.export_mcp_registry = AsyncMock(return_value=exported)
        mock_make_client.return_value = client

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            tmp_path = f.name

        try:
            main(["export-mcp-registry", "--output", tmp_path])

            with open(tmp_path, encoding="utf-8") as fh:
                data = json.loads(fh.read())
            assert data["servers"][0]["name"] == "test"
            assert "Exported to" in capsys.readouterr().out
        finally:
            os.unlink(tmp_path)

    @patch.dict(os.environ, _ENV, clear=False)
    @patch("taproot_sdk.toolbox.cli._make_client")
    def test_export_no_global(
        self, mock_make_client: MagicMock, capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.export_mcp_registry = AsyncMock(return_value={"servers": []})
        mock_make_client.return_value = client

        main(["export-mcp-registry", "--no-global"])

        kw = client.export_mcp_registry.call_args.kwargs
        assert kw["include_global"] is False
