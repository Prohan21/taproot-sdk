"""Tests for @client.tool() decorator and push_decorated_tools()."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.toolbox.models import ToolInfo

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


class TestToolDecoratorMetadata:
    """Decorator captures metadata correctly on the function."""

    def test_captures_explicit_name_and_description(self):
        client = _client()

        @client.tool(name="my_adder", description="Adds numbers together")
        def add(a: int, b: int) -> int:
            return a + b

        meta = add._toolbox_metadata
        assert meta["name"] == "my_adder"
        assert meta["description"] == "Adds numbers together"
        assert meta["entry_point"] == "add"
        assert "def add(a: int, b: int)" in meta["source_code"]

    def test_defaults_name_from_function(self):
        client = _client()

        @client.tool()
        def compute_sum(x: int, y: int) -> int:
            """Compute the sum of two integers."""
            return x + y

        meta = compute_sum._toolbox_metadata
        assert meta["name"] == "compute_sum"
        assert meta["description"] == "Compute the sum of two integers."

    def test_fallback_description_when_no_docstring(self):
        client = _client()

        @client.tool()
        def silent_func(x: int) -> int:
            return x * 2

        meta = silent_func._toolbox_metadata
        assert meta["description"] == "Tool: silent_func"

    def test_tags_and_requirements_captured(self):
        client = _client()

        @client.tool(tags=["math", "util"], requirements=["numpy>=1.24"])
        def matrix_mul(a: list, b: list) -> list:
            return a  # stub

        meta = matrix_mul._toolbox_metadata
        assert meta["tags"] == ["math", "util"]
        assert meta["requirements"] == ["numpy>=1.24"]

    def test_tags_and_requirements_default_to_empty(self):
        client = _client()

        @client.tool()
        def noop() -> None:
            pass

        meta = noop._toolbox_metadata
        assert meta["tags"] == []
        assert meta["requirements"] == []

    def test_decorated_function_still_callable(self):
        client = _client()

        @client.tool(name="double")
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10


class TestPushDecoratedTools:
    """push_decorated_tools() calls push_tool() for each decorated function."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_pushes_all_decorated_functions(self):
        client = _client()

        @client.tool(name="add", description="Add two numbers", tags=["math"])
        def add(a: int, b: int) -> int:
            return a + b

        @client.tool(name="mul", description="Multiply two numbers")
        def mul(a: int, b: int) -> int:
            return a * b

        resp_add = {**_TOOL_RESPONSE, "name": "add"}
        resp_mul = {**_TOOL_RESPONSE, "id": "tool-002", "name": "mul"}

        route = respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/push").mock(
            side_effect=[
                _json_resp(resp_add, 201),
                _json_resp(resp_mul, 201),
            ]
        )

        async with client:
            results = await client.push_decorated_tools(project_id="proj-1")

        assert len(results) == 2
        assert all(isinstance(r, ToolInfo) for r in results)
        assert results[0].name == "add"
        assert results[1].name == "mul"
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_push_uses_default_project_id(self):
        client = _client()  # project_id="proj-1"

        @client.tool(name="echo")
        def echo(msg: str) -> str:
            """Echo back."""
            return msg

        respx.post(f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/push").mock(
            return_value=_json_resp({**_TOOL_RESPONSE, "name": "echo"}, 201)
        )

        async with client:
            results = await client.push_decorated_tools()

        assert len(results) == 1
        assert results[0].name == "echo"

    @respx.mock
    @pytest.mark.asyncio
    async def test_push_empty_list_returns_empty(self):
        client = _client()

        async with client:
            results = await client.push_decorated_tools(project_id="proj-1")

        assert results == []


class TestDecoratedToolsIsolation:
    """Each client instance tracks its own decorated tools."""

    def test_separate_clients_have_separate_lists(self):
        client_a = _client()
        client_b = _client()

        @client_a.tool(name="tool_a")
        def func_a() -> None:
            pass

        @client_b.tool(name="tool_b")
        def func_b() -> None:
            pass

        assert len(client_a._decorated_tools) == 1
        assert len(client_b._decorated_tools) == 1
        assert client_a._decorated_tools[0]._toolbox_metadata["name"] == "tool_a"
        assert client_b._decorated_tools[0]._toolbox_metadata["name"] == "tool_b"
