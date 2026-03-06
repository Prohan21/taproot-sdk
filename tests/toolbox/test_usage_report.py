"""Tests for usage report client method and invoke_tool span attributes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.toolbox.models import UsageReport, ToolUsageStats

BASE = "https://gateway.test"


def _client() -> TaprootClient:
    return TaprootClient(
        base_url=BASE,
        api_key="test-key",
        project_id="proj-1",
    )


def _json_resp(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


_USAGE_REPORT_RESPONSE = {
    "project_id": "proj-1",
    "tools": [
        {
            "tool_id": "tool-001",
            "tool_name": "add",
            "tool_type": "hosted",
            "status": "active",
        },
        {
            "tool_id": "tool-002",
            "tool_name": "fetch_data",
            "tool_type": "external",
            "status": "deprecated",
        },
    ],
    "count": 2,
}

_INVOCATION_RESPONSE = {
    "invocation_id": "inv-001",
    "tool_name": "add",
    "success": True,
    "result": {"sum": 3},
    "error": None,
    "duration_ms": 42.0,
}


class TestGetUsageReport:
    @respx.mock
    @pytest.mark.asyncio
    async def test_returns_typed_report(self):
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/usage-report"
        ).mock(return_value=_json_resp(_USAGE_REPORT_RESPONSE))

        async with _client() as c:
            report = await c.get_usage_report()

        assert isinstance(report, UsageReport)
        assert report.project_id == "proj-1"
        assert report.count == 2
        assert len(report.tools) == 2
        assert isinstance(report.tools[0], ToolUsageStats)
        assert report.tools[0].tool_name == "add"
        assert report.tools[1].tool_type == "external"

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_report(self):
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/tools/usage-report"
        ).mock(
            return_value=_json_resp(
                {"project_id": "proj-1", "tools": [], "count": 0}
            )
        )

        async with _client() as c:
            report = await c.get_usage_report()

        assert report.count == 0
        assert report.tools == ()


class TestInvokeToolSpanAttributes:
    @respx.mock
    @pytest.mark.asyncio
    async def test_span_attributes_set_on_invocation(self):
        """Verify invoke_tool sets toolbox.* span attributes when OTel is active."""
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/invoke/add"
        ).mock(return_value=_json_resp(_INVOCATION_RESPONSE))

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        # Patch the real opentelemetry.trace module so the lazy import inside
        # invoke_tool picks up our mock span.
        from opentelemetry import trace as ot_trace

        with patch.object(ot_trace, "get_current_span", return_value=mock_span):
            async with _client() as c:
                result = await c.invoke_tool("add", {"a": 1, "b": 2})

        assert result.success is True
        mock_span.set_attribute.assert_any_call("toolbox.tool.name", "add")
        mock_span.set_attribute.assert_any_call("toolbox.tool.id", "inv-001")
        mock_span.set_attribute.assert_any_call("toolbox.invocation.success", True)

        # Check duration_ms was set (any float value)
        duration_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "toolbox.invocation.duration_ms"
        ]
        assert len(duration_calls) == 1
        assert isinstance(duration_calls[0][0][1], float)

    @respx.mock
    @pytest.mark.asyncio
    async def test_span_not_recording_skips_attributes(self):
        """When span is not recording, no attributes are set."""
        respx.post(
            f"{BASE}/api/v1/toolbox/v1/projects/proj-1/invoke/add"
        ).mock(return_value=_json_resp(_INVOCATION_RESPONSE))

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        from opentelemetry import trace as ot_trace

        with patch.object(ot_trace, "get_current_span", return_value=mock_span):
            async with _client() as c:
                result = await c.invoke_tool("add", {"a": 1, "b": 2})

        assert result.success is True
        mock_span.set_attribute.assert_not_called()
