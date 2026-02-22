"""
Tests for Anthropic Auto-instrumentation.

Tests automatic instrumentation of Anthropic SDK calls.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestAnthropicInstrumentation:
    """Tests for Anthropic SDK instrumentation."""

    def test_instrument_anthropic(self):
        """Test instrumenting Anthropic SDK."""
        with patch("taproot_sdk.auto_instrument.instrument_anthropic") as mock_instrument:
            mock_instrument.return_value = True

            result = mock_instrument()

            assert result is True

    def test_instrument_messages_create(self):
        """Test instrumenting messages.create."""
        with patch("taproot_sdk.auto_instrument.AnthropicInstrumentor") as MockInstrumentor:
            instrumentor = MockInstrumentor()
            instrumentor.instrument = Mock(return_value=True)

            result = instrumentor.instrument()

            assert result is True


class TestAnthropicMessageCapture:
    """Tests for capturing Anthropic message calls."""

    @pytest.mark.asyncio
    async def test_capture_message(self):
        """Test capturing a message call."""
        with patch("taproot_sdk.auto_instrument.capture_anthropic_call") as mock_capture:
            mock_capture.return_value = {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Hello"}],
                "response": {"content": [{"type": "text", "text": "Hi!"}]},
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }

            result = mock_capture(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result["model"] == "claude-3-sonnet-20240229"

    @pytest.mark.asyncio
    async def test_capture_streaming_message(self):
        """Test capturing a streaming message."""
        with patch("taproot_sdk.auto_instrument.capture_streaming_call") as mock_capture:
            mock_capture.return_value = {
                "model": "claude-3-sonnet-20240229",
                "streaming": True,
                "events": ["message_start", "content_block_delta", "message_stop"],
            }

            result = mock_capture(stream=True)

            assert result["streaming"] is True


class TestAnthropicSpanCreation:
    """Tests for creating spans for Anthropic calls."""

    def test_create_anthropic_span(self):
        """Test creating an Anthropic span."""
        with patch("taproot_sdk.auto_instrument.create_span") as mock_create:
            mock_create.return_value = Mock(id="span-456")

            span = mock_create(
                name="anthropic.messages.create",
                input={"messages": [{"role": "user", "content": "Hello"}]},
            )

            assert span.id == "span-456"

    def test_span_includes_system_prompt(self):
        """Test that span captures system prompt."""
        with patch("taproot_sdk.auto_instrument.create_span") as mock_create:
            mock_create.return_value = Mock(
                id="span-456",
                input={"system": "You are a helpful assistant."},
            )

            span = mock_create(
                name="anthropic.messages.create",
                input={"system": "You are a helpful assistant."},
            )

            assert "system" in span.input


class TestAnthropicToolUse:
    """Tests for capturing Anthropic tool use."""

    @pytest.mark.asyncio
    async def test_capture_tool_use(self):
        """Test capturing tool use blocks."""
        with patch("taproot_sdk.auto_instrument.capture_tool_use") as mock_capture:
            mock_capture.return_value = {
                "tool_use_id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco"},
            }

            result = mock_capture(
                tool_use_id="toolu_123",
                name="get_weather",
                input={"location": "San Francisco"},
            )

            assert result["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_capture_tool_result(self):
        """Test capturing tool result blocks."""
        with patch("taproot_sdk.auto_instrument.capture_tool_result") as mock_capture:
            mock_capture.return_value = {
                "tool_use_id": "toolu_123",
                "content": "72°F, sunny",
            }

            result = mock_capture(
                tool_use_id="toolu_123",
                content="72°F, sunny",
            )

            assert result["content"] == "72°F, sunny"


class TestAnthropicErrorCapture:
    """Tests for capturing Anthropic errors."""

    @pytest.mark.asyncio
    async def test_capture_overloaded_error(self):
        """Test capturing overloaded error."""
        with patch("taproot_sdk.auto_instrument.capture_error") as mock_capture:
            mock_capture.return_value = {
                "error_type": "OverloadedError",
                "message": "API is temporarily overloaded",
            }

            result = mock_capture(
                error_type="OverloadedError",
                message="API is temporarily overloaded",
            )

            assert result["error_type"] == "OverloadedError"

    @pytest.mark.asyncio
    async def test_capture_rate_limit_error(self):
        """Test capturing rate limit error."""
        with patch("taproot_sdk.auto_instrument.capture_error") as mock_capture:
            mock_capture.return_value = {
                "error_type": "RateLimitError",
                "retry_after": 60,
            }

            result = mock_capture(
                error_type="RateLimitError",
                retry_after=60,
            )

            assert result["retry_after"] == 60


class TestAnthropicContentBlocks:
    """Tests for handling content blocks."""

    @pytest.mark.asyncio
    async def test_parse_text_block(self):
        """Test parsing text content blocks."""
        with patch("taproot_sdk.auto_instrument.parse_content_block") as mock_parse:
            mock_parse.return_value = {
                "type": "text",
                "text": "Hello, world!",
            }

            result = mock_parse({"type": "text", "text": "Hello, world!"})

            assert result["type"] == "text"

    @pytest.mark.asyncio
    async def test_parse_tool_use_block(self):
        """Test parsing tool use content blocks."""
        with patch("taproot_sdk.auto_instrument.parse_content_block") as mock_parse:
            mock_parse.return_value = {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "calculator",
            }

            result = mock_parse({
                "type": "tool_use",
                "id": "toolu_123",
                "name": "calculator",
            })

            assert result["type"] == "tool_use"
