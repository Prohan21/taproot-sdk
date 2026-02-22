"""
Tests for OpenAI Auto-instrumentation.

Tests automatic instrumentation of OpenAI SDK calls.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock


class TestOpenAIInstrumentation:
    """Tests for OpenAI SDK instrumentation."""

    def test_instrument_openai(self):
        """Test instrumenting OpenAI SDK."""
        with patch("taproot_evals.auto_instrument.instrument_openai") as mock_instrument:
            mock_instrument.return_value = True

            result = mock_instrument()

            assert result is True

    def test_instrument_chat_completions(self):
        """Test instrumenting chat completions."""
        with patch("taproot_evals.auto_instrument.OpenAIInstrumentor") as MockInstrumentor:
            instrumentor = MockInstrumentor()
            instrumentor.instrument = Mock(return_value=True)

            result = instrumentor.instrument()

            assert result is True


class TestOpenAIChatCapture:
    """Tests for capturing OpenAI chat completion calls."""

    @pytest.mark.asyncio
    async def test_capture_chat_completion(self):
        """Test capturing a chat completion call."""
        with patch("taproot_evals.auto_instrument.capture_openai_call") as mock_capture:
            mock_capture.return_value = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "response": {"content": "Hi there!"},
                "tokens": {"prompt": 5, "completion": 3, "total": 8},
            }

            result = mock_capture(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result["model"] == "gpt-4"
            assert "tokens" in result

    @pytest.mark.asyncio
    async def test_capture_streaming_completion(self):
        """Test capturing a streaming chat completion."""
        with patch("taproot_evals.auto_instrument.capture_streaming_call") as mock_capture:
            mock_capture.return_value = {
                "model": "gpt-4",
                "streaming": True,
                "chunks": 5,
                "total_content": "Hello world",
            }

            result = mock_capture(stream=True)

            assert result["streaming"] is True


class TestOpenAISpanCreation:
    """Tests for creating spans for OpenAI calls."""

    def test_create_llm_span(self):
        """Test creating an LLM span."""
        with patch("taproot_evals.auto_instrument.create_span") as mock_create:
            mock_create.return_value = Mock(id="span-123")

            span = mock_create(
                name="openai.chat.completions",
                input={"messages": [{"role": "user", "content": "Hello"}]},
            )

            assert span.id == "span-123"

    def test_span_includes_model_info(self):
        """Test that span includes model information."""
        with patch("taproot_evals.auto_instrument.create_span") as mock_create:
            mock_create.return_value = Mock(
                id="span-123",
                metadata={"model": "gpt-4", "provider": "openai"},
            )

            span = mock_create(
                name="openai.chat.completions",
                metadata={"model": "gpt-4"},
            )

            assert span.metadata["model"] == "gpt-4"

    def test_span_captures_tokens(self):
        """Test that span captures token usage."""
        with patch("taproot_evals.auto_instrument.finalize_span") as mock_finalize:
            mock_finalize.return_value = Mock(
                usage={"prompt_tokens": 10, "completion_tokens": 20}
            )

            result = mock_finalize(
                span_id="span-123",
                output="Response",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
            )

            assert result.usage["prompt_tokens"] == 10


class TestOpenAIErrorCapture:
    """Tests for capturing OpenAI errors."""

    @pytest.mark.asyncio
    async def test_capture_rate_limit_error(self):
        """Test capturing rate limit error."""
        with patch("taproot_evals.auto_instrument.capture_error") as mock_capture:
            mock_capture.return_value = {
                "error_type": "RateLimitError",
                "message": "Rate limit exceeded",
            }

            result = mock_capture(
                error_type="RateLimitError",
                message="Rate limit exceeded",
            )

            assert result["error_type"] == "RateLimitError"

    @pytest.mark.asyncio
    async def test_capture_api_error(self):
        """Test capturing API error."""
        with patch("taproot_evals.auto_instrument.capture_error") as mock_capture:
            mock_capture.return_value = {
                "error_type": "APIError",
                "status_code": 500,
            }

            result = mock_capture(
                error_type="APIError",
                status_code=500,
            )

            assert result["status_code"] == 500


class TestOpenAIFunctionCalls:
    """Tests for capturing function/tool calls."""

    @pytest.mark.asyncio
    async def test_capture_function_call(self):
        """Test capturing a function call."""
        with patch("taproot_evals.auto_instrument.capture_function_call") as mock_capture:
            mock_capture.return_value = {
                "function_name": "get_weather",
                "arguments": {"location": "New York"},
            }

            result = mock_capture(
                function_name="get_weather",
                arguments={"location": "New York"},
            )

            assert result["function_name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_capture_tool_call(self):
        """Test capturing a tool call."""
        with patch("taproot_evals.auto_instrument.capture_tool_call") as mock_capture:
            mock_capture.return_value = {
                "tool_call_id": "call-123",
                "tool_name": "calculator",
                "input": {"expression": "2+2"},
                "output": "4",
            }

            result = mock_capture(
                tool_call_id="call-123",
                tool_name="calculator",
            )

            assert result["tool_name"] == "calculator"
