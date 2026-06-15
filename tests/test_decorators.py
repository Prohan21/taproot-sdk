"""Tests for taproot_sdk.decorators module."""

import pytest

import taproot_sdk as ev
from taproot_sdk.decorators import instrument


@pytest.fixture
def initialized_sdk():
    """Initialize SDK before tests."""
    ev.init(
        project_id="test-project",
        api_url="http://localhost:8000",
    )
    yield
    ev.shutdown()


class TestInstrumentDecorator:
    """Tests for @instrument decorator."""

    def test_instrument_sync_function(self, initialized_sdk):
        """Test instrumenting a synchronous function."""

        @instrument(spankind="tool")
        def my_function(x: int, y: int) -> int:
            return x + y

        result = my_function(2, 3)
        assert result == 5

    def test_instrument_async_function(self, initialized_sdk):
        """Test instrumenting an asynchronous function."""
        import asyncio

        @instrument(spankind="tool")
        async def my_async_function(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        result = asyncio.run(my_async_function(2, 3))
        assert result == 5

    def test_instrument_with_custom_name(self, initialized_sdk):
        """Test instrumenting with a custom span name."""

        @instrument(spankind="tool", name="custom-span-name")
        def my_function():
            return "hello"

        result = my_function()
        assert result == "hello"

    def test_instrument_preserves_function_metadata(self, initialized_sdk):
        """Test that decorator preserves function metadata."""

        @instrument(spankind="tool")
        def my_documented_function():
            """This is my docstring."""
            pass

        assert my_documented_function.__name__ == "my_documented_function"
        assert my_documented_function.__doc__ == "This is my docstring."

    def test_instrument_with_exception(self, initialized_sdk):
        """Test that exceptions are propagated correctly."""

        @instrument(spankind="tool")
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_function()

    def test_instrument_with_ignore_inputs(self, initialized_sdk):
        """Test instrumenting with ignored inputs."""

        @instrument(spankind="tool", ignore_inputs=True)
        def my_function(secret: str) -> str:
            return f"processed: {secret}"

        result = my_function("my-secret")
        assert result == "processed: my-secret"

    def test_instrument_with_ignore_specific_inputs(self, initialized_sdk):
        """Test instrumenting with specific ignored input parameters."""

        @instrument(spankind="tool", ignore_inputs=["password"])
        def my_function(username: str, password: str) -> str:
            return f"user: {username}"

        result = my_function("john", "secret123")
        assert result == "user: john"

    def test_instrument_with_ignore_outputs(self, initialized_sdk):
        """Test instrumenting with ignored outputs."""

        @instrument(spankind="tool", ignore_outputs=True)
        def my_function() -> dict:
            return {"sensitive": "data"}

        result = my_function()
        assert result == {"sensitive": "data"}

    def test_instrument_with_kwargs(self, initialized_sdk):
        """Test instrumenting function with kwargs."""

        @instrument(spankind="tool")
        def my_function(**kwargs) -> dict:
            return kwargs

        result = my_function(a=1, b=2, c=3)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_instrument_with_default_args(self, initialized_sdk):
        """Test instrumenting function with default arguments."""

        @instrument(spankind="tool")
        def my_function(x: int, y: int = 10) -> int:
            return x + y

        result = my_function(5)
        assert result == 15

    def test_all_span_kinds(self, initialized_sdk):
        """Test all supported span kinds."""
        span_kinds = [
            "workflow",
            "agent",
            "chain",
            "tool",
            "retrieval",
            "embedding",
            "completion",
            "chat",
            "rerank",
        ]

        def make_func(kind):
            @instrument(spankind=kind)
            def test_func():
                return kind

            return test_func

        for kind in span_kinds:
            test_func = make_func(kind)

            result = test_func()
            assert result == kind


class TestInstrumentWithoutInit:
    """Tests for @instrument decorator without SDK initialization."""

    def test_decorator_works_without_init(self):
        """Test that decorated functions still work even without SDK init.

        The decorator should gracefully handle the case where the SDK
        is not initialized (uses noop tracer).
        """

        @instrument(spankind="tool")
        def my_function(x: int) -> int:
            return x * 2

        # Should not raise, just won't create spans
        result = my_function(5)
        assert result == 10
