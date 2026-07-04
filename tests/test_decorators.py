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


class TestRedaction:
    """WO-013 T1: redact_by_default scrubs secrets from exported span attributes."""

    JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQs"

    def _local_tracing(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from taproot_sdk import decorators

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(decorators.trace, "get_tracer", lambda name: provider.get_tracer(name))
        return exporter

    def _set_config(self, monkeypatch, **config):
        from taproot_sdk import core

        monkeypatch.setattr(core, "_config", dict(config))

    def test_inputs_redacted_by_default(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool")
        def work(api_key: str, email: str) -> str:
            return "ok"

        work("sk-live-abc123XYZ789", "a@b.com")
        inputs = exporter.get_finished_spans()[-1].attributes["ev.data.inputs"]
        assert "sk-live-abc123XYZ789" not in inputs
        assert "a@b.com" not in inputs
        assert "redacted:" in inputs

    def test_outputs_redacted_by_default(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)
        jwt = self.JWT

        @instrument(spankind="tool")
        def work() -> dict:
            return {"token": "tok-value", "body": f"Bearer {jwt}"}

        work()
        outputs = exporter.get_finished_spans()[-1].attributes["ev.data.outputs"]
        assert jwt not in outputs
        assert "tok-value" not in outputs
        assert "redacted:" in outputs

    async def test_async_redaction(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool")
        async def work(password: str) -> str:
            return "done"

        await work("hunter2secret")
        inputs = exporter.get_finished_spans()[-1].attributes["ev.data.inputs"]
        assert "hunter2secret" not in inputs

    def test_config_opt_out_reproduces_plaintext(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)
        self._set_config(monkeypatch, redact_by_default=False)

        @instrument(spankind="tool")
        def work(api_key: str) -> dict:
            return {"email": "a@b.com"}

        work("sk-live-abc123XYZ789")
        span = exporter.get_finished_spans()[-1]
        assert "sk-live-abc123XYZ789" in span.attributes["ev.data.inputs"]
        assert "a@b.com" in span.attributes["ev.data.outputs"]

    def test_decorator_override_disables_redaction(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool", redact=False)
        def work(api_key: str) -> str:
            return "ok"

        work("sk-live-abc123XYZ789")
        inputs = exporter.get_finished_spans()[-1].attributes["ev.data.inputs"]
        assert "sk-live-abc123XYZ789" in inputs

    def test_decorator_override_enables_redaction(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)
        self._set_config(monkeypatch, redact_by_default=False)

        @instrument(spankind="tool", redact=True)
        def work(api_key: str) -> str:
            return "ok"

        work("sk-live-abc123XYZ789")
        inputs = exporter.get_finished_spans()[-1].attributes["ev.data.inputs"]
        assert "sk-live-abc123XYZ789" not in inputs

    def test_correlation_spine_survives_redaction(self, monkeypatch):
        """taproot.correlation_id / taproot.interaction_id / ev.meta.* pass through untouched."""
        from taproot_sdk._context import (
            TaprootInteractionContext,
            correlation_id_var,
            reset_interaction_context,
            set_interaction_context,
        )

        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool")
        def work(api_key: str) -> str:
            return "ok"

        cid_token = correlation_id_var.set("corr-abc-123")
        ictx_token = set_interaction_context(
            TaprootInteractionContext(interaction_id="int-xyz-789")
        )
        try:
            work("sk-live-abc123XYZ789")
        finally:
            reset_interaction_context(ictx_token)
            correlation_id_var.reset(cid_token)

        span = exporter.get_finished_spans()[-1]
        assert span.attributes["taproot.correlation_id"] == "corr-abc-123"
        assert span.attributes["taproot.interaction_id"] == "int-xyz-789"
        assert span.attributes["ev.meta.function"] == "work"
        assert "sk-live-abc123XYZ789" not in span.attributes["ev.data.inputs"]

    def test_ignore_inputs_still_wins(self, monkeypatch):
        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool", ignore_inputs=True)
        def work(api_key: str) -> str:
            return "ok"

        work("sk-live-abc123XYZ789")
        span = exporter.get_finished_spans()[-1]
        assert "ev.data.inputs" not in span.attributes
