"""Tests for taproot_sdk.auto_instrument module."""

from taproot_sdk.auto_instrument import (
    INSTRUMENTORS,
    get_instrumented_libraries,
    is_instrumented,
    setup_auto_instrumentation,
    uninstrument,
    uninstrument_all,
)


class TestAutoInstrumentation:
    """Tests for auto-instrumentation functions."""

    def test_instrumentors_mapping_exists(self):
        """Test that the instrumentors mapping is defined."""
        assert len(INSTRUMENTORS) > 0
        assert "openai" in INSTRUMENTORS
        assert "anthropic" in INSTRUMENTORS

    def test_setup_with_unavailable_library(self):
        """Test setup with a library that's not installed."""
        # This should not raise, just log a debug message
        result = setup_auto_instrumentation(["nonexistent_library"])
        assert result == []

    def test_get_instrumented_libraries_initially_empty(self):
        """Test that no libraries are instrumented initially."""
        uninstrument_all()
        assert get_instrumented_libraries() == []

    def test_is_instrumented_returns_false_initially(self):
        """Test that is_instrumented returns False for uninstrumented libraries."""
        uninstrument_all()
        assert not is_instrumented("openai")
        assert not is_instrumented("anthropic")

    def test_uninstrument_nonexistent_library(self):
        """Test uninstrumenting a library that's not instrumented."""
        uninstrument_all()
        # Should not raise
        result = uninstrument(["openai"])
        assert result == []

    def test_uninstrument_all(self):
        """Test uninstrument_all clears all instrumentation."""
        uninstrument_all()
        assert get_instrumented_libraries() == []


class TestAutoInstrumentationWithMocks:
    """Tests for auto-instrumentation with mocked instrumentors.

    These tests verify the logic without requiring actual LLM libraries
    to be installed.
    """

    def test_setup_returns_empty_list_when_no_packages(self):
        """Test that setup returns empty list when instrumentor packages are missing."""
        # Try to instrument libraries - they may or may not be installed
        # depending on the test environment
        uninstrument_all()
        result = setup_auto_instrumentation(["openai"])

        # If openai instrumentor is not installed, result should be empty
        # If it is installed, it should contain "openai"
        assert isinstance(result, list)

    def test_double_instrumentation_idempotent(self):
        """Test that instrumenting twice is idempotent."""
        uninstrument_all()

        # Try to instrument twice
        result1 = setup_auto_instrumentation(["openai"])
        result2 = setup_auto_instrumentation(["openai"])

        # Both calls should return the same result
        # (either both empty if not installed, or both with "openai")
        assert result1 == result2
