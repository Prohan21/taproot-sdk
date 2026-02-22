"""Tests for taproot_evals.core module."""

import pytest

import taproot_evals as ev
from taproot_evals.core import get_config, is_initialized


class TestInit:
    """Tests for ev.init() function."""

    def test_init_basic(self):
        """Test basic initialization."""
        tracer = ev.init(
            project_id="test-project",
            api_url="http://localhost:8000",
        )

        assert tracer is not None
        assert is_initialized()

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        ev.init(
            project_id="test-project",
            api_url="http://localhost:8000",
            api_key="sk-test-key",
        )

        config = get_config()
        assert config["api_key"] == "sk-test-key"

    def test_init_with_sampling_rate(self):
        """Test initialization with custom sampling rate."""
        ev.init(
            project_id="test-project",
            api_url="http://localhost:8000",
            sampling_rate=0.5,
        )

        config = get_config()
        assert config["sampling_rate"] == 0.5

    def test_init_twice_raises_error(self):
        """Test that initializing twice raises an error."""
        ev.init(
            project_id="test-project",
            api_url="http://localhost:8000",
        )

        with pytest.raises(RuntimeError, match="already initialized"):
            ev.init(
                project_id="test-project",
                api_url="http://localhost:8000",
            )

    def test_init_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from api_url."""
        ev.init(
            project_id="test-project",
            api_url="http://localhost:8000/",
        )

        config = get_config()
        assert config["api_url"] == "http://localhost:8000"


class TestShutdown:
    """Tests for ev.shutdown() function."""

    def test_shutdown_after_init(self):
        """Test shutdown after initialization."""
        ev.init(
            project_id="test-project",
            api_url="http://localhost:8000",
        )

        assert is_initialized()

        ev.shutdown()

        assert not is_initialized()

    def test_shutdown_without_init(self):
        """Test shutdown without initialization doesn't raise."""
        # Should not raise
        ev.shutdown()

    def test_reinit_after_shutdown(self):
        """Test that we can reinitialize after shutdown."""
        ev.init(
            project_id="test-project-1",
            api_url="http://localhost:8000",
        )

        ev.shutdown()

        # Should work now
        ev.init(
            project_id="test-project-2",
            api_url="http://localhost:8000",
        )

        config = get_config()
        assert config["project_id"] == "test-project-2"


class TestGetTracer:
    """Tests for ev.get_tracer() function."""

    def test_get_tracer_after_init(self):
        """Test getting tracer after initialization."""
        ev.init(
            project_id="test-project",
            api_url="http://localhost:8000",
        )

        tracer = ev.get_tracer()
        assert tracer is not None

    def test_get_tracer_without_init_raises(self):
        """Test getting tracer without initialization raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            ev.get_tracer()
