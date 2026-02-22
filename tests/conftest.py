"""Pytest configuration and fixtures."""

import pytest

from taproot_sdk import shutdown


@pytest.fixture(autouse=True)
def reset_sdk():
    """Reset SDK state before each test."""
    # Ensure clean state before test
    try:
        shutdown()
    except Exception:
        pass

    yield

    # Clean up after test
    try:
        shutdown()
    except Exception:
        pass
