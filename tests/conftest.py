"""Pytest configuration and fixtures."""

from contextlib import suppress

import pytest

from taproot_sdk import shutdown


@pytest.fixture(autouse=True)
def reset_sdk():
    """Reset SDK state before each test."""
    # Ensure clean state before test
    with suppress(Exception):
        shutdown()

    yield

    # Clean up after test
    with suppress(Exception):
        shutdown()
