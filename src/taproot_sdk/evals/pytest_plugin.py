"""Pytest fixtures for eval workflows.

Register as a pytest plugin via conftest.py or pyproject.toml:

    # conftest.py
    pytest_plugins = ["taproot_sdk.evals.pytest_plugin"]

Fixtures:
    eval_client: A TaprootClient configured from env vars
    eval_run: Factory fixture that triggers and waits for a run
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from taproot_sdk.client import TaprootClient
    from taproot_sdk.evals.models import EvalResult


@pytest.fixture
def eval_client() -> "TaprootClient":
    """Create a TaprootClient from environment variables.

    Requires:
        TAPROOT_EVAL_URL: Base URL (e.g., https://api.taproot.dev)
        TAPROOT_API_KEY_ID: API key ID for auth
        TAPROOT_PROJECT_ID: Project ID for scoping
    """
    from taproot_sdk.client import TaprootClient

    base_url = os.environ.get("TAPROOT_EVAL_URL")
    api_key = os.environ.get("TAPROOT_API_KEY_ID")
    project_id = os.environ.get("TAPROOT_PROJECT_ID")

    if not base_url:
        pytest.skip("TAPROOT_EVAL_URL not set")
    if not api_key:
        pytest.skip("TAPROOT_API_KEY_ID not set")
    if not project_id:
        pytest.skip("TAPROOT_PROJECT_ID not set")

    return TaprootClient(
        base_url=base_url,
        api_key=api_key,
        project_id=project_id,
        timeout=300.0,
    )


@pytest.fixture
def eval_run(eval_client: "TaprootClient"):
    """Factory fixture: triggers a test run and waits for completion.

    Usage:
        async def test_quality(eval_run):
            result = await eval_run("test-config-uuid")
            assert_eval(result, min_pass_rate=80)
    """
    import asyncio

    async def _run(
        test_config_id: str,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
        tags: list[str] | None = None,
        description: str | None = None,
    ) -> "EvalResult":
        handle = await eval_client.trigger_eval_run(
            test_config_id, tags=tags, description=description,
        )
        return await eval_client.wait_for_eval(
            handle.run_id, timeout=timeout, poll_interval=poll_interval,
        )

    return _run
