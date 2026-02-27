"""
Unified HTTP client for all Taproot platform services behind the APIM gateway.

Initialized once with base_url + api_key + project_id. Each method adds
the route path the load balancer expects. If the SDK has been initialized
via ev.init(), the client can pull config from there automatically.

APIM routes:
  /api/v1/retrieval/...  -> Retrieval-S
  /api/v1/evals/...      -> Evals-S
  /api/v1/guardrails/... -> Guardrail-S
  /api/v1/prompts/...    -> Prompt-S
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from taproot_sdk.core import get_config, is_initialized
from taproot_sdk.evals.models import EvalResult, RunHandle
from taproot_sdk.prompts.models import PromptResponse


class TaprootClient:
    """Single async client for all Taproot platform services."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        timeout: float = 30.0,
    ):
        # Pull from SDK config if available and not explicitly provided
        config = get_config() if is_initialized() else {}

        self.base_url = (base_url or config.get("api_url", "")).rstrip("/")
        self.api_key = api_key or config.get("api_key", "")
        self.project_id = project_id or config.get("project_id", "")

        if not self.base_url:
            raise ValueError(
                "base_url is required. Either pass it explicitly or call ev.init() first."
            )

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._http.aclose()

    # -- Retrieval-S --

    async def retrieval_query(
        self,
        store_name: str,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """Query a retrieval store for relevant documents."""
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if filters:
            body["filters"] = filters
        r = await self._http.post(
            f"/api/v1/retrieval/api/v1/stores/{store_name}/query", json=body,
        )
        r.raise_for_status()
        return r.json()

    # -- Prompt-S --

    async def get_prompt(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
        project_id: str | None = None,
    ) -> PromptResponse:
        """Fetch a prompt template from Prompt-S.

        GET /api/v1/prompts/serve/{project_id}/{name}

        Args:
            name: The prompt name identifier.
            version: Optional specific version number to fetch.
            label: Optional label (e.g. "production") to resolve.
            project_id: Override project ID (defaults to client's project_id).

        Returns:
            PromptResponse with template content, variables, and metadata.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        pid = project_id or self.project_id
        params: dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        if label is not None:
            params["label"] = label
        r = await self._http.get(
            f"/api/v1/prompts/serve/{pid}/{name}", params=params,
        )
        r.raise_for_status()
        data = r.json()
        return PromptResponse(
            schema_version=data.get("schema_version", 1),
            name=data["name"],
            version=data["version"],
            content=data["content"],
            content_hash=data["content_hash"],
            config=data.get("config", {}),
            required_variables=tuple(data.get("required_variables", [])),
            label=data.get("label"),
            cached_at=data.get("cached_at"),
        )

    # -- Health --

    async def health_retrieval(self) -> dict[str, Any]:
        r = await self._http.get("/api/v1/retrieval/health/ready")
        r.raise_for_status()
        return r.json()

    async def health_evals(self) -> dict[str, Any]:
        r = await self._http.get("/api/v1/evals/health/ready")
        r.raise_for_status()
        return r.json()

    async def health_guardrails(self) -> dict[str, Any]:
        r = await self._http.get("/api/v1/guardrails/health")
        r.raise_for_status()
        return r.json()

    # -- Evals-S --

    async def trigger_eval_run(
        self,
        test_config_id: str,
        *,
        tags: list[str] | None = None,
        description: str | None = None,
        project_id: str | None = None,
    ) -> RunHandle:
        """Trigger a new evaluation run.

        Args:
            test_config_id: UUID of the test configuration to run.
            tags: Optional tags for the run.
            description: Optional description for the run.
            project_id: Override project ID (defaults to client's project_id).

        Returns:
            RunHandle with run_id and status.
        """
        pid = project_id or self.project_id
        body: dict[str, Any] = {"test_config_id": test_config_id}
        if tags is not None:
            body["tags"] = tags
        if description is not None:
            body["description"] = description

        r = await self._http.post(
            f"/api/v1/evals/v1/projects/{pid}/test-runs/trigger",
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        return RunHandle(
            run_id=str(data["run_id"]),
            status=data["status"],
            message=data.get("message", ""),
        )

    async def get_eval_run(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> EvalResult:
        """Get current state of an evaluation run.

        Args:
            run_id: UUID of the test run.
            project_id: Override project ID.

        Returns:
            EvalResult with current run state.
        """
        pid = project_id or self.project_id
        r = await self._http.get(
            f"/api/v1/evals/v1/projects/{pid}/test-runs/{run_id}",
        )
        r.raise_for_status()
        return EvalResult.from_api_response(r.json())

    async def wait_for_eval(
        self,
        run_id: str,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
        project_id: str | None = None,
    ) -> EvalResult:
        """Wait for an evaluation run to complete.

        Polls the Evals-S API until the run reaches a terminal state
        (completed, failed, cancelled) or the timeout is exceeded.

        Args:
            run_id: UUID of the test run.
            timeout: Maximum seconds to wait (default 300).
            poll_interval: Seconds between polls (default 5).
            project_id: Override project ID.

        Returns:
            EvalResult with final run state.

        Raises:
            TimeoutError: If the run doesn't complete within the timeout.
        """
        start = time.monotonic()

        while True:
            result = await self.get_eval_run(run_id, project_id=project_id)

            if result.status in ("completed", "failed", "cancelled"):
                return result

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Eval run {run_id} did not complete within {timeout}s. "
                    f"Current status: {result.status}, "
                    f"progress: {result.completed_items}/{result.total_items}"
                )

            await asyncio.sleep(poll_interval)
