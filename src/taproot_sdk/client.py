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

from typing import Any

import httpx

from taproot_sdk.core import get_config, is_initialized


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
            f"/api/v1/retrieval/stores/{store_name}/query", json=body,
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
    ) -> dict[str, Any]:
        """Fetch a prompt template from Prompt-S.

        GET /api/v1/prompts/serve/{project_id}/{name}
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
        return r.json()

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
