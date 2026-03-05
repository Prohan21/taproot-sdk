"""
Unified HTTP client for all Taproot platform services.

Supports two modes:

  **APIM mode** (default, ``direct_mode=False``):
    Routes through the API Gateway / APIM which validates the raw API key
    and injects ``X-Api-Key-Id`` downstream. Paths include the APIM prefix
    (e.g. ``/api/v1/evals/v1/projects/...``).

  **Direct mode** (``direct_mode=True``):
    Hits the service directly (local dev, docker-compose, port-forward).
    Sends ``X-Api-Key-Id`` header and uses internal service paths
    (e.g. ``/v1/projects/...``).

APIM routes:
  /api/v1/retrieval/{proxy+}  -> Retrieval-S
  /api/v1/evals/{proxy+}      -> Evals-S
  /api/v1/guardrails/{proxy+} -> Guardrail-S
  /api/v1/prompts/{proxy+}    -> Prompt-S Management
  /serve/{proxy+}             -> Prompt-S Serving (Lambda / Azure Function)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import httpx

from taproot_sdk.core import get_config, is_initialized
from taproot_sdk.evals.models import (
    AlertHistory,
    AlertRule,
    DiscoverySession,
    DiscoverySuggestion,
    EvalResult,
    Experiment,
    ExportResult,
    GoldenDataset,
    GoldenDatasetItem,
    GoldenDatasetVersion,
    JobStatus,
    MetricComparison,
    PaginatedList,
    RunHandle,
    TestConfiguration,
    TraceInfo,
    TraceStats,
    Webhook,
    WebhookDelivery,
)
from taproot_sdk.exceptions import (
    AuthenticationError,
    ConflictError,
    PromptNotFoundError,
    RateLimitError,
    ServerError,
    TaprootAPIError,
    ValidationError,
)
from taproot_sdk.guardrails.check_results import AnalyticsSummary, CheckResult
from taproot_sdk.guardrails.configs import GuardrailConfig
from taproot_sdk.guardrails.models import GuardrailResponse
from taproot_sdk.prompts.cache import PromptCache
from taproot_sdk.prompts.models import (
    ChatMessage,
    PromptResponse,
    PromptType,
    ToolDefinition,
)
from taproot_sdk.toolbox.models import (
    CredentialInfo,
    CredentialList,
    ImportResult,
    InvocationResult,
    MCPServerInfo,
    MCPServerList,
    ToolInfo,
    ToolList,
)
from taproot_sdk.retrieval.models import (
    AccessGranted,
    AccessList,
    AccessRevoked,
    BatchCancelled,
    BatchCreated,
    BatchJobList,
    BatchStatus,
    ChunkInfo,
    ChunkList,
    ChunksDeleted,
    ChunksUploaded,
    DocumentDeleted,
    DocumentDetail,
    DocumentList,
    IngestionJob,
    JobCancelled,
    JobDetail,
    JobList,
    QueryResponse,
    StoreCreated,
    StoreDeleted,
    StoreInfo,
    StoreList,
    StoreStatistics,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_MAX_RETRY_WAIT = 10
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_SUPPORTED_SCHEMA_VERSION = 1


class TaprootClient:
    """Single async client for all Taproot platform services.

    Args:
        base_url: Base URL of the APIM gateway or direct service endpoint.
        api_key: API key (raw key for APIM, or key ID for direct mode).
        project_id: Default project ID for all requests.
        timeout: HTTP request timeout in seconds.
        direct_mode: When True, sends ``X-Api-Key-Id`` header and uses
            internal service paths (``/v1/...``). When False (default),
            sends ``x-api-key`` header and uses APIM gateway paths
            (``/api/v1/evals/v1/...``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        timeout: float = 30.0,
        direct_mode: bool = False,
        cache_ttl_seconds: float = 30.0,
        max_stale_seconds: float = 60.0,
    ):
        # Pull from SDK config if available and not explicitly provided
        config = get_config() if is_initialized() else {}

        self.base_url = (base_url or config.get("api_url", "")).rstrip("/")
        self.api_key = api_key or config.get("api_key", "")
        self.project_id = project_id or config.get("project_id", "")
        self.direct_mode = direct_mode

        if not self.base_url:
            raise ValueError(
                "base_url is required. "
                "Either pass it explicitly or call ev.init() first."
            )

        # APIM expects raw key in x-api-key; services expect X-Api-Key-Id
        auth_header = "X-Api-Key-Id" if direct_mode else "x-api-key"

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                auth_header: self.api_key,
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

        # L1 in-memory cache for prompt serving (stale-while-revalidate)
        self._prompt_cache = PromptCache(
            ttl_seconds=cache_ttl_seconds,
            max_stale_seconds=max_stale_seconds,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> TaprootClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal request wrapper with retry + error wrapping
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        service: str = "",
    ) -> httpx.Response:
        """Send an HTTP request with retry and SDK error wrapping.

        Retries on 429, 500, 502, 503, 504 and connection errors
        with exponential backoff: 1s, 2s, 4s, 8s, 10s (max).

        The total number of attempts and cumulative wait time are tracked
        and surfaced in ``ServerError`` when all retries are exhausted.
        """
        total_wait = 0.0
        r: httpx.Response | None = None

        for attempt in range(_MAX_RETRIES + 1):
            kwargs: dict[str, Any] = {}
            if json is not None:
                kwargs["json"] = json
            if params is not None:
                kwargs["params"] = params
            if content is not None:
                kwargs["content"] = content
            if headers is not None:
                kwargs["headers"] = headers

            try:
                r = await self._http.request(method, path, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt == _MAX_RETRIES:
                    raise ServerError(
                        0,
                        f"Connection failed: {exc}",
                        service=service,
                        request_url=f"{self.base_url}{path}",
                        attempts=attempt + 1,
                        total_wait_seconds=total_wait,
                    ) from exc
                wait = min(2 ** attempt, _MAX_RETRY_WAIT)
                total_wait += wait
                logger.debug(
                    "Connection error, retrying in %ds (attempt %d): %s",
                    wait, attempt + 1, exc,
                )
                await asyncio.sleep(wait)
                continue

            if r.status_code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_RETRIES:
                # Stash retry metadata on the response for _raise_for_status
                r._taproot_attempts = attempt + 1  # type: ignore[attr-defined]
                r._taproot_total_wait = total_wait  # type: ignore[attr-defined]
                return r

            wait = min(2 ** attempt, _MAX_RETRY_WAIT)
            total_wait += wait
            logger.debug(
                "Retryable status %d, retrying in %ds (attempt %d)",
                r.status_code, wait, attempt + 1,
            )
            await asyncio.sleep(wait)

        assert r is not None  # noqa: S101 — loop always executes at least once
        r._taproot_attempts = _MAX_RETRIES + 1  # type: ignore[attr-defined]
        r._taproot_total_wait = total_wait  # type: ignore[attr-defined]
        return r

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        service: str = "",
        project_id: str = "",
    ) -> None:
        """Raise a typed SDK exception for non-2xx responses.

        Args:
            response: The HTTP response to inspect.
            service: Which Taproot service this request targeted.
            project_id: Optional project context for better error messages.
        """
        if response.is_success:
            return

        status = response.status_code
        url = str(response.request.url) if response.request else ""
        body = response.text

        # Try to extract detail from JSON body
        detail = ""
        json_data: dict[str, Any] = {}
        try:
            json_data = response.json()

            # Guardrail-S uses two error formats:
            #   422: {"detail": "Validation error", "errors": [{field, message, type}]}
            #   4xx: {"detail": {"error": {"code": "...", "message": "..."}}}

            # 1. Check for Guardrail-S structured errors list
            errors_list = json_data.get("errors", [])
            if isinstance(errors_list, list) and errors_list:
                parts = []
                for err in errors_list:
                    if isinstance(err, dict) and "message" in err:
                        field = err.get("field", "")
                        parts.append(f"{field}: {err['message']}" if field else err["message"])
                    else:
                        parts.append(str(err))
                detail = "; ".join(parts)
            else:
                # 2. Parse the "detail" field
                raw = json_data.get(
                    "detail", json_data.get("message", json_data.get("error", "")),
                )

                if isinstance(raw, list):
                    # FastAPI Pydantic errors: [{loc: [...], msg: "...", type: "..."}]
                    parts = []
                    for item in raw:
                        if isinstance(item, dict) and "msg" in item:
                            loc = " -> ".join(str(x) for x in item.get("loc", []))
                            parts.append(f"{loc}: {item['msg']}" if loc else item["msg"])
                        else:
                            parts.append(str(item))
                    detail = "; ".join(parts) if parts else "Validation error"
                elif isinstance(raw, dict):
                    # Nested error envelope: {"error": {"code": ..., "message": ...}}
                    inner = raw.get("error", raw)
                    if isinstance(inner, dict) and "message" in inner:
                        code = inner.get("code", "")
                        detail = f"[{code}] {inner['message']}" if code else inner["message"]
                    elif "message" in raw:
                        code = raw.get("code", "")
                        detail = f"[{code}] {raw['message']}" if code else raw["message"]
                    elif "code" in raw:
                        detail = raw["code"]
                    else:
                        detail = str(raw)
                elif isinstance(raw, str):
                    detail = raw
                else:
                    detail = str(raw) if raw else ""
        except Exception:
            detail = body[:500] if body else ""

        if not detail:
            detail = ""

        # Detect serving-layer auth error and provide user-friendly message
        if status == 401 and "X-Api-Key-Id" in (detail or body):
            detail = (
                "Serving layer authentication failed. The APIM gateway is not "
                "forwarding your API key identity to the serving layer. "
                "Verify the APIM integration maps context.identity.apiKeyId "
                "to the X-Api-Key-Id header."
            )

        if status in (401, 403):
            raise AuthenticationError(
                status, detail, service=service, request_url=url, body=body,
                project_id=project_id or self.project_id,
            )
        if status == 404:
            raise TaprootAPIError(
                404, detail or "Not found", service=service, request_url=url, body=body,
            )
        if status == 409:
            raise ConflictError(
                detail or "Resource conflict", service=service, request_url=url, body=body,
            )
        if status == 422:
            # Collect structured errors from both Guardrail-S and FastAPI formats
            errors: list[dict[str, Any]] = []
            try:
                # Guardrail-S: {"errors": [{field, message, type}]}
                gs_errors = json_data.get("errors", [])
                if isinstance(gs_errors, list) and gs_errors:
                    errors = gs_errors
                else:
                    # FastAPI: {"detail": [{loc, msg, type}]}
                    raw_detail = json_data.get("detail", [])
                    if isinstance(raw_detail, list):
                        errors = raw_detail
            except Exception:
                pass
            raise ValidationError(
                detail or "Validation error",
                errors=errors, service=service, request_url=url, body=body,
            )
        if status == 429:
            retry_after: float | None = None
            raw = response.headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    pass
            raise RateLimitError(
                detail or "Rate limit exceeded", service=service, request_url=url,
                retry_after=retry_after, body=body,
            )

        # 5xx errors — include retry context
        if status >= 500:
            attempts = getattr(response, "_taproot_attempts", 1)
            total_wait = getattr(response, "_taproot_total_wait", 0.0)
            raise ServerError(
                status,
                detail or "Internal server error",
                service=service,
                request_url=url,
                body=body,
                attempts=attempts,
                total_wait_seconds=total_wait,
            )

        raise TaprootAPIError(
            status, detail or f"HTTP {status}", service=service, request_url=url, body=body,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _evals_path(self, path: str) -> str:
        """Build Evals-S path, accounting for APIM vs direct mode."""
        if self.direct_mode:
            return f"/v1{path}"
        return f"/api/v1/evals/v1{path}"

    def _retrieval_path(self, path: str) -> str:
        """Build Retrieval-S path."""
        if self.direct_mode:
            return f"/api/v1{path}"
        return f"/api/v1/retrieval/api/v1{path}"

    def _guardrail_path(self, path: str) -> str:
        """Build Guardrail-S path."""
        if self.direct_mode:
            return path
        return f"/api/v1/guardrails{path}"

    def _prompts_path(self, path: str) -> str:
        """Build Prompt-S Management path."""
        if self.direct_mode:
            return path
        return f"/api/v1/prompts{path}"

    def _project_prompts_path(
        self, suffix: str, project_id: str | None = None,
    ) -> str:
        """Build project-scoped Prompt-S Management path."""
        pid = project_id or self.project_id
        return self._prompts_path(f"/projects/{pid}/prompts{suffix}")

    def _project_guardrail_path(
        self, suffix: str, project_id: str | None = None,
    ) -> str:
        """Build project-scoped Guardrail-S path."""
        pid = project_id or self.project_id
        return self._guardrail_path(f"/projects/{pid}{suffix}")

    def _project_evals_path(
        self, suffix: str, project_id: str | None = None,
    ) -> str:
        """Build project-scoped Evals-S path."""
        pid = project_id or self.project_id
        return self._evals_path(f"/projects/{pid}{suffix}")

    def _toolbox_path(self, path: str) -> str:
        """Build ToolBox-S path."""
        if self.direct_mode:
            return f"/v1{path}"
        return f"/api/v1/toolbox/v1{path}"

    def _project_toolbox_path(
        self, suffix: str, project_id: str | None = None,
    ) -> str:
        """Build project-scoped ToolBox-S path."""
        pid = project_id or self.project_id
        return self._toolbox_path(f"/projects/{pid}/tools{suffix}")

    # ==================================================================
    # Retrieval-S — Store Management
    # ==================================================================

    async def create_store(
        self, name: str, *, display_name: str | None = None,
        embedding_provider: str | None = None, embedding_model: str | None = None,
        index_type: str | None = None, hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None, ivfflat_lists: int | None = None,
        default_ef_search: int | None = None, use_halfvec: bool | None = None,
        enable_fulltext: bool | None = None,
    ) -> StoreCreated:
        """Create a new vector store.

        Args:
            name: Unique store name matching ``^[a-z][a-z0-9_]{0,49}$``.
            display_name: Human-readable display name.
            embedding_provider: One of "openai", "azure_openai", "cohere", "google".
            embedding_model: Model name (default: "text-embedding-3-small").
            index_type: One of "hnsw", "ivfflat", "diskann", "scann".
            hnsw_m: HNSW M parameter (2-100, default 16).
            hnsw_ef_construction: HNSW ef_construction (16-500, default 128).
            ivfflat_lists: IVFFlat lists (1-10000, default 100).
            default_ef_search: Default ef_search for queries (10-500, default 100).
            use_halfvec: Use half-precision vectors (default True).
            enable_fulltext: Enable fulltext/hybrid search (default False).

        Raises:
            ValidationError: Invalid store name or parameter values.
            ConflictError: Store with this name already exists.
        """
        body: dict[str, Any] = {"name": name}
        if display_name is not None:
            body["display_name"] = display_name
        if embedding_provider is not None:
            body["embedding_provider"] = embedding_provider
        if embedding_model is not None:
            body["embedding_model"] = embedding_model
        if index_type is not None:
            body["index_type"] = index_type
        if hnsw_m is not None:
            body["hnsw_m"] = hnsw_m
        if hnsw_ef_construction is not None:
            body["hnsw_ef_construction"] = hnsw_ef_construction
        if ivfflat_lists is not None:
            body["ivfflat_lists"] = ivfflat_lists
        if default_ef_search is not None:
            body["default_ef_search"] = default_ef_search
        if use_halfvec is not None:
            body["use_halfvec"] = use_halfvec
        if enable_fulltext is not None:
            body["enable_fulltext"] = enable_fulltext
        r = await self._request(
            "POST", self._retrieval_path("/stores"), json=body, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreCreated.from_api_response(r.json())

    async def list_stores(self, *, offset: int = 0, limit: int = 50) -> StoreList:
        """List all accessible stores."""
        r = await self._request(
            "GET", self._retrieval_path("/stores"),
            params={"offset": offset, "limit": limit}, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreList.from_api_response(r.json())

    async def get_store(self, store_id: str) -> StoreInfo:
        """Get store details by UUID."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_id}"), service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreInfo.from_api_response(r.json())

    async def get_store_by_name(self, name: str) -> StoreInfo:
        """Get store details by name."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/by-name/{name}"), service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreInfo.from_api_response(r.json())

    async def update_store(
        self, store_id: str, *, display_name: str | None = None,
        default_ef_search: int | None = None,
    ) -> StoreInfo:
        """Update store settings."""
        body: dict[str, Any] = {}
        if display_name is not None:
            body["display_name"] = display_name
        if default_ef_search is not None:
            body["default_ef_search"] = default_ef_search
        r = await self._request(
            "PATCH", self._retrieval_path(f"/stores/{store_id}"),
            json=body, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreInfo.from_api_response(r.json())

    async def delete_store(self, store_id: str) -> StoreDeleted:
        """Delete a store (soft-delete: sets is_active=false)."""
        r = await self._request(
            "DELETE", self._retrieval_path(f"/stores/{store_id}"), service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreDeleted.from_api_response(r.json())

    async def get_store_stats(self, store_name: str) -> StoreStatistics:
        """Get usage statistics for a store."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/stats"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return StoreStatistics.from_api_response(r.json())

    # ==================================================================
    # Retrieval-S — Access Control
    # ==================================================================

    async def grant_store_access(
        self, store_id: str, api_key_id: str, *,
        access_level: str = "read_write",
    ) -> AccessGranted:
        """Grant an API key access to a store."""
        r = await self._request(
            "POST", self._retrieval_path(f"/stores/{store_id}/access"),
            json={"api_key_id": api_key_id, "access_level": access_level},
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return AccessGranted.from_api_response(r.json())

    async def revoke_store_access(self, store_id: str, api_key_id: str) -> AccessRevoked:
        """Revoke an API key's access to a store."""
        r = await self._request(
            "DELETE", self._retrieval_path(f"/stores/{store_id}/access"),
            json={"api_key_id": api_key_id}, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return AccessRevoked.from_api_response(r.json())

    async def list_store_access(self, store_id: str) -> AccessList:
        """List all access grants for a store."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_id}/access"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return AccessList.from_api_response(r.json())

    # ==================================================================
    # Retrieval-S — Query / Search
    # ==================================================================

    async def retrieval_query(
        self, store_name: str, query: str, top_k: int = 10,
        filters: dict | None = None, *, ef_search: int | None = None,
        search_mode: str = "vector_only", keyword_weight: float | None = None,
        rerank: bool = False, rerank_top_n: int | None = None,
    ) -> QueryResponse:
        """Query a retrieval store for relevant documents.

        Args:
            store_name: Target store name.
            query: Natural language search query.
            top_k: Maximum number of results (1-100, default 10).
            filters: Metadata key-value filters.
            ef_search: Override ef_search parameter (1-1000).
            search_mode: "vector_only", "hybrid", or "keyword_only".
            keyword_weight: Weight for keyword component in hybrid (0.0-1.0).
            rerank: Enable cross-encoder reranking.
            rerank_top_n: Number of results to rerank (1-100).

        Raises:
            TaprootAPIError: Store not found (404).
            ValidationError: Invalid search parameters (400/422).
            AuthenticationError: Access denied (403).
        """
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if filters:
            body["filters"] = filters
        if ef_search is not None:
            body["ef_search"] = ef_search
        if search_mode != "vector_only":
            body["search_mode"] = search_mode
        if keyword_weight is not None:
            body["keyword_weight"] = keyword_weight
        if rerank:
            body["rerank"] = rerank
        if rerank_top_n is not None:
            body["rerank_top_n"] = rerank_top_n
        r = await self._request(
            "POST", self._retrieval_path(f"/stores/{store_name}/query"),
            json=body, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return QueryResponse.from_api_response(r.json())

    # ==================================================================
    # Retrieval-S — Document Ingestion
    # ==================================================================

    async def ingest_document(
        self, store_name: str, *, index: str,
        source_uri: str | None = None, signed_url: str | None = None,
        content_base64: str | None = None, filename: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunking: dict[str, Any] | None = None,
        pipeline_id: str = "default",
    ) -> IngestionJob:
        """Trigger document ingestion into a store.

        Provide exactly one of ``source_uri``, ``signed_url``, or ``content_base64``.

        Raises:
            ValidationError: Missing/invalid source, bad base64 (422).
            TaprootAPIError: Store not found (404).
        """
        body: dict[str, Any] = {"index": index, "pipeline_id": pipeline_id}
        if source_uri is not None:
            body["source_uri"] = source_uri
        if signed_url is not None:
            body["signed_url"] = signed_url
        if content_base64 is not None:
            body["content_base64"] = content_base64
        if filename is not None:
            body["filename"] = filename
        if content_type is not None:
            body["content_type"] = content_type
        if metadata is not None:
            body["metadata"] = metadata
        if chunking is not None:
            body["chunking"] = chunking
        r = await self._request(
            "POST", self._retrieval_path(f"/stores/{store_name}/ingest"),
            json=body, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return IngestionJob.from_api_response(r.json())

    async def get_ingestion_job(self, store_name: str, job_id: str) -> JobDetail:
        """Get detailed status of an ingestion job."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/ingest/{job_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return JobDetail.from_api_response(r.json())

    async def list_ingestion_jobs(
        self, store_name: str, *, status: str | None = None,
        created_after: str | None = None, created_before: str | None = None,
        sort_by: str = "created_at", sort_order: str = "desc",
        offset: int = 0, limit: int = 50,
    ) -> JobList:
        """List ingestion jobs for a store.

        Args:
            store_name: Target store name.
            status: Filter by job status (pending, processing, completed, failed, cancelled).
            created_after: ISO timestamp — only return jobs created after this time.
            created_before: ISO timestamp — only return jobs created before this time.
            sort_by: Sort field (default "created_at").
            sort_order: Sort direction: "asc" or "desc" (default "desc").
            offset: Pagination offset (default 0).
            limit: Pagination limit (1-100, default 50).
        """
        params: dict[str, Any] = {
            "offset": offset, "limit": limit,
            "sort_by": sort_by, "sort_order": sort_order,
        }
        if status is not None:
            params["status"] = status
        if created_after is not None:
            params["created_after"] = created_after
        if created_before is not None:
            params["created_before"] = created_before
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/ingest/jobs"),
            params=params, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return JobList.from_api_response(r.json())

    async def cancel_ingestion_job(self, store_name: str, job_id: str) -> JobCancelled:
        """Cancel a pending or processing ingestion job."""
        r = await self._request(
            "DELETE", self._retrieval_path(f"/stores/{store_name}/ingest/{job_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return JobCancelled.from_api_response(r.json())

    # ==================================================================
    # Retrieval-S — Batch Ingestion
    # ==================================================================

    async def batch_ingest(
        self, store_name: str, *, source: dict[str, Any],
        pipeline_id: str = "default", chunking: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None, max_documents: int | None = None,
    ) -> BatchCreated:
        """Create a batch ingestion job."""
        body: dict[str, Any] = {"source": source, "pipeline_id": pipeline_id}
        if chunking is not None:
            body["chunking"] = chunking
        if metadata is not None:
            body["metadata"] = metadata
        if max_documents is not None:
            body["max_documents"] = max_documents
        r = await self._request(
            "POST", self._retrieval_path(f"/stores/{store_name}/ingest/batch"),
            json=body, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return BatchCreated.from_api_response(r.json())

    async def get_batch_status(self, store_name: str, batch_id: str) -> BatchStatus:
        """Get full status of a batch ingestion job."""
        r = await self._request(
            "GET",
            self._retrieval_path(f"/stores/{store_name}/ingest/batch/{batch_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return BatchStatus.from_api_response(r.json())

    async def list_batch_jobs(
        self, store_name: str, batch_id: str, *,
        status: str | None = None, offset: int = 0, limit: int = 50,
    ) -> BatchJobList:
        """List individual jobs within a batch."""
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if status is not None:
            params["status"] = status
        r = await self._request(
            "GET",
            self._retrieval_path(f"/stores/{store_name}/ingest/batch/{batch_id}/jobs"),
            params=params, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return BatchJobList.from_api_response(r.json())

    async def cancel_batch(self, store_name: str, batch_id: str) -> BatchCancelled:
        """Cancel a batch ingestion job."""
        r = await self._request(
            "DELETE",
            self._retrieval_path(f"/stores/{store_name}/ingest/batch/{batch_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return BatchCancelled.from_api_response(r.json())

    # ==================================================================
    # Retrieval-S — Document Management
    # ==================================================================

    async def list_documents(
        self, store_name: str, *, offset: int = 0, limit: int = 20,
        doc_id: str | None = None,
    ) -> DocumentList:
        """List documents in a store."""
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if doc_id is not None:
            params["doc_id"] = doc_id
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/documents"),
            params=params, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return DocumentList.from_api_response(r.json())

    async def get_document(self, store_name: str, doc_id: str) -> DocumentDetail:
        """Get document details."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/documents/{doc_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return DocumentDetail.from_api_response(r.json())

    async def delete_document(self, store_name: str, doc_id: str) -> DocumentDeleted:
        """Delete a document and all its chunks."""
        r = await self._request(
            "DELETE", self._retrieval_path(f"/stores/{store_name}/documents/{doc_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return DocumentDeleted.from_api_response(r.json())

    # ==================================================================
    # Retrieval-S — Chunk Management
    # ==================================================================

    async def upload_chunks(
        self, store_name: str, doc_id: str, chunks: list[dict[str, Any]],
    ) -> ChunksUploaded:
        """Upload pre-computed chunks directly to a store."""
        r = await self._request(
            "POST", self._retrieval_path(f"/stores/{store_name}/chunks"),
            json={"doc_id": doc_id, "chunks": chunks}, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return ChunksUploaded.from_api_response(r.json())

    async def list_chunks(
        self, store_name: str, *, doc_id: str | None = None,
        offset: int = 0, limit: int = 20,
    ) -> ChunkList:
        """List chunks in a store."""
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if doc_id is not None:
            params["doc_id"] = doc_id
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/chunks"),
            params=params, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return ChunkList.from_api_response(r.json())

    async def get_chunk(self, store_name: str, chunk_id: str) -> ChunkInfo:
        """Get a single chunk by ID."""
        r = await self._request(
            "GET", self._retrieval_path(f"/stores/{store_name}/chunks/{chunk_id}"),
            service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return ChunkInfo.from_dict(r.json())

    async def delete_chunks(self, store_name: str, doc_id: str) -> ChunksDeleted:
        """Delete all chunks for a document."""
        r = await self._request(
            "DELETE", self._retrieval_path(f"/stores/{store_name}/chunks"),
            params={"doc_id": doc_id}, service="retrieval",
        )
        self._raise_for_status(r, service="retrieval")
        return ChunksDeleted.from_api_response(r.json())

    # ==================================================================
    # Prompt-S — Serving (cached reads)
    # ==================================================================

    async def get_prompt(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
        project_id: str | None = None,
    ) -> PromptResponse:
        """Fetch a prompt template from the serving layer (cached).

        Uses an L1 in-memory cache with stale-while-revalidate semantics.
        For uncached management reads, use :meth:`get_version`.
        """
        if version is not None and label is not None:
            raise ValueError("Cannot specify both 'version' and 'label'")

        pid = project_id or self.project_id
        return await self._prompt_cache.get_or_fetch(
            pid, name, version=version, label=label,
            fetch_fn=self._fetch_prompt_serving,
        )

    def get_prompt_sync(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
        project_id: str | None = None,
    ) -> PromptResponse:
        """Synchronous wrapper around :meth:`get_prompt`."""
        coro = self.get_prompt(name, version=version, label=label, project_id=project_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()

        return asyncio.run(coro)

    async def _fetch_prompt_serving(
        self,
        project_id: str,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
    ) -> PromptResponse:
        """Low-level HTTP GET to the serving layer (no caching)."""
        params: dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        if label is not None:
            params["label"] = label

        r = await self._request(
            "GET", f"/serve/{project_id}/{name}", params=params,
            service="prompts-serving",
        )

        if r.status_code == 404:
            raise PromptNotFoundError(
                name, project_id=project_id,
                version=version, label=label,
                request_url=str(r.request.url) if r.request else "",
                body=r.text,
            )
        self._raise_for_status(r, service="prompts-serving", project_id=project_id)
        return self._parse_prompt_response(r.json())

    @staticmethod
    def _parse_prompt_response(data: dict[str, Any]) -> PromptResponse:
        """Parse a serving-layer JSON response into a PromptResponse."""
        schema_version = data.get("schema_version")
        if schema_version is not None and schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version: {schema_version}. "
                f"This SDK supports schema_version={_SUPPORTED_SCHEMA_VERSION}. "
                f"Please upgrade taproot-sdk."
            )

        raw_prompt_type = data.get("prompt_type", "text")
        try:
            prompt_type = PromptType(raw_prompt_type)
        except ValueError:
            logger.warning("Unknown prompt_type '%s', defaulting to TEXT", raw_prompt_type)
            prompt_type = PromptType.TEXT

        raw_messages = data.get("messages")
        messages: tuple[ChatMessage, ...] | None = None
        if raw_messages is not None:
            messages = tuple(
                ChatMessage(role=m["role"], content=m["content"], name=m.get("name"))
                for m in raw_messages
            )

        raw_tools = data.get("tools")
        tools: tuple[ToolDefinition, ...] | None = None
        if raw_tools is not None:
            tools = tuple(
                ToolDefinition(
                    name=t["name"], description=t["description"],
                    parameters=t.get("parameters", {}),
                    type=t.get("type", "function"),
                )
                for t in raw_tools
            )

        ab_test = bool(data.get("ab_test", False))
        raw_variant = data.get("selected_variant")
        selected_variant = int(raw_variant) if raw_variant is not None else None

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
            prompt_type=prompt_type,
            messages=messages,
            tools=tools,
            ab_test=ab_test,
            selected_variant=selected_variant,
        )

    @property
    def prompt_cache(self) -> PromptCache:
        """Access the prompt serving cache for manual invalidation."""
        return self._prompt_cache

    # ==================================================================
    # Prompt-S — Management API
    # ==================================================================

    async def create_prompt(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
        prompt_type: str = "text",
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new prompt definition."""
        body: dict[str, Any] = {"name": name, "prompt_type": prompt_type}
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        r = await self._request(
            "POST",
            self._project_prompts_path("", project_id),
            json=body,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def list_prompts(
        self,
        *,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """List prompts in a project."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_archived": include_archived,
        }
        r = await self._request(
            "GET",
            self._project_prompts_path("", project_id),
            params=params,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def get_prompt_metadata(
        self,
        name: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Get prompt metadata (not the content/versions)."""
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{name}", project_id),
            service="prompts",
        )
        if r.status_code == 404:
            raise PromptNotFoundError(
                name, project_id=project_id or self.project_id,
                request_url=str(r.request.url) if r.request else "", body=r.text,
            )
        self._raise_for_status(r, service="prompts", project_id=project_id or self.project_id)
        return r.json()

    async def update_prompt(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
        is_archived: bool | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Update a prompt's mutable fields."""
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        if is_archived is not None:
            body["is_archived"] = is_archived
        r = await self._request(
            "PATCH",
            self._project_prompts_path(f"/{name}", project_id),
            json=body,
            service="prompts",
        )
        if r.status_code == 404:
            raise PromptNotFoundError(
                name, project_id=project_id or self.project_id,
                request_url=str(r.request.url) if r.request else "", body=r.text,
            )
        self._raise_for_status(r, service="prompts", project_id=project_id or self.project_id)
        return r.json()

    async def archive_prompt(
        self,
        name: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Archive (soft-delete) a prompt."""
        return await self.update_prompt(name, is_archived=True, project_id=project_id)

    async def delete_prompt(
        self,
        name: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Hard-delete a prompt."""
        r = await self._request(
            "DELETE",
            self._project_prompts_path(f"/{name}", project_id),
            service="prompts",
        )
        if r.status_code == 404:
            raise PromptNotFoundError(
                name, project_id=project_id or self.project_id,
                request_url=str(r.request.url) if r.request else "", body=r.text,
            )
        self._raise_for_status(r, service="prompts", project_id=project_id or self.project_id)

    # -- Versions --

    async def create_version(
        self,
        prompt_name: str,
        *,
        content: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        config: dict | None = None,
        model_config: dict | None = None,
        change_description: str | None = None,
        source: str = "manual",
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new prompt version.

        Provide ``content`` for text prompts or ``messages`` for chat prompts.
        """
        body: dict[str, Any] = {"source": source}
        if content is not None:
            body["content"] = content
        if messages is not None:
            body["messages"] = messages
        if tools is not None:
            body["tools"] = tools
        if config is not None:
            body["config"] = config
        if model_config is not None:
            body["model_config"] = model_config
        if change_description is not None:
            body["change_description"] = change_description
        r = await self._request(
            "POST",
            self._project_prompts_path(f"/{prompt_name}/versions", project_id),
            json=body,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def list_versions(
        self,
        prompt_name: str,
        *,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """List versions for a prompt."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/versions", project_id),
            params=params,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def get_version(
        self,
        prompt_name: str,
        version: int,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Get a specific prompt version."""
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/versions/{version}", project_id),
            service="prompts",
        )
        if r.status_code == 404:
            raise PromptNotFoundError(
                prompt_name, project_id=project_id or self.project_id,
                version=version,
                request_url=str(r.request.url) if r.request else "", body=r.text,
            )
        self._raise_for_status(r, service="prompts", project_id=project_id or self.project_id)
        return r.json()

    async def diff_versions(
        self,
        prompt_name: str,
        v1: int,
        v2: int,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Get a diff between two prompt versions."""
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/versions/diff", project_id),
            params={"v1": v1, "v2": v2},
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    # -- Labels --

    async def set_label(
        self,
        prompt_name: str,
        label: str,
        version: int,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Set a label to point at a specific version."""
        r = await self._request(
            "PUT",
            self._project_prompts_path(f"/{prompt_name}/labels", project_id),
            json={"label": label, "version": version},
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def set_weighted_label(
        self,
        prompt_name: str,
        label: str,
        weights: list[dict[str, Any]],
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Set a weighted (A/B test) label.

        Args:
            weights: List of ``{"version": int, "weight": float}`` dicts
                that sum to 1.0.
        """
        r = await self._request(
            "PUT",
            self._project_prompts_path(f"/{prompt_name}/labels/weighted", project_id),
            json={"label": label, "weights": weights},
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def list_labels(
        self,
        prompt_name: str,
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all labels for a prompt."""
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/labels", project_id),
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def get_label(
        self,
        prompt_name: str,
        label: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Get a specific label."""
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/labels/{label}", project_id),
            service="prompts",
        )
        if r.status_code == 404:
            raise PromptNotFoundError(
                prompt_name, project_id=project_id or self.project_id,
                label=label,
                request_url=str(r.request.url) if r.request else "", body=r.text,
            )
        self._raise_for_status(r, service="prompts", project_id=project_id or self.project_id)
        return r.json()

    async def delete_label(
        self,
        prompt_name: str,
        label: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Delete a label from a prompt."""
        r = await self._request(
            "DELETE",
            self._project_prompts_path(f"/{prompt_name}/labels/{label}", project_id),
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")

    # -- Approval --

    async def approve_version(
        self,
        prompt_name: str,
        version: int,
        *,
        reason: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Approve a prompt version (four-eyes rule)."""
        body: dict[str, Any] = {}
        if reason is not None:
            body["reason"] = reason
        r = await self._request(
            "POST",
            self._project_prompts_path(
                f"/{prompt_name}/versions/{version}/approve", project_id,
            ),
            json=body,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def reject_version(
        self,
        prompt_name: str,
        version: int,
        *,
        reason: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Reject a prompt version."""
        body: dict[str, Any] = {}
        if reason is not None:
            body["reason"] = reason
        r = await self._request(
            "POST",
            self._project_prompts_path(
                f"/{prompt_name}/versions/{version}/reject", project_id,
            ),
            json=body,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    # -- Examples --

    async def add_examples(
        self,
        prompt_name: str,
        examples: list[dict[str, Any]],
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Add training examples to a prompt."""
        r = await self._request(
            "POST",
            self._project_prompts_path(f"/{prompt_name}/examples", project_id),
            json={"examples": examples},
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def list_examples(
        self,
        prompt_name: str,
        *,
        signal: str | None = None,
        consumed: bool | None = None,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """List examples for a prompt."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if signal is not None:
            params["signal"] = signal
        if consumed is not None:
            params["consumed"] = consumed
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/examples", project_id),
            params=params,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    # -- Optimization --

    async def trigger_optimization(
        self,
        prompt_name: str,
        *,
        optimizer_name: str = "MIPROv2",
        config: dict | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Trigger an automatic prompt optimization (APO) job."""
        body: dict[str, Any] = {"optimizer_name": optimizer_name}
        if config is not None:
            body["config"] = config
        r = await self._request(
            "POST",
            self._project_prompts_path(f"/{prompt_name}/optimize", project_id),
            json=body,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    async def list_optimization_jobs(
        self,
        prompt_name: str,
        *,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """List optimization jobs for a prompt."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        r = await self._request(
            "GET",
            self._project_prompts_path(f"/{prompt_name}/optimize", project_id),
            params=params,
            service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    # ==================================================================
    # Health
    # ==================================================================

    async def health_retrieval(self) -> dict[str, Any]:
        if self.direct_mode:
            r = await self._request("GET", "/health/ready", service="retrieval")
        else:
            r = await self._request("GET", "/api/v1/retrieval/health/ready", service="retrieval")
        self._raise_for_status(r, service="retrieval")
        return r.json()

    async def health_evals(self) -> dict[str, Any]:
        if self.direct_mode:
            r = await self._request("GET", "/health/ready", service="evals")
        else:
            r = await self._request("GET", "/api/v1/evals/health/ready", service="evals")
        self._raise_for_status(r, service="evals")
        return r.json()

    async def health_evals_basic(self) -> dict[str, Any]:
        """GET /health — basic liveness check."""
        if self.direct_mode:
            r = await self._request("GET", "/health", service="evals")
        else:
            r = await self._request("GET", "/api/v1/evals/health", service="evals")
        self._raise_for_status(r, service="evals")
        return r.json()

    async def health_guardrails(self) -> dict[str, Any]:
        r = await self._request("GET", self._guardrail_path("/health"), service="guardrails")
        self._raise_for_status(r, service="guardrails")
        return r.json()

    async def health_prompts(self) -> dict[str, Any]:
        """Check Prompt-S Management service health."""
        r = await self._request(
            "GET", self._prompts_path("/health"), service="prompts",
        )
        self._raise_for_status(r, service="prompts")
        return r.json()

    # ==================================================================
    # Guardrail-S
    # ==================================================================

    async def check_input(
        self,
        content: str,
        *,
        project_id: str | None = None,
        config_name: str | None = None,
        trace_id: str | None = None,
        scanner_overrides: dict[str, bool] | None = None,
    ) -> GuardrailResponse:
        """Check user input content before sending to an LLM."""
        pid = project_id or self.project_id
        body: dict[str, Any] = {
            "project_id": pid,
            "content": content,
            "trace_id": trace_id or str(uuid.uuid4()),
        }
        if config_name is not None:
            body["config_name"] = config_name
        if scanner_overrides is not None:
            body["scanner_overrides"] = scanner_overrides

        r = await self._request(
            "POST",
            self._guardrail_path("/guardrails/check/input"),
            json=body,
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return GuardrailResponse.from_api_response(r.json())

    async def check_output(
        self,
        content: str,
        *,
        project_id: str | None = None,
        config_name: str | None = None,
        trace_id: str | None = None,
        original_input: str | None = None,
        scanner_overrides: dict[str, bool] | None = None,
    ) -> GuardrailResponse:
        """Check LLM output content before returning to the user."""
        pid = project_id or self.project_id
        body: dict[str, Any] = {
            "project_id": pid,
            "content": content,
            "trace_id": trace_id or str(uuid.uuid4()),
        }
        if config_name is not None:
            body["config_name"] = config_name
        if original_input is not None:
            body["original_input"] = original_input
        if scanner_overrides is not None:
            body["scanner_overrides"] = scanner_overrides

        r = await self._request(
            "POST",
            self._guardrail_path("/guardrails/check/output"),
            json=body,
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return GuardrailResponse.from_api_response(r.json())

    # ------------------------------------------------------------------
    # Guardrail-S — Check Results
    # ------------------------------------------------------------------

    async def list_check_results(
        self,
        *,
        verdict: str | None = None,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[CheckResult]:
        """List stored check results for a project.

        Args:
            verdict: Filter by verdict (``ALLOW``, ``BLOCK``, etc.).
            limit: Max results to return (1-500, default 50).
            offset: Pagination offset.
            project_id: Override default project ID.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict is not None:
            params["verdict"] = verdict
        r = await self._request(
            "GET",
            self._project_guardrail_path("/check-results", project_id),
            params=params,
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        data = r.json()
        items = data if isinstance(data, list) else data.get("items", [])
        return [CheckResult.from_dict(d) for d in items]

    async def get_check_result(
        self,
        result_id: str,
        *,
        project_id: str | None = None,
    ) -> CheckResult:
        """Get a single check result by ID."""
        r = await self._request(
            "GET",
            self._project_guardrail_path(
                f"/check-results/{result_id}", project_id,
            ),
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return CheckResult.from_dict(r.json())

    async def delete_check_result(
        self,
        result_id: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Soft-delete a check result."""
        r = await self._request(
            "DELETE",
            self._project_guardrail_path(
                f"/check-results/{result_id}", project_id,
            ),
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")

    # ------------------------------------------------------------------
    # Guardrail-S — Analytics
    # ------------------------------------------------------------------

    async def get_guardrail_analytics(
        self,
        start_date: str,
        end_date: str,
        *,
        granularity: str = "day",
        project_id: str | None = None,
    ) -> AnalyticsSummary:
        """Get aggregated scan analytics for a project.

        Args:
            start_date: ISO 8601 UTC datetime (inclusive).
            end_date: ISO 8601 UTC datetime (exclusive).
            granularity: Bucket size — ``hour``, ``day``, or ``week``.
            project_id: Override default project ID.
        """
        r = await self._request(
            "GET",
            self._project_guardrail_path("/analytics", project_id),
            params={
                "start_date": start_date,
                "end_date": end_date,
                "granularity": granularity,
            },
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return AnalyticsSummary.from_dict(r.json())

    # ------------------------------------------------------------------
    # Guardrail-S — Named Configs
    # ------------------------------------------------------------------

    async def create_guardrail_config(
        self,
        name: str,
        *,
        company_policy_version: str = "default",
        description: str | None = None,
        scanner_overrides: dict[str, dict[str, Any]] | None = None,
        mode: str = "active",
        updated_by: str = "",
        project_id: str | None = None,
    ) -> GuardrailConfig:
        """Create a named guardrail configuration.

        Args:
            name: Config name (alphanumeric + underscores).
            company_policy_version: Policy version to pin (default ``"default"``).
            description: Human-readable description.
            scanner_overrides: Per-scanner settings. Uses the **config-level**
                schema: ``{"scanner_id": {"enabled": True, "threshold": 0.8}}``.
                This differs from check-level overrides which use ``Dict[str, bool]``.
            mode: ``"active"`` or ``"shadow"``.
            updated_by: Audit trail — who created this config.
            project_id: Override default project ID.
        """
        body: dict[str, Any] = {
            "name": name,
            "company_policy_version": company_policy_version,
            "mode": mode,
        }
        if description is not None:
            body["description"] = description
        if scanner_overrides is not None:
            body["scanner_overrides"] = scanner_overrides
        body["updated_by"] = updated_by
        r = await self._request(
            "POST",
            self._project_guardrail_path("/configs", project_id),
            json=body,
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return GuardrailConfig.from_dict(r.json())

    async def list_guardrail_configs(
        self,
        *,
        project_id: str | None = None,
    ) -> list[GuardrailConfig]:
        """List all named configs for a project."""
        r = await self._request(
            "GET",
            self._project_guardrail_path("/configs", project_id),
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        data = r.json()
        items = data if isinstance(data, list) else data.get("items", data.get("configs", []))
        return [GuardrailConfig.from_dict(d) for d in items]

    async def get_guardrail_config(
        self,
        config_name: str,
        *,
        version: int | None = None,
        project_id: str | None = None,
    ) -> GuardrailConfig:
        """Get a named config by name, optionally at a specific version."""
        params: dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        r = await self._request(
            "GET",
            self._project_guardrail_path(f"/configs/{config_name}", project_id),
            params=params or None,
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return GuardrailConfig.from_dict(r.json())

    async def update_guardrail_config(
        self,
        config_name: str,
        *,
        company_policy_version: str | None = None,
        description: str | None = None,
        scanner_overrides: dict[str, dict[str, Any]] | None = None,
        updated_by: str = "",
        project_id: str | None = None,
    ) -> GuardrailConfig:
        """Update a named config (creates a new version).

        Args:
            config_name: Name of the config to update.
            company_policy_version: Policy version to pin.
            description: Updated description.
            scanner_overrides: Config-level scanner settings
                (``{"scanner_id": {"enabled": True, "threshold": 0.8}}``).
            updated_by: Audit trail.
            project_id: Override default project ID.
        """
        body: dict[str, Any] = {}
        if company_policy_version is not None:
            body["company_policy_version"] = company_policy_version
        if description is not None:
            body["description"] = description
        if scanner_overrides is not None:
            body["scanner_overrides"] = scanner_overrides
        body["updated_by"] = updated_by
        r = await self._request(
            "PUT",
            self._project_guardrail_path(
                f"/configs/{config_name}", project_id,
            ),
            json=body,
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return GuardrailConfig.from_dict(r.json())

    async def delete_guardrail_config(
        self,
        config_name: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Delete a named config and all its versions."""
        r = await self._request(
            "DELETE",
            self._project_guardrail_path(
                f"/configs/{config_name}", project_id,
            ),
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")

    async def set_default_guardrail_config(
        self,
        config_name: str,
        *,
        project_id: str | None = None,
    ) -> GuardrailConfig:
        """Set a named config as the project default."""
        r = await self._request(
            "PUT",
            self._project_guardrail_path(
                f"/configs/{config_name}/default", project_id,
            ),
            service="guardrails",
        )
        self._raise_for_status(r, service="guardrails")
        return GuardrailConfig.from_dict(r.json())

    # ==================================================================
    # Evals-S — Golden Datasets
    # ==================================================================

    async def create_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        project_id: str | None = None,
    ) -> GoldenDataset:
        """Create a new golden dataset."""
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        if tags is not None:
            body["tags"] = tags
        r = await self._request(
            "POST",
            self._project_evals_path("/golden-datasets", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return GoldenDataset.from_api_response(r.json())

    async def list_datasets(
        self,
        *,
        is_active: bool | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[GoldenDataset]:
        """List golden datasets."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if is_active is not None:
            params["is_active"] = is_active
        if tag is not None:
            params["tag"] = tag
        r = await self._request(
            "GET",
            self._project_evals_path("/golden-datasets", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [GoldenDataset.from_api_response(d) for d in r.json()]

    async def get_dataset(
        self, dataset_id: str, *, project_id: str | None = None,
    ) -> GoldenDataset:
        """Get a single golden dataset by ID."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/golden-datasets/{dataset_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return GoldenDataset.from_api_response(r.json())

    async def update_dataset(
        self,
        dataset_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        is_active: bool | None = None,
        project_id: str | None = None,
    ) -> GoldenDataset:
        """Update a golden dataset."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if tags is not None:
            body["tags"] = tags
        if is_active is not None:
            body["is_active"] = is_active
        r = await self._request(
            "PATCH",
            self._project_evals_path(f"/golden-datasets/{dataset_id}", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return GoldenDataset.from_api_response(r.json())

    async def delete_dataset(
        self, dataset_id: str, *, project_id: str | None = None,
    ) -> None:
        """Soft-delete a golden dataset."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(f"/golden-datasets/{dataset_id}", project_id),
        )
        self._raise_for_status(r, service="evals")

    # -- Dataset Items --

    async def add_dataset_item(
        self,
        dataset_id: str,
        input_query: str,
        *,
        expected_output: str | None = None,
        expected_output_json: dict[str, Any] | None = None,
        expected_tool_calls: list[dict[str, Any]] | None = None,
        expected_context_ids: list[str] | None = None,
        expected_contexts: list[str] | None = None,
        input_metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        session_group: str | None = None,
        sequence_order: int | None = None,
        project_id: str | None = None,
    ) -> GoldenDatasetItem:
        """Add an item to a golden dataset."""
        body: dict[str, Any] = {"input_query": input_query}
        if expected_output is not None:
            body["expected_output"] = expected_output
        if expected_output_json is not None:
            body["expected_output_json"] = expected_output_json
        if expected_tool_calls is not None:
            body["expected_tool_calls"] = expected_tool_calls
        if expected_context_ids is not None:
            body["expected_context_ids"] = expected_context_ids
        if expected_contexts is not None:
            body["expected_contexts"] = expected_contexts
        if input_metadata is not None:
            body["input_metadata"] = input_metadata
        if tags is not None:
            body["tags"] = tags
        if notes is not None:
            body["notes"] = notes
        if session_group is not None:
            body["session_group"] = session_group
        if sequence_order is not None:
            body["sequence_order"] = sequence_order
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/items", project_id,
            ),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return GoldenDatasetItem.from_api_response(r.json())

    async def list_dataset_items(
        self,
        dataset_id: str,
        *,
        session_group: str | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> PaginatedList[GoldenDatasetItem]:
        """List items in a golden dataset."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if session_group is not None:
            params["session_group"] = session_group
        r = await self._request(
            "GET",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/items", project_id,
            ),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        data = r.json()
        return PaginatedList(
            items=[GoldenDatasetItem.from_api_response(i) for i in data["items"]],
            total=data["total"],
            offset=data.get("offset", offset),
            limit=data.get("limit", limit),
        )

    async def update_dataset_item(
        self,
        dataset_id: str,
        item_id: str,
        *,
        input_query: str | None = None,
        expected_output: str | None = None,
        expected_output_json: dict[str, Any] | None = None,
        expected_tool_calls: list[dict[str, Any]] | None = None,
        expected_context_ids: list[str] | None = None,
        expected_contexts: list[str] | None = None,
        input_metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        session_group: str | None = None,
        sequence_order: int | None = None,
        project_id: str | None = None,
    ) -> GoldenDatasetItem:
        """Update a dataset item."""
        body: dict[str, Any] = {}
        if input_query is not None:
            body["input_query"] = input_query
        if expected_output is not None:
            body["expected_output"] = expected_output
        if expected_output_json is not None:
            body["expected_output_json"] = expected_output_json
        if expected_tool_calls is not None:
            body["expected_tool_calls"] = expected_tool_calls
        if expected_context_ids is not None:
            body["expected_context_ids"] = expected_context_ids
        if expected_contexts is not None:
            body["expected_contexts"] = expected_contexts
        if input_metadata is not None:
            body["input_metadata"] = input_metadata
        if tags is not None:
            body["tags"] = tags
        if notes is not None:
            body["notes"] = notes
        if session_group is not None:
            body["session_group"] = session_group
        if sequence_order is not None:
            body["sequence_order"] = sequence_order
        r = await self._request(
            "PATCH",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/items/{item_id}", project_id,
            ),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return GoldenDatasetItem.from_api_response(r.json())

    async def delete_dataset_item(
        self,
        dataset_id: str,
        item_id: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Delete a dataset item."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/items/{item_id}", project_id,
            ),
        )
        self._raise_for_status(r, service="evals")

    async def bulk_session_assign(
        self,
        dataset_id: str,
        item_ids: list[str],
        session_group: str,
        *,
        auto_order: bool = True,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Assign items to a session group for multi-turn testing."""
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/items/session-assign", project_id,
            ),
            json={
                "item_ids": item_ids,
                "session_group": session_group,
                "auto_order": auto_order,
            },
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    async def promote_trace_to_dataset(
        self,
        dataset_id: str,
        trace_ids: list[str],
        *,
        extract_tool_calls: bool = True,
        extract_contexts: bool = True,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Promote traces into golden dataset items."""
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/items/from-trace", project_id,
            ),
            json={
                "trace_ids": trace_ids,
                "extract_tool_calls": extract_tool_calls,
                "extract_contexts": extract_contexts,
            },
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    # -- Dataset Versioning --

    async def create_dataset_version(
        self,
        dataset_id: str,
        reason: str,
        *,
        changed_by: str | None = None,
        project_id: str | None = None,
    ) -> GoldenDatasetVersion:
        """Create a version snapshot of a golden dataset."""
        body: dict[str, Any] = {"reason": reason}
        if changed_by is not None:
            body["changed_by"] = changed_by
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/versions", project_id,
            ),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return GoldenDatasetVersion.from_api_response(r.json())

    async def list_dataset_versions(
        self,
        dataset_id: str,
        *,
        project_id: str | None = None,
    ) -> list[GoldenDatasetVersion]:
        """List version history of a golden dataset."""
        r = await self._request(
            "GET",
            self._project_evals_path(
                f"/golden-datasets/{dataset_id}/versions", project_id,
            ),
        )
        self._raise_for_status(r, service="evals")
        return [GoldenDatasetVersion.from_api_response(v) for v in r.json()]

    # ==================================================================
    # Evals-S — Test Configurations
    # ==================================================================

    async def create_test_config(
        self,
        name: str,
        dataset_id: str,
        agent_target_id: str,
        metrics: list[str],
        *,
        description: str | None = None,
        pass_threshold: float = 0.7,
        is_active: bool = True,
        project_id: str | None = None,
    ) -> TestConfiguration:
        """Create a test configuration."""
        body: dict[str, Any] = {
            "name": name,
            "dataset_id": dataset_id,
            "agent_target_id": agent_target_id,
            "metrics": metrics,
            "pass_threshold": pass_threshold,
            "is_active": is_active,
        }
        if description is not None:
            body["description"] = description
        r = await self._request(
            "POST",
            self._project_evals_path("/test-configurations", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return TestConfiguration.from_api_response(r.json())

    async def list_test_configs(
        self,
        *,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[TestConfiguration]:
        """List test configurations."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if is_active is not None:
            params["is_active"] = is_active
        r = await self._request(
            "GET",
            self._project_evals_path("/test-configurations", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [TestConfiguration.from_api_response(c) for c in r.json()]

    async def get_test_config(
        self, config_id: str, *, project_id: str | None = None,
    ) -> TestConfiguration:
        """Get a single test configuration."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/test-configurations/{config_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return TestConfiguration.from_api_response(r.json())

    async def update_test_config(
        self,
        config_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        metrics: list[str] | None = None,
        pass_threshold: float | None = None,
        is_active: bool | None = None,
        project_id: str | None = None,
    ) -> TestConfiguration:
        """Update a test configuration."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if metrics is not None:
            body["metrics"] = metrics
        if pass_threshold is not None:
            body["pass_threshold"] = pass_threshold
        if is_active is not None:
            body["is_active"] = is_active
        r = await self._request(
            "PATCH",
            self._project_evals_path(f"/test-configurations/{config_id}", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return TestConfiguration.from_api_response(r.json())

    async def delete_test_config(
        self, config_id: str, *, project_id: str | None = None,
    ) -> None:
        """Soft-delete a test configuration."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(f"/test-configurations/{config_id}", project_id),
        )
        self._raise_for_status(r, service="evals")

    # ==================================================================
    # Evals-S — Test Runs
    # ==================================================================

    async def trigger_eval_run(
        self,
        test_config_id: str,
        *,
        tags: list[str] | None = None,
        description: str | None = None,
        experiment_id: str | None = None,
        project_id: str | None = None,
    ) -> RunHandle:
        """Trigger a new evaluation run."""
        body: dict[str, Any] = {"test_config_id": test_config_id}
        if tags is not None:
            body["tags"] = tags
        if description is not None:
            body["description"] = description
        if experiment_id is not None:
            body["experiment_id"] = experiment_id
        r = await self._request(
            "POST",
            self._project_evals_path("/test-runs/trigger", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        data = r.json()
        return RunHandle(
            run_id=str(data.get("run_id", data.get("id", ""))),
            status=data["status"],
            message=data.get("message", ""),
        )

    async def list_eval_runs(
        self,
        *,
        test_config_id: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[EvalResult]:
        """List test runs with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if test_config_id is not None:
            params["test_config_id"] = test_config_id
        if status_filter is not None:
            params["status_filter"] = status_filter
        r = await self._request(
            "GET",
            self._project_evals_path("/test-runs", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [EvalResult.from_api_response(run) for run in r.json()]

    async def get_eval_run(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> EvalResult:
        """Get current state of an evaluation run."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/test-runs/{run_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return EvalResult.from_api_response(r.json())

    async def cancel_eval_run(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> EvalResult:
        """Cancel a pending or running evaluation run."""
        r = await self._request(
            "POST",
            self._project_evals_path(f"/test-runs/{run_id}/cancel", project_id),
        )
        self._raise_for_status(r, service="evals")
        return EvalResult.from_api_response(r.json())

    async def get_eval_run_results(
        self,
        run_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> PaginatedList[dict[str, Any]]:
        """Get individual item results for a test run."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        r = await self._request(
            "GET",
            self._project_evals_path(f"/test-runs/{run_id}/results", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        data = r.json()
        return PaginatedList(
            items=data.get("items", []),
            total=data.get("total", 0),
            offset=data.get("offset", offset),
            limit=data.get("limit", limit),
        )

    async def compare_eval_runs(
        self,
        run_id_a: str,
        run_id_b: str,
        *,
        project_id: str | None = None,
    ) -> list[MetricComparison]:
        """Compare metrics between two completed test runs."""
        r = await self._request(
            "POST",
            self._project_evals_path("/test-runs/compare", project_id),
            json={"run_id_a": run_id_a, "run_id_b": run_id_b},
        )
        self._raise_for_status(r, service="evals")
        data = r.json()
        return [
            MetricComparison.from_api_response(c)
            for c in data.get("comparisons", [])
        ]

    async def wait_for_eval(
        self,
        run_id: str,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
        project_id: str | None = None,
    ) -> EvalResult:
        """Wait for an evaluation run to complete.

        Polls until the run reaches a terminal state or the timeout is exceeded.
        """
        start = time.monotonic()

        while True:
            result = await self.get_eval_run(
                run_id, project_id=project_id,
            )

            if result.status in ("completed", "failed", "cancelled"):
                return result

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Eval run {run_id} did not complete "
                    f"within {timeout}s. "
                    f"Current status: {result.status}, "
                    f"progress: {result.completed_items}"
                    f"/{result.total_items}"
                )

            await asyncio.sleep(poll_interval)

    # ==================================================================
    # Evals-S — Experiments
    # ==================================================================

    async def create_experiment(
        self,
        name: str,
        metrics: list[str],
        *,
        description: str | None = None,
        dataset_id: str | None = None,
        dataset_version: int | None = None,
        agent_target_id: str | None = None,
        pass_threshold: float = 0.7,
        metadata: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> Experiment:
        """Create an experiment for grouping test runs."""
        body: dict[str, Any] = {
            "name": name,
            "metrics": metrics,
            "pass_threshold": pass_threshold,
        }
        if description is not None:
            body["description"] = description
        if dataset_id is not None:
            body["dataset_id"] = dataset_id
        if dataset_version is not None:
            body["dataset_version"] = dataset_version
        if agent_target_id is not None:
            body["agent_target_id"] = agent_target_id
        if metadata is not None:
            body["metadata"] = metadata
        r = await self._request(
            "POST",
            self._project_evals_path("/experiments", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return Experiment.from_api_response(r.json())

    async def list_experiments(
        self,
        *,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[Experiment]:
        """List experiments."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if is_active is not None:
            params["is_active"] = is_active
        r = await self._request(
            "GET",
            self._project_evals_path("/experiments", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [Experiment.from_api_response(e) for e in r.json()]

    async def get_experiment(
        self, experiment_id: str, *, project_id: str | None = None,
    ) -> Experiment:
        """Get an experiment by ID (includes run_count)."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/experiments/{experiment_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return Experiment.from_api_response(r.json())

    async def update_experiment(
        self,
        experiment_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        pass_threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
        is_active: bool | None = None,
        project_id: str | None = None,
    ) -> Experiment:
        """Update an experiment."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if pass_threshold is not None:
            body["pass_threshold"] = pass_threshold
        if metadata is not None:
            body["metadata"] = metadata
        if is_active is not None:
            body["is_active"] = is_active
        r = await self._request(
            "PATCH",
            self._project_evals_path(f"/experiments/{experiment_id}", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return Experiment.from_api_response(r.json())

    async def delete_experiment(
        self, experiment_id: str, *, project_id: str | None = None,
    ) -> None:
        """Deactivate an experiment."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(f"/experiments/{experiment_id}", project_id),
        )
        self._raise_for_status(r, service="evals")

    async def compare_experiment_runs(
        self, experiment_id: str, *, project_id: str | None = None,
    ) -> dict[str, Any]:
        """Compare all runs within an experiment."""
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/experiments/{experiment_id}/compare", project_id,
            ),
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    # ==================================================================
    # Evals-S — Alert Rules
    # ==================================================================

    async def create_alert_rule(
        self,
        name: str,
        condition_type: str,
        condition_config: dict[str, Any],
        notification_config: dict[str, Any],
        *,
        notification_channel: str = "webhook",
        cooldown_minutes: int = 60,
        project_id: str | None = None,
    ) -> AlertRule:
        """Create an alert rule."""
        r = await self._request(
            "POST",
            self._project_evals_path("/alert-rules", project_id),
            json={
                "name": name,
                "condition_type": condition_type,
                "condition_config": condition_config,
                "notification_channel": notification_channel,
                "notification_config": notification_config,
                "cooldown_minutes": cooldown_minutes,
            },
        )
        self._raise_for_status(r, service="evals")
        return AlertRule.from_api_response(r.json())

    async def list_alert_rules(
        self,
        *,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[AlertRule]:
        """List alert rules."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if is_active is not None:
            params["is_active"] = is_active
        r = await self._request(
            "GET",
            self._project_evals_path("/alert-rules", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [AlertRule.from_api_response(a) for a in r.json()]

    async def get_alert_rule(
        self, rule_id: str, *, project_id: str | None = None,
    ) -> AlertRule:
        """Get an alert rule by ID."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/alert-rules/{rule_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return AlertRule.from_api_response(r.json())

    async def update_alert_rule(
        self,
        rule_id: str,
        *,
        name: str | None = None,
        condition_config: dict[str, Any] | None = None,
        notification_config: dict[str, Any] | None = None,
        is_active: bool | None = None,
        cooldown_minutes: int | None = None,
        project_id: str | None = None,
    ) -> AlertRule:
        """Update an alert rule."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if condition_config is not None:
            body["condition_config"] = condition_config
        if notification_config is not None:
            body["notification_config"] = notification_config
        if is_active is not None:
            body["is_active"] = is_active
        if cooldown_minutes is not None:
            body["cooldown_minutes"] = cooldown_minutes
        r = await self._request(
            "PATCH",
            self._project_evals_path(f"/alert-rules/{rule_id}", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return AlertRule.from_api_response(r.json())

    async def delete_alert_rule(
        self, rule_id: str, *, project_id: str | None = None,
    ) -> None:
        """Soft-delete an alert rule."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(f"/alert-rules/{rule_id}", project_id),
        )
        self._raise_for_status(r, service="evals")

    async def get_alert_history(
        self,
        rule_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[AlertHistory]:
        """Get alert trigger history for a rule."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        r = await self._request(
            "GET",
            self._project_evals_path(f"/alert-rules/{rule_id}/history", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [AlertHistory.from_api_response(h) for h in r.json()]

    # ==================================================================
    # Evals-S — Webhooks
    # ==================================================================

    async def create_webhook(
        self,
        name: str,
        url: str,
        *,
        events: list[str] | None = None,
        headers: dict[str, str] | None = None,
        project_id: str | None = None,
    ) -> Webhook:
        """Create a webhook subscription."""
        body: dict[str, Any] = {"name": name, "url": url}
        if events is not None:
            body["events"] = events
        if headers is not None:
            body["headers"] = headers
        r = await self._request(
            "POST",
            self._project_evals_path("/webhooks", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return Webhook.from_api_response(r.json())

    async def list_webhooks(
        self, *, project_id: str | None = None,
    ) -> list[Webhook]:
        """List webhooks."""
        r = await self._request(
            "GET",
            self._project_evals_path("/webhooks", project_id),
        )
        self._raise_for_status(r, service="evals")
        return [Webhook.from_api_response(w) for w in r.json()]

    async def get_webhook(
        self, webhook_id: str, *, project_id: str | None = None,
    ) -> Webhook:
        """Get a webhook by ID."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/webhooks/{webhook_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return Webhook.from_api_response(r.json())

    async def update_webhook(
        self,
        webhook_id: str,
        *,
        name: str | None = None,
        url: str | None = None,
        events: list[str] | None = None,
        headers: dict[str, str] | None = None,
        is_active: bool | None = None,
        project_id: str | None = None,
    ) -> Webhook:
        """Update a webhook."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if url is not None:
            body["url"] = url
        if events is not None:
            body["events"] = events
        if headers is not None:
            body["headers"] = headers
        if is_active is not None:
            body["is_active"] = is_active
        r = await self._request(
            "PATCH",
            self._project_evals_path(f"/webhooks/{webhook_id}", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return Webhook.from_api_response(r.json())

    async def delete_webhook(
        self, webhook_id: str, *, project_id: str | None = None,
    ) -> None:
        """Delete a webhook."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(f"/webhooks/{webhook_id}", project_id),
        )
        self._raise_for_status(r, service="evals")

    async def list_webhook_deliveries(
        self,
        webhook_id: str,
        *,
        status_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[WebhookDelivery]:
        """List delivery attempts for a webhook."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status_filter is not None:
            params["status_filter"] = status_filter
        r = await self._request(
            "GET",
            self._project_evals_path(
                f"/webhooks/{webhook_id}/deliveries", project_id,
            ),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [WebhookDelivery.from_api_response(d) for d in r.json()]

    async def get_webhook_delivery(
        self,
        webhook_id: str,
        delivery_id: str,
        *,
        project_id: str | None = None,
    ) -> WebhookDelivery:
        """Get a single webhook delivery."""
        r = await self._request(
            "GET",
            self._project_evals_path(
                f"/webhooks/{webhook_id}/deliveries/{delivery_id}", project_id,
            ),
        )
        self._raise_for_status(r, service="evals")
        return WebhookDelivery.from_api_response(r.json())

    async def retry_webhook_delivery(
        self,
        webhook_id: str,
        delivery_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Retry a failed webhook delivery."""
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/webhooks/{webhook_id}/deliveries/{delivery_id}/retry",
                project_id,
            ),
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    # ==================================================================
    # Evals-S — Discovery
    # ==================================================================

    async def start_discovery(
        self,
        agent_target_id: str,
        *,
        agent_config: dict[str, Any] | None = None,
        target_dataset_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Start an agent capability discovery session.

        Returns a dict with ``session_id``, ``status``, and ``job_id``.
        Use :meth:`get_discovery_session` to fetch the full typed session.
        """
        body: dict[str, Any] = {"agent_target_id": agent_target_id}
        if agent_config is not None:
            body["agent_config"] = agent_config
        if target_dataset_id is not None:
            body["target_dataset_id"] = target_dataset_id
        r = await self._request(
            "POST",
            self._project_evals_path("/discovery/start", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    async def list_discovery_sessions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
    ) -> list[DiscoverySession]:
        """List discovery sessions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        r = await self._request(
            "GET",
            self._project_evals_path("/discovery", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [DiscoverySession.from_api_response(s) for s in r.json()]

    async def get_discovery_session(
        self, session_id: str, *, project_id: str | None = None,
    ) -> DiscoverySession:
        """Get a discovery session by ID."""
        r = await self._request(
            "GET",
            self._project_evals_path(f"/discovery/{session_id}", project_id),
        )
        self._raise_for_status(r, service="evals")
        return DiscoverySession.from_api_response(r.json())

    async def get_discovery_probes(
        self,
        session_id: str,
        *,
        level: int | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get probe results for a discovery session."""
        params: dict[str, Any] = {}
        if level is not None:
            params["level"] = level
        r = await self._request(
            "GET",
            self._project_evals_path(f"/discovery/{session_id}/probes", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    async def get_discovery_suggestions(
        self,
        session_id: str,
        *,
        status: str | None = None,
        category: str | None = None,
        project_id: str | None = None,
    ) -> list[DiscoverySuggestion]:
        """Get test case suggestions from a discovery session."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if category is not None:
            params["category"] = category
        r = await self._request(
            "GET",
            self._project_evals_path(
                f"/discovery/{session_id}/suggestions", project_id,
            ),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return [DiscoverySuggestion.from_api_response(s) for s in r.json()]

    async def update_discovery_suggestion(
        self,
        session_id: str,
        suggestion_id: str,
        *,
        user_edited_query: str | None = None,
        user_edited_expected_output: str | None = None,
        user_edited_tool_calls: list[dict[str, Any]] | None = None,
        user_notes: str | None = None,
        project_id: str | None = None,
    ) -> DiscoverySuggestion:
        """Edit a discovery suggestion."""
        body: dict[str, Any] = {}
        if user_edited_query is not None:
            body["user_edited_query"] = user_edited_query
        if user_edited_expected_output is not None:
            body["user_edited_expected_output"] = user_edited_expected_output
        if user_edited_tool_calls is not None:
            body["user_edited_tool_calls"] = user_edited_tool_calls
        if user_notes is not None:
            body["user_notes"] = user_notes
        r = await self._request(
            "PATCH",
            self._project_evals_path(
                f"/discovery/{session_id}/suggestions/{suggestion_id}",
                project_id,
            ),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return DiscoverySuggestion.from_api_response(r.json())

    async def approve_discovery_suggestions(
        self,
        session_id: str,
        suggestion_ids: list[str],
        target_dataset_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Approve suggestions and add them to a golden dataset."""
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/discovery/{session_id}/suggestions/approve", project_id,
            ),
            json={
                "suggestion_ids": suggestion_ids,
                "target_dataset_id": target_dataset_id,
            },
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    async def reject_discovery_suggestion(
        self,
        session_id: str,
        suggestion_id: str,
        *,
        project_id: str | None = None,
    ) -> DiscoverySuggestion:
        """Reject a discovery suggestion."""
        r = await self._request(
            "POST",
            self._project_evals_path(
                f"/discovery/{session_id}/suggestions/{suggestion_id}/reject",
                project_id,
            ),
        )
        self._raise_for_status(r, service="evals")
        return DiscoverySuggestion.from_api_response(r.json())

    async def cancel_discovery(
        self, session_id: str, *, project_id: str | None = None,
    ) -> dict[str, Any]:
        """Cancel a running discovery session.

        Returns a dict with ``session_id`` and ``status`` (``"cancelled"``).
        Use :meth:`get_discovery_session` for the full typed session.
        """
        r = await self._request(
            "POST",
            self._project_evals_path(f"/discovery/{session_id}/cancel", project_id),
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    # ==================================================================
    # Evals-S — Traces
    # ==================================================================

    async def ingest_traces(
        self,
        otlp_payload: dict[str, Any],
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest OTLP-formatted trace data."""
        r = await self._request(
            "POST",
            self._project_evals_path("/traces", project_id),
            json=otlp_payload,
        )
        self._raise_for_status(r, service="evals")
        return r.json()

    async def list_traces(
        self,
        *,
        agent_name: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 50,
        project_id: str | None = None,
    ) -> PaginatedList[TraceInfo]:
        """List traces with pagination.

        Returns a :class:`PaginatedList` of :class:`TraceInfo` objects.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if agent_name is not None:
            params["agent_name"] = agent_name
        if status is not None:
            params["status"] = status
        r = await self._request(
            "GET",
            self._project_evals_path("/traces", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        data = r.json()
        items_raw = data.get("items", data if isinstance(data, list) else [])
        return PaginatedList(
            items=[TraceInfo.from_api_response(t) for t in items_raw],
            total=data.get("total", len(items_raw)),
            offset=(page - 1) * page_size,
            limit=page_size,
        )

    async def get_trace(
        self,
        trace_id: str,
        *,
        include_spans: bool = True,
        project_id: str | None = None,
    ) -> TraceInfo:
        """Get a single trace with optional span detail.

        Returns a :class:`TraceInfo`.  When *include_spans* is ``True``
        the server may include span data in the response — access it
        via the raw ``metadata`` field if needed.
        """
        params: dict[str, Any] = {"include_spans": include_spans}
        r = await self._request(
            "GET",
            self._project_evals_path(f"/traces/{trace_id}", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return TraceInfo.from_api_response(r.json())

    async def get_trace_stats(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        project_id: str | None = None,
    ) -> TraceStats:
        """Get aggregate trace statistics."""
        params: dict[str, Any] = {}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        r = await self._request(
            "GET",
            self._project_evals_path("/traces/stats", project_id),
            params=params,
        )
        self._raise_for_status(r, service="evals")
        return TraceStats.from_api_response(r.json())

    async def delete_trace(
        self, trace_id: str, *, project_id: str | None = None,
    ) -> None:
        """Permanently delete a trace and its spans."""
        r = await self._request(
            "DELETE",
            self._project_evals_path(f"/traces/{trace_id}", project_id),
        )
        self._raise_for_status(r, service="evals")

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    async def export_eval_results(
        self,
        *,
        format: str = "jsonl",
        from_date: str | None = None,
        to_date: str | None = None,
        include: list[str] | None = None,
        project_id: str | None = None,
    ) -> ExportResult:
        """Request a bulk export of evaluation results.

        Args:
            format: Export format — ``"jsonl"`` (default) or ``"csv"``.
            from_date: ISO-8601 start date filter (inclusive).
            to_date: ISO-8601 end date filter (inclusive).
            include: Optional list of extra field groups to include.
                Supported values: ``"scores"``, ``"trace_refs"``.
                Defaults to ``["scores", "trace_refs"]``.
            project_id: Override the default project_id.

        Returns:
            An :class:`ExportResult` containing a presigned download URL
            (valid for 24 hours).
        """
        pid = project_id or self.project_id
        body: dict[str, Any] = {"project_id": pid, "format": format}
        if from_date is not None:
            body["from_date"] = from_date
        if to_date is not None:
            body["to_date"] = to_date
        if include is not None:
            body["include"] = include
        r = await self._request(
            "POST",
            self._project_evals_path("/exports", project_id),
            json=body,
        )
        self._raise_for_status(r, service="evals")
        return ExportResult.from_api_response(r.json())

    # ------------------------------------------------------------------
    # Job Status
    # ------------------------------------------------------------------

    async def get_eval_job_status(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> JobStatus:
        """Poll the cloud job dispatcher status for a test run.

        This is a low-level method that checks the underlying cloud job
        (Step Functions, Container Apps Job, etc.) rather than the database
        status returned by :meth:`get_eval_run`.  Useful for debugging
        stuck runs or monitoring job orchestration.

        Args:
            run_id: The test run UUID.
            project_id: Override the default project_id.

        Returns:
            A :class:`JobStatus` with the dispatcher-reported state.
        """
        r = await self._request(
            "GET",
            self._project_evals_path(f"/test-runs/{run_id}/job-status", project_id),
        )
        self._raise_for_status(r, service="evals")
        return JobStatus.from_api_response(r.json())

    # ==================================================================
    # ToolBox-S — Tool Management
    # ==================================================================

    async def push_tool(
        self,
        name: str,
        source_code: str,
        entry_point: str,
        *,
        description: str = "",
        requirements: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        timeout_ms: int = 30000,
        memory_mb: int = 256,
        scope: str = "project",
        project_id: str | None = None,
    ) -> ToolInfo:
        """Push a hosted Python tool to ToolBox-S.

        Validates source code, auto-generates input schema from function
        signature if not provided, and triggers async venv build for
        tools with custom requirements.

        Args:
            name: Tool name (must be a valid Python identifier).
            source_code: Python source code containing the tool function.
            entry_point: Name of the function to invoke.
            description: Human-readable description of the tool.
            requirements: pip requirements (e.g. ``["pandas>=2.0"]``).
            input_schema: JSON Schema for input validation. Auto-generated
                from function signature if not provided.
            output_schema: JSON Schema for output documentation.
            tags: Searchable tags.
            timeout_ms: Execution timeout in milliseconds (100–300000).
            memory_mb: Memory limit in MB (64–2048).
            scope: ``"project"`` or ``"global"``.
            project_id: Override the default project_id.

        Returns:
            A :class:`ToolInfo` with the stored tool definition.
        """
        body: dict[str, Any] = {
            "name": name,
            "description": description or f"Tool {name}",
            "source_code": source_code,
            "entry_point": entry_point,
            "requirements": requirements or [],
            "tags": tags or [],
            "timeout_ms": timeout_ms,
            "memory_mb": memory_mb,
            "scope": scope,
        }
        if input_schema is not None:
            body["input_schema"] = input_schema
        if output_schema is not None:
            body["output_schema"] = output_schema

        r = await self._request(
            "POST",
            self._project_toolbox_path("/push", project_id),
            json=body,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=project_id or self.project_id)
        return ToolInfo.from_api_response(r.json())

    async def register_tool(
        self,
        name: str,
        endpoint_url: str,
        *,
        description: str = "",
        http_method: str = "POST",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        auth_type: str = "none",
        tags: list[str] | None = None,
        scope: str = "project",
        project_id: str | None = None,
    ) -> ToolInfo:
        """Register an external HTTP tool.

        Args:
            name: Tool name.
            endpoint_url: URL to proxy invocations to.
            description: Human-readable description.
            http_method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            input_schema: JSON Schema for input validation.
            output_schema: JSON Schema for output documentation.
            auth_type: Auth type (none, api_key, bearer, oauth2).
            tags: Searchable tags.
            scope: ``"project"`` or ``"global"``.
            project_id: Override the default project_id.

        Returns:
            A :class:`ToolInfo` with the registered tool definition.
        """
        body: dict[str, Any] = {
            "name": name,
            "description": description or f"External tool {name}",
            "endpoint_url": endpoint_url,
            "http_method": http_method,
            "input_schema": input_schema or {"type": "object"},
            "auth_type": auth_type,
            "tags": tags or [],
            "scope": scope,
        }
        if output_schema is not None:
            body["output_schema"] = output_schema

        r = await self._request(
            "POST",
            self._project_toolbox_path("/register", project_id),
            json=body,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=project_id or self.project_id)
        return ToolInfo.from_api_response(r.json())

    async def invoke_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any] | None = None,
        *,
        version: int | None = None,
        project_id: str | None = None,
    ) -> InvocationResult:
        """Invoke a tool by name.

        Routes to the correct executor (subprocess for hosted, HTTP proxy
        for external) based on the tool's type.

        Args:
            tool_name: Name of the tool to invoke.
            input_data: Input arguments as a dict.
            version: Specific version to invoke (latest if not specified).
            project_id: Override the default project_id.

        Returns:
            An :class:`InvocationResult` with success/error and the result.
        """
        pid = project_id or self.project_id
        body: dict[str, Any] = {
            "input": input_data or {},
        }
        if version is not None:
            body["version"] = version

        r = await self._request(
            "POST",
            self._toolbox_path(f"/projects/{pid}/invoke/{tool_name}"),
            json=body,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return InvocationResult.from_api_response(r.json())

    async def list_tools(
        self,
        *,
        tags: list[str] | None = None,
        tool_type: str | None = None,
        scope: str | None = None,
        status: str | None = None,
        include_global: bool = True,
        project_id: str | None = None,
    ) -> ToolList:
        """List tools available to a project.

        Args:
            tags: Filter by tags (comma-joined).
            tool_type: Filter by type (hosted, external, mcp).
            scope: Filter by scope (project, global).
            status: Filter by status (active, building, build_failed, etc.).
            include_global: Include global-scoped tools (default True).
            project_id: Override the default project_id.

        Returns:
            A :class:`ToolList` with matching tools.
        """
        params: dict[str, Any] = {"include_global": include_global}
        if tags:
            params["tags"] = ",".join(tags)
        if tool_type:
            params["tool_type"] = tool_type
        if scope:
            params["scope"] = scope
        if status:
            params["status"] = status

        r = await self._request(
            "GET",
            self._project_toolbox_path("", project_id),
            params=params,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=project_id or self.project_id)
        return ToolList.from_api_response(r.json())

    async def get_tool(
        self,
        tool_id: str,
        *,
        project_id: str | None = None,
    ) -> ToolInfo:
        """Get a tool by ID.

        Args:
            tool_id: The tool UUID.
            project_id: Override the default project_id.

        Returns:
            A :class:`ToolInfo` with the tool definition.
        """
        r = await self._request(
            "GET",
            self._project_toolbox_path(f"/{tool_id}", project_id),
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=project_id or self.project_id)
        return ToolInfo.from_api_response(r.json())

    async def update_tool(
        self,
        tool_id: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        timeout_ms: int | None = None,
        memory_mb: int | None = None,
        scope: str | None = None,
        status: str | None = None,
        project_id: str | None = None,
    ) -> ToolInfo:
        """Update tool metadata (non-code fields).

        Args:
            tool_id: The tool UUID.
            description: New description.
            tags: New tags list.
            timeout_ms: New timeout.
            memory_mb: New memory limit.
            scope: New scope (project/global).
            status: New status (active/deprecated/disabled).
            project_id: Override the default project_id.

        Returns:
            A :class:`ToolInfo` with the updated tool.
        """
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if tags is not None:
            body["tags"] = tags
        if timeout_ms is not None:
            body["timeout_ms"] = timeout_ms
        if memory_mb is not None:
            body["memory_mb"] = memory_mb
        if scope is not None:
            body["scope"] = scope
        if status is not None:
            body["status"] = status

        r = await self._request(
            "PUT",
            self._project_toolbox_path(f"/{tool_id}", project_id),
            json=body,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=project_id or self.project_id)
        return ToolInfo.from_api_response(r.json())

    async def delete_tool(
        self,
        tool_id: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Soft-delete a tool.

        Args:
            tool_id: The tool UUID.
            project_id: Override the default project_id.
        """
        r = await self._request(
            "DELETE",
            self._project_toolbox_path(f"/{tool_id}", project_id),
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=project_id or self.project_id)

    async def import_openapi(
        self,
        namespace: str,
        *,
        spec_url: str | None = None,
        spec_body: dict[str, Any] | None = None,
        base_url: str | None = None,
        tags: list[str] | None = None,
        auth_type: str = "none",
        scope: str = "project",
        project_id: str | None = None,
    ) -> ImportResult:
        """Import tools from an OpenAPI spec.

        Provide either *spec_url* (to fetch remotely) or *spec_body*
        (inline JSON).  Each endpoint in the spec becomes an external
        tool with the given *namespace* prefix.

        Args:
            namespace: Prefix for tool names (e.g. ``"stripe"``).
            spec_url: URL to fetch the OpenAPI spec from.
            spec_body: Inline OpenAPI spec as a dict.
            base_url: Override the base URL for endpoints.
            tags: Tags to apply to imported tools.
            auth_type: Auth type (none, api_key, bearer, oauth2).
            scope: ``"project"`` or ``"global"``.
            project_id: Override the default project_id.

        Returns:
            An :class:`ImportResult` with created tools and counts.
        """
        body: dict[str, Any] = {
            "namespace": namespace,
            "auth_type": auth_type,
            "scope": scope,
        }
        if spec_url is not None:
            body["spec_url"] = spec_url
        if spec_body is not None:
            body["spec_body"] = spec_body
        if base_url is not None:
            body["base_url"] = base_url
        if tags is not None:
            body["tags"] = tags

        r = await self._request(
            "POST",
            self._project_toolbox_path("/import-openapi", project_id),
            json=body,
            service="toolbox",
        )
        self._raise_for_status(
            r, service="toolbox", project_id=project_id or self.project_id,
        )
        return ImportResult.from_api_response(r.json())

    # ==================================================================
    # ToolBox-S — Credentials
    # ==================================================================

    async def set_tool_credential(
        self,
        tool_definition_id: str,
        credential_type: str,
        name: str,
        credential_payload: dict[str, Any],
        *,
        expires_at: str | None = None,
        project_id: str | None = None,
    ) -> CredentialInfo:
        """Store an encrypted credential for a tool.

        Args:
            tool_definition_id: The tool definition UUID this credential belongs to.
            credential_type: Credential type (e.g. ``api_key``, ``oauth2``).
            name: Human-readable credential name.
            credential_payload: Secret key-value pairs (encrypted server-side).
            expires_at: Optional ISO-8601 expiration timestamp.
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        body: dict[str, Any] = {
            "tool_definition_id": tool_definition_id,
            "credential_type": credential_type,
            "name": name,
            "credential_payload": credential_payload,
        }
        if expires_at is not None:
            body["expires_at"] = expires_at

        r = await self._request(
            "POST",
            self._toolbox_path(f"/projects/{pid}/credentials"),
            json=body,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return CredentialInfo.from_api_response(r.json())

    async def list_credentials(
        self,
        *,
        tool_definition_id: str | None = None,
        project_id: str | None = None,
    ) -> CredentialList:
        """List credentials for the current project.

        Args:
            tool_definition_id: Optional filter by tool definition.
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        params: dict[str, Any] = {}
        if tool_definition_id:
            params["tool_definition_id"] = tool_definition_id

        r = await self._request(
            "GET",
            self._toolbox_path(f"/projects/{pid}/credentials"),
            params=params,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return CredentialList.from_api_response(r.json())

    async def revoke_credential(
        self,
        credential_id: str,
        version: int,
        *,
        project_id: str | None = None,
    ) -> CredentialInfo:
        """Revoke a credential.

        Args:
            credential_id: The credential UUID.
            version: Optimistic concurrency version.
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        r = await self._request(
            "POST",
            self._toolbox_path(f"/projects/{pid}/credentials/{credential_id}/revoke"),
            json={"version": version},
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return CredentialInfo.from_api_response(r.json())

    # ==================================================================
    # ToolBox-S — MCP Servers
    # ==================================================================

    async def register_mcp_server(
        self,
        name: str,
        transport_type: str,
        url: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        scope: str = "project",
        project_id: str | None = None,
    ) -> MCPServerInfo:
        """Register an MCP server.

        Args:
            name: Server name.
            transport_type: Transport type (e.g. ``sse``, ``stdio``).
            url: Server URL or connection string.
            description: Optional description.
            tags: Optional tags.
            scope: Visibility scope (default ``project``).
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        r = await self._request(
            "POST",
            self._toolbox_path(f"/projects/{pid}/mcp-servers"),
            json={
                "name": name,
                "description": description,
                "transport_type": transport_type,
                "url": url,
                "tags": tags or [],
                "scope": scope,
            },
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return MCPServerInfo.from_api_response(r.json())

    async def list_mcp_servers(
        self,
        *,
        status: str | None = None,
        tags: list[str] | None = None,
        project_id: str | None = None,
    ) -> MCPServerList:
        """List MCP servers.

        Args:
            status: Optional filter by health status.
            tags: Optional filter by tags.
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = ",".join(tags)

        r = await self._request(
            "GET",
            self._toolbox_path(f"/projects/{pid}/mcp-servers"),
            params=params,
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return MCPServerList.from_api_response(r.json())

    async def get_mcp_server(
        self,
        server_id: str,
        *,
        project_id: str | None = None,
    ) -> MCPServerInfo:
        """Get MCP server by ID.

        Args:
            server_id: The MCP server UUID.
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        r = await self._request(
            "GET",
            self._toolbox_path(f"/projects/{pid}/mcp-servers/{server_id}"),
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)
        return MCPServerInfo.from_api_response(r.json())

    async def delete_mcp_server(
        self,
        server_id: str,
        *,
        project_id: str | None = None,
    ) -> None:
        """Delete (soft) an MCP server.

        Args:
            server_id: The MCP server UUID.
            project_id: Override the default project_id.
        """
        pid = project_id or self.project_id
        r = await self._request(
            "DELETE",
            self._toolbox_path(f"/projects/{pid}/mcp-servers/{server_id}"),
            service="toolbox",
        )
        self._raise_for_status(r, service="toolbox", project_id=pid)

    async def health_toolbox(self) -> dict[str, Any]:
        """Check ToolBox-S service health."""
        r = await self._request(
            "GET", self._toolbox_path("/health"), service="toolbox",
        )
        self._raise_for_status(r, service="toolbox")
        return r.json()
