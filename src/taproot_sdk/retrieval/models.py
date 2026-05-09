"""
Typed response models for Retrieval-S API.

Field names match the Retrieval-S Pydantic response models exactly
(src/retrieval_service/api/v1/endpoints/ in Retrieval-S).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ===================================================================
# Store Models
# ===================================================================


@dataclass(frozen=True)
class IndexParams:
    """HNSW / IVFFlat index configuration."""

    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    ivfflat_lists: int = 100
    default_ef_search: int = 100

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexParams:
        return cls(
            hnsw_m=data.get("hnsw_m", 16),
            hnsw_ef_construction=data.get("hnsw_ef_construction", 128),
            ivfflat_lists=data.get("ivfflat_lists", 100),
            default_ef_search=data.get("default_ef_search", 100),
        )


@dataclass(frozen=True)
class StoreInfo:
    """Full store configuration and metadata."""

    id: str
    name: str
    table_name: str
    display_name: str | None
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    index_type: str
    index_params: IndexParams
    use_halfvec: bool
    enable_fulltext: bool
    created_at: str
    is_active: bool

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> StoreInfo:
        raw_params = data.get("index_params") or {}
        return cls(
            id=str(data["id"]),
            name=data["name"],
            table_name=data.get("table_name", f"{data['name']}_vector_store"),
            display_name=data.get("display_name"),
            embedding_provider=data.get("embedding_provider", "openai"),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            embedding_dimensions=data.get("embedding_dimensions", 0),
            index_type=data.get("index_type", "hnsw"),
            index_params=IndexParams.from_dict(raw_params),
            use_halfvec=data.get("use_halfvec", True),
            enable_fulltext=data.get("enable_fulltext", False),
            created_at=data.get("created_at", ""),
            is_active=data.get("is_active", True),
        )


@dataclass(frozen=True)
class StoreList:
    """Paginated list of stores."""

    stores: tuple[StoreInfo, ...]
    total: int
    offset: int
    limit: int
    has_more: bool

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> StoreList:
        raw_stores = data.get("stores", [])
        return cls(
            stores=tuple(StoreInfo.from_api_response(s) for s in raw_stores),
            total=data.get("total", len(raw_stores)),
            offset=data.get("offset", 0),
            limit=data.get("limit", 50),
            has_more=data.get("has_more", False),
        )


@dataclass(frozen=True)
class StoreCreated:
    """Response from creating a store."""

    store: StoreInfo
    message: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> StoreCreated:
        raw_store = data.get("store", data)
        return cls(
            store=StoreInfo.from_api_response(raw_store),
            message=data.get("message", ""),
        )


@dataclass(frozen=True)
class StoreDeleted:
    """Response from deleting a store."""

    success: bool
    message: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> StoreDeleted:
        return cls(
            success=data.get("success", True),
            message=data.get("message", ""),
        )


@dataclass(frozen=True)
class StoreStatistics:
    """Store usage statistics."""

    store_name: str
    chunk_count: int
    document_count: int
    total_size_bytes: int
    embedding_dimensions: int
    query_count: int
    ingestion_count: int
    last_queried_at: str | None
    last_ingested_at: str | None
    last_recount_at: str | None
    precision: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> StoreStatistics:
        return cls(
            store_name=data.get("store_name", ""),
            chunk_count=data.get("chunk_count", 0),
            document_count=data.get("document_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            embedding_dimensions=data.get("embedding_dimensions", 0),
            query_count=data.get("query_count", 0),
            ingestion_count=data.get("ingestion_count", 0),
            last_queried_at=data.get("last_queried_at"),
            last_ingested_at=data.get("last_ingested_at"),
            last_recount_at=data.get("last_recount_at"),
            precision=data.get("precision", "approximate"),
        )


# ===================================================================
# Access Control Models
# ===================================================================


@dataclass(frozen=True)
class AccessGrant:
    """A single access grant for an API key on a store."""

    api_key_id: str
    store_id: str
    access_level: str
    granted_at: str
    granted_by_api_key_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessGrant:
        return cls(
            api_key_id=data["api_key_id"],
            store_id=str(data.get("store_id", "")),
            access_level=data.get("access_level", "read_write"),
            granted_at=data.get("granted_at", ""),
            granted_by_api_key_id=data.get("granted_by_api_key_id", ""),
        )


@dataclass(frozen=True)
class AccessGranted:
    """Response from granting access."""

    access: AccessGrant
    message: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> AccessGranted:
        return cls(
            access=AccessGrant.from_dict(data.get("access", data)),
            message=data.get("message", ""),
        )


@dataclass(frozen=True)
class AccessList:
    """List of access grants for a store."""

    access: tuple[AccessGrant, ...]
    count: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> AccessList:
        raw = data.get("access", [])
        return cls(
            access=tuple(AccessGrant.from_dict(a) for a in raw),
            count=data.get("count", len(raw)),
        )


@dataclass(frozen=True)
class AccessRevoked:
    """Response from revoking access."""

    success: bool
    message: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> AccessRevoked:
        return cls(
            success=data.get("success", True),
            message=data.get("message", ""),
        )


# ===================================================================
# Query Models
# ===================================================================


@dataclass(frozen=True)
class QueryHit:
    """A single search result from a vector query."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryHit:
        return cls(
            chunk_id=data.get("chunk_id", ""),
            score=data.get("score", 0.0),
            text=data.get("text", ""),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class QueryResponse:
    """Response from a retrieval query."""

    hits: tuple[QueryHit, ...]
    total_hits: int
    search_time_ms: float

    @property
    def is_empty(self) -> bool:
        """True when no results were returned."""
        return len(self.hits) == 0

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> QueryResponse:
        raw_hits = data.get("hits", [])
        return cls(
            hits=tuple(QueryHit.from_dict(h) for h in raw_hits),
            total_hits=data.get("total_hits", len(raw_hits)),
            search_time_ms=data.get("search_time_ms", 0.0),
        )


# ===================================================================
# Ingestion Models
# ===================================================================


@dataclass(frozen=True)
class IngestionJob:
    """Handle returned when a document ingestion is triggered."""

    job_id: str
    status: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> IngestionJob:
        return cls(
            job_id=data.get("job_id", data.get("id", "")),
            status=data.get("status", "pending"),
        )


@dataclass(frozen=True)
class JobDetail:
    """Full detail for a single ingestion job."""

    job_id: str
    status: str
    source_uri: str
    index: str
    pipeline_id: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    documents_processed: int
    chunks_created: int

    @property
    def is_terminal(self) -> bool:
        """True when the job has reached a final state."""
        return self.status in ("completed", "failed", "cancelled")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> JobDetail:
        return cls(
            job_id=data.get("job_id", data.get("id", "")),
            status=data.get("status", ""),
            source_uri=data.get("source_uri", ""),
            index=data.get("index", ""),
            pipeline_id=data.get("pipeline_id", "default"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            documents_processed=data.get("documents_processed", 0),
            chunks_created=data.get("chunks_created", 0),
        )


@dataclass(frozen=True)
class JobSummary:
    """Summary of an ingestion job (used in list responses)."""

    job_id: str
    status: str
    store_name: str
    source_uri: str
    pipeline_id: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    documents_processed: int
    chunks_created: int
    batch_id: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSummary:
        return cls(
            job_id=data.get("job_id", data.get("id", "")),
            status=data.get("status", ""),
            store_name=data.get("store_name", ""),
            source_uri=data.get("source_uri", ""),
            pipeline_id=data.get("pipeline_id", "default"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            documents_processed=data.get("documents_processed", 0),
            chunks_created=data.get("chunks_created", 0),
            batch_id=data.get("batch_id"),
        )


@dataclass(frozen=True)
class JobList:
    """Paginated list of ingestion jobs."""

    items: tuple[JobSummary, ...]
    total: int
    offset: int
    limit: int
    has_more: bool

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> JobList:
        raw = data.get("items", [])
        return cls(
            items=tuple(JobSummary.from_dict(j) for j in raw),
            total=data.get("total", len(raw)),
            offset=data.get("offset", 0),
            limit=data.get("limit", 50),
            has_more=data.get("has_more", False),
        )


@dataclass(frozen=True)
class JobCancelled:
    """Response from cancelling an ingestion job."""

    job_id: str
    status: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> JobCancelled:
        return cls(
            job_id=data.get("job_id", data.get("id", "")),
            status=data.get("status", "cancelled"),
        )


# ===================================================================
# Batch Ingestion Models
# ===================================================================


@dataclass(frozen=True)
class BatchProgress:
    """Progress breakdown for a batch job."""

    total: int
    pending: int
    in_progress: int
    completed: int
    failed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchProgress:
        return cls(
            total=data.get("total", 0),
            pending=data.get("pending", 0),
            in_progress=data.get("in_progress", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
        )


@dataclass(frozen=True)
class BatchResults:
    """Aggregate results for a batch job."""

    total_chunks_created: int
    cancelled_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchResults:
        return cls(
            total_chunks_created=data.get("total_chunks_created", 0),
            cancelled_count=data.get("cancelled_count", 0),
        )


@dataclass(frozen=True)
class BatchErrorSample:
    """A sample error from a batch job."""

    job_id: str
    source_uri: str
    error: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchErrorSample:
        return cls(
            job_id=data.get("job_id", ""),
            source_uri=data.get("source_uri", ""),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class BatchCreated:
    """Response from creating a batch ingestion job."""

    batch_id: str
    status: str
    created_at: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> BatchCreated:
        return cls(
            batch_id=data.get("batch_id", data.get("id", "")),
            status=data.get("status", "listing"),
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True)
class BatchStatus:
    """Full status of a batch ingestion job."""

    batch_id: str
    status: str
    source_type: str
    source_summary: str
    progress: BatchProgress
    results: BatchResults
    errors_sample: tuple[BatchErrorSample, ...]
    created_at: str
    updated_at: str
    completed_at: str | None

    @property
    def is_terminal(self) -> bool:
        """True when the batch has reached a final state."""
        return self.status in ("completed", "cancelled", "failed")

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> BatchStatus:
        return cls(
            batch_id=data.get("batch_id", data.get("id", "")),
            status=data.get("status", ""),
            source_type=data.get("source_type", ""),
            source_summary=data.get("source_summary", ""),
            progress=BatchProgress.from_dict(data.get("progress") or {}),
            results=BatchResults.from_dict(data.get("results") or {}),
            errors_sample=tuple(
                BatchErrorSample.from_dict(e)
                for e in data.get("errors_sample", [])
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            completed_at=data.get("completed_at"),
        )


@dataclass(frozen=True)
class BatchJobInfo:
    """Summary of a single job within a batch."""

    job_id: str
    status: str
    source_uri: str
    created_at: str
    completed_at: str | None
    error: str | None
    chunks_created: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJobInfo:
        return cls(
            job_id=data.get("job_id", data.get("id", "")),
            status=data.get("status", ""),
            source_uri=data.get("source_uri", ""),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            chunks_created=data.get("chunks_created", 0),
        )


@dataclass(frozen=True)
class BatchJobList:
    """Paginated list of jobs within a batch."""

    batch_id: str
    jobs: tuple[BatchJobInfo, ...]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> BatchJobList:
        raw = data.get("jobs", [])
        return cls(
            batch_id=data.get("batch_id", ""),
            jobs=tuple(BatchJobInfo.from_dict(j) for j in raw),
            total=data.get("total", len(raw)),
            limit=data.get("limit", 50),
            offset=data.get("offset", 0),
        )


@dataclass(frozen=True)
class BatchCancelled:
    """Response from cancelling a batch."""

    batch_id: str
    status: str
    cancelled_jobs: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> BatchCancelled:
        return cls(
            batch_id=data.get("batch_id", data.get("id", "")),
            status=data.get("status", "cancelled"),
            cancelled_jobs=data.get("cancelled_jobs", 0),
        )


# ===================================================================
# Document Models
# ===================================================================


@dataclass(frozen=True)
class DocumentInfo:
    """Summary of a document (used in list responses)."""

    doc_id: str
    chunk_count: int
    source: str | None
    first_ingested_at: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentInfo:
        return cls(
            doc_id=data.get("doc_id", ""),
            chunk_count=data.get("chunk_count", 0),
            source=data.get("source"),
            first_ingested_at=data.get("first_ingested_at"),
        )


@dataclass(frozen=True)
class DocumentDetail:
    """Full document details."""

    doc_id: str
    store_name: str
    chunk_count: int
    source: str | None
    first_ingested_at: str | None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> DocumentDetail:
        return cls(
            doc_id=data.get("doc_id", ""),
            store_name=data.get("store_name", ""),
            chunk_count=data.get("chunk_count", 0),
            source=data.get("source"),
            first_ingested_at=data.get("first_ingested_at"),
        )


@dataclass(frozen=True)
class DocumentList:
    """Paginated list of documents."""

    documents: tuple[DocumentInfo, ...]
    total: int
    offset: int
    limit: int
    has_more: bool

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> DocumentList:
        raw = data.get("documents", [])
        return cls(
            documents=tuple(DocumentInfo.from_dict(d) for d in raw),
            total=data.get("total", len(raw)),
            offset=data.get("offset", 0),
            limit=data.get("limit", 20),
            has_more=data.get("has_more", False),
        )


@dataclass(frozen=True)
class DocumentDeleted:
    """Response from deleting a document."""

    doc_id: str
    store_name: str
    chunks_deleted: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> DocumentDeleted:
        return cls(
            doc_id=data.get("doc_id", ""),
            store_name=data.get("store_name", ""),
            chunks_deleted=data.get("chunks_deleted", 0),
        )


@dataclass(frozen=True)
class DocumentOperationResult:
    """Response from a document create, replace, or selector-delete operation."""

    operation_id: str
    operation: str
    status: str
    job_id: str | None
    document_id: str | None
    doc_id: str | None
    polling_url: str | None
    idempotency_key: str | None

    @property
    def is_terminal(self) -> bool:
        """True when the operation no longer has queued work."""
        return self.status in ("completed", "failed")

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> DocumentOperationResult:
        return cls(
            operation_id=data.get("operation_id", ""),
            operation=data.get("operation", ""),
            status=data.get("status", ""),
            job_id=data.get("job_id"),
            document_id=data.get("document_id"),
            doc_id=data.get("doc_id"),
            polling_url=data.get("polling_url"),
            idempotency_key=data.get("idempotency_key"),
        )


# ===================================================================
# Chunk Models
# ===================================================================


@dataclass(frozen=True)
class ChunkInfo:
    """A single chunk with content and metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    store_name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkInfo:
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata") or {},
            store_name=data.get("store_name", ""),
        )


@dataclass(frozen=True)
class ChunkList:
    """Paginated list of chunks."""

    chunks: tuple[ChunkInfo, ...]
    total: int
    offset: int
    limit: int
    has_more: bool

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ChunkList:
        raw = data.get("chunks", [])
        return cls(
            chunks=tuple(ChunkInfo.from_dict(c) for c in raw),
            total=data.get("total", len(raw)),
            offset=data.get("offset", 0),
            limit=data.get("limit", 20),
            has_more=data.get("has_more", False),
        )


@dataclass(frozen=True)
class ChunksUploaded:
    """Response from uploading chunks."""

    store_name: str
    doc_id: str
    chunk_count: int
    status: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ChunksUploaded:
        return cls(
            store_name=data.get("store_name", ""),
            doc_id=data.get("doc_id", ""),
            chunk_count=data.get("chunk_count", 0),
            status=data.get("status", "success"),
        )


@dataclass(frozen=True)
class ChunksDeleted:
    """Response from deleting chunks by document."""

    store_name: str
    doc_id: str
    chunks_deleted: int
    status: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ChunksDeleted:
        return cls(
            store_name=data.get("store_name", ""),
            doc_id=data.get("doc_id", ""),
            chunks_deleted=data.get("chunks_deleted", 0),
            status=data.get("status", "success"),
        )
