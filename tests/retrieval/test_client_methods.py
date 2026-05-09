"""Tests for TaprootClient retrieval methods (mocked HTTP)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.exceptions import AuthenticationError, TaprootAPIError, ValidationError
from taproot_sdk.retrieval.models import (
    BatchCancelled,
    BatchCreated,
    BatchStatus,
    ChunkInfo,
    ChunkList,
    ChunksDeleted,
    ChunksUploaded,
    DocumentDeleted,
    DocumentDetail,
    DocumentList,
    DocumentOperationResult,
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

BASE = "https://gateway.test"


def _client() -> TaprootClient:
    return TaprootClient(
        base_url=BASE,
        api_key="test-key",
        project_id="proj-1",
    )


def _json_resp(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


# ==================================================================
# Store Management
# ==================================================================


class TestCreateStore:
    @respx.mock
    @pytest.mark.asyncio
    async def test_create_store_returns_typed(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores").mock(
            return_value=_json_resp({
                "store": {"id": "s-1", "name": "my_store"},
                "message": "Store created",
            })
        )
        c = _client()
        result = await c.create_store("my_store", display_name="My Store")
        assert isinstance(result, StoreCreated)
        assert result.store.name == "my_store"
        assert result.message == "Store created"

    @respx.mock
    @pytest.mark.asyncio
    async def test_create_store_conflict_raises(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores").mock(
            return_value=httpx.Response(409, json={"detail": "already exists"})
        )
        c = _client()
        with pytest.raises(Exception, match="already exists"):
            await c.create_store("dup_store")


class TestListStores:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_stores(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores").mock(
            return_value=_json_resp({
                "stores": [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}],
                "total": 2,
                "offset": 0,
                "limit": 50,
                "has_more": False,
            })
        )
        c = _client()
        result = await c.list_stores()
        assert isinstance(result, StoreList)
        assert len(result.stores) == 2


class TestGetStore:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_store(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s-1").mock(
            return_value=_json_resp({"id": "s-1", "name": "test_store"})
        )
        c = _client()
        result = await c.get_store("s-1")
        assert isinstance(result, StoreInfo)
        assert result.id == "s-1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_store_not_found(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/missing").mock(
            return_value=httpx.Response(404, json={"detail": "Store not found"})
        )
        c = _client()
        with pytest.raises(TaprootAPIError):
            await c.get_store("missing")


class TestDeleteStore:
    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_store(self):
        respx.delete(f"{BASE}/api/v1/retrieval/api/v1/stores/s-1").mock(
            return_value=_json_resp({"success": True, "message": "Store deleted"})
        )
        c = _client()
        result = await c.delete_store("s-1")
        assert isinstance(result, StoreDeleted)
        assert result.success is True


class TestGetStoreStats:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_store_stats(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/my_store/stats").mock(
            return_value=_json_resp({
                "store_name": "my_store",
                "chunk_count": 500,
                "document_count": 10,
                "total_size_bytes": 1_000_000,
                "embedding_dimensions": 1536,
                "query_count": 42,
                "ingestion_count": 5,
            })
        )
        c = _client()
        result = await c.get_store_stats("my_store")
        assert isinstance(result, StoreStatistics)
        assert result.chunk_count == 500


# ==================================================================
# Query
# ==================================================================


class TestRetrievalQuery:
    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_query(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/test_store/query").mock(
            return_value=_json_resp({
                "hits": [
                    {"chunk_id": "c1", "score": 0.95, "text": "relevant"},
                ],
                "total_hits": 1,
                "search_time_ms": 8.2,
            })
        )
        c = _client()
        result = await c.retrieval_query("test_store", "what is taproot?")
        assert isinstance(result, QueryResponse)
        assert len(result.hits) == 1
        assert result.hits[0].score == 0.95
        assert result.is_empty is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_auth_error(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/private/query").mock(
            return_value=httpx.Response(403, json={"detail": "Access denied"})
        )
        c = _client()
        with pytest.raises(AuthenticationError):
            await c.retrieval_query("private", "secret query")

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_with_hybrid_mode(self):
        route = respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/query").mock(
            return_value=_json_resp({"hits": [], "total_hits": 0, "search_time_ms": 1.0})
        )
        c = _client()
        result = await c.retrieval_query(
            "s", "test", search_mode="hybrid", keyword_weight=0.3, rerank=True,
        )
        assert result.is_empty is True
        body = route.calls[0].request.content
        assert b"hybrid" in body
        assert b"keyword_weight" in body
        assert b"rerank" in body


# ==================================================================
# Document Ingestion
# ==================================================================


class TestIngestDocument:
    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest").mock(
            return_value=_json_resp({"job_id": "j-1", "status": "pending"})
        )
        c = _client()
        result = await c.ingest_document(
            "s", index="s", source_uri="s3://bucket/doc.pdf",
        )
        assert isinstance(result, IngestionJob)
        assert result.job_id == "j-1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_validation_error(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest").mock(
            return_value=httpx.Response(422, json={
                "detail": [{"loc": ["body", "source_uri"], "msg": "required"}],
            })
        )
        c = _client()
        with pytest.raises(ValidationError):
            await c.ingest_document("s", index="s")


class TestGetIngestionJob:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_job(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest/j-1").mock(
            return_value=_json_resp({
                "job_id": "j-1",
                "status": "completed",
                "source_uri": "s3://b/f",
                "index": "s",
                "pipeline_id": "default",
                "created_at": "2025-01-01T00:00:00Z",
                "documents_processed": 1,
                "chunks_created": 25,
            })
        )
        c = _client()
        result = await c.get_ingestion_job("s", "j-1")
        assert isinstance(result, JobDetail)
        assert result.chunks_created == 25
        assert result.is_terminal is True


class TestListIngestionJobs:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_jobs(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest/jobs").mock(
            return_value=_json_resp({
                "items": [{"job_id": "j-1", "status": "completed"}],
                "total": 1,
                "offset": 0,
                "limit": 50,
                "has_more": False,
            })
        )
        c = _client()
        result = await c.list_ingestion_jobs("s")
        assert isinstance(result, JobList)
        assert len(result.items) == 1


class TestCancelIngestionJob:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cancel(self):
        respx.delete(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest/j-1").mock(
            return_value=_json_resp({"job_id": "j-1", "status": "cancelled"})
        )
        c = _client()
        result = await c.cancel_ingestion_job("s", "j-1")
        assert isinstance(result, JobCancelled)
        assert result.status == "cancelled"


# ==================================================================
# Batch Ingestion
# ==================================================================


class TestBatchIngest:
    @respx.mock
    @pytest.mark.asyncio
    async def test_batch_ingest(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest/batch").mock(
            return_value=_json_resp({
                "batch_id": "b-1",
                "status": "listing",
                "created_at": "2025-01-01T00:00:00Z",
            })
        )
        c = _client()
        result = await c.batch_ingest(
            "s", source={"type": "s3_prefix", "uri": "s3://bucket/docs/"},
        )
        assert isinstance(result, BatchCreated)
        assert result.batch_id == "b-1"


class TestGetBatchStatus:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_status(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest/batch/b-1").mock(
            return_value=_json_resp({
                "batch_id": "b-1",
                "status": "completed",
                "source_type": "s3_prefix",
                "source_summary": "s3://b/d/",
                "progress": {"total": 5, "completed": 5},
                "results": {"total_chunks_created": 100},
                "errors_sample": [],
                "created_at": "",
                "updated_at": "",
            })
        )
        c = _client()
        result = await c.get_batch_status("s", "b-1")
        assert isinstance(result, BatchStatus)
        assert result.progress.completed == 5


class TestCancelBatch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cancel(self):
        respx.delete(f"{BASE}/api/v1/retrieval/api/v1/stores/s/ingest/batch/b-1").mock(
            return_value=_json_resp({
                "batch_id": "b-1",
                "status": "cancelled",
                "cancelled_jobs": 3,
            })
        )
        c = _client()
        result = await c.cancel_batch("s", "b-1")
        assert isinstance(result, BatchCancelled)
        assert result.cancelled_jobs == 3


# ==================================================================
# Document Management
# ==================================================================


class TestListDocuments:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_documents(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/documents").mock(
            return_value=_json_resp({
                "documents": [{"doc_id": "d1"}, {"doc_id": "d2"}],
                "total": 2,
                "offset": 0,
                "limit": 20,
                "has_more": False,
            })
        )
        c = _client()
        result = await c.list_documents("s")
        assert isinstance(result, DocumentList)
        assert len(result.documents) == 2


class TestGetDocument:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_document(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/documents/d-1").mock(
            return_value=_json_resp({
                "doc_id": "d-1",
                "store_name": "s",
                "chunk_count": 10,
            })
        )
        c = _client()
        result = await c.get_document("s", "d-1")
        assert isinstance(result, DocumentDetail)
        assert result.chunk_count == 10


class TestDeleteDocument:
    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_document(self):
        respx.delete(f"{BASE}/api/v1/retrieval/api/v1/stores/s/documents/d-1").mock(
            return_value=_json_resp({
                "doc_id": "d-1",
                "store_name": "s",
                "chunks_deleted": 10,
            })
        )
        c = _client()
        result = await c.delete_document("s", "d-1")
        assert isinstance(result, DocumentDeleted)
        assert result.chunks_deleted == 10


class TestDocumentOperations:
    @respx.mock
    @pytest.mark.asyncio
    async def test_create_document_operation(self):
        route = respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/documents").mock(
            return_value=_json_resp({
                "operation_id": "idem-1",
                "operation": "create",
                "status": "queued",
                "job_id": "j-1",
                "doc_id": "d-1",
                "idempotency_key": "idem-1",
            })
        )
        c = _client()
        result = await c.create_document(
            "s",
            doc_id="d-1",
            source_uri="s3://bucket/doc.pdf",
            idempotency_key="idem-1",
        )

        assert isinstance(result, DocumentOperationResult)
        assert result.operation == "create"
        assert result.job_id == "j-1"
        assert route.calls[0].request.headers["Idempotency-Key"] == "idem-1"
        assert b'"doc_id":"d-1"' in route.calls[0].request.content
        assert b'"source_uri":"s3://bucket/doc.pdf"' in route.calls[0].request.content

    @respx.mock
    @pytest.mark.asyncio
    async def test_replace_document_operation(self):
        route = respx.put(f"{BASE}/api/v1/retrieval/api/v1/stores/s/documents").mock(
            return_value=_json_resp({
                "operation_id": "op-1",
                "operation": "replace",
                "status": "queued",
                "job_id": "j-1",
                "doc_id": "d-1",
            })
        )
        c = _client()
        result = await c.replace_document(
            "s",
            doc_id="d-1",
            source_uri="s3://bucket/new.pdf",
        )

        assert isinstance(result, DocumentOperationResult)
        assert result.operation == "replace"
        assert b'"selector":{"doc_id":"d-1"}' in route.calls[0].request.content

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_document_by_selector_operation(self):
        route = respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/documents/delete").mock(
            return_value=_json_resp({
                "operation_id": "op-1",
                "operation": "delete",
                "status": "completed",
                "doc_id": "d-1",
            })
        )
        c = _client()
        result = await c.delete_document_by_selector("s", filename="doc.pdf")

        assert result.is_terminal is True
        assert b'"selector":{"filename":"doc.pdf"}' in route.calls[0].request.content


# ==================================================================
# Chunk Management
# ==================================================================


class TestUploadChunks:
    @respx.mock
    @pytest.mark.asyncio
    async def test_upload(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/chunks").mock(
            return_value=_json_resp({
                "store_name": "s",
                "doc_id": "d1",
                "chunk_count": 3,
                "status": "success",
            })
        )
        c = _client()
        result = await c.upload_chunks("s", "d1", [
            {"content": "chunk 1"},
            {"content": "chunk 2"},
            {"content": "chunk 3"},
        ])
        assert isinstance(result, ChunksUploaded)
        assert result.chunk_count == 3


class TestListChunks:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_chunks(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/chunks").mock(
            return_value=_json_resp({
                "chunks": [{"id": "c1", "content": "text"}],
                "total": 1,
                "offset": 0,
                "limit": 20,
                "has_more": False,
            })
        )
        c = _client()
        result = await c.list_chunks("s")
        assert isinstance(result, ChunkList)
        assert len(result.chunks) == 1


class TestGetChunk:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_chunk(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/s/chunks/c-1").mock(
            return_value=_json_resp({
                "id": "c-1",
                "content": "Hello world",
                "metadata": {"page": 1},
                "store_name": "s",
            })
        )
        c = _client()
        result = await c.get_chunk("s", "c-1")
        assert isinstance(result, ChunkInfo)
        assert result.content == "Hello world"


class TestDeleteChunks:
    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_chunks(self):
        respx.delete(f"{BASE}/api/v1/retrieval/api/v1/stores/s/chunks").mock(
            return_value=_json_resp({
                "store_name": "s",
                "doc_id": "d1",
                "chunks_deleted": 5,
                "status": "success",
            })
        )
        c = _client()
        result = await c.delete_chunks("s", "d1")
        assert isinstance(result, ChunksDeleted)
        assert result.chunks_deleted == 5


# ==================================================================
# Error Verbosity
# ==================================================================


class TestErrorVerbosity:
    """Verify that SDK errors provide actionable developer context."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error_includes_service_and_url(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores/s/query").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        c = _client()
        with pytest.raises(AuthenticationError) as exc_info:
            await c.retrieval_query("s", "test query")
        err = exc_info.value
        assert "retrieval" in str(err).lower() or hasattr(err, "service")

    @respx.mock
    @pytest.mark.asyncio
    async def test_validation_error_includes_field_details(self):
        respx.post(f"{BASE}/api/v1/retrieval/api/v1/stores").mock(
            return_value=httpx.Response(422, json={
                "detail": [
                    {"loc": ["body", "name"], "msg": "does not match pattern", "type": "value_error"},
                ],
            })
        )
        c = _client()
        with pytest.raises(ValidationError) as exc_info:
            await c.create_store("INVALID NAME!")
        err = exc_info.value
        assert hasattr(err, "errors") or "pattern" in str(err).lower()

    @respx.mock
    @pytest.mark.asyncio
    async def test_404_error_is_descriptive(self):
        respx.get(f"{BASE}/api/v1/retrieval/api/v1/stores/nonexistent/stats").mock(
            return_value=httpx.Response(404, json={"detail": "Store 'nonexistent' not found"})
        )
        c = _client()
        with pytest.raises(TaprootAPIError) as exc_info:
            await c.get_store_stats("nonexistent")
        assert "not found" in str(exc_info.value).lower()
