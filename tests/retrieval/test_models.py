"""Tests for retrieval response models."""

from __future__ import annotations

import pytest

from taproot_sdk.retrieval.models import (
    AccessGrant,
    AccessGranted,
    AccessList,
    AccessRevoked,
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
    IndexParams,
    IngestionJob,
    JobCancelled,
    JobDetail,
    JobList,
    QueryHit,
    QueryResponse,
    StoreCreated,
    StoreDeleted,
    StoreInfo,
    StoreList,
    StoreStatistics,
)


# ===================================================================
# Store Models
# ===================================================================


class TestIndexParams:
    def test_defaults(self):
        p = IndexParams()
        assert p.hnsw_m == 16
        assert p.hnsw_ef_construction == 128

    def test_from_dict(self):
        p = IndexParams.from_dict({"hnsw_m": 32, "default_ef_search": 200})
        assert p.hnsw_m == 32
        assert p.default_ef_search == 200

    def test_from_empty_dict(self):
        p = IndexParams.from_dict({})
        assert p.hnsw_m == 16

    def test_frozen(self):
        p = IndexParams()
        with pytest.raises(AttributeError):
            p.hnsw_m = 99  # type: ignore[misc]


class TestStoreInfo:
    SAMPLE = {
        "id": "abc-123",
        "name": "my_store",
        "table_name": "my_store_vector_store",
        "display_name": "My Store",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "index_type": "hnsw",
        "index_params": {"hnsw_m": 32},
        "use_halfvec": True,
        "enable_fulltext": False,
        "created_at": "2025-01-01T00:00:00Z",
        "is_active": True,
    }

    def test_from_api_response(self):
        s = StoreInfo.from_api_response(self.SAMPLE)
        assert s.id == "abc-123"
        assert s.name == "my_store"
        assert s.index_params.hnsw_m == 32
        assert s.embedding_dimensions == 1536

    def test_frozen(self):
        s = StoreInfo.from_api_response(self.SAMPLE)
        with pytest.raises(AttributeError):
            s.name = "other"  # type: ignore[misc]

    def test_missing_optional_fields(self):
        minimal = {"id": "1", "name": "test"}
        s = StoreInfo.from_api_response(minimal)
        assert s.display_name is None
        assert s.is_active is True


class TestStoreList:
    def test_from_api_response(self):
        data = {
            "stores": [
                {"id": "1", "name": "a"},
                {"id": "2", "name": "b"},
            ],
            "total": 2,
            "offset": 0,
            "limit": 50,
            "has_more": False,
        }
        sl = StoreList.from_api_response(data)
        assert len(sl.stores) == 2
        assert sl.stores[0].name == "a"
        assert sl.has_more is False

    def test_empty_list(self):
        sl = StoreList.from_api_response({"stores": []})
        assert len(sl.stores) == 0
        assert sl.total == 0


class TestStoreCreated:
    def test_from_api_response(self):
        data = {
            "store": {"id": "1", "name": "new_store"},
            "message": "Store created",
        }
        sc = StoreCreated.from_api_response(data)
        assert sc.store.name == "new_store"
        assert sc.message == "Store created"


class TestStoreDeleted:
    def test_from_api_response(self):
        sd = StoreDeleted.from_api_response({"success": True, "message": "Deleted"})
        assert sd.success is True
        assert sd.message == "Deleted"


class TestStoreStatistics:
    def test_from_api_response(self):
        data = {
            "store_name": "test_store",
            "chunk_count": 1000,
            "document_count": 50,
            "total_size_bytes": 5_000_000,
            "embedding_dimensions": 1536,
            "query_count": 200,
            "ingestion_count": 10,
            "last_queried_at": "2025-06-01T12:00:00Z",
            "last_ingested_at": None,
            "last_recount_at": "2025-06-01T11:00:00Z",
            "precision": "approximate",
        }
        st = StoreStatistics.from_api_response(data)
        assert st.chunk_count == 1000
        assert st.last_ingested_at is None
        assert st.last_recount_at == "2025-06-01T11:00:00Z"


# ===================================================================
# Access Control Models
# ===================================================================


class TestAccessGrant:
    def test_from_dict(self):
        data = {
            "api_key_id": "key-1",
            "store_id": "store-1",
            "access_level": "read",
            "granted_at": "2025-01-01T00:00:00Z",
            "granted_by_api_key_id": "admin-key",
        }
        g = AccessGrant.from_dict(data)
        assert g.api_key_id == "key-1"
        assert g.access_level == "read"


class TestAccessGranted:
    def test_from_api_response(self):
        data = {
            "access": {
                "api_key_id": "key-1",
                "store_id": "s1",
                "access_level": "read_write",
                "granted_at": "2025-01-01T00:00:00Z",
                "granted_by_api_key_id": "admin",
            },
            "message": "Access granted",
        }
        ag = AccessGranted.from_api_response(data)
        assert ag.access.api_key_id == "key-1"
        assert ag.message == "Access granted"


class TestAccessList:
    def test_from_api_response(self):
        data = {
            "access": [
                {
                    "api_key_id": "k1",
                    "store_id": "s1",
                    "access_level": "read",
                    "granted_at": "",
                    "granted_by_api_key_id": "",
                },
            ],
            "count": 1,
        }
        al = AccessList.from_api_response(data)
        assert len(al.access) == 1
        assert al.count == 1


class TestAccessRevoked:
    def test_from_api_response(self):
        ar = AccessRevoked.from_api_response({"success": True, "message": "Revoked"})
        assert ar.success is True


# ===================================================================
# Query Models
# ===================================================================


class TestQueryHit:
    def test_from_dict(self):
        h = QueryHit.from_dict({
            "chunk_id": "c-1",
            "score": 0.95,
            "text": "Hello world",
            "metadata": {"source": "doc.pdf"},
        })
        assert h.chunk_id == "c-1"
        assert h.score == 0.95
        assert h.metadata["source"] == "doc.pdf"

    def test_defaults(self):
        h = QueryHit.from_dict({})
        assert h.chunk_id == ""
        assert h.score == 0.0
        assert h.metadata == {}


class TestQueryResponse:
    def test_from_api_response(self):
        data = {
            "hits": [
                {"chunk_id": "c1", "score": 0.9, "text": "match"},
                {"chunk_id": "c2", "score": 0.8, "text": "other"},
            ],
            "total_hits": 2,
            "search_time_ms": 12.5,
        }
        qr = QueryResponse.from_api_response(data)
        assert len(qr.hits) == 2
        assert qr.total_hits == 2
        assert qr.search_time_ms == 12.5
        assert qr.is_empty is False

    def test_empty_response(self):
        qr = QueryResponse.from_api_response({"hits": []})
        assert qr.is_empty is True
        assert qr.total_hits == 0


# ===================================================================
# Ingestion Models
# ===================================================================


class TestIngestionJob:
    def test_from_api_response(self):
        j = IngestionJob.from_api_response({"job_id": "j-1", "status": "pending"})
        assert j.job_id == "j-1"
        assert j.status == "pending"

    def test_fallback_id_field(self):
        j = IngestionJob.from_api_response({"id": "j-2"})
        assert j.job_id == "j-2"


class TestJobDetail:
    SAMPLE = {
        "job_id": "j-1",
        "status": "completed",
        "source_uri": "s3://bucket/doc.pdf",
        "index": "my_store",
        "pipeline_id": "default",
        "created_at": "2025-01-01T00:00:00Z",
        "started_at": "2025-01-01T00:00:01Z",
        "completed_at": "2025-01-01T00:00:10Z",
        "error": None,
        "documents_processed": 1,
        "chunks_created": 42,
    }

    def test_from_api_response(self):
        jd = JobDetail.from_api_response(self.SAMPLE)
        assert jd.chunks_created == 42
        assert jd.is_terminal is True
        assert jd.succeeded is True

    def test_pending_not_terminal(self):
        jd = JobDetail.from_api_response({**self.SAMPLE, "status": "processing"})
        assert jd.is_terminal is False
        assert jd.succeeded is False

    def test_failed_is_terminal(self):
        jd = JobDetail.from_api_response({
            **self.SAMPLE,
            "status": "failed",
            "error": "Out of memory",
        })
        assert jd.is_terminal is True
        assert jd.succeeded is False
        assert jd.error == "Out of memory"


class TestJobList:
    def test_from_api_response(self):
        data = {
            "items": [
                {
                    "job_id": "j-1",
                    "status": "completed",
                    "store_name": "test_store",
                    "source_uri": "s3://b/f",
                    "pipeline_id": "default",
                    "created_at": "2025-01-01T00:00:00Z",
                    "started_at": "2025-01-01T00:00:01Z",
                    "completed_at": "2025-01-01T00:00:10Z",
                    "documents_processed": 1,
                    "chunks_created": 42,
                },
            ],
            "total": 1,
            "offset": 0,
            "limit": 50,
            "has_more": False,
        }
        jl = JobList.from_api_response(data)
        assert len(jl.items) == 1
        assert jl.items[0].job_id == "j-1"
        assert jl.items[0].store_name == "test_store"
        assert jl.items[0].chunks_created == 42


class TestJobCancelled:
    def test_from_api_response(self):
        jc = JobCancelled.from_api_response({"job_id": "j-1", "status": "cancelled"})
        assert jc.status == "cancelled"


# ===================================================================
# Batch Ingestion Models
# ===================================================================


class TestBatchCreated:
    def test_from_api_response(self):
        bc = BatchCreated.from_api_response({
            "batch_id": "b-1",
            "status": "listing",
            "created_at": "2025-01-01T00:00:00Z",
        })
        assert bc.batch_id == "b-1"


class TestBatchStatus:
    SAMPLE = {
        "batch_id": "b-1",
        "status": "completed",
        "source_type": "s3_prefix",
        "source_summary": "s3://bucket/docs/",
        "progress": {
            "total": 10,
            "pending": 0,
            "in_progress": 0,
            "completed": 9,
            "failed": 1,
        },
        "results": {"total_chunks_created": 450, "cancelled_count": 0},
        "errors_sample": [
            {"job_id": "j-5", "source_uri": "s3://bucket/docs/bad.pdf", "error": "parse error"},
        ],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:01:00Z",
        "completed_at": "2025-01-01T00:01:00Z",
    }

    def test_from_api_response(self):
        bs = BatchStatus.from_api_response(self.SAMPLE)
        assert bs.progress.completed == 9
        assert bs.results.total_chunks_created == 450
        assert len(bs.errors_sample) == 1
        assert bs.is_terminal is True

    def test_in_progress_not_terminal(self):
        bs = BatchStatus.from_api_response({**self.SAMPLE, "status": "processing"})
        assert bs.is_terminal is False


class TestBatchCancelled:
    def test_from_api_response(self):
        bc = BatchCancelled.from_api_response({
            "batch_id": "b-1",
            "status": "cancelled",
            "cancelled_jobs": 3,
        })
        assert bc.cancelled_jobs == 3


# ===================================================================
# Document Models
# ===================================================================


class TestDocumentDetail:
    def test_from_api_response(self):
        dd = DocumentDetail.from_api_response({
            "doc_id": "doc-1",
            "store_name": "test_store",
            "chunk_count": 25,
            "source": "upload",
            "first_ingested_at": "2025-01-01T00:00:00Z",
        })
        assert dd.doc_id == "doc-1"
        assert dd.chunk_count == 25


class TestDocumentList:
    def test_from_api_response(self):
        dl = DocumentList.from_api_response({
            "documents": [{"doc_id": "d1"}, {"doc_id": "d2"}],
            "total": 2,
            "offset": 0,
            "limit": 20,
            "has_more": False,
        })
        assert len(dl.documents) == 2


class TestDocumentDeleted:
    def test_from_api_response(self):
        dd = DocumentDeleted.from_api_response({
            "doc_id": "d-1",
            "store_name": "test",
            "chunks_deleted": 10,
        })
        assert dd.chunks_deleted == 10


# ===================================================================
# Chunk Models
# ===================================================================


class TestChunkInfo:
    def test_from_dict(self):
        c = ChunkInfo.from_dict({
            "id": "ck-1",
            "content": "Hello world",
            "metadata": {"page": 1},
            "store_name": "test",
        })
        assert c.id == "ck-1"
        assert c.content == "Hello world"

    def test_defaults(self):
        c = ChunkInfo.from_dict({})
        assert c.id == ""
        assert c.metadata == {}


class TestChunkList:
    def test_from_api_response(self):
        cl = ChunkList.from_api_response({
            "chunks": [{"id": "c1", "content": "text"}],
            "total": 1,
            "offset": 0,
            "limit": 20,
            "has_more": False,
        })
        assert len(cl.chunks) == 1


class TestChunksUploaded:
    def test_from_api_response(self):
        cu = ChunksUploaded.from_api_response({
            "store_name": "test",
            "doc_id": "d1",
            "chunk_count": 5,
            "status": "success",
        })
        assert cu.chunk_count == 5


class TestChunksDeleted:
    def test_from_api_response(self):
        cd = ChunksDeleted.from_api_response({
            "store_name": "test",
            "doc_id": "d1",
            "chunks_deleted": 5,
            "status": "success",
        })
        assert cd.chunks_deleted == 5
