"""WO-013 T2: client routing/parse tests for the eval-loop Evals-S methods.

Covers ingest_traces (trace-ingest hot path) plus trigger_eval_run and
wait_for_eval (the CI eval-gate surface), in both APIM and direct mode.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.evals.models import EvalResult, RunHandle
from taproot_sdk.exceptions import AuthenticationError, ValidationError

BASE = "https://gateway.test"


def _client(direct_mode: bool = False) -> TaprootClient:
    return TaprootClient(
        base_url=BASE,
        api_key="test-key",
        project_id="proj-1",
        direct_mode=direct_mode,
    )


def _json_resp(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


OTLP_PAYLOAD = {
    "resourceSpans": [
        {
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "a"}}]},
            "scopeSpans": [{"spans": [{"traceId": "abc", "spanId": "def", "name": "work"}]}],
        }
    ]
}


class TestIngestTraces:
    @respx.mock
    async def test_apim_route_and_body(self):
        route = respx.post(f"{BASE}/api/v1/evals/v1/projects/proj-1/traces").mock(
            return_value=_json_resp({"status": "accepted", "spans_ingested": 1})
        )
        result = await _client().ingest_traces(OTLP_PAYLOAD)
        assert result == {"status": "accepted", "spans_ingested": 1}
        sent = json.loads(route.calls[0].request.content)
        assert sent == OTLP_PAYLOAD

    @respx.mock
    async def test_direct_route(self):
        route = respx.post(f"{BASE}/v1/projects/proj-1/traces").mock(
            return_value=_json_resp({"status": "accepted"})
        )
        result = await _client(direct_mode=True).ingest_traces(OTLP_PAYLOAD)
        assert result["status"] == "accepted"
        assert route.calls[0].request.headers["X-Api-Key-Id"] == "test-key"

    @respx.mock
    async def test_project_id_override(self):
        respx.post(f"{BASE}/api/v1/evals/v1/projects/other/traces").mock(
            return_value=_json_resp({"status": "accepted"})
        )
        result = await _client().ingest_traces(OTLP_PAYLOAD, project_id="other")
        assert result["status"] == "accepted"

    @respx.mock
    async def test_validation_error_maps(self):
        respx.post(f"{BASE}/api/v1/evals/v1/projects/proj-1/traces").mock(
            return_value=httpx.Response(
                422,
                json={"detail": [{"loc": ["body", "resourceSpans"], "msg": "field required"}]},
            )
        )
        with pytest.raises(ValidationError):
            await _client().ingest_traces({"bad": "payload"})


class TestTriggerEvalRun:
    @respx.mock
    async def test_apim_route_body_and_parse(self):
        route = respx.post(f"{BASE}/api/v1/evals/v1/projects/proj-1/test-runs/trigger").mock(
            return_value=_json_resp({"run_id": "run-1", "status": "pending", "message": "queued"})
        )
        handle = await _client().trigger_eval_run("cfg-1", tags=["ci"], description="gate")
        assert isinstance(handle, RunHandle)
        assert handle.run_id == "run-1"
        assert handle.status == "pending"
        body = json.loads(route.calls[0].request.content)
        assert body == {"test_config_id": "cfg-1", "tags": ["ci"], "description": "gate"}

    @respx.mock
    async def test_direct_route(self):
        respx.post(f"{BASE}/v1/projects/proj-1/test-runs/trigger").mock(
            return_value=_json_resp({"id": "run-2", "status": "pending"})
        )
        handle = await _client(direct_mode=True).trigger_eval_run("cfg-1")
        assert handle.run_id == "run-2"

    @respx.mock
    async def test_auth_error_maps(self):
        respx.post(f"{BASE}/api/v1/evals/v1/projects/proj-1/test-runs/trigger").mock(
            return_value=httpx.Response(403, json={"detail": "Access denied"})
        )
        with pytest.raises(AuthenticationError):
            await _client().trigger_eval_run("cfg-1")


class TestWaitForEval:
    RUNNING = {
        "id": "run-1",
        "status": "running",
        "total_items": 10,
        "completed_items": 3,
        "failed_items": 0,
    }
    COMPLETED = {
        "id": "run-1",
        "status": "completed",
        "total_items": 10,
        "completed_items": 10,
        "failed_items": 1,
        "aggregate_scores": {
            "answer_relevancy": {"mean": 0.9, "min": 0.5, "max": 1.0, "passed": 9, "failed": 1}
        },
        "tags": ["ci"],
    }

    @respx.mock
    async def test_polls_until_terminal_apim(self):
        route = respx.get(f"{BASE}/api/v1/evals/v1/projects/proj-1/test-runs/run-1").mock(
            side_effect=[_json_resp(self.RUNNING), _json_resp(self.COMPLETED)]
        )
        result = await _client().wait_for_eval("run-1", timeout=30, poll_interval=0.01)
        assert isinstance(result, EvalResult)
        assert result.status == "completed"
        assert result.pass_rate == 90.0
        assert result.aggregate_scores["answer_relevancy"].mean == 0.9
        assert route.call_count == 2

    @respx.mock
    async def test_direct_route(self):
        respx.get(f"{BASE}/v1/projects/proj-1/test-runs/run-1").mock(
            return_value=_json_resp(self.COMPLETED)
        )
        result = await _client(direct_mode=True).wait_for_eval("run-1", timeout=5)
        assert result.status == "completed"

    @respx.mock
    async def test_timeout_raises(self):
        respx.get(f"{BASE}/api/v1/evals/v1/projects/proj-1/test-runs/run-1").mock(
            return_value=_json_resp(self.RUNNING)
        )
        with pytest.raises(TimeoutError, match="run-1"):
            await _client().wait_for_eval("run-1", timeout=0.05, poll_interval=0.01)
