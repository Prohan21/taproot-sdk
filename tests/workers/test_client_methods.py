"""WO-013 T2: client routing/parse tests for Worker-S session methods.

Worker-S is the loop's remediation edge: session creation (trusted proxy),
message send, and write-action approval.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from taproot_sdk.client import TaprootClient
from taproot_sdk.exceptions import AuthenticationError, ValidationError
from taproot_sdk.workers.models import PendingAction, SessionCreated, SessionMessage

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


SESSION_CREATED = {
    "session_id": "sess-1",
    "session_token": "tok-abc",
    "stream_url": "https://gateway.test/api/v1/workers/v1/sessions/sess-1/stream",
}


class TestCreateWorkerSession:
    @respx.mock
    async def test_apim_route_body_and_parse(self):
        route = respx.post(f"{BASE}/api/v1/workers/v1/sessions").mock(
            return_value=_json_resp(SESSION_CREATED)
        )
        result = await _client().create_worker_session(
            "reset the meter", user_id="u-1", email="u@corp.com"
        )
        assert isinstance(result, SessionCreated)
        assert result.session_id == "sess-1"
        assert result.session_token == "tok-abc"
        body = json.loads(route.calls[0].request.content)
        assert body == {
            "user_id": "u-1",
            "email": "u@corp.com",
            "message": "reset the meter",
            "project_ids": ["proj-1"],
        }

    @respx.mock
    async def test_direct_route(self):
        route = respx.post(f"{BASE}/v1/sessions").mock(return_value=_json_resp(SESSION_CREATED))
        result = await _client(direct_mode=True).create_worker_session(
            "hi", user_id="u-1", email="u@corp.com", project_ids=["p-a", "p-b"]
        )
        assert result.stream_url.endswith("/stream")
        body = json.loads(route.calls[0].request.content)
        assert body["project_ids"] == ["p-a", "p-b"]
        assert route.calls[0].request.headers["X-Api-Key-Id"] == "test-key"

    @respx.mock
    async def test_validation_error_maps(self):
        respx.post(f"{BASE}/api/v1/workers/v1/sessions").mock(
            return_value=httpx.Response(
                422, json={"detail": [{"loc": ["body", "email"], "msg": "field required"}]}
            )
        )
        with pytest.raises(ValidationError):
            await _client().create_worker_session("hi", user_id="u-1", email="")


class TestSendWorkerMessage:
    MESSAGE = {
        "id": "msg-1",
        "session_id": "sess-1",
        "role": "user",
        "content": "do the thing",
        "turn_index": 2,
    }

    @respx.mock
    async def test_apim_route_token_and_parse(self):
        route = respx.post(f"{BASE}/api/v1/workers/v1/sessions/sess-1/messages").mock(
            return_value=_json_resp(self.MESSAGE)
        )
        result = await _client().send_worker_message(
            "sess-1", "do the thing", session_token="tok-abc"
        )
        assert isinstance(result, SessionMessage)
        assert result.turn_index == 2
        req = route.calls[0].request
        assert req.headers["Authorization"] == "Bearer tok-abc"
        assert json.loads(req.content) == {"message": "do the thing"}

    @respx.mock
    async def test_direct_route(self):
        respx.post(f"{BASE}/v1/sessions/sess-1/messages").mock(
            return_value=_json_resp(self.MESSAGE)
        )
        result = await _client(direct_mode=True).send_worker_message(
            "sess-1", "do the thing", session_token="tok-abc"
        )
        assert result.content == "do the thing"

    @respx.mock
    async def test_invalid_token_maps_to_auth_error(self):
        respx.post(f"{BASE}/api/v1/workers/v1/sessions/sess-1/messages").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid session token"})
        )
        with pytest.raises(AuthenticationError):
            await _client().send_worker_message("sess-1", "hi", session_token="bad")


class TestApproveWorkerAction:
    ACTION = {
        "id": "act-1",
        "session_id": "sess-1",
        "step_index": 3,
        "tool_id": "tool-9",
        "tool_name": "delete_record",
        "action_class": "write",
        "status": "approved",
        "resolved_payload": {"record_id": "r-1"},
    }

    @respx.mock
    async def test_apim_route_and_parse(self):
        route = respx.post(
            f"{BASE}/api/v1/workers/v1/sessions/sess-1/pending-actions/act-1/approve"
        ).mock(return_value=_json_resp(self.ACTION))
        result = await _client().approve_worker_action("sess-1", "act-1", session_token="tok-abc")
        assert isinstance(result, PendingAction)
        assert result.is_approved is True
        assert result.resolved_payload == {"record_id": "r-1"}
        req = route.calls[0].request
        assert req.headers["Authorization"] == "Bearer tok-abc"
        assert json.loads(req.content) == {}

    @respx.mock
    async def test_edited_payload_sent(self):
        route = respx.post(
            f"{BASE}/api/v1/workers/v1/sessions/sess-1/pending-actions/act-1/approve"
        ).mock(return_value=_json_resp(self.ACTION))
        await _client().approve_worker_action(
            "sess-1",
            "act-1",
            session_token="tok-abc",
            edited_payload={"record_id": "r-2"},
        )
        body = json.loads(route.calls[0].request.content)
        assert body == {"edited_payload": {"record_id": "r-2"}}

    @respx.mock
    async def test_direct_route(self):
        respx.post(f"{BASE}/v1/sessions/sess-1/pending-actions/act-1/approve").mock(
            return_value=_json_resp(self.ACTION)
        )
        result = await _client(direct_mode=True).approve_worker_action(
            "sess-1", "act-1", session_token="tok-abc"
        )
        assert result.tool_name == "delete_record"
