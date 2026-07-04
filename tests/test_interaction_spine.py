"""WO-002 T3: reliable interaction-id emission + OTLP span attribute."""

from unittest.mock import AsyncMock
from uuid import UUID

import httpx

from taproot_sdk import TaprootClient
from taproot_sdk.decorators import instrument
from taproot_sdk._context import (
    HEADER_ACTIVITY_VERSION,
    HEADER_CALLER_ID,
    HEADER_CALLER_TYPE,
    HEADER_CORRELATION_ID,
    HEADER_INTERACTION_ID,
    HEADER_INTERACTION_TYPE,
    HEADER_PARENT_ACTIVITY_ID,
    HEADER_PARENT_INTERACTION_ID,
    HEADER_ROOT_AGENT_ID,
    HEADER_SOURCE_AGENT_ID,
    TaprootInteractionContext,
    clear_interaction_context,
    propagation_headers,
    reset_interaction_context,
    set_interaction_context,
)


def _client(**kwargs: object) -> TaprootClient:
    client = TaprootClient(base_url="https://api.test", api_key="k", project_id="p", **kwargs)
    client._http = AsyncMock()
    client._http.request.return_value = httpx.Response(status_code=200)
    return client


class TestWireConstantParity:
    """SDK wire constants must match the taproot-common documented names."""

    def test_constants_match_documented_wire_names(self) -> None:
        assert HEADER_ACTIVITY_VERSION == "X-Taproot-Activity-Version"
        assert HEADER_INTERACTION_ID == "X-Taproot-Interaction-Id"
        assert HEADER_INTERACTION_TYPE == "X-Taproot-Interaction-Type"
        assert HEADER_CALLER_ID == "X-Taproot-Caller-Id"
        assert HEADER_CALLER_TYPE == "X-Taproot-Caller-Type"
        assert HEADER_SOURCE_AGENT_ID == "X-Taproot-Source-Agent-Id"
        assert HEADER_ROOT_AGENT_ID == "X-Taproot-Root-Agent-Id"
        assert HEADER_PARENT_INTERACTION_ID == "X-Taproot-Parent-Interaction-Id"
        assert HEADER_PARENT_ACTIVITY_ID == "X-Taproot-Parent-Activity-Id"
        assert HEADER_CORRELATION_ID == "X-Correlation-ID"


class TestAutoMint:
    async def test_apim_call_with_no_bound_context_sends_valid_interaction_id(
        self,
    ) -> None:
        clear_interaction_context()
        client = _client()
        await client._request("GET", "/v1/test")

        headers = client._http.request.mock_calls[0].kwargs["headers"]
        UUID(headers[HEADER_INTERACTION_ID])

    async def test_direct_call_with_no_bound_context_sends_valid_interaction_id(
        self,
    ) -> None:
        clear_interaction_context()
        client = _client(direct_mode=True)
        await client._request("GET", "/v1/test")

        headers = client._http.request.mock_calls[0].kwargs["headers"]
        UUID(headers[HEADER_INTERACTION_ID])

    async def test_auto_minted_id_is_stable_for_the_client_session(self) -> None:
        clear_interaction_context()
        client = _client()
        await client._request("GET", "/v1/one")
        await client._request("GET", "/v1/two")

        first = client._http.request.mock_calls[0].kwargs["headers"][HEADER_INTERACTION_ID]
        second = client._http.request.mock_calls[1].kwargs["headers"][HEADER_INTERACTION_ID]
        assert first == second

    async def test_bound_context_wins_over_auto_mint(self) -> None:
        client = _client()
        token = set_interaction_context(TaprootInteractionContext(interaction_id="bound-id"))
        try:
            await client._request("GET", "/v1/test")
        finally:
            reset_interaction_context(token)
            clear_interaction_context()

        headers = client._http.request.mock_calls[0].kwargs["headers"]
        assert headers[HEADER_INTERACTION_ID] == "bound-id"


class TestNoDeprecatedCallerHeaders:
    def test_propagation_headers_never_emit_caller_headers(self) -> None:
        from taproot_sdk._context import TaprootActorRef

        context = TaprootInteractionContext(
            interaction_id="int-1",
            caller=TaprootActorRef(actor_type="user", actor_id="user-1"),
        )
        headers = propagation_headers(context)
        assert HEADER_CALLER_ID not in headers
        assert HEADER_CALLER_TYPE not in headers


class TestSpanAttribute:
    def _local_tracing(self, monkeypatch):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from taproot_sdk import decorators

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(decorators.trace, "get_tracer", lambda name: provider.get_tracer(name))
        return exporter

    def test_decorated_span_carries_interaction_id(self, monkeypatch) -> None:
        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool")
        def work() -> str:
            return "ok"

        token = set_interaction_context(TaprootInteractionContext(interaction_id="span-int-1"))
        try:
            work()
        finally:
            reset_interaction_context(token)

        spans = exporter.get_finished_spans()
        assert spans[-1].attributes["taproot.interaction_id"] == "span-int-1"

    async def test_async_decorated_span_carries_interaction_id(self, monkeypatch) -> None:
        exporter = self._local_tracing(monkeypatch)

        @instrument(spankind="tool")
        async def work() -> str:
            return "ok"

        token = set_interaction_context(TaprootInteractionContext(interaction_id="span-int-2"))
        try:
            await work()
        finally:
            reset_interaction_context(token)

        spans = exporter.get_finished_spans()
        assert spans[-1].attributes["taproot.interaction_id"] == "span-int-2"

    def test_span_has_no_interaction_attribute_when_unbound(self, monkeypatch) -> None:
        exporter = self._local_tracing(monkeypatch)
        clear_interaction_context()

        @instrument(spankind="tool")
        def work() -> str:
            return "ok"

        work()
        spans = exporter.get_finished_spans()
        assert "taproot.interaction_id" not in spans[-1].attributes
