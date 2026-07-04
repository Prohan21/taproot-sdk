"""WO-013 T3: exporter drop-visibility, retry classification, honest flush."""

from __future__ import annotations

import httpx
import respx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from taproot_sdk.otlp_json_exporter import JsonOtlpSpanExporter

ENDPOINT = "https://gateway.test/api/v1/evals/v1/traces"


def _make_spans(count: int = 1):
    collector = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collector))
    tracer = provider.get_tracer("test")
    for i in range(count):
        with tracer.start_as_current_span(f"span-{i}"):
            pass
    return collector.get_finished_spans()


def _exporter(**kwargs) -> JsonOtlpSpanExporter:
    kwargs.setdefault("retry_backoff", 0.0)
    return JsonOtlpSpanExporter(endpoint=ENDPOINT, **kwargs)


class TestExportSuccess:
    @respx.mock
    def test_2xx_succeeds(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200))
        exporter = _exporter()
        assert exporter.export(_make_spans()) is SpanExportResult.SUCCESS
        assert route.call_count == 1
        assert exporter.spans_dropped_total == 0

    def test_empty_batch_short_circuits(self):
        assert _exporter().export([]) is SpanExportResult.SUCCESS


class TestRetryClassification:
    @respx.mock
    def test_transient_503_is_retried_then_succeeds(self):
        route = respx.post(ENDPOINT).mock(side_effect=[httpx.Response(503), httpx.Response(200)])
        exporter = _exporter()
        assert exporter.export(_make_spans()) is SpanExportResult.SUCCESS
        assert route.call_count == 2
        assert exporter.spans_dropped_total == 0

    @respx.mock
    def test_429_is_retried(self):
        route = respx.post(ENDPOINT).mock(side_effect=[httpx.Response(429), httpx.Response(200)])
        assert _exporter().export(_make_spans()) is SpanExportResult.SUCCESS
        assert route.call_count == 2

    @respx.mock
    def test_permanent_422_is_not_retried(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(422))
        exporter = _exporter()
        assert exporter.export(_make_spans(3)) is SpanExportResult.FAILURE
        assert route.call_count == 1
        assert exporter.spans_dropped_total == 3

    @respx.mock
    def test_exhausted_retries_drop_and_count(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(503))
        exporter = _exporter(max_retries=2)
        assert exporter.export(_make_spans(2)) is SpanExportResult.FAILURE
        assert route.call_count == 3  # initial + 2 retries
        assert exporter.spans_dropped_total == 2

    @respx.mock
    def test_connection_error_is_retried(self):
        route = respx.post(ENDPOINT).mock(
            side_effect=[httpx.ConnectError("boom"), httpx.Response(200)]
        )
        assert _exporter().export(_make_spans()) is SpanExportResult.SUCCESS
        assert route.call_count == 2

    @respx.mock
    def test_drop_counter_accumulates_and_is_logged(self, caplog):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(400))
        exporter = _exporter()
        with caplog.at_level("WARNING"):
            exporter.export(_make_spans(2))
            exporter.export(_make_spans(3))
        assert exporter.spans_dropped_total == 5
        assert any("otlp_json.spans_dropped" in r.message for r in caplog.records)


class TestLifecycle:
    def test_force_flush_returns_true_with_no_internal_buffer(self):
        # Honest no-op: the exporter holds no buffer (export() is synchronous,
        # queueing is owned by BatchSpanProcessor), so True is truthful.
        assert _exporter().force_flush() is True
        assert _exporter().force_flush(timeout_millis=5000) is True

    @respx.mock
    def test_export_after_shutdown_fails(self):
        exporter = _exporter()
        exporter.shutdown()
        assert exporter.export(_make_spans()) is SpanExportResult.FAILURE

    def test_shutdown_closes_http_client(self):
        exporter = _exporter()
        exporter.shutdown()
        assert exporter._client.is_closed


class TestEncoderImportCanary:
    def test_private_trace_encoder_path_still_exists(self):
        """Fails loudly if an OTel bump moves the private encoder module."""
        from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import (
            encode_spans,
        )

        assert callable(encode_spans)
