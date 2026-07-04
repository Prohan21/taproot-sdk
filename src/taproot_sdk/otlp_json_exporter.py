"""OTLP JSON span exporter for AWS API Gateway compatibility.

AWS API Gateway REST API v1 corrupts binary protobuf payloads because it
does not support binary media types by default. This exporter serializes
spans to OTLP JSON format instead, which passes through API Gateway cleanly.

Uses the standard OpenTelemetry protobuf encoder to build the
ExportTraceServiceRequest message, then converts to JSON via
google.protobuf.json_format.MessageToDict.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)

# 429 and 5xx are transient; other 4xx are permanent and never retried.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Protobuf bytes fields that need base64 -> hex normalization in JSON output.
# MessageToDict encodes bytes as base64, but OTLP JSON spec requires hex for IDs.
_ID_FIELDS = frozenset({"traceId", "spanId", "parentSpanId"})


class JsonOtlpSpanExporter(SpanExporter):
    """OTLP span exporter that sends JSON instead of protobuf.

    Drop-in replacement for OTLPSpanExporter when the transport layer
    (e.g. AWS API Gateway) cannot handle binary protobuf payloads.
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        compress: bool = True,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        self._endpoint = endpoint
        self._headers = dict(headers or {})
        self._headers["Content-Type"] = "application/json"
        self._compress = compress
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._shutdown = False
        self._spans_dropped_total = 0
        self._client = httpx.Client(timeout=timeout)

    @property
    def spans_dropped_total(self) -> int:
        """Monotonic count of spans this exporter has given up on delivering."""
        return self._spans_dropped_total

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            return SpanExportResult.FAILURE

        if not spans:
            return SpanExportResult.SUCCESS

        try:
            payload = self._encode_to_json(spans)
        except Exception:
            logger.exception("otlp_json.encode_failed")
            self._record_drop(len(spans), reason="encode_error")
            return SpanExportResult.FAILURE

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        headers = dict(self._headers)
        if self._compress:
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"

        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(
                    self._endpoint,
                    content=body,
                    headers=headers,
                )
            except Exception as exc:
                # Connection/timeout errors are transient — retry.
                last_status = None
                logger.warning(
                    "otlp_json.send_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
            else:
                if resp.status_code < 400:
                    return SpanExportResult.SUCCESS
                last_status = resp.status_code
                logger.warning(
                    "otlp_json.export_failed",
                    extra={
                        "status": resp.status_code,
                        "attempt": attempt + 1,
                        "retryable": resp.status_code in _RETRYABLE_STATUSES,
                        "body": resp.text[:500],
                    },
                )
                if resp.status_code not in _RETRYABLE_STATUSES:
                    break

            if attempt < self._max_retries and self._retry_backoff > 0:
                time.sleep(self._retry_backoff * (2**attempt))

        self._record_drop(len(spans), reason=f"status={last_status}")
        return SpanExportResult.FAILURE

    def _record_drop(self, count: int, reason: str) -> None:
        self._spans_dropped_total += count
        logger.warning(
            "otlp_json.spans_dropped",
            extra={
                "dropped": count,
                "spans_dropped_total": self._spans_dropped_total,
                "reason": reason,
            },
        )

    def shutdown(self) -> None:
        self._shutdown = True
        self._client.close()

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """No-op that truthfully returns True.

        This exporter holds no internal buffer: ``export()`` delivers (with
        bounded retries) or drops synchronously, and queueing is owned by the
        ``BatchSpanProcessor`` — its ``force_flush`` drains the queue through
        ``export()``, so there is never anything here left to flush.
        """
        return True

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_to_json(spans: Sequence[ReadableSpan]) -> dict[str, Any]:
        """Convert ReadableSpan objects to OTLP JSON dict.

        Uses the OTel protobuf encoder to build the canonical message,
        then converts to a JSON-compatible dict with hex-encoded IDs.
        """
        from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
        from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import (
            encode_spans,
        )

        # encode_spans returns an ExportTraceServiceRequest protobuf message
        message = encode_spans(spans)
        data: dict[str, Any] = MessageToDict(message, preserving_proto_field_name=False)

        # MessageToDict encodes bytes fields as base64.
        # OTLP JSON spec requires trace/span IDs as lowercase hex strings.
        _normalize_ids(data)
        return data


def _normalize_ids(obj: Any) -> None:
    """Recursively convert base64-encoded ID fields to hex strings."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in _ID_FIELDS and isinstance(obj[key], str):
                obj[key] = _b64_to_hex(obj[key])
            else:
                _normalize_ids(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _normalize_ids(item)


def _b64_to_hex(value: str) -> str:
    """Convert a base64-encoded string to lowercase hex."""
    try:
        return base64.b64decode(value).hex()
    except Exception:
        return value
