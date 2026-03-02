"""Typed models for Guardrail-S check results and analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    """A stored guardrail check result."""

    id: str
    project_id: str
    verdict: str
    content_hash: str = ""
    check_type: str = ""
    config_name: str | None = None
    trace_id: str = ""
    blocked_by: str | None = None
    block_reasons: tuple[str, ...] | None = None
    total_latency_ms: float = 0.0
    scanner_count: int = 0
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckResult:
        raw_reasons = data.get("block_reasons")
        return cls(
            id=data.get("id") or data.get("result_id", ""),
            project_id=data.get("project_id", ""),
            verdict=data.get("verdict", ""),
            content_hash=data.get("content_hash", ""),
            check_type=data.get("check_type", ""),
            config_name=data.get("config_name"),
            trace_id=data.get("trace_id", ""),
            blocked_by=data.get("blocked_by"),
            block_reasons=tuple(raw_reasons) if raw_reasons else None,
            total_latency_ms=data.get("total_latency_ms", 0.0),
            scanner_count=data.get("scanner_count", 0),
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True)
class TimeseriesBucket:
    """A single bucket in an analytics timeseries."""

    timestamp: str
    total_checks: int = 0
    total_blocks: int = 0
    total_allows: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeseriesBucket:
        return cls(
            timestamp=data.get("timestamp", ""),
            total_checks=data.get("total_checks") or data.get("checks", 0),
            total_blocks=data.get("total_blocks") or data.get("blocks", 0),
            total_allows=data.get("total_allows") or data.get("allows", 0),
        )


@dataclass(frozen=True)
class AnalyticsSummary:
    """Aggregated analytics from Guardrail-S."""

    total_checks: int = 0
    verdicts: dict[str, int] = field(default_factory=dict)
    scanner_triggers: dict[str, int] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    timeseries: tuple[TimeseriesBucket, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalyticsSummary:
        raw_ts = data.get("timeseries") or data.get("time_series") or []
        return cls(
            total_checks=data.get("total_checks", 0),
            verdicts=data.get("verdicts") or data.get("verdict_counts", {}),
            scanner_triggers=(
                data.get("scanner_triggers") or data.get("scanner_counts", {})
            ),
            latency_p50_ms=data.get("latency_p50_ms", 0.0),
            latency_p95_ms=data.get("latency_p95_ms", 0.0),
            latency_p99_ms=data.get("latency_p99_ms", 0.0),
            timeseries=tuple(TimeseriesBucket.from_dict(b) for b in raw_ts),
        )
