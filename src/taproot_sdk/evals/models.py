"""Eval result data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RunHandle:
    """Handle returned when a test run is triggered.

    Contains the run_id needed to poll for completion.
    """

    run_id: str
    status: str
    message: str


@dataclass(frozen=True)
class AggregateScore:
    """Aggregate score for a single metric across all items."""

    mean: float
    min: float
    max: float
    std_dev: float
    passed: int
    failed: int


@dataclass(frozen=True)
class EvalResult:
    """Result of a completed evaluation run.

    Contains overall status, pass rate, aggregate scores, and timing info.
    """

    run_id: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    aggregate_scores: dict[str, AggregateScore] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Percentage of items that passed (0-100)."""
        if self.completed_items == 0:
            return 0.0
        return ((self.completed_items - self.failed_items) / self.completed_items) * 100

    @property
    def duration_ms(self) -> float | None:
        """Duration of the run in milliseconds, or None if not available."""
        if not self.started_at or not self.completed_at:
            return None
        from datetime import datetime

        start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
        return (end - start).total_seconds() * 1000

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> EvalResult:
        """Create EvalResult from Evals-S API response."""
        raw_scores = data.get("aggregate_scores") or {}
        scores = {
            name: AggregateScore(
                mean=s.get("mean", 0),
                min=s.get("min", 0),
                max=s.get("max", 0),
                std_dev=s.get("std_dev", 0),
                passed=s.get("passed", 0),
                failed=s.get("failed", 0),
            )
            for name, s in raw_scores.items()
        }

        return cls(
            run_id=str(data["id"]),
            status=data["status"],
            total_items=data.get("total_items", 0),
            completed_items=data.get("completed_items", 0),
            failed_items=data.get("failed_items", 0),
            aggregate_scores=scores,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            tags=data.get("tags") or [],
        )


# ------------------------------------------------------------------
# Paginated wrapper
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PaginatedList(Generic[T]):
    """Generic paginated list returned by list endpoints."""

    items: list[T]
    total: int
    offset: int = 0
    limit: int = 100


# ------------------------------------------------------------------
# Golden Dataset models
# ------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenDataset:
    """A golden dataset for evaluation."""

    id: str
    project_id: str
    name: str
    description: str = ""
    version: int = 1
    is_active: bool = True
    tags: list[str] = field(default_factory=list)
    item_count: int = 0
    slug: str = ""
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GoldenDataset:
        return cls(
            id=str(data["id"]),
            project_id=data.get("project_id", ""),
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", 1),
            is_active=data.get("is_active", True),
            tags=data.get("tags") or [],
            item_count=data.get("item_count", 0),
            slug=data.get("slug", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class GoldenDatasetItem:
    """A single item within a golden dataset."""

    id: str
    dataset_id: str
    input_query: str
    expected_output: str | None = None
    expected_output_json: dict[str, Any] | None = None
    expected_tool_calls: list[dict[str, Any]] | None = None
    expected_context_ids: list[str] | None = None
    expected_contexts: list[str] | None = None
    input_metadata: dict[str, Any] | None = None
    has_placeholders: bool = False
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    source_type: str | None = None
    source_trace_id: str | None = None
    session_group: str | None = None
    sequence_order: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GoldenDatasetItem:
        return cls(
            id=str(data["id"]),
            dataset_id=str(data.get("dataset_id", "")),
            input_query=data["input_query"],
            expected_output=data.get("expected_output"),
            expected_output_json=data.get("expected_output_json"),
            expected_tool_calls=data.get("expected_tool_calls"),
            expected_context_ids=data.get("expected_context_ids"),
            expected_contexts=data.get("expected_contexts"),
            input_metadata=data.get("input_metadata"),
            has_placeholders=data.get("has_placeholders", False),
            tags=data.get("tags") or [],
            notes=data.get("notes"),
            source_type=data.get("source_type"),
            source_trace_id=data.get("source_trace_id"),
            session_group=data.get("session_group"),
            sequence_order=data.get("sequence_order"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class GoldenDatasetVersion:
    """A version snapshot of a golden dataset."""

    id: str
    dataset_id: str
    version: int
    item_count: int = 0
    change_reason: str | None = None
    changed_by: str | None = None
    created_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GoldenDatasetVersion:
        return cls(
            id=str(data["id"]),
            dataset_id=str(data.get("dataset_id", "")),
            version=data["version"],
            item_count=data.get("item_count", 0),
            change_reason=data.get("change_reason"),
            changed_by=data.get("changed_by"),
            created_at=data.get("created_at"),
        )


# ------------------------------------------------------------------
# Test Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TestConfiguration:
    """A test configuration linking a dataset to an agent with metrics."""

    id: str
    project_id: str
    name: str
    dataset_id: str
    agent_target_id: str
    metrics: list[str] = field(default_factory=list)
    pass_threshold: float = 0.7
    description: str | None = None
    is_active: bool = True
    last_run_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> TestConfiguration:
        return cls(
            id=str(data["id"]),
            project_id=data.get("project_id", ""),
            name=data["name"],
            dataset_id=str(data.get("dataset_id", "")),
            agent_target_id=str(data.get("agent_target_id", "")),
            metrics=data.get("metrics") or [],
            pass_threshold=data.get("pass_threshold", 0.7),
            description=data.get("description"),
            is_active=data.get("is_active", True),
            last_run_at=data.get("last_run_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """An experiment grouping multiple test runs for comparison."""

    id: str
    project_id: str
    name: str
    metrics: list[str] = field(default_factory=list)
    pass_threshold: float = 0.7
    description: str | None = None
    dataset_id: str | None = None
    dataset_version: int | None = None
    agent_target_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    run_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> Experiment:
        return cls(
            id=str(data["id"]),
            project_id=data.get("project_id", ""),
            name=data["name"],
            metrics=data.get("metrics") or [],
            pass_threshold=data.get("pass_threshold", 0.7),
            description=data.get("description"),
            dataset_id=str(data["dataset_id"]) if data.get("dataset_id") else None,
            dataset_version=data.get("dataset_version"),
            agent_target_id=(
                str(data["agent_target_id"]) if data.get("agent_target_id") else None
            ),
            metadata=data.get("metadata") or {},
            is_active=data.get("is_active", True),
            run_count=data.get("run_count", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# ------------------------------------------------------------------
# Alert Rules
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AlertRule:
    """An alert rule that fires on evaluation metric conditions."""

    id: str
    project_id: str
    name: str
    condition_type: str
    condition_config: dict[str, Any] = field(default_factory=dict)
    notification_channel: str = "webhook"
    notification_config: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    cooldown_minutes: int = 60
    last_triggered_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> AlertRule:
        return cls(
            id=str(data["id"]),
            project_id=data.get("project_id", ""),
            name=data["name"],
            condition_type=data["condition_type"],
            condition_config=data.get("condition_config") or {},
            notification_channel=data.get("notification_channel", "webhook"),
            notification_config=data.get("notification_config") or {},
            is_active=data.get("is_active", True),
            cooldown_minutes=data.get("cooldown_minutes", 60),
            last_triggered_at=data.get("last_triggered_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class AlertHistory:
    """A record of when an alert rule was triggered."""

    id: str
    alert_rule_id: str
    run_id: str
    triggered_at: str | None = None
    condition_snapshot: dict[str, Any] | None = None
    notification_sent: bool = False
    notification_error: str | None = None
    created_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> AlertHistory:
        return cls(
            id=str(data["id"]),
            alert_rule_id=str(data.get("alert_rule_id", "")),
            run_id=str(data.get("run_id", "")),
            triggered_at=data.get("triggered_at"),
            condition_snapshot=data.get("condition_snapshot"),
            notification_sent=data.get("notification_sent", False),
            notification_error=data.get("notification_error"),
            created_at=data.get("created_at"),
        )


# ------------------------------------------------------------------
# Webhooks
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Webhook:
    """A webhook subscription for evaluation events."""

    id: str
    project_id: str
    name: str
    url: str
    events: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> Webhook:
        return cls(
            id=str(data["id"]),
            project_id=data.get("project_id", ""),
            name=data["name"],
            url=data["url"],
            events=data.get("events") or [],
            is_active=data.get("is_active", True),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True)
class WebhookDelivery:
    """A record of a webhook delivery attempt."""

    id: str
    webhook_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    attempt_count: int = 0
    max_attempts: int = 3
    next_retry_at: str | None = None
    response_status_code: int | None = None
    response_body: str | None = None
    error_message: str | None = None
    last_attempt_at: str | None = None
    delivered_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> WebhookDelivery:
        return cls(
            id=str(data["id"]),
            webhook_id=str(data.get("webhook_id", "")),
            event_type=data.get("event_type", ""),
            payload=data.get("payload") or {},
            status=data.get("status", "pending"),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            next_retry_at=data.get("next_retry_at"),
            response_status_code=data.get("response_status_code"),
            response_body=data.get("response_body"),
            error_message=data.get("error_message"),
            last_attempt_at=data.get("last_attempt_at"),
            delivered_at=data.get("delivered_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoverySession:
    """An agent capability discovery session."""

    id: str
    project_id: str
    agent_target_id: str
    status: str = "pending"
    config_snapshot: dict[str, Any] | None = None
    total_l1_probes: int = 0
    completed_l1_probes: int = 0
    total_l2_probes: int = 0
    completed_l2_probes: int = 0
    discovered_capabilities: list[dict[str, Any]] | None = None
    total_suggestions: int = 0
    approved_suggestions: int = 0
    rejected_suggestions: int = 0
    target_dataset_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> DiscoverySession:
        return cls(
            id=str(data["id"]),
            project_id=data.get("project_id", ""),
            agent_target_id=str(data.get("agent_target_id", "")),
            status=data.get("status", "pending"),
            config_snapshot=data.get("config_snapshot"),
            total_l1_probes=data.get("total_l1_probes", 0),
            completed_l1_probes=data.get("completed_l1_probes", 0),
            total_l2_probes=data.get("total_l2_probes", 0),
            completed_l2_probes=data.get("completed_l2_probes", 0),
            discovered_capabilities=data.get("discovered_capabilities"),
            total_suggestions=data.get("total_suggestions", 0),
            approved_suggestions=data.get("approved_suggestions", 0),
            rejected_suggestions=data.get("rejected_suggestions", 0),
            target_dataset_id=(
                str(data["target_dataset_id"]) if data.get("target_dataset_id") else None
            ),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class DiscoverySuggestion:
    """A test case suggestion generated by the discovery system."""

    id: str
    session_id: str
    status: str = "pending"
    input_query: str = ""
    expected_output: str | None = None
    expected_tool_calls: list[dict[str, Any]] | None = None
    expected_context_ids: list[str] | None = None
    expected_contexts: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    reasoning: str | None = None
    category: str | None = None
    capability_ref: str | None = None
    confidence: float | None = None
    source_probe_ids: list[str] | None = None
    user_edited_query: str | None = None
    user_edited_expected_output: str | None = None
    user_edited_tool_calls: list[dict[str, Any]] | None = None
    user_notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> DiscoverySuggestion:
        return cls(
            id=str(data["id"]),
            session_id=str(data.get("session_id", "")),
            status=data.get("status", "pending"),
            input_query=data.get("input_query", ""),
            expected_output=data.get("expected_output"),
            expected_tool_calls=data.get("expected_tool_calls"),
            expected_context_ids=data.get("expected_context_ids"),
            expected_contexts=data.get("expected_contexts"),
            tags=data.get("tags") or [],
            notes=data.get("notes"),
            reasoning=data.get("reasoning"),
            category=data.get("category"),
            capability_ref=data.get("capability_ref"),
            confidence=data.get("confidence"),
            source_probe_ids=data.get("source_probe_ids"),
            user_edited_query=data.get("user_edited_query"),
            user_edited_expected_output=data.get("user_edited_expected_output"),
            user_edited_tool_calls=data.get("user_edited_tool_calls"),
            user_notes=data.get("user_notes"),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# ------------------------------------------------------------------
# Traces
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TraceInfo:
    """Summary information about a single trace."""

    trace_id: str
    project_id: str
    name: str | None = None
    agent_name: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: float | None = None
    status: str = "ok"
    error_message: str | None = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    span_count: int = 0
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> TraceInfo:
        return cls(
            trace_id=data.get("trace_id", str(data.get("id", ""))),
            project_id=data.get("project_id", ""),
            name=data.get("name"),
            agent_name=data.get("agent_name"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            duration_ms=data.get("duration_ms"),
            status=data.get("status", "ok"),
            error_message=data.get("error_message"),
            total_tokens=data.get("total_tokens", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            span_count=data.get("span_count", 0),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class TraceStats:
    """Aggregate trace statistics for a project."""

    total_traces: int = 0
    total_spans: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    error_rate: float = 0.0
    avg_duration_ms: float = 0.0

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> TraceStats:
        return cls(
            total_traces=data.get("total_traces", 0),
            total_spans=data.get("total_spans", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_duration_ms=data.get("avg_duration_ms", 0.0),
        )


# ------------------------------------------------------------------
# Run Comparison
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MetricComparison:
    """Comparison of a single metric between two test runs."""

    metric_name: str
    run_a_score: float
    run_b_score: float
    delta: float
    improved: bool
    percent_change: float = 0.0

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> MetricComparison:
        return cls(
            metric_name=data["metric_name"],
            run_a_score=data.get("run_a_score", 0.0),
            run_b_score=data.get("run_b_score", 0.0),
            delta=data.get("delta", 0.0),
            improved=data.get("improved", False),
            percent_change=data.get("percent_change", 0.0),
        )


# ------------------------------------------------------------------
# Export Result
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ExportResult:
    """Result of a bulk export request (JSONL or CSV)."""

    export_id: str
    export_url: str
    format: str = "jsonl"
    created_at: str | None = None
    expires_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> ExportResult:
        return cls(
            export_id=str(data.get("export_id", data.get("id", ""))),
            export_url=data.get("export_url", ""),
            format=data.get("format", "jsonl"),
            created_at=data.get("created_at"),
            expires_at=data.get("expires_at"),
        )


# ------------------------------------------------------------------
# Job Status
# ------------------------------------------------------------------


@dataclass(frozen=True)
class JobStatus:
    """Status of a cloud job executing a test run."""

    run_id: str
    job_id: str | None = None
    status: str = "unknown"
    source: str = "database"
    message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> JobStatus:
        return cls(
            run_id=str(data.get("run_id", "")),
            job_id=data.get("job_id"),
            status=data.get("status", "unknown"),
            source=data.get("source", "database"),
            message=data.get("message"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )
