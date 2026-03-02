"""
Eval SDK - Helpers for CI/CD evaluation workflows.

Usage:
    from taproot_sdk import TaprootClient
    from taproot_sdk.evals import assert_eval

    client = TaprootClient(base_url="...", api_key="...", project_id="my-project")
    result = await client.wait_for_eval(run_id, timeout=300)
    assert_eval(result, min_pass_rate=80)
"""

from taproot_sdk.evals.assertions import assert_eval
from taproot_sdk.evals.exceptions import EvalAssertionError
from taproot_sdk.evals.models import (
    AlertHistory,
    AlertRule,
    DiscoverySession,
    DiscoverySuggestion,
    EvalResult,
    Experiment,
    ExportResult,
    GoldenDataset,
    GoldenDatasetItem,
    GoldenDatasetVersion,
    JobStatus,
    MetricComparison,
    PaginatedList,
    RunHandle,
    TestConfiguration,
    TraceInfo,
    TraceStats,
    Webhook,
    WebhookDelivery,
)

__all__ = [
    "AlertHistory",
    "AlertRule",
    "DiscoverySession",
    "DiscoverySuggestion",
    "EvalAssertionError",
    "EvalResult",
    "Experiment",
    "ExportResult",
    "GoldenDataset",
    "GoldenDatasetItem",
    "GoldenDatasetVersion",
    "JobStatus",
    "MetricComparison",
    "PaginatedList",
    "RunHandle",
    "TestConfiguration",
    "TraceInfo",
    "TraceStats",
    "Webhook",
    "WebhookDelivery",
    "assert_eval",
]
