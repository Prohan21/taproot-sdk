"""Eval result data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
