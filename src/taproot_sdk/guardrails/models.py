"""
Typed response models for Guardrail-S API.

Field names match the Guardrail-S Pydantic response models exactly
(src/models/responses.py in Guardrail-S).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScannerSignal:
    """Individual scanner result from a guardrail check."""

    scanner_id: str
    scanner_version: str
    is_valid: bool
    score: float | None = None
    labels: tuple[str, ...] = ()
    reasoning: str | None = None
    latency_ms: float = 0.0

    @property
    def flagged(self) -> bool:
        """True when the scanner flagged the content (is_valid == False)."""
        return not self.is_valid

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScannerSignal:
        return cls(
            scanner_id=data["scanner_id"],
            scanner_version=data.get("scanner_version", ""),
            is_valid=data.get("is_valid", True),
            score=data.get("score"),
            labels=tuple(data.get("labels", [])),
            reasoning=data.get("reasoning"),
            latency_ms=data.get("latency_ms", 0.0),
        )


@dataclass(frozen=True)
class PolicySegmentResult:
    """LLM Judge result for a single policy segment."""

    segment_id: str
    violated: bool
    reasoning: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySegmentResult:
        return cls(
            segment_id=data["segment_id"],
            violated=data.get("violated", False),
            reasoning=data.get("reasoning"),
        )


@dataclass(frozen=True)
class RedactionAction:
    """Redaction instruction for ALLOW_WITH_REDACTIONS verdict."""

    start_index: int
    end_index: int
    replacement: str
    reason: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedactionAction:
        return cls(
            start_index=data["start_index"],
            end_index=data["end_index"],
            replacement=data["replacement"],
            reason=data["reason"],
        )


@dataclass(frozen=True)
class GuardrailResponse:
    """Typed representation of a Guardrail-S check response.

    Mirrors the server-side ``GuardrailResponse`` Pydantic model.
    """

    verdict: str
    signals: tuple[ScannerSignal, ...] = ()
    policy_signals: tuple[PolicySegmentResult, ...] = ()
    redactions: tuple[RedactionAction, ...] | None = None
    blocked_by: str | None = None
    block_reasons: tuple[str, ...] | None = None
    company_policy_version: str = ""
    project_policy_version: str = ""
    total_latency_ms: float = 0.0
    request_id: str = ""
    shadow_verdict: dict[str, Any] | None = None
    review_id: str | None = None
    # WO-005 degraded-mode contract: high-assurance callers SHOULD treat
    # degraded=True as fail-closed. Defaults preserve older-server responses.
    degraded: bool = False
    unevaluated: tuple[str, ...] = ()

    # ---- convenience helpers ----

    @property
    def is_blocked(self) -> bool:
        return self.verdict == "BLOCK"

    @property
    def is_allowed(self) -> bool:
        return self.verdict == "ALLOW"

    @property
    def flagged_scanners(self) -> tuple[ScannerSignal, ...]:
        """Scanners that flagged the content."""
        return tuple(s for s in self.signals if s.flagged)

    @property
    def violated_policies(self) -> tuple[PolicySegmentResult, ...]:
        """Policy segments that were violated."""
        return tuple(p for p in self.policy_signals if p.violated)

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GuardrailResponse:
        """Parse a raw JSON dict into a typed GuardrailResponse."""
        signals = tuple(ScannerSignal.from_dict(s) for s in data.get("signals", []))
        policy_signals = tuple(
            PolicySegmentResult.from_dict(p) for p in data.get("policy_signals", [])
        )
        raw_redactions = data.get("redactions")
        redactions = (
            tuple(RedactionAction.from_dict(r) for r in raw_redactions) if raw_redactions else None
        )
        raw_reasons = data.get("block_reasons")
        block_reasons = tuple(raw_reasons) if raw_reasons else None

        return cls(
            verdict=data["verdict"],
            signals=signals,
            policy_signals=policy_signals,
            redactions=redactions,
            blocked_by=data.get("blocked_by"),
            block_reasons=block_reasons,
            company_policy_version=data.get("company_policy_version", ""),
            project_policy_version=data.get("project_policy_version", ""),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            request_id=data.get("request_id", ""),
            shadow_verdict=data.get("shadow_verdict"),
            review_id=data.get("review_id"),
            degraded=data.get("degraded", False),
            unevaluated=tuple(data.get("unevaluated", ())),
        )
