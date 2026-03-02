"""Typed models for Guardrail-S named configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScannerOverride:
    """Per-scanner configuration within a named guardrail config.

    This is the **config-level** schema (used in ``/projects/{pid}/configs``).
    Note: check-level overrides (``check/input``, ``check/output``) use a
    simpler ``Dict[str, bool]`` to enable/disable scanners per request.
    """

    enabled: bool = True
    threshold: float | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScannerOverride:
        return cls(
            enabled=data.get("enabled", True),
            threshold=data.get("threshold"),
            config=data.get("config", {}),
        )


@dataclass(frozen=True)
class GuardrailConfig:
    """A named guardrail configuration for a project."""

    name: str
    version: int = 1
    description: str = ""
    company_policy_version: str = ""
    scanner_overrides: dict[str, ScannerOverride] = field(default_factory=dict)
    mode: str = "active"
    is_default: bool = False
    updated_by: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardrailConfig:
        raw_overrides = data.get("scanner_overrides", {})
        overrides = {}
        for scanner_id, value in raw_overrides.items():
            if isinstance(value, dict):
                overrides[scanner_id] = ScannerOverride.from_dict(value)
            else:
                overrides[scanner_id] = ScannerOverride(enabled=bool(value))
        return cls(
            name=data.get("name", ""),
            version=data.get("version", 1),
            description=data.get("description", ""),
            company_policy_version=data.get("company_policy_version", ""),
            scanner_overrides=overrides,
            mode=data.get("mode", "active"),
            is_default=data.get("is_default", False),
            updated_by=data.get("updated_by", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
