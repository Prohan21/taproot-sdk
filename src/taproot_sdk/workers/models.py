"""Worker-S typed models — frozen dataclasses for SDK consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkerSession:
    """A virtual worker session."""

    id: str
    user_id: str
    user_email: str
    project_ids: tuple[str, ...] = ()
    status: str = "active"
    started_at: str | None = None
    last_activity_at: str | None = None
    completed_at: str | None = None
    token_count: int = 0
    tool_call_count: int = 0

    @property
    def is_active(self) -> bool:
        return self.status in ("active", "running")

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> WorkerSession:
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            user_email=data["user_email"],
            project_ids=tuple(data.get("project_ids", [])),
            status=data.get("status", "active"),
            started_at=data.get("started_at"),
            last_activity_at=data.get("last_activity_at"),
            completed_at=data.get("completed_at"),
            token_count=data.get("token_count", 0),
            tool_call_count=data.get("tool_call_count", 0),
        )


@dataclass(frozen=True)
class SessionCreated:
    """Response from creating a worker session."""

    session_id: str
    session_token: str
    stream_url: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> SessionCreated:
        return cls(
            session_id=data["session_id"],
            session_token=data["session_token"],
            stream_url=data["stream_url"],
        )


@dataclass(frozen=True)
class SessionMessage:
    """A message in a worker session."""

    id: str
    session_id: str
    role: str
    content: str
    visible_to_user: bool = True
    turn_index: int = 0
    created_at: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> SessionMessage:
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            visible_to_user=data.get("visible_to_user", True),
            turn_index=data.get("turn_index", 0),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True)
class PendingAction:
    """A pending write-action requiring user approval."""

    id: str
    session_id: str
    step_index: int
    tool_id: str
    tool_name: str
    action_class: str
    proposed_payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    resolved_payload: dict[str, Any] | None = None
    created_at: str | None = None
    resolved_at: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    @property
    def is_approved(self) -> bool:
        return self.status == "approved"

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> PendingAction:
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            step_index=data["step_index"],
            tool_id=data["tool_id"],
            tool_name=data["tool_name"],
            action_class=data.get("action_class", "write"),
            proposed_payload=data.get("proposed_payload", {}),
            status=data.get("status", "pending"),
            resolved_payload=data.get("resolved_payload"),
            created_at=data.get("created_at"),
            resolved_at=data.get("resolved_at"),
        )
