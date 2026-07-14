"""
memory_manager.py

Memory layer for the IT Help-Desk Diagnostic Agent.

This module handles:

1. Short-term memory:
   - user name
   - user id
   - recent conversation messages inside the active session

2. Working memory:
   - current intent
   - collected information
   - missing required fields
   - pending confirmation
   - latest tool result
   - current workflow state

3. Long-term memory:
   - persistent user information stored in SQLite
   - stable preferences and useful facts that survive application restarts
"""

from __future__ import annotations
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.memory.long_term_memory import LongTermMemoryStore


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChatMessage:
    """Represents one message in the conversation."""

    role: str
    content: str
    timestamp: str = field(default_factory=utc_now)
    tool: Optional[str] = None


@dataclass
class ShortTermMemory:
    """
    Stores session-level memory.

    This memory is temporary and should only last during the active session.
    """

    user_id: Optional[str] = None
    user_name: Optional[str] = None
    messages: list[ChatMessage] = field(default_factory=list)

    def add_message(
        self,
        role: str,
        content: str,
        tool: Optional[str] = None,
    ) -> None:
        """
        Add a message to the current conversation.

        role should usually be:
        - "user"
        - "assistant"
        - "system"
        - "tool"
        """
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"Invalid message role: {role}")

        if not content or not isinstance(content, str):
            raise ValueError("Message content must be a non-empty string.")

        self.messages.append(
            ChatMessage(
                role=role,
                content=content,
                tool=tool,
            )
        )

    def get_recent_messages(self, limit: int = 8) -> list[dict[str, Any]]:
        """
        Return the most recent messages as dictionaries.

        This is useful for passing context to the orchestrator or LLM.
        """
        return [asdict(message) for message in self.messages[-limit:]]

    def clear_messages(self) -> None:
        """Clear the conversation history for the active session."""
        self.messages.clear()


@dataclass
class WorkingMemory:
    """
    Stores explicit task state.

    This is required by the project because the agent must track:
    - current intent
    - collected info
    - missing fields
    - pending confirmation
    - latest tool result
    - workflow state
    """

    current_intent: Optional[str] = None
    collected_info: dict[str, Any] = field(default_factory=dict)
    missing_required_fields: list[str] = field(default_factory=list)
    pending_confirmation: Optional[dict[str, Any]] = None
    latest_tool_result: Optional[dict[str, Any]] = None
    workflow_state: str = "awaiting_user"

    def update_collected_info(self, key: str, value: Any) -> None:
        """Store one collected field."""
        if not key or not isinstance(key, str):
            raise ValueError("Collected info key must be a non-empty string.")

        self.collected_info[key] = value

    def set_missing_fields(self, fields: list[str]) -> None:
        """Update the list of missing required fields."""
        self.missing_required_fields = fields

    def set_pending_confirmation(
        self,
        action: str,
        details: dict[str, Any],
    ) -> None:
        """
        Store a pending confirmation before a state-changing action.

        Example:
            action = "create_ticket"
            details = {"title": "...", "category": "Network"}
        """
        if not action:
            raise ValueError("Confirmation action cannot be empty.")

        self.pending_confirmation = {
            "action": action,
            "details": details,
            "requested_at": utc_now(),
        }

        self.workflow_state = "awaiting_confirmation"

    def clear_pending_confirmation(self) -> None:
        """Clear pending confirmation after user confirms or cancels."""
        self.pending_confirmation = None

    def set_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Store the latest tool result."""
        self.latest_tool_result = {
            "tool_name": tool_name,
            "result": result,
            "timestamp": utc_now(),
        }

    def reset_task(self) -> None:
        """Reset current task state but keep conversation memory."""
        self.current_intent = None
        self.collected_info.clear()
        self.missing_required_fields.clear()
        self.pending_confirmation = None
        self.latest_tool_result = None
        self.workflow_state = "awaiting_user"


class MemoryManager:
    """
    Main memory interface used by the UI and orchestrator.

    The UI should create one MemoryManager per Streamlit session.
    Long-term data is stored separately in SQLite and loaded after set_user().
    """

    def __init__(
        self,
        db_path: str | Path = "data/memory.db",
    ) -> None:
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.long_term_store = LongTermMemoryStore(db_path)
        self.long_term: dict[str, Any] = {}

    def set_user(
        self,
        user_name: str,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Store the current user's identity and load their
        persistent long-term memory.
        """
        if not isinstance(user_name, str):
            raise ValueError("user_name must be a string.")

        cleaned_name = " ".join(user_name.split())

        if not cleaned_name:
            raise ValueError("user_name cannot be empty.")

        self.short_term.user_name = cleaned_name

        if user_id is not None:
            if not isinstance(user_id, str):
                raise ValueError("user_id must be a string.")

            cleaned_user_id = user_id.strip()

            if not cleaned_user_id:
                raise ValueError("user_id cannot be empty.")

            resolved_user_id = cleaned_user_id

        else:
            safe_name = re.sub(
                r"[^\w]+",
                "_",
                cleaned_name.casefold(),
            ).strip("_")

            if not safe_name:
                raise ValueError(
                    "The name must contain at least one letter or number."
                )

            resolved_user_id = f"user_{safe_name}"

        self.short_term.user_id = resolved_user_id

        self.long_term = self.long_term_store.get_all(
            resolved_user_id
        )

    def add_user_message(self, content: str) -> None:
        """Add a user message to short-term memory."""
        self.short_term.add_message(role="user", content=content)

    def add_assistant_message(
        self,
        content: str,
        tool: Optional[str] = None,
    ) -> None:
        """Add an assistant message to short-term memory."""
        self.short_term.add_message(
            role="assistant",
            content=content,
            tool=tool,
        )

    def set_intent(self, intent: str) -> None:
        """Update the current intent."""
        if not intent or not isinstance(intent, str):
            raise ValueError("Intent must be a non-empty string.")

        self.working.current_intent = intent

    def set_workflow_state(self, state: str) -> None:
        """Update the current workflow state."""
        if not state:
            raise ValueError("Workflow state cannot be empty.")

        self.working.workflow_state = state

    def collect_info(self, key: str, value: Any) -> None:
        """Store collected information about the user's current issue."""
        self.working.update_collected_info(key, value)

    def require_fields(self, fields: list[str]) -> None:
        """Store missing fields that the agent still needs."""
        self.working.set_missing_fields(fields)

    def request_confirmation(
        self,
        action: str,
        details: dict[str, Any],
    ) -> None:
        """Create a confirmation gate before a state-changing action."""
        self.working.set_pending_confirmation(action, details)

    def confirm_action(self) -> dict[str, Any]:
        """Return and clear the pending action after explicit confirmation."""
        pending = self.working.pending_confirmation
        if pending is None:
            raise ValueError("No pending confirmation to confirm.")

        self.working.clear_pending_confirmation()
        self.working.missing_required_fields = [
            field
            for field in self.working.missing_required_fields
            if field != "user_confirmation"
        ]
        self.working.workflow_state = "confirmed_action"
        return pending

    def cancel_action(self) -> None:
        """Cancel a pending action."""
        self.working.clear_pending_confirmation()
        self.working.missing_required_fields = [
            field
            for field in self.working.missing_required_fields
            if field != "user_confirmation"
        ]
        self.working.workflow_state = "action_cancelled"

    def store_tool_result(
        self,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        """Store the result returned by a tool."""
        self.working.set_tool_result(tool_name, result)

    def _require_current_user_id(self) -> str:
        """Return the active user ID or raise a clear error."""
        user_id = self.short_term.user_id
        if user_id is None:
            raise ValueError(
                "A user must be set before accessing long-term memory."
            )
        return user_id

    def remember(self, key: str, value: Any) -> None:
        """Save or update persistent information for the current user."""
        user_id = self._require_current_user_id()
        self.long_term_store.save(user_id=user_id, key=key, value=value)
        self.long_term[key.strip()] = value

    def recall(self, key: str, default: Any = None) -> Any:
        """Return one value from the current user's loaded long-term memory."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Memory key must be a non-empty string.")
        return self.long_term.get(key.strip(), default)

    def forget(self, key: str) -> None:
        """Delete one persistent memory value for the current user."""
        user_id = self._require_current_user_id()
        self.long_term_store.delete(user_id=user_id, key=key)
        self.long_term.pop(key.strip(), None)

    def clear_long_term_memory(self) -> None:
        """Permanently delete all persistent memory for the current user."""
        user_id = self._require_current_user_id()
        self.long_term_store.clear_user(user_id)
        self.long_term.clear()

    def get_context(self) -> dict[str, Any]:
        """Return short-term, working, and loaded long-term memory."""
        return {
            "short_term": {
                "user_id": self.short_term.user_id,
                "user_name": self.short_term.user_name,
                "recent_messages": self.short_term.get_recent_messages(),
            },
            "working": asdict(self.working),
            "long_term": self.long_term.copy(),
        }

    def reset_current_task(self) -> None:
        """Reset only the active task memory."""
        self.working.reset_task()

    def reset_all(self) -> None:
        """
        Reset session memory without deleting persistent database records.

        Long-term memory is loaded again when set_user() is called.
        """
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.long_term = {}