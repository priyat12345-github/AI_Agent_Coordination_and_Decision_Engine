"""
Shared memory module for the AI Agent Coordination & Decision Engine.
Provides short-term and long-term context storage for agents.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class SharedMemory:
    """
    A simple in-memory context store shared across all agents in a workflow.
    Stores messages, intermediate results, and metadata for the current session.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._message_history: List[Dict[str, Any]] = []
        self._session_id: str = datetime.now().strftime("%Y%m%d%H%M%S")

    @property
    def session_id(self) -> str:
        return self._session_id

    # ─── Key-Value Store ──────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Store any value under a given key."""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored value by key."""
        return self._store.get(key, default)

    def delete(self, key: str) -> None:
        """Remove a key from the store."""
        self._store.pop(key, None)

    def keys(self) -> List[str]:
        return list(self._store.keys())

    # ─── Message History ─────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, agent: Optional[str] = None) -> None:
        """Append a message to the shared conversation history."""
        self._message_history.append(
            {
                "role": role,
                "content": content,
                "agent": agent,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the message history, optionally limited to the last N entries."""
        if last_n is not None:
            return self._message_history[-last_n:]
        return list(self._message_history)

    def clear_history(self) -> None:
        self._message_history.clear()

    # ─── Snapshot ────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a full snapshot of current memory state."""
        return {
            "session_id": self._session_id,
            "store": dict(self._store),
            "message_count": len(self._message_history),
        }

    def __repr__(self) -> str:
        return f"<SharedMemory session={self._session_id} keys={list(self._store.keys())} messages={len(self._message_history)}>"
