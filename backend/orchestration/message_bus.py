"""
Message Bus — Handles inter-agent communication and event routing.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from loguru import logger

from backend.agents.base_agent import AgentEvent

class MessageBus:
    """
    Central message bus for pub/sub event distribution between agents,
    the workflow engine, and WebSocket clients.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[AgentEvent] = []
        self._history_max_size = 1000

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            
    def subscribe_all(self, callback: Callable):
        """Subscribe to all events."""
        self.subscribe("*", callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Remove a subscription."""
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            
    async def publish(self, event: AgentEvent):
        """Publish an event to all relevant subscribers."""
        # Store in history
        self._history.append(event)
        if len(self._history) > self._history_max_size:
            self._history.pop(0)
            
        callbacks = []
        # Get specific subscribers
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])
        # Get global subscribers
        if "*" in self._subscribers:
            callbacks.extend(self._subscribers["*"])
            
        if callbacks:
            # Execute all callbacks concurrently
            tasks = []
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(asyncio.create_task(callback(event)))
                else:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"[MessageBus] Sync callback error: {e}")
                        
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def get_history(self, workflow_id: Optional[str] = None) -> List[AgentEvent]:
        """Get event history, optionally filtered by workflow ID."""
        if workflow_id:
            return [e for e in self._history if e.workflow_id == workflow_id]
        return list(self._history)


# Global message bus instance
message_bus = MessageBus()
