"""
Base Agent — Abstract foundation for all specialized AI agents.
Defines the standard interface, lifecycle hooks, logging, and tool integration patterns.
"""

import uuid
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from loguru import logger

from backend.core.llm_factory import get_llm, BaseLLM
from backend.core.config import AgentRole, AGENT_CONFIGS


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class AgentEvent:
    """Represents an event emitted by an agent during execution."""

    def __init__(
        self,
        agent_role: str,
        event_type: str,
        message: str,
        data: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        self.agent_role = agent_role
        self.event_type = event_type
        self.message = message
        self.data = data or {}
        self.workflow_id = workflow_id
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "agent_role": self.agent_role,
            "event_type": self.event_type,
            "message": self.message,
            "data": self.data,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp,
        }


class AgentResult:
    """Standardized result object returned by agent execution."""

    def __init__(
        self,
        agent_role: str,
        task: str,
        output: str,
        success: bool = True,
        metadata: Optional[Dict] = None,
        tools_used: Optional[List[str]] = None,
        execution_time: float = 0.0,
        confidence: float = 0.85,
    ):
        self.id = str(uuid.uuid4())
        self.agent_role = agent_role
        self.task = task
        self.output = output
        self.success = success
        self.metadata = metadata or {}
        self.tools_used = tools_used or []
        self.execution_time = execution_time
        self.confidence = confidence
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "agent_role": self.agent_role,
            "task": self.task,
            "output": self.output,
            "success": self.success,
            "metadata": self.metadata,
            "tools_used": self.tools_used,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all specialized AI agents.

    Provides:
    - Standardized run() lifecycle
    - LLM integration with retry logic
    - Tool binding and invocation
    - Event emission for real-time monitoring
    - Execution logging and audit trail
    """

    def __init__(
        self,
        role: AgentRole,
        llm: Optional[BaseLLM] = None,
        tools: Optional[List] = None,
        event_callback: Optional[Callable] = None,
    ):
        self.role = role
        self.config = AGENT_CONFIGS[role]
        self.name = self.config["name"]
        self.description = self.config["description"]
        self.llm = llm or get_llm()
        self.tools = tools or []
        self.tool_map: Dict[str, Any] = {}
        self.event_callback = event_callback
        self.status = AgentStatus.IDLE
        self.execution_count = 0
        self.total_tokens_used = 0
        self._register_tools()
        logger.info(f"Initialized {self.name}")

    def _register_tools(self):
        """Register tools into a lookup map."""
        for tool in self.tools:
            if hasattr(tool, "name"):
                self.tool_map[tool.name] = tool
            elif isinstance(tool, dict) and "name" in tool:
                self.tool_map[tool["name"]] = tool
        logger.debug(f"{self.name}: Registered {len(self.tool_map)} tools: {list(self.tool_map.keys())}")

    def bind_tools(self, tools: List):
        """Add additional tools to this agent."""
        self.tools.extend(tools)
        self._register_tools()

    async def emit_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ):
        """Emit an event for real-time monitoring."""
        event = AgentEvent(
            agent_role=self.role.value,
            event_type=event_type,
            message=message,
            data=data,
            workflow_id=workflow_id,
        )
        logger.info(f"[{self.name}] Event: {event_type} — {message}")
        if self.event_callback:
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(event)
            else:
                self.event_callback(event)
        return event

    async def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a registered tool by name."""
        if tool_name not in self.tool_map:
            raise ValueError(f"Tool '{tool_name}' not available to {self.name}")

        tool = self.tool_map[tool_name]
        await self.emit_event("tool_invoked", f"Using tool: {tool_name}", {"tool": tool_name, "args": kwargs})

        try:
            if hasattr(tool, "arun"):
                result = await tool.arun(**kwargs)
            elif hasattr(tool, "run"):
                result = tool.run(**kwargs)
            elif callable(tool):
                result = tool(**kwargs)
            else:
                raise ValueError(f"Tool '{tool_name}' is not callable")

            await self.emit_event("tool_completed", f"Tool '{tool_name}' completed", {"tool": tool_name})
            return result
        except Exception as e:
            await self.emit_event("tool_error", f"Tool '{tool_name}' failed: {str(e)}", {"tool": tool_name, "error": str(e)})
            raise

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Call the LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                response = await self.llm.ainvoke(messages)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"{self.name}: LLM call failed after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"{self.name}: LLM call failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    async def process(
        self,
        task: str,
        context: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ) -> AgentResult:
        """Core processing logic — implemented by each specialized agent."""
        pass

    async def run(
        self,
        task: str,
        context: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Main entry point for agent execution.
        Handles lifecycle: status updates, timing, logging, error handling.
        """
        start_time = datetime.utcnow()
        self.status = AgentStatus.RUNNING
        self.execution_count += 1

        await self.emit_event(
            "agent_started",
            f"{self.name} started processing task",
            {"task": task[:100], "execution_count": self.execution_count},
            workflow_id=workflow_id,
        )

        try:
            result = await self.process(task, context or {}, workflow_id)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            self.status = AgentStatus.COMPLETED

            await self.emit_event(
                "agent_completed",
                f"{self.name} completed in {execution_time:.2f}s",
                {
                    "execution_time": execution_time,
                    "confidence": result.confidence,
                    "tools_used": result.tools_used,
                    "success": result.success,
                },
                workflow_id=workflow_id,
            )

            logger.success(f"{self.name}: Task completed in {execution_time:.2f}s (confidence: {result.confidence:.0%})")
            return result

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.status = AgentStatus.ERROR
            logger.error(f"{self.name}: Task failed after {execution_time:.2f}s — {e}")

            await self.emit_event(
                "agent_error",
                f"{self.name} encountered an error: {str(e)[:100]}",
                {"error": str(e), "execution_time": execution_time},
                workflow_id=workflow_id,
            )

            return AgentResult(
                agent_role=self.role.value,
                task=task,
                output=f"Agent encountered an error: {str(e)}",
                success=False,
                execution_time=execution_time,
                confidence=0.0,
            )

    def describe(self) -> Dict:
        """Return a description of this agent."""
        return {
            "role": self.role.value,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "execution_count": self.execution_count,
            "tools": list(self.tool_map.keys()),
            "icon": self.config["icon"],
            "color": self.config["color"],
        }
