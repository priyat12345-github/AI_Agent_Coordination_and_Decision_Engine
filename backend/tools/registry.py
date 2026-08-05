"""
Tool Registry — Central registry for all enterprise tools available to agents.
Handles tool discovery, validation, and lifecycle management.
"""

from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ToolDefinition:
    """Metadata and implementation for a registered tool."""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    implementation: Any
    is_async: bool = True
    requires_approval: bool = False
    timeout_seconds: int = 30
    tags: List[str] = field(default_factory=list)

    async def arun(self, **kwargs) -> Any:
        """Async invocation of the tool."""
        import asyncio
        if self.is_async and asyncio.iscoroutinefunction(self.implementation):
            return await self.implementation(**kwargs)
        elif asyncio.iscoroutinefunction(self.implementation):
            return await self.implementation(**kwargs)
        else:
            return self.implementation(**kwargs)

    def run(self, **kwargs) -> Any:
        """Sync invocation of the tool."""
        return self.implementation(**kwargs)


class ToolRegistry:
    """
    Central registry managing all tools available to agents.
    Supports registration, discovery, and grouped access.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool_def: ToolDefinition):
        """Register a tool in the registry."""
        self._tools[tool_def.name] = tool_def
        if tool_def.category not in self._categories:
            self._categories[tool_def.category] = []
        if tool_def.name not in self._categories[tool_def.category]:
            self._categories[tool_def.category].append(tool_def.name)
        logger.debug(f"Registered tool: {tool_def.name} [{tool_def.category}]")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def get_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_for_agent(self, agent_role: str) -> List[ToolDefinition]:
        """Get tools appropriate for a specific agent role."""
        role_tool_map = {
            "planner": ["web_search", "database_query"],
            "research": ["web_search", "database_query", "document_reader", "enterprise_api"],
            "analysis": ["calculator", "database_query", "enterprise_api"],
            "decision": ["database_query", "enterprise_api"],
            "executor": ["report_generator", "email_sender", "calendar_scheduler", "database_writer", "enterprise_api"],
        }
        tool_names = role_tool_map.get(agent_role, [])
        return [self._tools[n] for n in tool_names if n in self._tools]

    def list_all(self) -> List[Dict]:
        """List all registered tools with metadata."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "is_async": t.is_async,
                "requires_approval": t.requires_approval,
                "tags": t.tags,
            }
            for t in self._tools.values()
        ]

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name: str):
        return name in self._tools


# Global registry instance
registry = ToolRegistry()
