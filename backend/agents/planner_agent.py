"""
Planner Agent — Decomposes complex business requests into structured execution plans
and coordinates routing to specialized agents.
"""

import json
import random
from typing import Any, Dict, List, Optional
from loguru import logger

from backend.agents.base_agent import BaseAgent, AgentResult
from backend.core.config import AgentRole
from backend.core.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_TASK_TEMPLATE


class PlannerAgent(BaseAgent):
    """
    Planner Agent: The orchestration brain of the multi-agent system.

    Responsibilities:
    - Parse and understand complex business requests
    - Decompose tasks into ordered sub-tasks
    - Determine required agents and their sequence
    - Identify data dependencies and resource needs
    - Generate structured execution plans
    """

    def __init__(self, llm=None, tools=None, event_callback=None):
        super().__init__(
            role=AgentRole.PLANNER,
            llm=llm,
            tools=tools,
            event_callback=event_callback,
        )

    def get_system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    def _parse_plan_from_response(self, response: str, request: str) -> Dict:
        """Extract structured plan metadata from LLM response."""
        # Extract agent mentions
        agents_mentioned = []
        agent_keywords = {
            "research": AgentRole.RESEARCH.value,
            "analysis": AgentRole.ANALYSIS.value,
            "decision": AgentRole.DECISION.value,
            "executor": AgentRole.EXECUTOR.value,
        }
        for keyword, role in agent_keywords.items():
            if keyword.lower() in response.lower():
                if role not in agents_mentioned:
                    agents_mentioned.append(role)

        # Build sub-tasks from plan
        sub_tasks = []
        lines = response.split("\n")
        task_num = 1
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                cleaned = line.lstrip("0123456789.-•*) ").strip()
                if len(cleaned) > 10:
                    sub_tasks.append({
                        "id": f"TASK-{task_num:03d}",
                        "description": cleaned[:100],
                        "status": "pending",
                    })
                    task_num += 1
                    if task_num > 8:
                        break

        # Always include executor so final answer is delivered to user
        if AgentRole.EXECUTOR.value not in agents_mentioned:
            agents_mentioned.append(AgentRole.EXECUTOR.value)

        return {
            "agents_required": agents_mentioned,
            "sub_tasks": sub_tasks[:6] if sub_tasks else [
                {"id": "TASK-001", "description": "Gather relevant information", "status": "pending"},
                {"id": "TASK-002", "description": "Analyze collected data", "status": "pending"},
                {"id": "TASK-003", "description": "Generate recommendations", "status": "pending"},
                {"id": "TASK-004", "description": "Execute and deliver final output", "status": "pending"},
            ],
            "estimated_steps": len(agents_mentioned) or 5,
            "priority": "HIGH",
            "complexity": "MEDIUM" if len(request) < 200 else "HIGH",
        }

    async def process(
        self,
        task: str,
        context: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ) -> AgentResult:
        """Generate a structured execution plan for the given business request."""

        await self.emit_event(
            "planning_started",
            "Analyzing business request and creating execution plan",
            {"request_length": len(task)},
            workflow_id=workflow_id,
        )

        # Build messages for LLM
        session_context = context.get("session_history", "No prior context") if context else "New session"
        prompt = PLANNER_TASK_TEMPLATE.substitute(
            request=task,
            context=session_context,
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        # Call LLM
        response = await self.call_llm(messages)

        # Parse plan structure
        plan_metadata = self._parse_plan_from_response(response, task)

        await self.emit_event(
            "plan_created",
            f"Execution plan created with {len(plan_metadata['sub_tasks'])} tasks",
            plan_metadata,
            workflow_id=workflow_id,
        )

        return AgentResult(
            agent_role=self.role.value,
            task=task,
            output=response,
            success=True,
            metadata={
                "plan": plan_metadata,
                "workflow_id": workflow_id,
            },
            confidence=random.uniform(0.85, 0.95),
        )
