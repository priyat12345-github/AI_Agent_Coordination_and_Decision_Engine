"""
Planner Agent — decomposes user requests into structured execution plans.
"""

from typing import Any, Dict

from agents.base_agent import BaseAgent
from prompts import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_TEMPLATE


class PlannerAgent(BaseAgent):
    """
    Takes a high-level user request and produces a structured JSON plan
    assigning sub-tasks to the appropriate specialist agents.
    """

    def __init__(self, **kwargs):
        super().__init__(name="PlannerAgent", **kwargs)

    @property
    def system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    def run(self, task: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Args:
            task: The original user request string.
            context: Shared memory snapshot or relevant key-value pairs.

        Returns:
            A parsed plan dict with 'goal' and 'tasks' keys.
        """
        context_str = "\n".join(
            f"{k}: {v}" for k, v in context.items()
        ) or "No prior context."

        human_msg = PLANNER_HUMAN_TEMPLATE.format(
            user_request=task,
            context=context_str,
        )

        raw = self._call_llm(human_msg)
        plan = self._parse_json(raw)

        # Attach metadata
        plan["agent"] = self.name
        plan["raw_request"] = task
        return plan
