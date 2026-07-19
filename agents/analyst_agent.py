"""
Analyst Agent — analyses information and produces structured findings.
"""

from typing import Any, Dict

from agents.base_agent import BaseAgent
from prompts import ANALYST_SYSTEM_PROMPT, ANALYST_HUMAN_TEMPLATE
from tools import search_knowledge_base


class AnalystAgent(BaseAgent):
    """
    Processes a sub-task by analysing available context and returning
    structured findings with key points, confidence level, and recommendations.
    """

    def __init__(self, **kwargs):
        tools = [search_knowledge_base]
        super().__init__(name="AnalystAgent", tools=tools, **kwargs)

    @property
    def system_prompt(self) -> str:
        return ANALYST_SYSTEM_PROMPT

    def run(self, task: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Args:
            task: Description of the analysis sub-task.
            context: Relevant context from shared memory.
            kwargs:
                task_id (int): The plan task ID this maps to.

        Returns:
            Parsed findings dict with key_points, confidence, recommendations.
        """
        task_id = kwargs.get("task_id", 0)
        context_str = "\n".join(
            f"{k}: {v}" for k, v in context.items()
        ) or "No prior context."

        human_msg = ANALYST_HUMAN_TEMPLATE.format(
            task_id=task_id,
            task_description=task,
            context=context_str,
        )

        raw = self._call_llm(human_msg)
        result = self._parse_json(raw)
        result["agent"] = self.name
        return result
