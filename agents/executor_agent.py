"""
Executor Agent — performs concrete tasks based on plan + analyst findings.
"""

import json
from typing import Any, Dict

from agents.base_agent import BaseAgent
from prompts import EXECUTOR_SYSTEM_PROMPT, EXECUTOR_HUMAN_TEMPLATE


class ExecutorAgent(BaseAgent):
    """
    Takes a specific task and any analysis results, then carries out the work
    (e.g., data transformation, decision-making, computation) and returns a result.
    """

    def __init__(self, **kwargs):
        super().__init__(name="ExecutorAgent", **kwargs)

    @property
    def system_prompt(self) -> str:
        return EXECUTOR_SYSTEM_PROMPT

    def run(self, task: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Args:
            task: The specific task description to execute.
            context: Shared memory / key-value context.
            kwargs:
                task_id (int): Plan task ID.
                analysis (dict): Findings from the AnalystAgent, if available.

        Returns:
            Result dict with status, result, output, and errors.
        """
        task_id = kwargs.get("task_id", 0)
        analysis = kwargs.get("analysis", {})
        analysis_str = json.dumps(analysis, indent=2) if analysis else "No analysis provided."
        context_str = "\n".join(
            f"{k}: {v}" for k, v in context.items()
        ) or "No prior context."

        human_msg = EXECUTOR_HUMAN_TEMPLATE.format(
            task_id=task_id,
            task_description=task,
            analysis=analysis_str,
            context=context_str,
        )

        raw = self._call_llm(human_msg)
        result = self._parse_json(raw)
        result["agent"] = self.name
        return result
