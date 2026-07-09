"""
Responder Agent — synthesises all agent results into a user-facing response.
"""

import json
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from prompts import RESPONDER_SYSTEM_PROMPT, RESPONDER_HUMAN_TEMPLATE


class ResponderAgent(BaseAgent):
    """
    Reads the original request, the execution plan, and all intermediate
    results, then composes a clear, professional, Markdown-formatted reply.
    """

    def __init__(self, **kwargs):
        super().__init__(name="ResponderAgent", **kwargs)

    @property
    def system_prompt(self) -> str:
        return RESPONDER_SYSTEM_PROMPT

    def run(self, task: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Args:
            task: The original user request.
            context: Shared memory context (unused directly).
            kwargs:
                plan (dict): The plan produced by PlannerAgent.
                results (list[dict]): List of results from Analyst/Executor agents.

        Returns:
            Dict with 'response' key containing the final Markdown text.
        """
        plan: Dict[str, Any] = kwargs.get("plan", {})
        results: List[Dict[str, Any]] = kwargs.get("results", [])

        plan_str = json.dumps(plan, indent=2)
        results_str = json.dumps(results, indent=2)

        human_msg = RESPONDER_HUMAN_TEMPLATE.format(
            user_request=task,
            plan=plan_str,
            results=results_str,
        )

        response_text = self._call_llm(human_msg)
        return {
            "agent": self.name,
            "response": response_text,
        }
