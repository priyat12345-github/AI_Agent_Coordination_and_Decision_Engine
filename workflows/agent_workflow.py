"""
Agent Workflow Orchestrator.
Coordinates the Planner → Analyst/Executor → Responder pipeline.
"""

import json
from typing import Any, Dict, List

from agents import AnalystAgent, ExecutorAgent, PlannerAgent, ResponderAgent
from memory.shared_memory import SharedMemory


class AgentWorkflow:
    """
    Orchestrates a full multi-agent workflow:

    1. PlannerAgent   → decomposes the user request into tasks
    2. AnalystAgent   → analyses tasks marked 'ANALYST'
    3. ExecutorAgent  → executes tasks marked 'EXECUTOR'
    4. ResponderAgent → synthesises all results into a final reply
    """

    def __init__(self, memory: SharedMemory | None = None):
        self.memory = memory or SharedMemory()
        self.planner = PlannerAgent()
        self.analyst = AnalystAgent()
        self.executor = ExecutorAgent()
        self.responder = ResponderAgent()

    # ─── Public API ──────────────────────────────────────────────────────────

    def run(self, user_request: str) -> Dict[str, Any]:
        """
        Run a full agent workflow for the given user request.

        Returns a dict with:
            - plan       : the structured execution plan
            - task_results: list of per-task agent outputs
            - response   : the final Markdown-formatted user reply
            - session_id : memory session identifier
        """
        self.memory.add_message("user", user_request)
        context = self.memory.snapshot()["store"]

        # ── Step 1: Plan ──────────────────────────────────────────────────
        print(f"\n[PlannerAgent] Decomposing request …")
        plan = self.planner.run(task=user_request, context=context)
        self.memory.set("plan", plan)
        self._log("Plan", plan)

        tasks: List[Dict[str, Any]] = plan.get("tasks", [])
        task_results: List[Dict[str, Any]] = []

        # ── Step 2: Execute each task ─────────────────────────────────────
        for task in tasks:
            task_id = task.get("id", 0)
            description = task.get("description", "")
            agent_role = task.get("agent", "EXECUTOR").upper()
            context = self.memory.snapshot()["store"]

            print(f"\n[{agent_role}] Running task {task_id}: {description}")

            if agent_role == "ANALYST":
                result = self.analyst.run(
                    task=description,
                    context=context,
                    task_id=task_id,
                )
            elif agent_role == "EXECUTOR":
                # Pull any prior analysis for this task if available
                prior_analysis = self._find_analysis(task_results, task_id)
                result = self.executor.run(
                    task=description,
                    context=context,
                    task_id=task_id,
                    analysis=prior_analysis,
                )
            else:
                # Default to executor for unknown roles
                result = self.executor.run(
                    task=description,
                    context=context,
                    task_id=task_id,
                )

            task_results.append(result)
            self.memory.set(f"task_{task_id}_result", result)
            self.memory.add_message("assistant", json.dumps(result), agent=result.get("agent"))

        # ── Step 3: Respond ───────────────────────────────────────────────
        print(f"\n[ResponderAgent] Composing final response …")
        context = self.memory.snapshot()["store"]
        final = self.responder.run(
            task=user_request,
            context=context,
            plan=plan,
            results=task_results,
        )
        self.memory.add_message("assistant", final.get("response", ""), agent="ResponderAgent")

        return {
            "plan": plan,
            "task_results": task_results,
            "response": final.get("response", ""),
            "session_id": self.memory.session_id,
        }

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_analysis(results: List[Dict], task_id: int) -> Dict:
        """Return the most recent ANALYST result that matches task_id, if any."""
        for r in reversed(results):
            if r.get("task_id") == task_id and r.get("agent") == "AnalystAgent":
                return r
        return {}

    @staticmethod
    def _log(label: str, data: Any) -> None:
        try:
            print(f"  └─ {label}: {json.dumps(data, indent=2)[:300]} …")
        except Exception:
            print(f"  └─ {label}: {str(data)[:300]} …")
