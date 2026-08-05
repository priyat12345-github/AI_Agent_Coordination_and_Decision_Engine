"""
Workflow Engine — Orchestrates multi-agent execution using LangGraph principles.
"""

import uuid
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from backend.core.config import AgentRole, WorkflowStatus, WORKFLOW_TEMPLATES
from backend.agents.planner_agent import PlannerAgent
from backend.agents.research_agent import ResearchAgent
from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.decision_agent import DecisionAgent
from backend.agents.executor_agent import ExecutorAgent
from backend.tools.registry import registry
from backend.memory.memory_manager import memory_manager
from backend.orchestration.message_bus import message_bus


class WorkflowState:
    """Manages the state of a running workflow."""
    
    def __init__(self, request: str, workflow_type: str = "custom"):
        self.id = f"WF-{str(uuid.uuid4())[:8].upper()}"
        self.request = request
        self.workflow_type = workflow_type
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.utcnow().isoformat()
        self.started_at = None
        self.completed_at = None
        self.context: Dict[str, Any] = {
            "workflow_id": self.id,
            "workflow_type": workflow_type,
            "request": request,
        }
        self.plan: Optional[Dict] = None
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "request": self.request,
            "workflow_type": self.workflow_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "plan": self.plan,
        }


class WorkflowEngine:
    """
    Executes and coordinates complex multi-agent workflows.
    """

    def __init__(self):
        self._active_workflows: Dict[str, WorkflowState] = {}
        self._completed_workflows: Dict[str, WorkflowState] = {}
        
        # Initialize agents
        self.planner = PlannerAgent(event_callback=message_bus.publish)
        self.research = ResearchAgent(tools=registry.get_for_agent("research"), event_callback=message_bus.publish)
        self.analysis = AnalysisAgent(tools=registry.get_for_agent("analysis"), event_callback=message_bus.publish)
        self.decision = DecisionAgent(tools=registry.get_for_agent("decision"), event_callback=message_bus.publish)
        self.executor = ExecutorAgent(tools=registry.get_for_agent("executor"), event_callback=message_bus.publish)
        
        # Agent map for dynamic routing
        self.agent_map = {
            AgentRole.PLANNER.value: self.planner,
            AgentRole.RESEARCH.value: self.research,
            AgentRole.ANALYSIS.value: self.analysis,
            AgentRole.DECISION.value: self.decision,
            AgentRole.EXECUTOR.value: self.executor,
        }

    def create_workflow(self, request: str, workflow_type: str = "custom") -> WorkflowState:
        """Create a new workflow instance."""
        state = WorkflowState(request, workflow_type)
        self._active_workflows[state.id] = state
        return state

    async def execute_workflow(self, workflow_id: str) -> Dict:
        """Execute a workflow through the standard pipeline."""
        if workflow_id not in self._active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        state = self._active_workflows[workflow_id]
        state.status = WorkflowStatus.RUNNING
        state.started_at = datetime.utcnow().isoformat()
        
        try:
            # STEP 1: PLANNING
            logger.info(f"[{state.id}] Starting Planning Phase")
            planner_result = await self.planner.run(
                task=state.request,
                context=state.context,
                workflow_id=state.id
            )
            state.plan = planner_result.metadata.get("plan", {})
            state.context["plan_output"] = planner_result.output
            memory_manager.record_agent_output(state.id, "planner", state.request, planner_result.output, state.id)
            
            # STEP 2: RESEARCH
            if "research" in state.plan.get("agents_required", ["research"]):
                logger.info(f"[{state.id}] Starting Research Phase")
                research_task = f"Research and retrieve data for: {state.request}"
                research_result = await self.research.run(
                    task=research_task,
                    context=state.context,
                    workflow_id=state.id
                )
                state.context["research_output"] = research_result.output
                memory_manager.record_agent_output(state.id, "research", research_task, research_result.output, state.id)
            
            # STEP 3: ANALYSIS
            if "analysis" in state.plan.get("agents_required", ["analysis"]):
                logger.info(f"[{state.id}] Starting Analysis Phase")
                analysis_task = "Analyze the research findings."
                analysis_result = await self.analysis.run(
                    task=analysis_task,
                    context=state.context,
                    workflow_id=state.id
                )
                state.context["analysis_output"] = analysis_result.output
                memory_manager.record_agent_output(state.id, "analysis", analysis_task, analysis_result.output, state.id)
                
            # STEP 4: DECISION
            if "decision" in state.plan.get("agents_required", ["decision"]):
                logger.info(f"[{state.id}] Starting Decision Phase")
                decision_task = "Evaluate analysis and provide recommendation."
                decision_result = await self.decision.run(
                    task=decision_task,
                    context=state.context,
                    workflow_id=state.id
                )
                state.context["decision_output"] = decision_result.output
                state.context["decision_metadata"] = decision_result.metadata
                memory_manager.record_agent_output(state.id, "decision", decision_task, decision_result.output, state.id)
                
            # STEP 5: EXECUTION
            if "executor" in state.plan.get("agents_required", ["executor"]):
                logger.info(f"[{state.id}] Starting Execution Phase")
                executor_task = f"Generate final answer and deliverable for: {state.request}"
                executor_result = await self.executor.run(
                    task=executor_task,
                    context=state.context,
                    workflow_id=state.id
                )
                state.context["execution_output"] = executor_result.output
                state.results["deliverables"] = executor_result.metadata.get("deliverables", [])
                memory_manager.record_agent_output(state.id, "executor", executor_task, executor_result.output, state.id)
                
            state.status = WorkflowStatus.COMPLETED
            logger.success(f"[{state.id}] Workflow completed successfully")
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.errors.append(str(e))
            logger.error(f"[{state.id}] Workflow failed: {e}")
            
        finally:
            state.completed_at = datetime.utcnow().isoformat()
            self._completed_workflows[state.id] = state
            del self._active_workflows[state.id]
            
        return {
            "workflow_id": state.id,
            "status": state.status.value,
            "deliverables": state.results.get("deliverables", []),
            "errors": state.errors,
            "final_context": state.context
        }

    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow status and details."""
        if workflow_id in self._active_workflows:
            return self._active_workflows[workflow_id].to_dict()
        if workflow_id in self._completed_workflows:
            return self._completed_workflows[workflow_id].to_dict()
        return None

# Global workflow engine instance
engine = WorkflowEngine()
