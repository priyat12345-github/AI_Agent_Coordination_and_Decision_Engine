"""
Workflows API Router
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List

from backend.orchestration.workflow_engine import engine
from backend.workflows.business_workflows import (
    run_market_analysis,
    run_vendor_evaluation,
    run_financial_review,
    run_customer_escalation
)

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

class WorkflowRequest(BaseModel):
    request: str
    workflow_type: str = "custom"

@router.post("/")
async def launch_workflow(req: WorkflowRequest, background_tasks: BackgroundTasks):
    """Launch a new multi-agent workflow."""
    
    # Create the workflow state synchronously so we return the ID
    state = engine.create_workflow(req.request, req.workflow_type)
    
    # Run the actual execution in the background
    background_tasks.add_task(engine.execute_workflow, state.id)
    
    return {
        "status": "launched",
        "workflow_id": state.id,
        "message": "Workflow executing in background"
    }

@router.get("/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get the status and results of a specific workflow."""
    workflow = engine.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Also attach the full context from completed workflows
    if workflow_id in engine._completed_workflows:
        state = engine._completed_workflows[workflow_id]
        result = workflow.copy()
        result["context"] = state.context
        return result
    return workflow
