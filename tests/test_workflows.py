"""
Integration tests for the AgentWorkflow orchestrator.
Run with: python -m pytest tests/test_workflows.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


MOCK_PLAN = {
    "goal": "Analyse Q2 sales and generate a report",
    "tasks": [
        {"id": 1, "description": "Analyse Q2 sales figures", "agent": "ANALYST", "depends_on": []},
        {"id": 2, "description": "Generate summary report", "agent": "EXECUTOR", "depends_on": [1]},
    ],
    "agent": "PlannerAgent",
    "raw_request": "Analyse Q2 sales and generate a report",
}

MOCK_ANALYSIS = {
    "task_id": 1,
    "findings": "Q2 sales grew 15% YoY",
    "key_points": ["Growth", "New markets"],
    "confidence": "HIGH",
    "recommendations": ["Expand marketing"],
    "agent": "AnalystAgent",
}

MOCK_EXECUTION = {
    "task_id": 2,
    "status": "SUCCESS",
    "result": "Report generated",
    "output": "Sales grew 15% YoY. Recommend expanding marketing.",
    "errors": [],
    "agent": "ExecutorAgent",
}

MOCK_RESPONSE = "## Q2 Sales Report\n\nSales grew **15% YoY**. We recommend expanding marketing efforts."


@pytest.fixture
def workflow_with_mocks():
    """Return an AgentWorkflow with all agents replaced by mocks."""
    from workflows.agent_workflow import AgentWorkflow

    wf = AgentWorkflow()
    wf.planner.run = MagicMock(return_value=MOCK_PLAN)
    wf.analyst.run = MagicMock(return_value=MOCK_ANALYSIS)
    wf.executor.run = MagicMock(return_value=MOCK_EXECUTION)
    wf.responder.run = MagicMock(return_value={"agent": "ResponderAgent", "response": MOCK_RESPONSE})
    return wf


def test_workflow_run_returns_all_keys(workflow_with_mocks):
    result = workflow_with_mocks.run("Analyse Q2 sales and generate a report")
    assert "plan" in result
    assert "task_results" in result
    assert "response" in result
    assert "session_id" in result


def test_workflow_calls_planner(workflow_with_mocks):
    workflow_with_mocks.run("Test request")
    workflow_with_mocks.planner.run.assert_called_once()


def test_workflow_calls_analyst_for_analyst_tasks(workflow_with_mocks):
    workflow_with_mocks.run("Test request")
    workflow_with_mocks.analyst.run.assert_called_once()


def test_workflow_calls_executor_for_executor_tasks(workflow_with_mocks):
    workflow_with_mocks.run("Test request")
    workflow_with_mocks.executor.run.assert_called_once()


def test_workflow_calls_responder(workflow_with_mocks):
    workflow_with_mocks.run("Test request")
    workflow_with_mocks.responder.run.assert_called_once()


def test_workflow_response_content(workflow_with_mocks):
    result = workflow_with_mocks.run("Test request")
    assert "Q2 Sales Report" in result["response"]


def test_workflow_memory_stores_plan(workflow_with_mocks):
    workflow_with_mocks.run("Test request")
    stored_plan = workflow_with_mocks.memory.get("plan")
    assert stored_plan is not None
    assert stored_plan["goal"] == MOCK_PLAN["goal"]
