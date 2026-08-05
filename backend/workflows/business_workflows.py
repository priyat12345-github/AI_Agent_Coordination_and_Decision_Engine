"""
Business Workflows — Predefined templates for common enterprise tasks.
"""

from typing import Dict, Any

from backend.orchestration.workflow_engine import engine
from backend.core.config import WORKFLOW_TEMPLATES

async def run_market_analysis(topic: str) -> Dict[str, Any]:
    """Execute a market analysis workflow."""
    request = f"Perform a comprehensive market analysis on: {topic}. I need to understand market size, key competitors, growth trends, and whether we should invest in this space."
    state = engine.create_workflow(request, "market_analysis")
    return await engine.execute_workflow(state.id)

async def run_vendor_evaluation(vendor_details: str) -> Dict[str, Any]:
    """Execute a vendor evaluation workflow."""
    request = f"Evaluate the following vendor proposal and recommend whether we should select them: {vendor_details}. Consider our enterprise standards for security and financial stability."
    state = engine.create_workflow(request, "vendor_evaluation")
    return await engine.execute_workflow(state.id)

async def run_financial_review(financial_data_context: str) -> Dict[str, Any]:
    """Execute a financial review workflow."""
    request = f"Review our current financial performance data: {financial_data_context}. Identify any risk areas, margin compressions, and provide strategic recommendations."
    state = engine.create_workflow(request, "financial_review")
    return await engine.execute_workflow(state.id)

async def run_customer_escalation(ticket_details: str) -> Dict[str, Any]:
    """Execute a customer escalation workflow."""
    request = f"Resolve this customer escalation: {ticket_details}. Assess churn risk, define priority, and draft a response plan."
    state = engine.create_workflow(request, "customer_escalation")
    return await engine.execute_workflow(state.id)
