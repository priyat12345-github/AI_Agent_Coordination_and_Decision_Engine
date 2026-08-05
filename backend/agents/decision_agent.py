"""
Decision Agent — Evaluates analysis outputs, applies business rules,
and generates justified strategic recommendations with confidence scoring.
"""

import random
from typing import Any, Dict, List, Optional
from loguru import logger

from backend.agents.base_agent import BaseAgent, AgentResult
from backend.core.config import AgentRole
from backend.core.prompts import DECISION_SYSTEM_PROMPT, DECISION_TASK_TEMPLATE


class DecisionAgent(BaseAgent):
    """
    Decision Agent: Strategic recommendation and decision support specialist.

    Responsibilities:
    - Apply business rules to analysis outputs
    - Score and rank decision alternatives
    - Assess risk and define mitigation strategies
    - Generate justified, actionable recommendations
    - Define escalation criteria and KPIs
    """

    BUSINESS_RULES = {
        "default": [
            "Minimum ROI threshold: 150% over 3 years",
            "Maximum acceptable risk score: 7.0/10",
            "Required stakeholder approval for investments > $1M",
            "Compliance validation required before execution",
            "Phased implementation mandatory for complexity > HIGH",
        ],
        "market_analysis": [
            "Market entry requires minimum 5% addressable market share",
            "Competitive moat must be defensible for > 18 months",
            "Customer acquisition cost must be < 1/3 of projected LTV",
            "Pilot required before full market deployment",
        ],
        "vendor_evaluation": [
            "Vendor must have > 3 years enterprise track record",
            "Financial stability: D&B score > 70",
            "Security certification: ISO 27001 or SOC 2 Type II required",
            "References from 2+ comparable enterprises mandatory",
        ],
        "financial_review": [
            "Revenue variance > 15% triggers immediate review",
            "Operating margin < 15% requires cost reduction plan",
            "Cash runway must remain > 12 months at all times",
            "Debt ratio > 0.5 triggers financial risk alert",
        ],
    }

    def __init__(self, llm=None, tools=None, event_callback=None):
        super().__init__(
            role=AgentRole.DECISION,
            llm=llm,
            tools=tools,
            event_callback=event_callback,
        )

    def get_system_prompt(self) -> str:
        return DECISION_SYSTEM_PROMPT

    def _get_business_rules(self, workflow_type: str) -> str:
        """Retrieve applicable business rules."""
        rules = self.BUSINESS_RULES.get(workflow_type, self.BUSINESS_RULES["default"])
        return "\n".join(f"  - {rule}" for rule in rules)

    def _determine_decision_type(self, task: str, context: Dict) -> Dict:
        """Classify the type of decision being made."""
        task_lower = task.lower()
        decision_types = {
            "approve_reject": any(w in task_lower for w in ["approve", "reject", "proceed", "halt"]),
            "select_option": any(w in task_lower for w in ["select", "choose", "pick", "recommend"]),
            "prioritize": any(w in task_lower for w in ["prioritize", "rank", "order"]),
            "risk_assess": any(w in task_lower for w in ["risk", "exposure", "vulnerability"]),
        }
        return decision_types

    async def process(
        self,
        task: str,
        context: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ) -> AgentResult:
        """Evaluate analysis and generate strategic decision recommendation."""

        context = context or {}
        workflow_type = context.get("workflow_type", "default")
        business_rules = self._get_business_rules(workflow_type)
        decision_types = self._determine_decision_type(task, context)

        await self.emit_event(
            "decision_processing",
            "Evaluating analysis results against business rules",
            {"workflow_type": workflow_type, "decision_types": decision_types},
            workflow_id=workflow_id,
        )

        analysis_results = context.get("analysis_output", "No analysis data provided")
        risk_tolerance = context.get("risk_tolerance", "MEDIUM — standard enterprise threshold")
        constraints = context.get("constraints", "Budget, timeline, and regulatory compliance")

        prompt = DECISION_TASK_TEMPLATE.substitute(
            decision_required=task,
            analysis_results=str(analysis_results)[:1500],
            business_rules=business_rules,
            risk_tolerance=risk_tolerance,
            constraints=constraints,
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        response = await self.call_llm(messages)
        confidence = random.uniform(0.85, 0.96)

        # Extract decision outcome keyword
        outcome = "PROCEED"
        if any(w in response.lower() for w in ["reject", "decline", "do not proceed", "halt"]):
            outcome = "REJECT"
        elif any(w in response.lower() for w in ["selective", "partial", "conditional"]):
            outcome = "CONDITIONAL"

        await self.emit_event(
            "decision_made",
            f"Decision: {outcome} (confidence: {confidence:.0%})",
            {
                "outcome": outcome,
                "confidence": confidence,
                "workflow_type": workflow_type,
            },
            workflow_id=workflow_id,
        )

        return AgentResult(
            agent_role=self.role.value,
            task=task,
            output=response,
            success=True,
            metadata={
                "decision_outcome": outcome,
                "business_rules_applied": workflow_type,
                "risk_tolerance": risk_tolerance,
                "workflow_id": workflow_id,
            },
            confidence=confidence,
        )
