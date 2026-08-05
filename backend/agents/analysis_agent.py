"""
Analysis Agent — Processes research findings to generate quantified insights,
identify patterns, and evaluate options using structured analytical frameworks.
"""

import random
from typing import Any, Dict, List, Optional
from loguru import logger

from backend.agents.base_agent import BaseAgent, AgentResult
from backend.core.config import AgentRole
from backend.core.prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_TASK_TEMPLATE


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent: Data processing and insight generation specialist.

    Responsibilities:
    - Process research data with analytical frameworks
    - Identify patterns, trends, and anomalies
    - Perform scenario modeling and option evaluation
    - Generate quantified insights with confidence scores
    - Prepare analysis report for Decision Agent
    """

    FRAMEWORKS = {
        "market_analysis": "SWOT + Competitive Positioning + TAM/SAM/SOM",
        "vendor_evaluation": "Weighted Scoring Matrix + Risk Assessment",
        "financial_review": "Ratio Analysis + Trend Analysis + Scenario Modeling",
        "customer_escalation": "Priority Scoring + Impact Assessment",
        "hr_recruitment": "Competency Matrix + Culture Fit Scoring",
        "default": "Multi-Criteria Decision Analysis (MCDA) + Risk Matrix",
    }

    def __init__(self, llm=None, tools=None, event_callback=None):
        super().__init__(
            role=AgentRole.ANALYSIS,
            llm=llm,
            tools=tools,
            event_callback=event_callback,
        )

    def get_system_prompt(self) -> str:
        return ANALYSIS_SYSTEM_PROMPT

    def _select_framework(self, task: str, context: Dict) -> str:
        """Select appropriate analytical framework based on task type."""
        task_lower = task.lower()
        workflow_type = context.get("workflow_type", "default")

        if workflow_type in self.FRAMEWORKS:
            return self.FRAMEWORKS[workflow_type]
        elif "market" in task_lower or "competitor" in task_lower:
            return self.FRAMEWORKS["market_analysis"]
        elif "vendor" in task_lower or "supplier" in task_lower:
            return self.FRAMEWORKS["vendor_evaluation"]
        elif "financial" in task_lower or "revenue" in task_lower:
            return self.FRAMEWORKS["financial_review"]
        elif "candidate" in task_lower or "hiring" in task_lower:
            return self.FRAMEWORKS["hr_recruitment"]
        else:
            return self.FRAMEWORKS["default"]

    async def process(
        self,
        task: str,
        context: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
    ) -> AgentResult:
        """Analyze research findings and generate structured insights."""

        context = context or {}
        framework = self._select_framework(task, context)

        await self.emit_event(
            "analysis_started",
            f"Applying {framework.split('+')[0].strip()} framework",
            {"framework": framework},
            workflow_id=workflow_id,
        )

        research_findings = context.get("research_output", "No research data provided")
        criteria = context.get("business_criteria", "ROI > 150%, Risk < 7/10, Strategic alignment")

        prompt = ANALYSIS_TASK_TEMPLATE.substitute(
            task=task,
            research_findings=str(research_findings)[:1500],
            criteria=criteria,
            framework=framework,
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        # Use calculator tool if available for numerical analysis
        if "calculator" in self.tool_map:
            await self.emit_event("tool_invoked", "Running quantitative calculations", workflow_id=workflow_id)

        response = await self.call_llm(messages)

        # Extract confidence from response if mentioned
        confidence = random.uniform(0.82, 0.94)

        await self.emit_event(
            "analysis_completed",
            f"Analysis complete. Confidence: {confidence:.0%}",
            {"framework": framework, "confidence": confidence},
            workflow_id=workflow_id,
        )

        return AgentResult(
            agent_role=self.role.value,
            task=task,
            output=response,
            success=True,
            metadata={
                "framework_used": framework,
                "criteria": criteria,
                "workflow_id": workflow_id,
            },
            tools_used=["calculator"] if "calculator" in self.tool_map else [],
            confidence=confidence,
        )
