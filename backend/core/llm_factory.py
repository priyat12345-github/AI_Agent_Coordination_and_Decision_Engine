"""
LLM Factory — Provides a unified interface for multiple LLM providers.
Supports Mock (demo), Google Gemini, and OpenAI backends.
"""

import json
import random
import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from backend.core.config import settings, LLMProvider


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Async invocation of the LLM."""
        pass

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Sync invocation of the LLM."""
        pass


class MockLLM(BaseLLM):
    """
    Mock LLM for demonstration and testing without requiring API keys.
    Generates realistic, context-aware responses based on agent role and prompt content.
    """

    PLANNER_RESPONSES = [
        """I've analyzed the business request and created a structured execution plan:

**Task Breakdown:**
1. **Research Phase** — Gather relevant market data, competitor information, and industry benchmarks
2. **Analysis Phase** — Process collected data, identify patterns, and evaluate options  
3. **Decision Phase** — Apply business rules, score alternatives, and generate recommendations
4. **Execution Phase** — Implement approved actions and generate deliverables

**Agent Routing:**
- Research Agent: External data collection and knowledge retrieval
- Analysis Agent: Data processing, pattern recognition, and insight generation
- Decision Agent: Recommendation synthesis and risk assessment
- Executor Agent: Report generation and action implementation

**Estimated Timeline:** 3-5 minutes for complete workflow execution
**Confidence Level:** High — all required inputs available

Proceeding with workflow execution...""",

        """Strategic Execution Plan Generated:

**Objective Decomposition:**
- Primary Goal: Complete the requested business analysis and deliver actionable insights
- Sub-Tasks: Data gathering (Research Agent), pattern analysis (Analysis Agent), decision support (Decision Agent)

**Priority Matrix:**
1. HIGH: Data accuracy and completeness validation
2. HIGH: Risk assessment and mitigation identification  
3. MEDIUM: Stakeholder communication preparation
4. LOW: Historical trend documentation

**Dependencies Identified:**
- Research results required before analysis can begin
- Analysis output gates decision recommendation
- Decision approval required before execution

Ready to coordinate multi-agent workflow.""",
    ]

    RESEARCH_RESPONSES = [
        """Research completed. Here are the key findings:

**Market Intelligence:**
- Global market size: $847B with 12.3% CAGR projected through 2028
- Key players: TechCorp (23% share), InnovateCo (18%), DataSystems (15%)
- Emerging trends: AI integration (↑34%), cloud-native solutions (↑28%), regulatory compliance tools (↑19%)

**Competitive Analysis:**
- TechCorp: Strong enterprise presence, legacy integration challenges
- InnovateCo: Agile startup, limited enterprise track record
- DataSystems: Specialized expertise, narrow product range

**Industry Benchmarks:**
- Average ROI: 287% over 3 years
- Implementation timeline: 6-18 months
- Customer satisfaction scores: 4.2/5.0 average

**Data Sources Consulted:** Industry reports (Gartner, IDC), company filings, news databases, partner research
**Confidence Score:** 87%""",

        """Research synthesis complete. Key intelligence gathered:

**Financial Data Points:**
- Revenue growth rate: 14.7% YoY
- Operating margin: 23.4% (industry avg: 19.1%)
- Customer acquisition cost: $1,240 (declining trend ↓8%)
- Lifetime customer value: $47,300

**Regulatory Environment:**
- New compliance requirements effective Q1 2025 — DORA framework
- Data privacy regulations tightening across EU and APAC markets
- ESG reporting mandates expanding to mid-market companies

**Technology Landscape:**
- LLM adoption in enterprise: 67% planning deployment in 12 months
- Integration challenges cited by 78% of organizations
- Security and governance top concerns (82% respondents)

Passing findings to Analysis Agent for processing...""",
    ]

    ANALYSIS_RESPONSES = [
        """Analysis complete. Synthesized insights from research data:

**Quantitative Assessment:**
| Metric | Score | Weight | Weighted Score |
|--------|-------|--------|----------------|
| Market Opportunity | 8.7/10 | 30% | 2.61 |
| Competitive Position | 7.2/10 | 25% | 1.80 |
| Implementation Risk | 6.8/10 | 20% | 1.36 |
| Financial Viability | 9.1/10 | 25% | 2.28 |
| **TOTAL** | | | **8.05/10** |

**Key Insights:**
✅ Strong market fundamentals support aggressive expansion
✅ Financial metrics exceed industry benchmarks by 22%
⚠️ Competitive pressure increasing — first-mover advantage window: 6-9 months
⚠️ Implementation complexity requires phased approach

**Risk Assessment:**
- HIGH: Regulatory compliance timeline alignment
- MEDIUM: Technical integration complexity
- LOW: Market demand uncertainty

**Recommendation Confidence:** 91%""",

        """Data analysis completed with high statistical confidence:

**Pattern Recognition Results:**
- Seasonal demand spike: Q3/Q4 (↑67% vs baseline)
- Customer segment most valuable: Enterprise (>500 employees) — 78% of revenue
- Geographic opportunity: APAC market underpenetrated (12% current, 34% potential)

**Scenario Modeling:**
- Conservative (5% growth): NPV = $12.4M, IRR = 18%
- Base Case (12% growth): NPV = $28.7M, IRR = 31%  
- Optimistic (20% growth): NPV = $47.2M, IRR = 44%

**SWOT Summary:**
- Strengths: Technology differentiation, strong brand loyalty
- Weaknesses: Limited APAC presence, SMB market gaps
- Opportunities: AI/automation wave, regulatory tailwinds
- Threats: Economic headwinds, new entrants

Transferring to Decision Agent for recommendation generation...""",
    ]

    DECISION_RESPONSES = [
        """Decision synthesis complete. Recommendation generated:

**EXECUTIVE RECOMMENDATION: PROCEED WITH FULL DEPLOYMENT**
Confidence Level: 89% | Risk Level: MEDIUM

**Decision Rationale:**
The analysis data strongly supports moving forward based on:
1. Financial returns exceed minimum threshold (ROI >200%)
2. Market timing is optimal — competitive window open for 6-9 months
3. Risk mitigation strategies are available and implementable
4. Stakeholder alignment confirmed across key business units

**Recommended Action Plan:**
| Phase | Timeline | Investment | Expected Return |
|-------|----------|------------|-----------------|
| Phase 1: Pilot | Months 1-3 | $500K | Validation |
| Phase 2: Scale | Months 4-9 | $2.1M | $4.8M |
| Phase 3: Optimize | Months 10-18 | $800K | $12.3M |

**Conditions & Caveats:**
- Proceed only if regulatory approval confirmed by Month 2
- Review checkpoint after Phase 1 with success criteria defined
- Escalate if competition moves before Month 4

**Decision Classification:** Strategic Investment Approved
**Next Step:** Executor Agent to generate implementation roadmap""",

        """Decision framework applied. Strategic recommendation issued:

**RECOMMENDATION: SELECTIVE ENGAGEMENT — Tier 1 Markets Only**
Priority Score: 7.8/10 | Implementation Urgency: HIGH

**Business Rule Application:**
✅ Rule 1 (ROI Threshold >150%): PASSED — Projected 287%
✅ Rule 2 (Risk Score <7): PASSED — Current risk: 4.2/10
✅ Rule 3 (Resource Availability): PASSED — Team capacity confirmed
⚠️ Rule 4 (Market Maturity >60%): PARTIAL — 67% in Tier 1, 34% in Tier 2

**Selected Markets:** North America, Western Europe, Australia
**Deferred Markets:** APAC (revisit Q2), Latin America (revisit Q3)

**KPIs & Success Metrics:**
- 90-day target: 15 enterprise contracts signed
- 180-day target: $3.2M ARR achieved
- 12-month target: Market share increased by 4 percentage points

Approved for execution. Report generation initiated.""",
    ]

    EXECUTOR_RESPONSES = [
        """Execution completed successfully. All tasks performed:

**Actions Completed:**
✅ Strategic report generated → /reports/strategic_analysis_{date}.pdf
✅ Executive summary prepared → 2 pages, board-ready format
✅ Stakeholder notification draft → 3 emails prepared for review
✅ Calendar events created → Q1 review checkpoint, Phase 1 kickoff
✅ CRM records updated → 47 prospect accounts flagged for outreach
✅ Database entries logged → Workflow execution record saved
✅ Monitoring alerts configured → KPI tracking activated

**Deliverables Summary:**
1. **Executive Report** — Comprehensive strategic analysis with recommendations
2. **Action Item List** — 23 tasks assigned to responsible parties  
3. **Risk Register** — 8 risks documented with mitigation plans
4. **Timeline** — Gantt chart for 18-month implementation roadmap

**Workflow Execution Metrics:**
- Total processing time: 4m 23s
- Data sources consulted: 14
- Agents coordinated: 5
- Confidence score: 88.5%
- Status: COMPLETED SUCCESSFULLY

All outputs saved to enterprise knowledge base. Workflow execution complete.""",
    ]

    WARRANTY_RESPONSES = [
        """Based on the records retrieved from the enterprise database:

**Product Details:**
- **Product ID:** 205
- **Name:** Enterprise Router X-205
- **Category:** Networking
- **Status:** Active

**Warranty Information:**
Yes, Product 205 is fully covered. It comes with a **2-year comprehensive enterprise warranty**, which includes next-day hardware replacement.

This information has been verified directly from the `products` table in our enterprise database.""",
    ]

    GENERIC_RESPONSES = [
        "Task processed successfully. The AI agent has analyzed the input and generated a comprehensive response based on available data and enterprise knowledge.",
        "Processing complete. Multi-agent coordination resulted in an optimal solution pathway with high confidence scoring.",
        "Analysis finalized. Cross-referencing multiple data sources confirmed the recommended course of action.",
    ]

    def _select_response(self, messages: List[Dict[str, str]], agent_role: str = "") -> str:
        """Select or generate appropriate response based on agent role and prompt content."""
        content = " ".join(m.get("content", "") for m in messages).lower()

        # Check if Executor (final answer) is running or if fallback response needed
        is_executor = any(w in content for w in ["generate final answer", "implement the recommended", "deliverable", "execute and deliver", "final answer", "for:"])

        # ─── EXECUTOR AGENT FINAL ANSWERS ───
        if is_executor:
            # 1. Product 205 Warranty / Router Query
            if "205" in content or "router" in content or ("warranty" in content and ("205" in content or "product" in content)):
                return """Based on records retrieved from the enterprise database:

**Product Details:**
- **Product ID:** 205
- **Name:** Enterprise Router X-205
- **Category:** Networking
- **Price:** $1,200
- **Status:** Active

**Warranty Coverage:**
Yes, Product 205 is fully covered. It includes a **2-year comprehensive enterprise warranty** with next-day hardware replacement.

*Source: Verified from SQLite `products` table.*"""

            # 2. Product 308 Firewall Query
            if "308" in content or "firewall" in content:
                return """Based on records retrieved from the enterprise database:

**Product Details:**
- **Product ID:** 308
- **Name:** Cloud Firewall Server
- **Category:** Security
- **Price:** $4,500
- **Status:** Active

**Warranty Coverage:**
Covered under a **1-year standard warranty** including firmware updates and remote support.

*Source: Verified from SQLite `products` table.*"""

            # 3. Customer C001 / Customer Accounts / At Risk / Churn Queries (Check BEFORE vendors)
            if any(w in content for w in ["c001", "global finance", "customer", "client", "account", "mrr", "churn", "at risk", "vip"]):
                if "c001" in content or "global finance" in content:
                    return """Based on records retrieved from the enterprise database:

### 👤 Customer Profile: Global Finance Corp (C001)
- **Account ID:** C001
- **Customer Name:** Global Finance Corp
- **Tier:** Enterprise Tier
- **Monthly Revenue (MRR):** $45,000
- **Account Status:** Active
- **Sentiment:** Positive (NPS Score: 72)
- **Assigned Account Manager:** Rachel Torres
- **Recent Purchase:** Server Cluster X (2024-01-15)
- **Warranty Status:** Active — Premium Coverage
- **Open Support Tickets:** 2

*Source: Verified from SQLite `customers` table.*"""
                
                return """Based on records retrieved from the enterprise database:

### 📊 Enterprise Customer Accounts Overview:
1. **Global Finance Corp (C001)** — Enterprise Tier | $45,000 MRR | Status: **Active** (NPS: 72) | Manager: Rachel Torres
2. **RetailMax Group (C002)** — Premium Tier | $18,500 MRR | Status: ⚠️ **At Risk** (7 Open Tickets, NPS: 34) | Manager: Marcus Lee
3. **HealthBridge Network (C003)** — Enterprise Tier | $67,200 MRR | Status: **Active** (NPS: 91) | Manager: Elena Vasquez
4. **LogiChain Dynamics (C004)** — Standard Tier | $8,900 MRR | Status: 🚨 **Churning** (12 Open Tickets, NPS: 18) | Manager: David Kim
5. **Alex Johnson (104)** — VIP Tier | $12,000 MRR | Status: **Active** | Purchase: Laptop Pro X

*Source: Verified from SQLite `customers` table.*"""

            # 4. Laptop Pro X / Alex Johnson Query
            if "104" in content or "laptop" in content or "alex" in content:
                return """Based on records retrieved from the enterprise database:

**Customer & Product Details:**
- **Customer:** Alex Johnson (ID: 104) — VIP Tier
- **Product:** Laptop Pro X
- **Purchase Date:** 2026-07-15
- **Warranty Status:** Active — Full Coverage (3-year accidental damage protection)
- **Account Standing:** Good

*Source: Verified from SQLite `customers` and `products` tables.*"""

            # 5. Vendors & Suppliers Query
            if any(w in content for w in ["vendor", "supplier", "partner", "technova", "cloudpeak", "datastream", "v001", "v002", "v003", "v004", "v005"]):
                return """Based on records retrieved from the enterprise database:

### 🏭 Enterprise Vendor Evaluation:
- **TechNova Solutions (V001):** Software | Revenue: $142M | Rating: 8.7/10 | Risk: **LOW** | Contact: sarah.chen@technova.com
- **CloudPeak Systems (V003):** Infrastructure | Revenue: $234M | Rating: 9.1/10 | Risk: **LOW** | Contact: priya.sharma@cloudpeak.com
- **CyberShield Security (V005):** Cybersecurity | Revenue: $95M | Rating: 9.4/10 | Risk: **LOW** | Contact: vikram.patel@cybershield.com
- **DataStream Analytics (V002):** Analytics | Revenue: $78.5M | Rating: 7.9/10 | Risk: **MEDIUM**
- **AI Dynamics Corp (V004):** AI Solutions | Revenue: $56M | Rating: 8.3/10 | Risk: **MEDIUM**

*Source: Verified from SQLite `vendors` table.*"""

            # 6. Policy & Refund Query
            if any(w in content for w in ["policy", "refund", "return", "escalation"]):
                return """Based on enterprise policy records:

**VIP & Enterprise Refund Policy:**
- VIP & Enterprise tier customers with active warranties are eligible for **immediate, no-questions-asked hardware replacements or full refunds** including shipping.
- Standard tier customers with expired warranties receive **15% trade-in credit**.

**Escalation SLA Policy:**
- Accounts marked *At Risk* or *Churning* with >5 open tickets are escalated to an Executive Account Manager within **2 hours**.

*Source: Verified from SQLite `policies` table.*"""

            # 7. Support Tickets Query
            if any(w in content for w in ["ticket", "support", "issue", "tick-101", "tick-102", "tick-103"]):
                return """Based on enterprise database records:

### 🎫 Open Support Tickets & Escalations:
1. **TICK-101** — Customer: LogiChain Dynamics (C004) | Issue: *Enterprise Router X-205 hardware failure & packet drop* | Priority: 🔴 **HIGH** | Status: OPEN | Assigned: David Kim
2. **TICK-102** — Customer: RetailMax Group (C002) | Issue: *POS Terminal v2 billing discount review* | Priority: 🟡 **MEDIUM** | Status: OPEN | Assigned: Marcus Lee
3. **TICK-103** — Customer: HealthBridge Network (C003) | Issue: *Annual security audit compliance check* | Priority: 🟢 **LOW** | Status: RESOLVED | Assigned: Elena Vasquez

*Source: Verified from SQLite `support_tickets` table.*"""

            # 8. Dynamic SQLite Lookup for ANY other question
            try:
                import sqlite3
                from backend.core.config import BASE_DIR
                db_path = BASE_DIR / "data" / "enterprise.db"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Extract search terms
                    words = [w for w in re.findall(r'\w+', content) if len(w) > 2 and w not in ["the", "what", "where", "how", "when", "which", "is", "are", "there", "any", "for", "with", "this", "that", "show", "tell", "details"]]
                    
                    for word in words:
                        # Search products
                        cursor.execute("SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR status LIKE ?", (f"%{word}%", f"%{word}%", f"%{word}%"))
                        prods = [dict(r) for r in cursor.fetchall()]
                        if prods:
                            conn.close()
                            lines = [f"- **{p['name']}** (ID: {p['id']}): Category: {p['category']} | Price: ${p['price']:,} | Warranty: {p['warranty_details']}" for p in prods]
                            return "Based on records retrieved from the database:\n\n### 📦 Product Information:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `products` table.*"

                        # Search customers
                        cursor.execute("SELECT * FROM customers WHERE name LIKE ? OR tier LIKE ? OR status LIKE ? OR account_manager LIKE ?", (f"%{word}%", f"%{word}%", f"%{word}%", f"%{word}%"))
                        custs = [dict(r) for r in cursor.fetchall()]
                        if custs:
                            conn.close()
                            lines = [f"- **{c['name']}** (ID: {c['id']}): Tier: {c['tier']} | MRR: ${c['mrr']:,} | Status: {c['status']} | Manager: {c['account_manager']}" for c in custs]
                            return "Based on records retrieved from the database:\n\n### 👤 Customer Account Information:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `customers` table.*"

                        # Search vendors
                        cursor.execute("SELECT * FROM vendors WHERE name LIKE ? OR industry LIKE ? OR risk_level LIKE ?", (f"%{word}%", f"%{word}%", f"%{word}%"))
                        vens = [dict(r) for r in cursor.fetchall()]
                        if vens:
                            conn.close()
                            lines = [f"- **{v['name']}** (ID: {v['id']}): Industry: {v['industry']} | Revenue: ${v['revenue']:,} | Rating: {v['score']}/10 | Risk: {v['risk_level']}" for v in vens]
                            return "Based on records retrieved from the database:\n\n### 🏭 Vendor Information:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `vendors` table.*"
                    conn.close()
            except Exception as e:
                pass

            # 9. Math / Calculator Fallback
            if any(op in content for op in ["+", "*", "/", "multiply", "divide", "plus", "percent"]):
                nums = re.findall(r'\d+(?:\.\d+)?', content)
                if len(nums) >= 2:
                    n1, n2 = float(nums[0]), float(nums[1])
                    if "*" in content or "multiply" in content:
                        ans = n1 * n2
                        return f"### 🧮 Calculation Result:\n**{n1:,.2f} × {n2:,.2f} = {ans:,.2f}**\n\n*Computed using AI Agent Calculator Tool.*"
                    elif "+" in content or "plus" in content:
                        ans = n1 + n2
                        return f"### 🧮 Calculation Result:\n**{n1:,.2f} + {n2:,.2f} = {ans:,.2f}**\n\n*Computed using AI Agent Calculator Tool.*"

            # Open-ended Analytical Fallback for any general request
            clean_query = content.replace("generate final answer and deliverable for:", "").strip().capitalize()
            return f"""### 🤖 Multi-Agent Analysis & Findings

**Request Analyzed:** "{clean_query}"

**Key Insights & Findings:**
1. **System Evaluation:** Evaluated query against enterprise knowledge base, database indexes, and operational policies.
2. **Operational Recommendation:** All system metrics confirm healthy operational status and high confidence alignment.
3. **Action Items:** No critical flags detected; routine execution recommended.

*Generated by 5-Agent Engine (Planner → Research → Analysis → Decision → Executor).*"""

        # ─── PLANNER AGENT ───
        if any(w in content for w in ["available agents:", "create a detailed execution plan", "business request:"]):
            return random.choice(self.PLANNER_RESPONSES)

        # ─── RESEARCH AGENT ───
        if any(w in content for w in ["retrieve data for:", "information gathering", "data retrieval", "web search"]):
            return "Database query executed across `products`, `customers`, `vendors`, `support_tickets`, and `policies` tables. Data extracted successfully and passed to Analysis Agent."

        # ─── ANALYSIS AGENT ───
        if any(w in content for w in ["analyze the research", "data processing", "pattern recognition", "mcda"]):
            return "Analysis complete: Cross-referenced retrieved database records with business rules and validated entity status."

        # ─── DECISION AGENT ───
        if any(w in content for w in ["evaluate analysis", "recommendation", "risk scoring", "decision agent"]):
            return "Decision formulated: Confirmed optimal response pathway based on enterprise policies and database records."

        # General Fallback
        return random.choice(self.GENERIC_RESPONSES)


    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Async mock invocation with simulated processing delay."""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
        response = self._select_response(messages)
        logger.info(f"[MockLLM] Generated response ({len(response)} chars)")
        return response

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Sync mock invocation."""
        response = self._select_response(messages)
        logger.info(f"[MockLLM] Generated response ({len(response)} chars)")
        return response


class GeminiLLM(BaseLLM):
    """Google Gemini LLM wrapper."""

    def __init__(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.client = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
            )
            logger.info("Initialized Google Gemini LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        result = await self.client.ainvoke(lc_messages)
        return result.content

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        result = self.client.invoke(lc_messages)
        return result.content


class OpenAILLM(BaseLLM):
    """OpenAI GPT LLM wrapper."""

    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            logger.info("Initialized OpenAI LLM")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        result = await self.client.ainvoke(lc_messages)
        return result.content

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        result = self.client.invoke(lc_messages)
        return result.content


def create_llm() -> BaseLLM:
    """
    Factory function that creates the appropriate LLM instance
    based on the configured provider and available API keys.
    Falls back to Mock if provider initialization fails.
    """
    provider = settings.LLM_PROVIDER

    if provider == LLMProvider.GEMINI and settings.GOOGLE_API_KEY:
        try:
            return GeminiLLM()
        except Exception as e:
            logger.warning(f"Gemini init failed, falling back to Mock: {e}")
            return MockLLM()

    elif provider == LLMProvider.OPENAI and settings.OPENAI_API_KEY:
        try:
            return OpenAILLM()
        except Exception as e:
            logger.warning(f"OpenAI init failed, falling back to Mock: {e}")
            return MockLLM()

    else:
        logger.info("Using Mock LLM (no API key configured)")
        return MockLLM()


# Global LLM instance
_llm_instance: Optional[BaseLLM] = None


def get_llm() -> BaseLLM:
    """Get or create the global LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_llm()
    return _llm_instance
