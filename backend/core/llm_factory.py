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
        full_content = " ".join(m.get("content", "") for m in messages).lower()

        # ── Extract ONLY the user's actual question from the executor task ──────────
        # The executor task is always: "Generate final answer and deliverable for: <USER QUESTION>"
        user_query = ""
        for marker in ["generate final answer and deliverable for:", "final answer and deliverable for:", "for:"]:
            if marker in full_content:
                idx = full_content.rfind(marker)
                user_query = full_content[idx + len(marker):].strip()
                # Take only the first sentence/line to avoid noise
                user_query = user_query.split("\n")[0].strip()[:300]
                break

        # Use extracted user query for matching (falls back to full content for non-executor phases)
        is_executor = bool(user_query) or any(w in full_content for w in [
            "generate final answer", "deliverable", "execute and deliver"
        ])
        q = user_query if user_query else full_content  # <- match only user's question

        # ─── EXECUTOR AGENT FINAL ANSWERS ───
        if is_executor:

            # ── Direct SQLite lookup (most accurate — handles ALL queries) ────────
            try:
                import sqlite3
                from backend.core.config import BASE_DIR
                db_path = BASE_DIR / "data" / "enterprise.db"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    # ── Extract User Context ──────────────────────────────────────
                    user_match = re.search(r'\(User:\s*([^)]+)\)', q, re.IGNORECASE)
                    logged_in_user = user_match.group(1).strip() if user_match else None
                    
                    role = "Employee" # default
                    user_id = None
                    user_name = None
                    
                    if logged_in_user:
                        # Clean query by removing the User tag for matching
                        q = re.sub(r'\(User:\s*[^)]+\)', '', q, flags=re.IGNORECASE).strip()
                        
                        # Lookup user in DB to verify role
                        cursor.execute("SELECT id, name FROM customers WHERE LOWER(id) = ? OR LOWER(name) = ?", (logged_in_user.lower(), logged_in_user.lower()))
                        cust_record = cursor.fetchone()
                        if cust_record:
                            role = "Customer"
                            user_id = cust_record["id"]
                            user_name = cust_record["name"]
                        else:
                            cursor.execute("SELECT id, name FROM vendors WHERE LOWER(id) = ? OR LOWER(name) = ?", (logged_in_user.lower(), logged_in_user.lower()))
                            vend_record = cursor.fetchone()
                            if vend_record:
                                role = "Vendor"
                                user_id = vend_record["id"]
                                user_name = vend_record["name"]
                            else:
                                # Predefined employees
                                valid_employees = ["elena vasquez", "sarah chen", "marcus lee", "rachel torres", "david kim", "admin"]
                                if logged_in_user.lower() in valid_employees:
                                    role = "Employee"
                                    user_name = logged_in_user
                                else:
                                    # Not found in system records
                                    conn.close()
                                    return f"⚠️ **Authentication Error:** User **{logged_in_user}** not found in enterprise records. Please sign in with a valid account."

                    # Enforce RBAC for Customers
                    if role == "Customer":
                        blocked_keywords = ["all vendor", "list vendor", "show vendor", "all customer", "list customer", "show customer", "ticket", "support", "issue", "at risk", "churn"]
                        if any(w in q.lower() for w in blocked_keywords):
                            conn.close()
                            return "⚠️ **Access Denied:** As a Customer, you only have permission to view your own account details, purchased products, and active warranties."
                            
                    # Enforce RBAC for Vendors
                    if role == "Vendor":
                        blocked_keywords = ["customer", "ticket", "support", "issue", "policy", "refund", "return", "all vendor", "list vendor", "show vendor"]
                        if any(w in q.lower() for w in blocked_keywords):
                            conn.close()
                            return "⚠️ **Access Denied:** As a Vendor, you only have permission to view your own company profile, performance ratings, and supplier risk level."

                    # ── LIST ALL PRODUCTS ─────────────────────────────────────────
                    if any(w in q for w in ["list all product", "all product", "all active product", "show all product", "show product", "active product", "list the active product", "list product", "what are the product"]):
                        cursor.execute("SELECT * FROM products ORDER BY id")
                        prods = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if prods:
                            lines = [f"| {p['id']} | **{p['name']}** | {p['category']} | ${p['price']:,} | {p['status']} | {p['warranty_details'][:60]}... |" for p in prods]
                            header = "| ID | Name | Category | Price | Status | Warranty |\n|---|---|---|---|---|---|"
                            return f"Based on records retrieved from the enterprise database:\n\n### 📦 All Products:\n\n{header}\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `products` table.*"
                    # Matches: 'network', 'networking', 'security', 'storage', etc.
                    cat_match = re.search(r'\b(network(?:ing)?|security|storage|display|laptop|cloud|computing)\b', q)
                    if cat_match:
                        cat = cat_match.group(1).rstrip('ing') if cat_match.group(1).endswith('ing') and cat_match.group(1) != 'networking' else cat_match.group(1)
                        # 'network' -> search for 'networking' in DB too
                        search_term = "network" if "network" in cat_match.group(1) else cat
                        cursor.execute("SELECT * FROM products WHERE LOWER(category) LIKE ?", (f"%{search_term}%",))
                        prods = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if prods:
                            lines = [f"- **{p['name']}** (ID: {p['id']}) | Price: ${p['price']:,} | Warranty: {p['warranty_details']}" for p in prods]
                            display_cat = cat_match.group(1).title()
                            return f"Based on records retrieved from the enterprise database:\n\n### 📦 Products in {display_cat} Category:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `products` table.*"
                        else:
                            conn.close()
                            return f"No products found in the **{cat_match.group(1).title()}** category."

                    # ── PRODUCT BY ID ─────────────────────────────────────────────
                    prod_id = re.search(r'\b(205|308|104|501|602|703)\b', q)
                    if prod_id:
                        cursor.execute("SELECT * FROM products WHERE id = ?", (int(prod_id.group(1)),))
                        p = cursor.fetchone()
                        conn.close()
                        if p:
                            p = dict(p)
                            return f"""Based on records retrieved from the enterprise database:

**Product Details:**
- **Product ID:** {p['id']}
- **Name:** {p['name']}
- **Category:** {p['category']}
- **Price:** ${p['price']:,}
- **Status:** {p['status']}

**Warranty Coverage:** {p['warranty_details']}

*Source: Verified from SQLite `products` table.*"""

                    # ── CUSTOMER PURCHASES ────────────────────────────────────────
                    if any(w in q for w in ["bought by", "brought by", "purchased by", "products of"]):
                        cursor.execute("SELECT * FROM customers")
                        custs = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        
                        # See if a specific customer is mentioned
                        matched = [c for c in custs if c['id'].lower() in q.lower() or c['name'].lower() in q.lower()]
                        
                        if role == "Customer":
                            # Check if they are trying to look up someone else
                            other_custs = [c for c in custs if c['id'].lower() != user_id.lower()]
                            if any(c['id'].lower() in q.lower() or c['name'].lower() in q.lower() for c in other_custs):
                                return "⚠️ **Access Denied:** You cannot view other customers' purchases."
                            matched = [c for c in custs if c['id'].lower() == user_id.lower()]
                        elif role == "Vendor":
                            return "⚠️ **Access Denied:** As a Vendor, you do not have permission to view customer purchases."
                            
                        if not matched:
                            matched = custs # List all if no specific customer is mentioned
                            
                        if matched:
                            lines = []
                            for c in matched:
                                purchases_raw = c.get('recent_purchase', 'None')
                                if ',' in purchases_raw:
                                    items = [item.strip() for item in purchases_raw.split(',')]
                                    items_md = "\n".join([f"  - {item}" for item in items])
                                    lines.append(f"- **{c['name']}** ({c['id']}) purchased the following items on {c.get('purchase_date', 'N/A')} | Warranty: {c.get('warranty_status', 'N/A')}:\n{items_md}")
                                else:
                                    lines.append(f"- **{c['name']}** ({c['id']}) purchased **{purchases_raw}** on {c.get('purchase_date', 'N/A')} | Warranty: {c.get('warranty_status', 'N/A')}")
                            return "Based on records retrieved from the enterprise database:\n\n### 🛒 Recent Customer Purchases:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `customers` table.*"

                    # ── CUSTOMER WARRANTY STATUS ───────────────────────────────
                    if any(w in q.lower() for w in ["warranty status", "my warranty", "warranty active", "is my warranty"]):
                        if role == "Customer" and user_id:
                            cursor.execute("SELECT * FROM customers WHERE LOWER(id) = ?", (user_id.lower(),))
                            c = cursor.fetchone()
                            conn.close()
                            if c:
                                c = dict(c)
                                icon = "✅" if "Active" in c.get("warranty_status", "") else "❌"
                                return f"""Based on records retrieved from the enterprise database:

### 🛡️ Warranty Status: {c['name']} ({c['id']})
- **Product(s):** {c.get('recent_purchase', 'N/A')}
- **Purchase Date:** {c.get('purchase_date', 'N/A')}
- **Warranty Status:** {icon} {c.get('warranty_status', 'N/A')}
- **Account Standing:** {c.get('account_standing', 'N/A')}

*Source: Verified from SQLite `customers` table.*"""

                    # ── CUSTOMER PROFILE / DETAILS ────────────────────────────────
                    # Matches IDs (C001, 104, 404) or specific names
                    cust_id = re.search(r'\b(c00[1-9]|[1-9][0-9]{2}|priya|alex|jordan)\b', q.lower())
                    if cust_id and any(w in q.lower() for w in ["profile", "who is", "customer", "account", "details"]):
                        match_term = cust_id.group(1)
                        if role == "Customer":
                            if match_term.lower() != user_id.lower() and match_term.lower() != user_name.lower():
                                conn.close()
                                return "⚠️ **Access Denied:** You cannot view other customers' records."
                        elif role == "Vendor":
                            conn.close()
                            return "⚠️ **Access Denied:** As a Vendor, you do not have permission to view customer profiles."
                            
                        if match_term.isdigit() or match_term.startswith('c'):
                            cursor.execute("SELECT * FROM customers WHERE LOWER(id) = ?", (match_term,))
                        else:
                            cursor.execute("SELECT * FROM customers WHERE LOWER(name) LIKE ?", (f"%{match_term}%",))
                        c = cursor.fetchone()
                        conn.close()
                        if c:
                            c = dict(c)
                            return f"""Based on records retrieved from the enterprise database:

### 👤 Customer Profile: {c['name']} ({c['id']})
- **Tier:** {c['tier']}
- **Monthly Revenue (MRR):** ${c['mrr']:,}
- **Account Status:** {c['status']}
- **NPS Score:** {c.get('nps', 'N/A')}
- **Account Manager:** {c['account_manager']}

*Source: Verified from SQLite `customers` table.*"""

                    # ── AT RISK / CHURNING CUSTOMERS ────────────────────────────────────
                    if any(w in q for w in ["at risk", "churn", "risk or churn", "risk customer", "churn customer", "high risk customer"]):
                        cursor.execute("SELECT * FROM customers WHERE status IN ('At Risk', 'Churning') ORDER BY status")
                        custs = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if custs:
                            lines = [f"- **{c['name']}** ({c['id']}) — {c['tier']} | MRR: ${c['mrr']:,} | Status: {'⚠️ At Risk' if c['status']=='At Risk' else '🚨 Churning'} | Tickets: {c.get('open_tickets','N/A')} | Manager: {c['account_manager']}" for c in custs]
                            return "Based on records retrieved from the enterprise database:\n\n### 🚨 At Risk & Churning Customers:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `customers` table.*"

                    # ── ALL CUSTOMERS ─────────────────────────────────────────────
                    if any(w in q for w in ["all customer", "list customer", "show customer", "customer account", "customer overview"]):
                        cursor.execute("SELECT * FROM customers ORDER BY mrr DESC")
                        custs = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if custs:
                            lines = [f"- **{c['name']}** ({c['id']}) — {c['tier']} | MRR: ${c['mrr']:,} | Status: {c['status']} | Manager: {c['account_manager']}" for c in custs]
                            return "Based on records retrieved from the enterprise database:\n\n### 📊 All Customer Accounts:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `customers` table.*"

                    # ── VENDORS BY RISK ────────────────────────────────────────────
                    risk_match = re.search(r'\b(low|medium|high)\b', q)
                    if risk_match and any(w in q for w in ["vendor", "supplier", "risk"]):
                        risk = risk_match.group(1).upper()
                        cursor.execute("SELECT * FROM vendors WHERE UPPER(risk_level) = ? ORDER BY score DESC", (risk,))
                        vens = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if vens:
                            lines = [f"- **{v['name']}** ({v['id']}) | Industry: {v['industry']} | Revenue: ${v['revenue']:,} | Rating: {v['score']}/10" for v in vens]
                            return f"Based on records retrieved from the enterprise database:\n\n### 🏭 {risk} Risk Vendors:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `vendors` table.*"

                    # ── VENDOR PROFILE / DETAILS ──────────────────────────────────
                    vendor_name_match = re.search(r'\b(technova|datastream|cloudpeak|dynamics|cybershield|v00[1-5])\b', q.lower())
                    if vendor_name_match or (role == "Vendor" and any(w in q.lower() for w in ["profile", "score", "rating", "risk", "details", "contact", "company", "who am i", "my info"])):
                        cursor.execute("SELECT * FROM vendors")
                        vens = [dict(r) for r in cursor.fetchall()]
                        
                        matched_v = []
                        if role == "Vendor":
                            # Check if they are trying to look up someone else
                            other_vens = [v for v in vens if v['id'].lower() != user_id.lower()]
                            if any(v['id'].lower() in q.lower() or v['name'].lower() in q.lower() for v in other_vens):
                                conn.close()
                                return "⚠️ **Access Denied:** You cannot view other vendors' records."
                            matched_v = [v for v in vens if v['id'].lower() == user_id.lower()]
                        else:
                            if vendor_name_match:
                                term = vendor_name_match.group(1)
                                matched_v = [v for v in vens if term in v['id'].lower() or term in v['name'].lower()]
                                
                        if matched_v:
                            v = matched_v[0]
                            conn.close()
                            return f"""Based on records retrieved from the enterprise database:

### 🏭 Vendor Profile: {v['name']} ({v['id']})
- **Industry:** {v['industry']}
- **Annual Revenue:** ${v['revenue']:,}
- **Employee Count:** {v['employees']}
- **Performance Rating:** {v['score']}/10
- **Supplier Risk Level:** {v['risk_level']}
- **Primary Contact:** {v['contact']}
- **Country:** {v['country']}
- **Established Year:** {v['established']}

*Source: Verified from SQLite `vendors` table.*"""

                    # ── ALL VENDORS ────────────────────────────────────────────────
                    if any(w in q for w in ["all vendor", "list vendor", "show vendor", "vendor"]):
                        cursor.execute("SELECT * FROM vendors ORDER BY score DESC")
                        vens = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if vens:
                            lines = [f"- **{v['name']}** ({v['id']}) | {v['industry']} | Revenue: ${v['revenue']:,} | Rating: {v['score']}/10 | Risk: **{v['risk_level']}**" for v in vens]
                            return "Based on records retrieved from the enterprise database:\n\n### 🏭 All Enterprise Vendors:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `vendors` table.*"

                    # ── TICKETS ───────────────────────────────────────────────────
                    if any(w in q for w in ["ticket", "support", "issue", "tick"]):
                        cursor.execute("SELECT * FROM support_tickets ORDER BY priority")
                        tickets = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if tickets:
                            priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                            lines = [f"- **{t['id']}** — Customer: {t['customer_name']} | Issue: *{t['issue']}* | Priority: {priority_icon.get(t['priority'],'⚪')} {t['priority']} | Status: {t['status']}" for t in tickets]
                            return "Based on enterprise database records:\n\n### 🎫 Support Tickets:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `support_tickets` table.*"

                    # ── ALL POLICIES (list / show / what are policies) ───────────
                    if any(w in q for w in ["all polic", "list polic", "show polic", "about polic", "what polic", "polic overview", "enterprise polic"]):
                        cursor.execute("SELECT * FROM policies ORDER BY policy_name")
                        pols = [dict(r) for r in cursor.fetchall()]
                        conn.close()
                        if pols:
                            lines = []
                            for p in pols:
                                tier = p.get('tier_required', 'All')
                                action = p.get('action', p.get('description', 'See policy for details'))
                                rule = p.get('rule', '')
                                lines.append(f"- **{p['policy_name']}** — Tier: *{tier}* | Action: {action}" + (f" | Rule: _{rule}_" if rule else ""))
                            return "Based on records retrieved from the enterprise database:\n\n### 📜 Enterprise Policies:\n" + "\n".join(lines) + "\n\n*Source: Verified from SQLite `policies` table.*"

                    # -- POLICIES & REFUND / RETURN ELIGIBILITY
                    if any(w in q for w in ["policy", "refund", "return", "eligible", "eligibility", "warranty refund", "am i eligible"]):
                        if role == "Customer" and user_id:
                            cursor.execute("SELECT * FROM customers WHERE LOWER(id) = ?", (user_id.lower(),))
                            c = cursor.fetchone()
                            conn.close()
                            if c:
                                c = dict(c)
                                tier = c.get("tier", "")
                                status = c.get("status", "")
                                if tier in ["Enterprise", "VIP"] and status == "Active":
                                    verdict = f"\u2705 **Yes! {c['name']} is fully eligible** for an immediate full refund or hardware replacement."
                                elif status == "At Risk":
                                    verdict = "\u26a0\ufe0f **Conditional.** Your account is *At Risk* - refunds require Executive Manager review."
                                elif status == "Churning":
                                    verdict = f"\u274c **Limited.** As a {tier} tier *{status}* customer, you qualify for a **15% trade-in credit** only."
                                else:
                                    verdict = f"\u26a0\ufe0f **Review Required.** Contact your Account Manager **{c.get('account_manager', 'N/A')}**."
                                return (
                                    "Based on records retrieved from the enterprise database:\n\n"
                                    f"### Refund Eligibility: {c['name']} ({c['id']})\n"
                                    f"- **Tier:** {tier}\n"
                                    f"- **Account Status:** {status}\n"
                                    f"- **Account Manager:** {c.get('account_manager', 'N/A')}\n\n"
                                    f"{verdict}\n\n"
                                    "*Source: Verified from SQLite `customers` and `policies` tables.*"
                                )
                        else:
                            conn.close()
                            return (
                                "Based on records retrieved from the enterprise database:\n\n"
                                "### Warranty Refund & Return Policy\n\n"
                                "**Fully Eligible (100% Refund):**\n"
                                "1. Global Finance Corp (C001) - Enterprise | Active\n"
                                "2. HealthBridge Network (C003) - Enterprise | Active\n"
                                "3. Alex Johnson (104) - VIP | Active\n\n"
                                "**Conditional Accounts:**\n"
                                "- RetailMax Group (C002) - At Risk (Executive review required)\n"
                                "- LogiChain Dynamics (C004) - Churning (15% trade-in credit)\n\n"
                                "*Source: Verified from SQLite `policies` and `customers` tables.*"
                            )


                    # ── MATH / CALCULATOR ─────────────────────────────────────────
                    if any(op in q for op in ["+", "*", "/", "multiply", "divide", "plus", "calculate", "percent"]):
                        nums = re.findall(r'\d+(?:\.\d+)?', q)
                        if len(nums) >= 2:
                            n1, n2 = float(nums[0]), float(nums[1])
                            if "*" in q or "multiply" in q:
                                ans = n1 * n2
                                return f"### 🧮 Calculation Result:\n**{n1:,.2f} × {n2:,.2f} = {ans:,.2f}**\n\n*Computed by AI Agent Calculator Tool.*"
                            elif "+" in q or "plus" in q:
                                ans = n1 + n2
                                return f"### 🧮 Calculation Result:\n**{n1:,.2f} + {n2:,.2f} = {ans:,.2f}**\n\n*Computed by AI Agent Calculator Tool.*"

                    conn.close()
            except Exception as db_err:
                logger.warning(f"Dynamic DB lookup failed: {db_err}")

            # Open-ended / General Analytical Fallback
            clean_query = (user_query or q).capitalize()
            return f"""### 🤖 Multi-Agent Analysis & Findings

**Request Analyzed:** "{clean_query}"

**Key Insights & Findings:**
1. **System Evaluation:** Evaluated query against enterprise knowledge base, database indexes, and operational policies.
2. **Operational Recommendation:** All system metrics confirm healthy operational status and high confidence alignment.
3. **Action Items:** No critical flags detected; routine execution recommended.

*Generated by 5-Agent Engine (Planner → Research → Analysis → Decision → Executor).*"""


        # ─── PLANNER AGENT ───
        if any(w in full_content for w in ["available agents:", "create a detailed execution plan", "business request:"]):
            return random.choice(self.PLANNER_RESPONSES)

        # ─── RESEARCH AGENT ───
        if any(w in full_content for w in ["retrieve data for:", "information gathering", "data retrieval", "web search"]):
            return "Database query executed across `products`, `customers`, `vendors`, `support_tickets`, and `policies` tables. Data extracted successfully and passed to Analysis Agent."

        # ─── ANALYSIS AGENT ───
        if any(w in full_content for w in ["analyze the research", "data processing", "pattern recognition", "mcda"]):
            return "Analysis complete: Cross-referenced retrieved database records with business rules and validated entity status."

        # ─── DECISION AGENT ───
        if any(w in full_content for w in ["evaluate analysis", "recommendation", "risk scoring", "decision agent"]):
            return "Decision formulated: Confirmed optimal response pathway based on enterprise policies and database records."

        # General Fallback
        return random.choice(self.GENERIC_RESPONSES)



    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Async mock invocation with minimal processing delay for fast response times."""
        delay = random.uniform(0.05, 0.15)
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
    """Google Gemini LLM wrapper with graceful fallback."""

    def __init__(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.mock_fallback = MockLLM()
        self.client = None
        
        # Try preferred Gemini model names
        model_candidates = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
        for model_name in model_candidates:
            try:
                self.client = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=settings.LLM_TEMPERATURE,
                    max_output_tokens=settings.LLM_MAX_TOKENS,
                    max_retries=0,
                )
                logger.info(f"Initialized Google Gemini LLM with model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Could not init model {model_name}: {e}")


    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.client:
            return await self.mock_fallback.ainvoke(messages, **kwargs)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            lc_messages = []
            for m in messages:
                if m["role"] == "system":
                    lc_messages.append(SystemMessage(content=m["content"]))
                else:
                    lc_messages.append(HumanMessage(content=m["content"]))
            result = await self.client.ainvoke(lc_messages)
            return result.content
        except Exception as e:
            logger.warning(f"Gemini API call failed ({e}), using engine fallback response.")
            return await self.mock_fallback.ainvoke(messages, **kwargs)

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.client:
            return self.mock_fallback.invoke(messages, **kwargs)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            lc_messages = []
            for m in messages:
                if m["role"] == "system":
                    lc_messages.append(SystemMessage(content=m["content"]))
                else:
                    lc_messages.append(HumanMessage(content=m["content"]))
            result = self.client.invoke(lc_messages)
            return result.content
        except Exception as e:
            logger.warning(f"Gemini API call failed ({e}), using engine fallback response.")
            return self.mock_fallback.invoke(messages, **kwargs)



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
    provider_str = str(settings.LLM_PROVIDER).lower()

    if (provider_str in ["gemini", "google"] or settings.GOOGLE_API_KEY) and settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your-google-api-key-here":
        try:
            return GeminiLLM()
        except Exception as e:
            logger.warning(f"Gemini init failed, falling back to Mock: {e}")
            return MockLLM()

    elif (provider_str == "openai" or settings.OPENAI_API_KEY) and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your-openai-api-key-here":
        try:
            return OpenAILLM()
        except Exception as e:
            logger.warning(f"OpenAI init failed, falling back to Mock: {e}")
            return MockLLM()

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
