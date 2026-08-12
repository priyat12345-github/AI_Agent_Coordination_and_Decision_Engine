"""
Mock enterprise data for demonstration and testing.
Provides realistic business data across multiple domains.
"""

import json
from datetime import datetime, timedelta
import random

# ─── Companies / Vendors ────────────────────────────────
VENDORS = [
    {
        "id": "V001",
        "name": "TechNova Solutions",
        "industry": "Enterprise Software",
        "revenue": 142_000_000,
        "employees": 850,
        "score": 8.7,
        "risk_level": "LOW",
        "contact": "sarah.chen@technova.com",
        "country": "USA",
        "established": 2010,
    },
    {
        "id": "V002",
        "name": "DataStream Analytics",
        "industry": "Data & Analytics",
        "revenue": 78_500_000,
        "employees": 320,
        "score": 7.9,
        "risk_level": "MEDIUM",
        "contact": "james.hall@datastream.io",
        "country": "UK",
        "established": 2015,
    },
    {
        "id": "V003",
        "name": "CloudPeak Systems",
        "industry": "Cloud Infrastructure",
        "revenue": 234_000_000,
        "employees": 1200,
        "score": 9.1,
        "risk_level": "LOW",
        "contact": "priya.sharma@cloudpeak.com",
        "country": "India",
        "established": 2008,
    },
    {
        "id": "V004",
        "name": "AI Dynamics Corp",
        "industry": "Artificial Intelligence",
        "revenue": 56_000_000,
        "employees": 180,
        "score": 8.3,
        "risk_level": "MEDIUM",
        "contact": "alex.rivera@aidynamics.ai",
        "country": "Canada",
        "established": 2018,
    },
]

# ─── Customers / CRM ─────────────────────────────────────
CUSTOMERS = [
    {
        "id": "C001",
        "name": "Global Finance Corp",
        "tier": "Enterprise",
        "mrr": 45_000,
        "status": "Active",
        "sentiment": "Positive",
        "account_manager": "Rachel Torres",
        "last_contact": "2024-11-15",
        "open_tickets": 2,
        "nps_score": 72,
    },
    {
        "id": "C002",
        "name": "RetailMax Group",
        "tier": "Premium",
        "mrr": 18_500,
        "status": "At Risk",
        "sentiment": "Neutral",
        "account_manager": "Marcus Lee",
        "last_contact": "2024-10-28",
        "open_tickets": 7,
        "nps_score": 34,
    },
    {
        "id": "C003",
        "name": "HealthBridge Network",
        "tier": "Enterprise",
        "mrr": 67_200,
        "status": "Active",
        "sentiment": "Positive",
        "account_manager": "Elena Vasquez",
        "last_contact": "2024-11-20",
        "open_tickets": 0,
        "nps_score": 91,
    },
    {
        "id": "C004",
        "name": "LogiChain Dynamics",
        "tier": "Standard",
        "mrr": 8_900,
        "status": "Churning",
        "sentiment": "Negative",
        "account_manager": "David Kim",
        "last_contact": "2024-10-05",
        "open_tickets": 12,
        "nps_score": 18,
    },
]

# ─── Financial Data ─────────────────────────────────────
FINANCIAL_DATA = {
    "company": "Acme Enterprise Corp",
    "fiscal_year": 2024,
    "currency": "USD",
    "revenue": {
        "Q1": 12_400_000,
        "Q2": 14_800_000,
        "Q3": 16_200_000,
        "Q4": 18_900_000,
        "total": 62_300_000,
    },
    "expenses": {
        "Q1": 9_800_000,
        "Q2": 11_200_000,
        "Q3": 12_100_000,
        "Q4": 14_300_000,
        "total": 47_400_000,
    },
    "gross_margin": "23.9%",
    "operating_margin": "18.2%",
    "ebitda": "11_380_000",
    "cash_position": 28_500_000,
    "debt_ratio": 0.24,
    "burn_rate": "N/A (profitable)",
    "yoy_growth": "22.4%",
    "headcount": 347,
    "cost_per_employee": 136_600,
}

# ─── Market Data ─────────────────────────────────────────
MARKET_DATA = {
    "sector": "Enterprise AI Software",
    "total_addressable_market": "USD 847 Billion",
    "cagr": "12.3%",
    "forecast_year": 2028,
    "key_players": [
        {"name": "AlphaAI", "market_share": "23%", "revenue": "12.4B"},
        {"name": "TechGiant AI", "market_share": "18%", "revenue": "9.7B"},
        {"name": "DataCore Systems", "market_share": "15%", "revenue": "8.1B"},
        {"name": "InnovateCorp", "market_share": "11%", "revenue": "5.9B"},
        {"name": "Others", "market_share": "33%", "revenue": "17.7B"},
    ],
    "trends": [
        "Generative AI integration: +67% adoption YoY",
        "Multi-agent systems: emerging rapidly",
        "Edge AI deployment: 34% growth",
        "AI governance frameworks: regulatory push",
    ],
    "growth_drivers": [
        "Digital transformation acceleration",
        "Workforce automation demand",
        "Data volume explosion",
        "Cloud-native AI services maturity",
    ],
}

# ─── HR / Candidates ─────────────────────────────────────
CANDIDATES = [
    {
        "id": "HR001",
        "name": "Ananya Krishnan",
        "role": "Senior ML Engineer",
        "experience_years": 7,
        "skills": ["Python", "PyTorch", "LLMs", "MLOps", "AWS"],
        "education": "M.S. Computer Science, MIT",
        "salary_expectation": 145_000,
        "fit_score": 9.2,
        "availability": "30 days",
        "location": "Remote",
    },
    {
        "id": "HR002",
        "name": "Carlos Martinez",
        "role": "Senior ML Engineer",
        "experience_years": 5,
        "skills": ["Python", "TensorFlow", "NLP", "Docker", "GCP"],
        "education": "B.S. Data Science, UC Berkeley",
        "salary_expectation": 128_000,
        "fit_score": 8.1,
        "availability": "Immediate",
        "location": "San Francisco, CA",
    },
    {
        "id": "HR003",
        "name": "Sophie Laurent",
        "role": "Senior ML Engineer",
        "experience_years": 9,
        "skills": ["Python", "PyTorch", "Computer Vision", "MLOps", "Azure"],
        "education": "Ph.D. Machine Learning, ETH Zurich",
        "salary_expectation": 165_000,
        "fit_score": 9.7,
        "availability": "60 days",
        "location": "Remote",
    },
]

# ─── Support Tickets / Escalations ─────────────────────
SUPPORT_TICKETS = [
    {
        "id": "TKT-8842",
        "customer": "RetailMax Group",
        "priority": "HIGH",
        "category": "Integration Failure",
        "description": "API integration with their ERP system failing intermittently since Nov 12",
        "created": "2024-11-12",
        "age_days": 8,
        "assigned_to": None,
        "estimated_revenue_impact": 18_500,
    },
    {
        "id": "TKT-9103",
        "customer": "LogiChain Dynamics",
        "priority": "CRITICAL",
        "category": "Data Loss",
        "description": "Customer reports missing shipment tracking records for last 14 days",
        "created": "2024-11-08",
        "age_days": 12,
        "assigned_to": "Engineering Team",
        "estimated_revenue_impact": 8_900,
    },
    {
        "id": "TKT-9287",
        "customer": "Global Finance Corp",
        "priority": "MEDIUM",
        "category": "Feature Request",
        "description": "Request for custom dashboard with regulatory compliance reporting",
        "created": "2024-11-18",
        "age_days": 2,
        "assigned_to": "Product Team",
        "estimated_revenue_impact": 0,
    },
]

# ─── KPIs ─────────────────────────────────────────────
COMPANY_KPIS = {
    "arr": 62_300_000,
    "mrr": 5_191_667,
    "customer_count": 247,
    "enterprise_customers": 89,
    "churn_rate": "2.3%",
    "nrr": "118%",
    "avg_deal_size": 252_228,
    "sales_cycle_days": 67,
    "cac": 24_500,
    "ltv": 287_000,
    "ltv_cac_ratio": 11.7,
    "nps": 68,
    "csat": "4.4/5.0",
}


def get_all_data() -> dict:
    """Return all mock data as a single dictionary."""
    return {
        "vendors": VENDORS,
        "customers": CUSTOMERS,
        "financial_data": FINANCIAL_DATA,
        "market_data": MARKET_DATA,
        "candidates": CANDIDATES,
        "support_tickets": SUPPORT_TICKETS,
        "company_kpis": COMPANY_KPIS,
    }


def get_random_metric() -> dict:
    """Generate a random business metric for dashboard animation."""
    metrics = [
        {"name": "ARR", "value": f"${COMPANY_KPIS['arr']:,}", "change": "+22.4%", "trend": "up"},
        {"name": "NRR", "value": COMPANY_KPIS['nrr'], "change": "+3.1%", "trend": "up"},
        {"name": "Churn Rate", "value": COMPANY_KPIS['churn_rate'], "change": "-0.4%", "trend": "down"},
        {"name": "NPS Score", "value": str(COMPANY_KPIS['nps']), "change": "+5pts", "trend": "up"},
        {"name": "LTV:CAC", "value": f"{COMPANY_KPIS['ltv_cac_ratio']}x", "change": "+1.2x", "trend": "up"},
    ]
    return random.choice(metrics)
