"""
Mock Enterprise Tools using LangChain's @tool decorator.
These simulate connections to internal ERPs, databases, and APIs.
"""

import json
from langchain_core.tools import tool


@tool
def fetch_sales_data(quarter: str) -> str:
    """
    Simulates fetching financial sales records from an ERP system for a given quarter.
    
    Args:
        quarter (str): The quarter to fetch data for (e.g., 'Q1', 'Q2', 'Q3', 'Q4').
        
    Returns:
        JSON string containing sales data and revenue.
    """
    mock_db = {
        "Q1": {"revenue": 1500000, "units_sold": 5000, "top_region": "North America"},
        "Q2": {"revenue": 1725000, "units_sold": 5800, "top_region": "Europe"},
        "Q3": {"revenue": 1600000, "units_sold": 5200, "top_region": "Asia"},
        "Q4": {"revenue": 2100000, "units_sold": 7100, "top_region": "North America"},
    }
    
    data = mock_db.get(quarter.upper())
    if not data:
        return json.dumps({"error": f"No data found for quarter: {quarter}"})
    
    return json.dumps({"quarter": quarter.upper(), "data": data})


@tool
def search_knowledge_base(query: str) -> str:
    """
    Simulates querying an internal wiki or document store for company policies, 
    historical data, or structural information.
    
    Args:
        query (str): The search query to look up.
        
    Returns:
        String containing the search results.
    """
    query = query.lower()
    if "q3 projection" in query or "projection" in query:
        return "Internal Wiki: Q3 projections indicate a slight dip in consumer spending, expecting $1.6M revenue. Focus marketing on Asia region."
    elif "marketing" in query or "strategy" in query:
        return "Internal Wiki: The current strategy relies heavily on digital ad spend in emerging markets. Expected ROI is 12%."
    elif "supply chain" in query or "risk" in query:
        return "Internal Wiki: Supply chain risks currently involve semiconductor shortages affecting Q4 deliverables. Mitigation plan: onboard secondary suppliers."
    
    return f"No specific internal documents found for query: '{query}'."


@tool
def calculate_growth_metrics(current: float, previous: float) -> str:
    """
    Calculates percentage growth between two numbers.
    Useful for precise mathematical operations that LLMs might struggle with.
    
    Args:
        current (float): The current period's value.
        previous (float): The previous period's value.
        
    Returns:
        String detailing the percentage growth.
    """
    if previous == 0:
        return "Error: Previous value cannot be zero."
    
    growth = ((current - previous) / previous) * 100
    return f"{growth:.2f}%"
