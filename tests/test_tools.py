"""
Unit tests for the new LangChain enterprise tools.
Run with: python -m pytest tests/test_tools.py -v
"""

import pytest
import json
from tools.enterprise_tools import (
    fetch_sales_data,
    search_knowledge_base,
    calculate_growth_metrics,
)

def test_fetch_sales_data_valid_quarter():
    """Test fetching data for a valid quarter (Q1)."""
    result_str = fetch_sales_data.invoke({"quarter": "Q1"})
    result = json.loads(result_str)
    
    assert "quarter" in result
    assert result["quarter"] == "Q1"
    assert "data" in result
    assert result["data"]["revenue"] == 1500000

def test_fetch_sales_data_invalid_quarter():
    """Test fetching data for an invalid quarter (Q5)."""
    result_str = fetch_sales_data.invoke({"quarter": "Q5"})
    result = json.loads(result_str)
    
    assert "error" in result
    assert "No data found" in result["error"]

def test_search_knowledge_base():
    """Test querying the internal wiki."""
    result = search_knowledge_base.invoke({"query": "Q3 projection"})
    assert "Internal Wiki" in result
    assert "$1.6M" in result

    result_not_found = search_knowledge_base.invoke({"query": "random unknown string"})
    assert "No specific internal documents found" in result_not_found

def test_calculate_growth_metrics():
    """Test growth calculations."""
    # (150 - 100) / 100 = 50%
    result = calculate_growth_metrics.invoke({"current": 150.0, "previous": 100.0})
    assert result == "50.00%"

def test_calculate_growth_metrics_zero_division():
    """Test gracefully handling division by zero."""
    result = calculate_growth_metrics.invoke({"current": 150.0, "previous": 0.0})
    assert "Error" in result
    assert "cannot be zero" in result
