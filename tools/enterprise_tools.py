"""
Enterprise Tools for Customer Support & Resolution Engine.
Connects directly to the live enterprise.db SQLite database.
"""

import json
import sqlite3
import os
from langchain_core.tools import tool

DB_PATH = "enterprise.db"

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database {DB_PATH} not found. Please run setup_db.py first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@tool
def fetch_customer_data(customer_id: str) -> str:
    """
    Fetches a live customer profile from the enterprise SQLite database.
    
    Args:
        customer_id (str): The ID of the customer (e.g., '104', 'C-992').
        
    Returns:
        JSON string containing customer details, purchase history, and warranty status.
    """
    clean_id = ''.join(filter(str.isdigit, customer_id))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE id = ?", (clean_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return json.dumps({"error": f"Customer ID {customer_id} not found in database."})
            
        data = dict(row)
        return json.dumps({"customer_id": customer_id, "data": data})
    except Exception as e:
        return json.dumps({"error": f"Database error: {str(e)}"})


@tool
def search_policy_wiki(query: str) -> str:
    """
    Queries the live enterprise database for return policies, 
    troubleshooting steps, or legal guidelines.
    
    Args:
        query (str): The search query to look up.
        
    Returns:
        String containing the policy or guidelines.
    """
    query = query.lower()
    search_keywords = ["refund", "return", "warranty", "troubleshoot", "screen", "broken"]
    
    found_keyword = None
    for kw in search_keywords:
        if kw in query:
            found_keyword = kw
            break
            
    if not found_keyword:
        return f"No specific policy documents found for query: '{query}'."
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT policy_text FROM policies WHERE keyword LIKE ?", (f"%{found_keyword}%",))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row["policy_text"]
        return f"No policy text found matching '{found_keyword}'."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def process_refund(customer_id: str, amount: float) -> str:
    """
    Simulates an action tool that connects to the billing API to issue a refund.
    
    Args:
        customer_id (str): The customer receiving the refund.
        amount (float): The dollar amount to refund.
        
    Returns:
        String confirming the billing action.
    """
    if amount <= 0:
        return "Error: Refund amount must be greater than zero."
        
    # In a real system, this would UPDATE the database to log the refund
    return f"SUCCESS: Refund of ${amount:.2f} has been processed and credited to customer {customer_id}'s original payment method. Transaction ID: REF-{customer_id}-8992"
