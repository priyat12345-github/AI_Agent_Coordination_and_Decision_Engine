import sqlite3
import os

DB_PATH = "enterprise.db"

def setup_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Customers Table
    cursor.execute("""
        CREATE TABLE customers (
            id TEXT PRIMARY KEY,
            name TEXT,
            tier TEXT,
            recent_purchase TEXT,
            purchase_date TEXT,
            warranty_status TEXT,
            account_standing TEXT
        )
    """)

    # Insert Customer Data
    customers_data = [
        ("104", "Alex Johnson", "VIP", "Laptop Pro X", "2026-07-15", "Active - Full Coverage", "Good"),
        ("205", "Sam Smith", "Standard", "Wireless Mouse", "2025-01-10", "Expired", "Good"),
        ("306", "Jordan Lee", "VIP", "Desktop Workstation", "2026-05-20", "Active - Basic", "Good")
    ]
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", 
        customers_data
    )

    # Create Policies Table
    cursor.execute("""
        CREATE TABLE policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT,
            policy_text TEXT
        )
    """)

    # Insert Policy Data
    policies_data = [
        ("refund, return, warranty", "Internal Policy: VIP customers with 'Active - Full Coverage' warranty are eligible for immediate, no-questions-asked refunds for hardware damage including shipping accidents. Standard customers with expired warranties are not eligible."),
        ("troubleshoot, screen, broken", "Tech Support Guide: If a screen is physically broken, it cannot be fixed via software. Hardware replacement or refund is required depending on warranty status.")
    ]
    cursor.executemany(
        "INSERT INTO policies (keyword, policy_text) VALUES (?, ?)", 
        policies_data
    )

    # Commit and close
    conn.commit()
    conn.close()
    print("Successfully generated enterprise.db with customers and policies tables.")

if __name__ == "__main__":
    setup_database()
