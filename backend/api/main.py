"""
FastAPI Server — Main entry point for the AI Agent Coordination Engine API.
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from backend.core.config import settings
from backend.orchestration.message_bus import message_bus

from backend.api.routers import workflows

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Multi-Agent Coordination & Decision Engine API",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflows.router)

from pydantic import BaseModel
import sqlite3
from backend.tools.enterprise_tools import _get_db_connection

class LoginRequest(BaseModel):
    username: str

@app.post("/api/v1/auth/login")
async def login(req: LoginRequest):
    username = req.username.strip()
    username_lower = username.lower()
    
    # Predefined employees
    valid_employees = ["elena vasquez", "sarah chen", "marcus lee", "rachel torres", "david kim", "admin"]
    if username_lower in valid_employees:
        return {"status": "success", "role": "Employee", "username": username}
        
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        # Check customers
        cursor.execute("SELECT name, id FROM customers WHERE LOWER(name) = ? OR LOWER(id) = ?", (username_lower, username_lower))
        cust = cursor.fetchone()
        if cust:
            conn.close()
            return {"status": "success", "role": "Customer", "username": cust[0], "id": cust[1]}
            
        # Check vendors
        cursor.execute("SELECT name, id FROM vendors WHERE LOWER(name) = ? OR LOWER(id) = ?", (username_lower, username_lower))
        vend = cursor.fetchone()
        if vend:
            conn.close()
            return {"status": "success", "role": "Vendor", "username": vend[0], "id": vend[1]}
            
        conn.close()
    except Exception as e:
        logger.error(f"Login DB check failed: {e}")
        
    return {"status": "error", "message": "User not found in enterprise records."}

@app.on_event("startup")
async def on_startup():
    """Ensure database and tools are fully initialized on server start."""
    from backend.tools.enterprise_tools import initialize_all_tools
    initialize_all_tools()
    logger.info("Application startup: Enterprise database and tools ready.")


# Active WebSocket connections
active_connections = set()

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent events."""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info("New WebSocket connection established")
    
    # Callback to push events to this websocket
    async def event_handler(event):
        try:
            await websocket.send_json(event.to_dict())
        except Exception:
            pass # Connection might be closed
            
    # Subscribe to all events
    message_bus.subscribe_all(event_handler)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    finally:
        active_connections.remove(websocket)
        message_bus.unsubscribe("*", event_handler)

@app.get("/api/health")
async def health_check():
    """System health check."""
    from backend.memory.memory_manager import memory_manager
    from backend.tools.registry import registry
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "llm_provider": settings.LLM_PROVIDER,
        "tools_loaded": len(registry),
        "memory_status": memory_manager.get_stats(),
    }

# Mount frontend static files
import os
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
