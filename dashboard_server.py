"""
dashboard_server.py — FastAPI backend that powers the web testing dashboard.

Run with: uvicorn dashboard_server:app --reload --port 8000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict

from memory.shared_memory import SharedMemory
from workflows.agent_workflow import AgentWorkflow

app = FastAPI(title="AI Agent Engine — Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared memory across requests in this server session
_memory = SharedMemory()


class QueryRequest(BaseModel):
    query: str
    reset_memory: bool = False


class QueryResponse(BaseModel):
    plan: Dict[str, Any]
    task_results: list
    response: str
    session_id: str


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "dashboard" / "index.html")


@app.post("/api/run", response_model=QueryResponse)
async def run_query(body: QueryRequest):
    global _memory
    if body.reset_memory:
        _memory = SharedMemory()

    workflow = AgentWorkflow(memory=_memory)
    result = workflow.run(body.query)
    return QueryResponse(**result)


@app.get("/api/history")
async def get_history():
    return {"history": _memory.get_history(), "session_id": _memory.session_id}


@app.delete("/api/history")
async def clear_history():
    _memory.clear_history()
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "session_id": _memory.session_id}


# Serve dashboard static files
app.mount("/dashboard", StaticFiles(directory=str(Path(__file__).parent / "dashboard")), name="dashboard")
