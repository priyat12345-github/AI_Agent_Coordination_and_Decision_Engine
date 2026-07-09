"""
Unit tests for individual agent classes.
Run with: python -m pytest tests/test_agents.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Return a mock LLM that returns a preset JSON string."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='{"goal": "test", "tasks": [{"id": 1, "description": "analyse data", "agent": "ANALYST", "depends_on": []}]}'
    )
    return llm


# ─── BaseAgent ────────────────────────────────────────────────────────────────

def test_base_agent_parse_json_plain():
    from agents.base_agent import BaseAgent

    class DummyAgent(BaseAgent):
        @property
        def system_prompt(self):
            return "system"

        def run(self, task, context, **kwargs):
            return {}

    agent = DummyAgent(name="dummy", llm=MagicMock())
    result = agent._parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_base_agent_parse_json_with_fences():
    from agents.base_agent import BaseAgent

    class DummyAgent(BaseAgent):
        @property
        def system_prompt(self):
            return "system"

        def run(self, task, context, **kwargs):
            return {}

    agent = DummyAgent(name="dummy", llm=MagicMock())
    raw = "```json\n{\"key\": \"value\"}\n```"
    result = agent._parse_json(raw)
    assert result == {"key": "value"}


# ─── PlannerAgent ─────────────────────────────────────────────────────────────

def test_planner_agent_run(mock_llm):
    from agents.planner_agent import PlannerAgent

    agent = PlannerAgent(llm=mock_llm)
    plan = agent.run(task="Analyse sales data", context={})

    assert "goal" in plan or "agent" in plan
    assert plan["agent"] == "PlannerAgent"
    mock_llm.invoke.assert_called_once()


# ─── AnalystAgent ────────────────────────────────────────────────────────────

def test_analyst_agent_run():
    from agents.analyst_agent import AnalystAgent

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"task_id": 1, "findings": "Revenue grew 20%", "key_points": ["Growth"], "confidence": "HIGH", "recommendations": []}'
    )

    agent = AnalystAgent(llm=mock_llm)
    result = agent.run(task="Analyse Q1 revenue", context={}, task_id=1)

    assert result["agent"] == "AnalystAgent"
    assert result.get("findings") == "Revenue grew 20%"


# ─── ExecutorAgent ────────────────────────────────────────────────────────────

def test_executor_agent_run():
    from agents.executor_agent import ExecutorAgent

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"task_id": 2, "status": "SUCCESS", "result": "Done", "output": "Report generated", "errors": []}'
    )

    agent = ExecutorAgent(llm=mock_llm)
    result = agent.run(task="Generate report", context={}, task_id=2, analysis={})

    assert result["agent"] == "ExecutorAgent"
    assert result.get("status") == "SUCCESS"


# ─── ResponderAgent ───────────────────────────────────────────────────────────

def test_responder_agent_run():
    from agents.responder_agent import ResponderAgent

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="## Final Report\nAll tasks complete.")

    agent = ResponderAgent(llm=mock_llm)
    result = agent.run(
        task="Summarise sales",
        context={},
        plan={"goal": "test", "tasks": []},
        results=[],
    )

    assert result["agent"] == "ResponderAgent"
    assert "Final Report" in result["response"]


# ─── SharedMemory ─────────────────────────────────────────────────────────────

def test_shared_memory_set_get():
    from memory.shared_memory import SharedMemory

    mem = SharedMemory()
    mem.set("foo", 42)
    assert mem.get("foo") == 42
    assert mem.get("missing", "default") == "default"


def test_shared_memory_message_history():
    from memory.shared_memory import SharedMemory

    mem = SharedMemory()
    mem.add_message("user", "Hello")
    mem.add_message("assistant", "Hi there", agent="ResponderAgent")

    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["agent"] == "ResponderAgent"


def test_shared_memory_snapshot():
    from memory.shared_memory import SharedMemory

    mem = SharedMemory()
    mem.set("x", 99)
    snap = mem.snapshot()

    assert "session_id" in snap
    assert snap["store"]["x"] == 99
    assert snap["message_count"] == 0
