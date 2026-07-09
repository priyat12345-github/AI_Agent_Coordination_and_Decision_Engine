# AI Agent Coordination & Decision Engine
### Milestone 1 — Agent Foundation Development

> A multi-agent coordination system built with LangChain, supporting OpenAI, Google Gemini, and Anthropic Claude.

---

## Project Structure

```
ai-agent-engine/
├── agents/
│   ├── base_agent.py       # Abstract base + LLM factory
│   ├── planner_agent.py    # Decomposes requests into task plans
│   ├── analyst_agent.py    # Analyses information & produces findings
│   ├── executor_agent.py   # Executes specific tasks
│   └── responder_agent.py  # Synthesises final user-facing response
├── prompts/
│   ├── planner_prompts.py
│   ├── analyst_prompts.py
│   └── executor_prompts.py
├── workflows/
│   └── agent_workflow.py   # Planner → Analyst/Executor → Responder pipeline
├── memory/
│   └── shared_memory.py    # In-memory key-value + message history store
├── dashboard/
│   ├── index.html          # Web testing UI
│   ├── style.css
│   └── app.js
├── tests/
│   ├── test_agents.py      # Unit tests for all agents
│   └── test_workflows.py   # Integration tests for the workflow
├── dashboard_server.py     # FastAPI backend for the dashboard
├── main.py                 # Rich CLI entry point
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and fill in your provider + API key
# e.g. for OpenAI:
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
```

### 3a. Run the CLI

```bash
# Interactive mode
python main.py

# Single query mode
python main.py --query "Analyse our Q2 sales and suggest growth strategies"
```

### 3b. Run the Web Dashboard

```bash
# Start the API server
uvicorn dashboard_server:app --reload --port 8000

# Open dashboard/index.html in your browser
```

---

## Agents

| Agent | Role | Output Format |
|---|---|---|
| **PlannerAgent** | Decomposes user request into numbered sub-tasks | JSON: `{ goal, tasks[] }` |
| **AnalystAgent** | Analyses a sub-task and produces structured findings | JSON: `{ findings, key_points, confidence, recommendations }` |
| **ExecutorAgent** | Carries out a specific task and returns the result | JSON: `{ status, result, output, errors }` |
| **ResponderAgent** | Synthesises all results into a Markdown reply | Plain Markdown text |

---

## Supported LLM Providers

Set `LLM_PROVIDER` in your `.env` file:

| Provider | Value | Model Env Var |
|---|---|---|
| OpenAI | `openai` | `OPENAI_MODEL` (default: `gpt-4o`) |
| Google | `google` | `GOOGLE_MODEL` (default: `gemini-1.5-pro`) |
| Anthropic | `anthropic` | `ANTHROPIC_MODEL` (default: `claude-3-5-sonnet-20241022`) |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Workflow Diagram

```
User Request
     │
     ▼
┌─────────────┐
│  Planner    │  → JSON execution plan
└─────────────┘
     │
     ├──► ANALYST tasks  → AnalystAgent
     │
     ├──► EXECUTOR tasks → ExecutorAgent
     │
     ▼
┌─────────────┐
│  Responder  │  → Final Markdown reply
└─────────────┘
     │
     ▼
  User Output
```

---

## Milestone 1 Checklist

- [x] Configure LangChain and required dependencies
- [x] Develop foundational AI agents (Planner, Analyst, Executor, Responder)
- [x] Implement prompt templates and interaction workflows
- [x] Create basic testing interfaces (CLI + Web Dashboard)
- [x] Shared memory system (key-value store + message history)
- [x] Unit & integration tests
