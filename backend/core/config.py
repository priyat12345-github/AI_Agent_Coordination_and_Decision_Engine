"""
Core configuration module for AI Agent Coordination & Decision Engine.
Manages settings, environment variables, and system-wide constants.
"""

import os
from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


# Project root directory
BASE_DIR = Path(__file__).parent.parent.parent


class LLMProvider(str, Enum):
    """Supported LLM provider modes."""
    MOCK = "mock"
    GEMINI = "gemini"
    OPENAI = "openai"
    GROQ = "groq"


class AgentRole(str, Enum):
    """Defined roles for specialized agents."""
    PLANNER = "planner"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DECISION = "decision"
    EXECUTOR = "executor"


class WorkflowStatus(str, Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "AI Agent Coordination & Decision Engine"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development", alias="APP_ENV")
    DEBUG: bool = Field(default=True)

    # LLM Configuration
    LLM_PROVIDER: LLMProvider = Field(default=LLMProvider.MOCK)
    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    LLM_TEMPERATURE: float = Field(default=0.7)
    LLM_MAX_TOKENS: int = Field(default=2048)

    # API Server
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")

    # Memory
    SHORT_TERM_MEMORY_SIZE: int = Field(default=10)
    CHROMA_PERSIST_DIR: str = Field(
        default=str(BASE_DIR / "data" / "chroma_store")
    )
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    # Database
    DATABASE_URL: str = Field(
        default=f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'enterprise.db'}"
    )

    # Reports
    REPORTS_DIR: str = Field(default=str(BASE_DIR / "reports"))

    # Security
    SECRET_KEY: str = Field(default="ai-agent-engine-secret-key-2024-enterprise")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = Field(default=30)

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings instance
settings = Settings()


# Agent configuration defaults
AGENT_CONFIGS = {
    AgentRole.PLANNER: {
        "name": "Planner Agent",
        "description": "Breaks down complex business requests into structured sub-tasks and routes them to specialized agents.",
        "max_iterations": 5,
        "timeout_seconds": 60,
        "icon": "🗺️",
        "color": "#6366f1",
    },
    AgentRole.RESEARCH: {
        "name": "Research Agent",
        "description": "Retrieves and synthesizes information from external sources, databases, and knowledge bases.",
        "max_iterations": 8,
        "timeout_seconds": 90,
        "icon": "🔍",
        "color": "#06b6d4",
    },
    AgentRole.ANALYSIS: {
        "name": "Analysis Agent",
        "description": "Processes gathered data, identifies patterns, evaluates options, and generates insights.",
        "max_iterations": 6,
        "timeout_seconds": 75,
        "icon": "📊",
        "color": "#10b981",
    },
    AgentRole.DECISION: {
        "name": "Decision Agent",
        "description": "Evaluates analysis outputs against business rules and generates actionable recommendations.",
        "max_iterations": 4,
        "timeout_seconds": 45,
        "icon": "⚖️",
        "color": "#f59e0b",
    },
    AgentRole.EXECUTOR: {
        "name": "Executor Agent",
        "description": "Carries out approved actions including API calls, report generation, and system updates.",
        "max_iterations": 10,
        "timeout_seconds": 120,
        "icon": "⚡",
        "color": "#ef4444",
    },
}

# Workflow templates mapping
WORKFLOW_TEMPLATES = {
    "market_analysis": {
        "name": "Market Analysis Workflow",
        "description": "Research market trends, analyse competitive landscape, and generate strategic recommendations.",
        "agents": [AgentRole.PLANNER, AgentRole.RESEARCH, AgentRole.ANALYSIS, AgentRole.DECISION, AgentRole.EXECUTOR],
        "estimated_duration": "2-5 minutes",
        "icon": "📈",
    },
    "vendor_evaluation": {
        "name": "Vendor Evaluation Workflow",
        "description": "Evaluate and score vendor proposals based on criteria, generate procurement recommendations.",
        "agents": [AgentRole.PLANNER, AgentRole.RESEARCH, AgentRole.ANALYSIS, AgentRole.DECISION, AgentRole.EXECUTOR],
        "estimated_duration": "3-6 minutes",
        "icon": "🏭",
    },
    "customer_escalation": {
        "name": "Customer Escalation Workflow",
        "description": "Classify and resolve customer escalations using CRM data and automated response generation.",
        "agents": [AgentRole.PLANNER, AgentRole.RESEARCH, AgentRole.DECISION, AgentRole.EXECUTOR],
        "estimated_duration": "1-3 minutes",
        "icon": "🎯",
    },
    "financial_review": {
        "name": "Financial Review Workflow",
        "description": "Analyse financial data, assess risk exposure, and generate executive summary reports.",
        "agents": [AgentRole.PLANNER, AgentRole.RESEARCH, AgentRole.ANALYSIS, AgentRole.DECISION, AgentRole.EXECUTOR],
        "estimated_duration": "3-5 minutes",
        "icon": "💰",
    },
    "hr_recruitment": {
        "name": "HR Recruitment Workflow",
        "description": "Screen candidates, evaluate fit against job requirements, and recommend hiring decisions.",
        "agents": [AgentRole.PLANNER, AgentRole.RESEARCH, AgentRole.ANALYSIS, AgentRole.DECISION],
        "estimated_duration": "2-4 minutes",
        "icon": "👥",
    },
}
