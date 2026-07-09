"""
Base Agent — abstract foundation for all specialised agents.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def build_llm() -> BaseChatModel:
    """
    Factory that instantiates the appropriate LangChain chat model
    based on the LLM_PROVIDER environment variable.
    Defaults to Groq (llama-3.3-70b-versatile).
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    temperature = float(os.getenv("TEMPERATURE", "0.3"))

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-pro"),
            temperature=temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Choose from: groq, openai, google, anthropic"
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the coordination engine.

    Subclasses must implement:
        - system_prompt (property)
        - run(task, context, **kwargs)
    """

    def __init__(self, name: str, llm: Optional[BaseChatModel] = None):
        self.name = name
        self.llm: BaseChatModel = llm or build_llm()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt string."""
        ...

    @abstractmethod
    def run(self, task: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the agent's primary function and return a result dict."""
        ...

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _call_llm(self, human_message: str) -> str:
        """Send a system + human message pair to the LLM and return the raw text."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_message),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from an LLM response.
        Handles responses that wrap JSON in markdown code fences.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            return {"error": str(exc), "raw_response": raw}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
