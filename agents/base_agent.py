"""
Base Agent — abstract foundation for all specialised agents.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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

    def __init__(self, name: str, llm: Optional[BaseChatModel] = None, tools: Optional[List[Callable]] = None):
        self.name = name
        self.llm: BaseChatModel = llm or build_llm()
        self.tools = tools or []

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
        
        if self.tools:
            return self._call_llm_with_tools(messages)
            
        response = self.llm.invoke(messages)
        return response.content

    def _call_llm_with_tools(self, messages: List[Any]) -> str:
        """
        Handles the tool-calling loop:
        1. Invokes the LLM bound with tools.
        2. Executes requested tools and feeds results back.
        3. Repeats until a final text response is produced.
        """
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Tool dictionary for quick execution lookup
        tool_map = {tool.name: tool for tool in self.tools}
        
        max_iterations = 5
        for _ in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                # Handle Groq's 'tool_use_failed' error where it returns a failed_generation string
                error_str = str(e)
                if 'failed_generation' in error_str:
                    try:
                        # Extract the failed generation text which usually contains the JSON
                        failed_gen = error_str.split("'failed_generation': '")[1].split("'}}")[0]
                        failed_gen = failed_gen.replace("\\n", "\n").replace('\\"', '"')
                        # Remove any trailing <function=...> tag
                        if "<function=" in failed_gen:
                            failed_gen = failed_gen.split("<function=")[0]
                        return failed_gen.strip()
                    except:
                        pass
                raise e

            messages.append(response)
            
            # If there are no tool calls, the LLM gave a final answer
            if not response.tool_calls:
                return response.content
                
            # Execute each requested tool
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                try:
                    print(f"    [Tool] {tool_name}({tool_args})")
                    tool_instance = tool_map.get(tool_name)
                    if not tool_instance:
                        raise ValueError(f"Tool {tool_name} not found")
                        
                    tool_output = tool_instance.invoke(tool_args)
                    
                except Exception as e:
                    tool_output = f"Error executing {tool_name}: {str(e)}"
                    print(f"    [Tool Error] {tool_output}")
                    
                messages.append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call_id
                ))
        
        return "Error: Exceeded maximum tool call iterations."

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
