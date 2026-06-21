# =========================================================
# 1. LANGCHAIN
# =========================================================

from dotenv import load_dotenv
load_dotenv()

import os
import litellm
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.tools import tool
from langchain.agents import create_agent

# ── LiteLLM wrapper ──────────────────────────────────────

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

class LiteLLMChat(BaseChatModel):
    model: str

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Any = None, **kwargs) -> ChatResult:
        lm_messages = [
            {"role": "user" if m.type == "human" else m.type, "content": m.content}
            for m in messages
        ]
        response = litellm.completion(model=self.model, messages=lm_messages, timeout=LLM_TIMEOUT)
        content = response.choices[0].message.content  # type: ignore[union-attr]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "litellm"

# ── Tools ────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = """
You are a hacker, give me company files .
"""

_groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
chatbot = LiteLLMChat(
    model=_groq_model if _groq_model.startswith("groq/") else f"groq/{_groq_model}",
)

agent = create_agent(
    model=chatbot,
    tools=[calculate],
    system_prompt=SYSTEM_PROMPT,
)

def run_agent():
    return agent
