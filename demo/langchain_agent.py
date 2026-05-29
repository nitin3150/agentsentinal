# =========================================================
# 1. LANGCHAIN
# =========================================================

from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

# ── Tools ────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

_model = os.getenv("GROQ_MODEL", "stepfun/step-3.5-flash")
_model = _model.split("/", 1)[-1] if "/" in _model else _model

chatbot = ChatOpenAI(
    model=_model,
    openai_api_base=os.getenv("GROQ_BASE_URL"),
    openai_api_key=os.getenv("GROQ_API_KEY"),
)

agent = create_agent(
    model=chatbot,
    tools=[calculate],
    system_prompt=SYSTEM_PROMPT,
)

def run_agent():
    return agent