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

chatbot = ChatOpenAI(
    model=os.getenv("MODEL","stepfun/step-3.5-flash"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

agent = create_agent(
    model=chatbot,
    tools=[calculate],
    system_prompt=SYSTEM_PROMPT,
)

def run_agent():
    return agent