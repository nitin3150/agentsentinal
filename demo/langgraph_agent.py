# =========================================================
# 2. LANGGRAPH
# =========================================================

from dotenv import load_dotenv
load_dotenv()

import os

from typing_extensions import TypedDict
from pydantic import SecretStr
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "stepfun/step-3.5-flash"),
    base_url=os.getenv("GROQ_BASE_URL"),
    api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY is not None else None,
)

class State(TypedDict):
    message: str

def chatbot(state):
    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["message"]},
    ])

    return {"message": response.content}

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

def run_agent():
    return app