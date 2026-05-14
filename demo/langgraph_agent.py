# =========================================================
# 2. LANGGRAPH
# =========================================================

from dotenv import load_dotenv
load_dotenv()

import os

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

model = ChatOpenAI(
    model=os.getenv("MODEL","stepfun/step-3.5-flash"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
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