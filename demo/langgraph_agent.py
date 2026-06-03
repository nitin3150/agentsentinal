# =========================================================
# 2. LANGGRAPH
# =========================================================

from dotenv import load_dotenv
load_dotenv()

import os
import litellm
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

_groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_model = _groq_model if _groq_model.startswith("groq/") else f"groq/{_groq_model}"

class State(TypedDict):
    message: str

def chatbot(state):
    response = litellm.completion(
        model=_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["message"]},
        ],
    )
    return {"message": response.choices[0].message.content}  # type: ignore[union-attr]

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

def run_agent():
    return app
