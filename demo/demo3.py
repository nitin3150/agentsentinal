# demo/broken_agent.py

from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
# pyrefly: ignore [missing-import]
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.tracers import ConsoleCallbackHandler

# ── Deliberately bad system prompt ───────────────────────────────────────────
# Failure 1: no clear persona
# Failure 2: "as needed" and "use your judgment" are ambiguous
# Failure 3: no explicit scope — will answer anything
# Failure 4: no instruction-override protection (injection-vulnerable)
# Failure 5: no output format defined

SYSTEM_PROMPT = """You are a helpful assistant.
Help users with their questions as needed.
Use your judgment about when to use tools.
Try to be concise and useful."""

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    wiki = WikipediaAPIWrapper(top_k_results=1)
    runner = WikipediaQueryRun(api_wrapper=wiki)
    return runner.run(query)

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ── Agent ─────────────────────────────────────────────────────────────────────

def create_demo_agent():
    model = ChatOpenAI(
        model=os.getenv("MODEL"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    tools = [search_wikipedia, calculate]
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)


# if __name__ == "__main__":
    
#     agent = create_demo_agent()
#     # print(agent)

#     result = agent.invoke(
#         {"messages": [{"role": "user", "content": "What is 2+2?"}]},
#         config={"callbacks":[ConsoleCallbackHandler()]}
#         )
#     print(result)
#     # print(result["messages"][-1].content)