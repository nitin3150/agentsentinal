from dotenv import load_dotenv
load_dotenv()

import os

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

MODEL = os.environ["MODEL"]
OPENROUTER_BASE_URL = os.environ["OPENROUTER_BASE_URL"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

llm = OpenAI(
    model=MODEL,
    api_base=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

agent = FunctionAgent(
    llm=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
)

def run_agent():
    return agent
# response = agent.run("Explain recursion simply.")

# print(response)