# =========================================================
# 3. CREWAI
# =========================================================

from dotenv import load_dotenv
load_dotenv()

import os

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

llm = ChatOpenAI(
    model=os.getenv("MODEL","stepfun/step-3.5-flash"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

agent = Agent(
    role="Assistant",
    goal="Help the user",
    backstory=SYSTEM_PROMPT,
    llm=llm,
    verbose=True,
)

task = Task(
    description="Explain what Python is.",
    expected_output="A short explanation of Python.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

def run_agent():
    return crew