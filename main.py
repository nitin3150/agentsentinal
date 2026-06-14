from dotenv import load_dotenv
load_dotenv()

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)  # LangChainPendingDeprecationWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentsentinel.sentinel import AgentSentinel
from demo.langchain_agent import run_agent

policies = str(Path(__file__).resolve().parent / "sample_policies.pdf")

if __name__ =="__main__":
    agent = run_agent()

    try:
        GROQ_API_KEY = os.environ["GROQ_API_KEY"]
        OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    except:
        raise

    sentinel = AgentSentinel(providers=[
        {"api_key": GROQ_API_KEY,       "model": os.environ["GROQ_MODEL"]},
        {"api_key": OPENROUTER_API_KEY,  "model": os.environ["OPENROUTER_MODEL"]},
    ])

    profile = sentinel.inspect(
        agent,
        source = "demo/langchain_agent.py",
        domain="personal agent",
        policies=policies,
    )

    result = sentinel.optimize(
        agent_profile=profile,
        policies=policies,
    )

    sentinel.stress_test(agent, profile, policies=policies)