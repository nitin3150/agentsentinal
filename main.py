from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentsentinal.sentinal import AgentSentinel
from demo.langchain_agent import run_agent

policies = str(Path(__file__).resolve().parent / "sample_policies.pdf")

agent = run_agent()

sentinel = AgentSentinel(providers=[
    {"api_key": os.environ["GROQ_API_KEY"],       "model": os.environ["GROQ_MODEL"]},
    {"api_key": os.environ["OPENROUTER_API_KEY"],  "model": os.environ["OPENROUTER_MODEL"]},
])

profile = sentinel.inspect(
    agent,
    domain="personal agent",
    policies=policies,
)

result = sentinel.improve(
    agent_profile=profile,
    policies=policies,
)

sentinel.stress_test(agent, profile, policies=policies)
