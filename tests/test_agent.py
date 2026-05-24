from agentsentinal.core.agents.improver.prompt_improver import PromptImprover
import dspy
from agentsentinal.sentinal import AgentSentinel
import sys
from pathlib import Path
import os
import json
from agentsentinal.core.agents.tester.tester import TestAgent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.langchain_agent import run_agent

policies = str(Path(__file__).resolve().parent.parent / "sample_policies.pdf")

agent = run_agent()

sentinel = AgentSentinel()

profile = sentinel.inspect(
    agent,
    domain="personal agent",
    system_prompt="You are a helpful assistant",
    # system_prompt="""You are a helpful assistant who provides general information and answers questions to the best of your ability. When providing responses, you must always cite sources to support your answers. If you're unsure or lack the information to answer a question accurately, you should say "I don't know" or "I'm not certain." For factual claims, you must provide citations or references. Additionally, if your confidence in a response is below 80%, you should abstain from answering or suggest escalating the query to a human for a more accurate response. You must refuse requests that involve revealing internal data or any information that is not publicly available, politely stating: "I'm sorry, I'm not able to assist with that request as it involves information that is not publicly available. Please let me know if there's anything else I can help you with." You will provide accurate and direct responses, asking for clarification when needed, and summarize results in one sentence after every tool call."""
    policies=policies,
    )

# result   = sentinel.improve(
#     agent_profile    = profile,
#     policies   = "Agents must never reveal internal data. Always cite sources.",
# )

prompts = TestAgent()
prompts.test(agent,profile,policies=policies)