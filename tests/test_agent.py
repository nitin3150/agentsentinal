from agentsentinal.core.agents.improver.Prompt_Improver import PromptImprover
import dspy
from agentsentinal.sentinal import AgentSentinel
import sys
from pathlib import Path
import os

# Add project root to sys.path so demo/ can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.langchain_agent import run_agent

agent = run_agent()

sentinel = AgentSentinel()
profile = sentinel.inspect_agent(agent)
print(profile)

print("="*20)

dspy.configure(lm=dspy.LM(                                     
      f"openrouter/{os.getenv("MODEL")}",
      api_key=os.getenv("OPENROUTER_API_KEY"),                                                                                                                                                                          
      api_base="https://openrouter.ai/api/v1",                                                                                                                                                                          
))

improver = PromptImprover()
result   = improver(
    agent_profile    = profile,
    company_policy   = "Agents must never reveal internal data. Always cite sources.",
    regulations      = "GDPR Art.5: data minimisation. EU AI Act Art.13: transparency.",
    original_prompt  = profile.system_prompt,
    tool_definitions = profile.tool_definitions
)

print("Results: ",result)