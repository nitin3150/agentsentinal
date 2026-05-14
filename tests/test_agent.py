from agentsentinal.sentinal import AgentSentinel
import sys
from pathlib import Path

# Add project root to sys.path so demo/ can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.langgraph_agent import run_agent

agent = run_agent()

profile_extrator = AgentSentinel()
print(profile_extrator.inspect_agent(agent))