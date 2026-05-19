import asyncio

from agentsentinal.core.agents.inspector import InspectorAgent
from agentsentinal.intake.agent_intake import AgentIntake
from agentsentinal.models.agent import InspectedAgentProfile


class AgentSentinel:
    def __init__(self):
        self._intake = AgentIntake()
        self._inspector = InspectorAgent()

    def inspect_agent(self, agent) -> InspectedAgentProfile:
        profile = self._intake.extract_profile(agent)
        profile.source_object = agent
        return asyncio.run(self._inspector.inspect(profile))
