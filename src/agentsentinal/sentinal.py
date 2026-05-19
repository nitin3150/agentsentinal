from agentsentinal.intake.agent_intake import AgentIntake

class AgentSentinel:
    def __init__(self):
        self._intake = AgentIntake()
    
    def inspect_agent(self,agent_graph):
        agent_profile = self._intake.extract_profile(agent_graph)
        return agent_profile