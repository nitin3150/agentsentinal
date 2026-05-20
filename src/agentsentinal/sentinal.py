import asyncio

from agentsentinal.core.agents.inspector import InspectorAgent
from agentsentinal.intake.agent_intake import AgentIntake
from agentsentinal.models.agent import InspectedAgentProfile


class AgentSentinel:
    def __init__(self):
        self._intake = AgentIntake()
        self._inspector = InspectorAgent()

    def inspect_agent(self, agent,agenda:str="",system_prompt:str="") -> InspectedAgentProfile:
        profile = self._intake.extract_profile(agent)
        profile.source_object = agent
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop (Jupyter, FastAPI, etc.) — schedule as
            # a coroutine and block via a new thread so we don't nest loops.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._inspector.inspect(profile))
                return future.result()

        return asyncio.run(self._inspector.inspect(profile))
