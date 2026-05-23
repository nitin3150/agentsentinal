import asyncio
import concurrent.futures
from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, START,END
from agentsentinal.core.agents.inspector import InspectorAgent
from agentsentinal.core.agents.intake.agent_intake import AgentIntake
from agentsentinal.models.agent import InspectedAgentProfile, AgentProfile
from agentsentinal.core.agents.improver.Prompt_Improver import PromptImprover
import logging

logger = logging.getLogger(__name__)

class SentinelState(TypedDict):
    agent: Any
    agent_profile: Optional[AgentProfile]
    inspected_profile: Optional[InspectedAgentProfile]
    # improved_profile: Optional[] 

class AgentSentinel:
    def __init__(self):
        self._intake = AgentIntake()
        self._inspector = InspectorAgent()
        self._improver = PromptImprover()
        self._workflow = self._build_workflow()
    
    def _build_workflow(self):
        builder = StateGraph(SentinelState)

        builder.add_node("intake", self._intake_node)
        builder.add_node("inspector", self._inspector_node)

        builder.add_edge(START, "intake")
        builder.add_edge("intake", "inspector")
        builder.add_edge("inspector", END)

        return builder.compile()

    def _intake_node(self, state: SentinelState) -> dict:
        # logger.info("Starting Agent Profiling....")
        profile = self._intake.extract_profile(
            state["agent"],
            state["agent_profile"]
            )
        profile.source_object = state["agent"]
        # logger.info("Profile: %s", profile.to_log_str())
        return {"agent_profile": profile}
    
    async def _inspector_node(self, state: SentinelState) -> dict:
        # logger.info("Inspection Starts...")
        if state["agent_profile"] is None:
            raise ValueError("intake node must run before inspector")
        inspected_profile = await self._inspector.inspect(state["agent_profile"])
        return {"inspected_profile": inspected_profile}

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()

        return asyncio.run(coro)

    def _invoke(self, initial_state: SentinelState) -> SentinelState:
        return self._run_async(self._workflow.ainvoke(initial_state))  # type: ignore[return-value]

    def _inital_state(self, agent:Any, domain: str = "", system_prompt:str = "") -> SentinelState:
        return {
            "agent": agent,
            "agent_profile": AgentProfile(
                domain = domain,
                system_prompt = system_prompt,
            ),
            "inspected_profile":None,
        }

    def inspect(self, agent, domain: str = "", system_prompt: str = "") -> InspectedAgentProfile:
        agent_profile = AgentProfile(
            domain=domain,
            system_prompt=system_prompt,
        )
        profile = self._intake.extract_profile(agent, agent_profile)
        return self._run_async(self._inspector.inspect(profile))

    def improve(self):
        pass

    def stress_test(self):
        pass

    def certify(self,agent, domain:str="",system_prompt:str=""):
        result = self._invoke(self._inital_state(agent, domain=domain, system_prompt=system_prompt))
        return result["inspected_profile"]  # type: ignore[return-value]