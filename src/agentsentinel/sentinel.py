import asyncio
import concurrent.futures
from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, START, END
from agentsentinel.core.agents.inspector import InspectorAgent
from agentsentinel.core.agents.intake.agent_intake import AgentIntake
from agentsentinel.models.agent import InspectedAgentProfile, AgentProfile
from agentsentinel.core.agents.improver.prompt_improver import PromptImprover
from agentsentinel.core.agents.tester.tester import TestAgent
import logging
import os
import dspy
import litellm

logger = logging.getLogger(__name__)

class SentinelState(TypedDict):
    agent: Any
    agent_profile: Optional[AgentProfile]
    inspected_profile: Optional[InspectedAgentProfile]

class AgentSentinel:
    def __init__(self, providers: list[dict] | None = None):
        if providers:
            self._configure_providers(providers)
        else:
            self._configure_from_env()
        self._intake = AgentIntake()
        self._inspector = InspectorAgent()
        self._improver = PromptImprover()
        self._workflow = self._build_workflow()
    
    def _configure_from_env(self):
        model = os.getenv("GROQ_MODEL") or os.getenv("OPENROUTER_MODEL") or ""
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
        if model and api_key:
            dspy.configure(lm=dspy.LM(model, api_key=api_key, num_retries=3))

    def _configure_providers(self, providers: list[dict]):
        primary = providers[0]
        fallbacks = providers[1:]

        dspy.configure(lm=dspy.LM(
            primary["model"],
            api_key=primary["api_key"],
            num_retries=3,
        ))

        litellm.num_retries = 3
        if fallbacks:
            litellm.fallbacks = fallbacks
        
    def _build_workflow(self):
        builder = StateGraph(SentinelState)

        builder.add_node("intake", self._intake_node)
        builder.add_node("inspector", self._inspector_node)

        builder.add_edge(START, "intake")
        builder.add_edge("intake", "inspector")
        builder.add_edge("inspector", END)

        return builder.compile()

    def _intake_node(self, state: SentinelState) -> dict:
        profile = self._intake.extract_profile(
            state["agent"],
            state["agent_profile"]
            )
        profile.source_object = state["agent"]
        return {"agent_profile": profile}
    
    async def _inspector_node(self, state: SentinelState) -> dict:
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

    def _initial_state(self, agent:Any, domain: str = "", system_prompt:str = "") -> SentinelState:
        return {
            "agent": agent,
            "agent_profile": AgentProfile(
                domain = domain,
                system_prompt = system_prompt,
            ),
            "inspected_profile":None,
        }

    def inspect(self, agent, domain: str = "", system_prompt: str = "", policies:str="") -> InspectedAgentProfile:
        agent_profile = AgentProfile(
            domain=domain,
            system_prompt=system_prompt,
        )
        profile = self._intake.extract_profile(agent, agent_profile)
        return self._run_async(self._inspector.inspect(profile,policies))

    def improve(self, agent_profile: InspectedAgentProfile, policies: str = ""):
        policy_text = policies
        if policies and os.path.isfile(policies):
            from agentsentinel.utils.policies import parse_policy_pdf
            try:
                policy_text = parse_policy_pdf(policies)
                logger.info("Policy PDF parsed for improver: %d chars", len(policy_text))
            except Exception as exc:
                logger.warning("Failed to parse policy PDF '%s': %s", policies, exc)

        result = self._improver(
            agent_profile=agent_profile,
            policies=policy_text,
        )
        return result
    
    def stress_test(self, agent, profile: Optional[InspectedAgentProfile] = None, policies: str = ""):
        if profile is None:
            profile = self.inspect(agent, policies = policies)
        test_agent = TestAgent()
        test_agent.test(agent, profile, policies = policies)

    def audit(self,agent, domain:str="",system_prompt:str=""):
        result = self._invoke(self._initial_state(agent, domain=domain, system_prompt=system_prompt))
        return result["inspected_profile"]  # type: ignore[return-value]