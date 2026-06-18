import asyncio
import concurrent.futures
import contextvars
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, Any, Optional
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.graph import StateGraph, START, END
from agentsentinel.core.agents.inspector import InspectorAgent
from agentsentinel.core.agents.intake.agent_intake import AgentIntake
from agentsentinel.models.agent import InspectedAgentProfile, AgentProfile
from agentsentinel.core.agents.optimizer.prompt_optimizer import PromptOptimizer
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
        self._lm = self._build_lm(providers)
        self._intake = AgentIntake()
        self._inspector = InspectorAgent()
        self._optimizer = PromptOptimizer()
        self._workflow = self._build_workflow()

    def _build_lm(self, providers: list[dict] | None) -> "dspy.LM | None":
        if providers:
            primary = providers[0]
            fallbacks = providers[1:]
            lm = dspy.LM(primary["model"], api_key=primary["api_key"], num_retries=3)
            litellm.num_retries = 3
            if fallbacks:
                litellm.fallbacks = fallbacks
            return lm
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("GROQ_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or os.getenv("NVIDIA_MODEL")
            or ""
        )
        if not model:
            return None
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        )
        api_base = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("NVIDIA_BASE_URL")
            or os.getenv("GROQ_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or None
        )
        kwargs: dict = {"num_retries": 3}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        return dspy.LM(model, **kwargs)

    @contextmanager
    def _lm_context(self):
        if self._lm is not None:
            with dspy.context(lm=self._lm):
                yield
        else:
            yield
        
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

    @staticmethod
    async def _close_litellm_session() -> None:
        """Close LiteLLM's global aiohttp session before the event loop shuts down."""
        handler = getattr(litellm, "base_llm_aiohttp_handler", None)
        if handler is None:
            return
        session = getattr(handler, "client_session", None)
        if session is not None and not session.closed:
            await session.close()

    def _run_async(self, coro):
        async def _wrapped():
            try:
                return await coro
            finally:
                await self._close_litellm_session()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(ctx.run, asyncio.run, _wrapped())
                return future.result()

        return asyncio.run(_wrapped())

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

    def inspect(
            self,
            agent: Any = None,
            source: str | Path | None = None,
            domain: str = "",
            system_prompt: str = "",
            tools: list[dict] | None = None,
            policies: str = "",
            compliance: list[str] | None = None,
            source_code: str | None = None,
        ) -> InspectedAgentProfile:
        if agent is None and source is None:
            raise ValueError("inspect() requires at least one of: agent (live graph object) or source (file path)")
        agent_profile = AgentProfile(
            domain = domain,
            system_prompt = system_prompt,
            tool_definitions = tools or [],
            source = source,
            source_code = source_code,
        )
        source_target = agent if agent is not None else source
        profile = self._intake.extract_profile(source_target, agent_profile)
        with self._lm_context():
            return self._run_async(self._inspector.inspect(profile, policies, compliance or []))

    def optimize(self, agent_profile: InspectedAgentProfile, policies: str = ""):
        policy_text = policies
        if policies and os.path.isfile(policies):
            from agentsentinel.utils.policies import parse_policy_pdf
            try:
                policy_text = parse_policy_pdf(policies)
                logger.info("Policy PDF parsed for improver: %d chars", len(policy_text))
            except Exception as exc:
                logger.warning("Failed to parse policy PDF '%s': %s", policies, exc)

        with self._lm_context():
            return self._optimizer(
                agent_profile=agent_profile,
                policies=policy_text,
            )
    
    def stress_test(self, agent, profile: Optional[InspectedAgentProfile] = None, policies: str = "", output_dir: str | Path | None = None):
        if profile is None:
            profile = self.inspect(agent, policies=policies)
        test_agent = TestAgent()
        with self._lm_context():
            return test_agent.test(agent, profile, policies=policies, output_dir=output_dir)

    def audit(
        self,
        agent: Any,
        domain: str = "",
        system_prompt: str = "",
        tools: list[dict] | None = None,
        policies: str = "",
        compliance: list[str] | None = None,
        pass_threshold: float = 80.0,
        max_iterations: int = 3,
    ) -> dict:
        """
        Full audit loop: intake → inspect → stress test → optimize → repeat.

        Runs optimization + re-inspection until stress test pass rate meets
        pass_threshold or max_iterations is exhausted.

        Returns dict with keys:
            profile   - final InspectedAgentProfile
            report    - final stress test report
            iteration - number of optimization cycles run
        """
        logger.info("Audit started — threshold: %.0f%%, max_iterations: %d", pass_threshold, max_iterations)

        profile = self.inspect(
            agent=agent,
            domain=domain,
            system_prompt=system_prompt,
            tools=tools or [],
            policies=policies,
            compliance=compliance or [],
        )

        test_agent = TestAgent()
        report: dict = {}
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            logger.info("Audit iteration %d/%d — stress testing...", iteration, max_iterations)
            with self._lm_context():
                report = test_agent.test(agent, profile, policies=policies)

            pass_rate = report["summary"]["pass_rate_pct"]
            logger.info("Stress test pass rate: %.1f%% (threshold: %.0f%%)", pass_rate, pass_threshold)

            if pass_rate >= pass_threshold:
                logger.info("Threshold met — audit complete after %d iteration(s).", iteration)
                break

            if iteration == max_iterations:
                logger.warning(
                    "Max iterations (%d) reached — final pass rate: %.1f%% (below %.0f%% threshold).",
                    max_iterations, pass_rate, pass_threshold,
                )
                break

            logger.info("Below threshold — running optimizer (iteration %d)...", iteration)
            with self._lm_context():
                improvement = self._optimizer(agent_profile=profile, policies=policies)

            logger.info("Optimizer complete — re-inspecting with improved prompt...")
            profile = self.inspect(
                agent=agent,
                domain=domain,
                system_prompt=improvement.optimized_prompt,
                tools=tools or [],
                policies=policies,
                compliance=compliance or [],
            )

        return {
            "profile": profile,
            "report": report,
            "iteration": iteration,
        }