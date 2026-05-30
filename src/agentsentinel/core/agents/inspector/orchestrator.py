from agentsentinel.models.agent import InspectedAgentProfile
import asyncio
from typing import Any
from uuid import uuid4

from agentsentinel.core.agents.inspector.aggregator import aggregate
from agentsentinel.core.agents.inspector.analyzers.framework import (
    FrameworkAnalysis,
    analyze_framework,
)
from agentsentinel.core.agents.inspector.analyzers.memory import (
    MemoryAnalysis,
    analyze_memory,
)
from agentsentinel.core.agents.inspector.analyzers.policy import (
    PolicyAnalysis,
    analyze_policy,
)
from agentsentinel.core.agents.inspector.analyzers.prompt import (
    PromptAnalysis,
    analyze_prompt,
)
from agentsentinel.core.agents.inspector.analyzers.semantic import (
    SemanticAnalysis,
    analyze_semantic,
)
from agentsentinel.core.agents.inspector.analyzers.tools import (
    ToolsAnalysis,
    analyze_tools,
)
from agentsentinel.models.intake import ExtractionResult
from agentsentinel.models import AgentProfile
from agentsentinel.utils.policies import parse_policy_pdf
import logging

logger = logging.getLogger(__name__)

CONFIDENCE_WARNING_THRESHOLD = 0.6


class InspectorAgent:
    """Analyses an already-extracted AgentProfile and returns an InspectedAgentProfile.

    Never invokes the agent. Inspection only — static analysis + one optional LLM call.
    """

    def __init__(self, semantic_enabled: bool = True):
        self.semantic_enabled = semantic_enabled

    async def inspect(self, profile: AgentProfile, policies: str = "", compliance: list[str] = ['all']) -> InspectedAgentProfile:
        logger.info("Inspection Starts...")
        extraction = ExtractionResult(
            system_prompt=profile.system_prompt,
            tool_definitions=profile.tool_definitions,
            framework=str(profile.framework) if profile.framework else "unknown",
            source_object=profile.source_object,
            warnings=list(profile.warnings),
        )
        extraction.confidence = extraction.compute_confidence()

        logger.info("Extraction Confidence: %s", extraction.confidence)

        if extraction.confidence < CONFIDENCE_WARNING_THRESHOLD:
            extraction.warnings.append(
                f"Low extraction confidence ({extraction.confidence:.2f}). "
                "Pass system_prompt and tool_definitions explicitly for a more complete analysis."
            )

        policy_text = ""
        if policies:
            try:
                policy_text = await asyncio.to_thread(parse_policy_pdf, policies)
                logger.info("Policy PDF parsed: %d chars", len(policy_text))
            except Exception as exc:
                logger.warning("Failed to parse policy PDF '%s': %s", policies, exc)
                extraction.warnings.append(f"Policy PDF could not be parsed: {exc}")

        # Static analyzers are pure Python (no I/O, no GIL release) — run them
        # directly rather than via to_thread to avoid pointless thread overhead.
        try:
            logger.info("Analysing Prompt...")
            prompt_res = analyze_prompt(extraction.system_prompt)
        except Exception as exc:
            logger.error("Error Analysing Prompt!!")
            prompt_res = exc

        try:
            logger.info("Analysing Tools...")
            tools_res = analyze_tools(extraction.tool_definitions)
        except Exception as exc:
            logger.error("Error Analysing Tools!!")
            tools_res = exc

        try:
            logger.info("Analysing Memory...")
            memory_res = analyze_memory(extraction.source_object, extraction.source_code)
        except Exception as exc:
            logger.error("Error Analysing Memory!!")
            memory_res = exc

        try:
            logger.info("Analysing Framework...")
            framework_res = analyze_framework(extraction.source_object, extraction.source_code)
        except Exception as exc:
            logger.error("Error Analysing Framework!!")
            framework_res = exc

        static_findings = {}
        if isinstance(prompt_res, PromptAnalysis):
            static_findings = {
                "ambiguous_phrases": prompt_res.ambiguous_phrases,
                "constraint_count": prompt_res.constraint_count,
            }

        try:
            logger.info("Analysing Semantics...")
            semantic_res: Any = await self._run_semantic(extraction, static_findings)
        except Exception as exc:
            logger.error("Error Analysing Semantics!!")
            semantic_res = exc

        try:
            logger.info("Analysing Policy...")
            policy_res: Any = await analyze_policy(
                system_prompt=extraction.system_prompt,
                tool_definitions=extraction.tool_definitions,
                policy_text=policy_text,
            )
        except Exception as exc:
            logger.error("Error Analysing Policy!!")
            policy_res = exc

        prompt = _unwrap(prompt_res, PromptAnalysis, extraction, "prompt")
        tools = _unwrap(tools_res, ToolsAnalysis, extraction, "tools")
        memory = _unwrap(memory_res, MemoryAnalysis, extraction, "memory")
        framework = _unwrap(framework_res, FrameworkAnalysis, extraction, "framework")
        semantic = _unwrap(semantic_res, SemanticAnalysis, extraction, "semantic")
        policy = _unwrap(policy_res, PolicyAnalysis, extraction, "policy")

        return aggregate(
            agent_id=str(uuid4()),
            extraction=extraction,
            prompt=prompt,
            tools=tools,
            memory=memory,
            framework=framework,
            semantic=semantic,
            policy=policy,
        )

    async def _run_semantic(
        self,
        extraction: ExtractionResult,
        static_findings: dict[str, Any],
    ) -> SemanticAnalysis:
        if not self.semantic_enabled:
            return SemanticAnalysis()
        return await analyze_semantic(
            system_prompt=extraction.system_prompt,
            framework=extraction.framework,
            tool_count=len(extraction.tool_definitions),
            static_findings=static_findings,
        )


def _unwrap(value: Any, expected_type: type, extraction: ExtractionResult, label: str):
    if isinstance(value, BaseException):
        extraction.warnings.append(f"{label} analyzer failed: {type(value).__name__}: {value}")
        return None
    if isinstance(value, expected_type):
        return value
    return None
