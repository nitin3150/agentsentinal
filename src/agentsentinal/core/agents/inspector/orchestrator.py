import asyncio
from typing import Any, Optional

from agentsentinal.core.agents.inspector.aggregator import aggregate
from agentsentinal.core.agents.inspector.analyzers.framework import (
    FrameworkAnalysis,
    analyze_framework,
)
from agentsentinal.core.agents.inspector.analyzers.memory import (
    MemoryAnalysis,
    analyze_memory,
)
from agentsentinal.core.agents.inspector.analyzers.prompt import (
    PromptAnalysis,
    analyze_prompt,
)
from agentsentinal.core.agents.inspector.analyzers.semantic import (
    SemanticAnalysis,
    analyze_semantic,
)
from agentsentinal.core.agents.inspector.analyzers.tools import (
    ToolsAnalysis,
    analyze_tools,
)
from agentsentinal.intake import extract
from agentsentinal.intake.types import ExtractionResult
from agentsentinal.models import AgentProfile


CONFIDENCE_WARNING_THRESHOLD = 0.6


class InspectorAgent:
    """Reads an agent's configuration and produces an AgentProfile.

    Never invokes the agent. Inspection only — config + static + one LLM call.
    """

    def __init__(self, semantic_enabled: bool = True):
        self.semantic_enabled = semantic_enabled

    async def inspect(
        self,
        agent_id: str,
        agent: Any = None,
        system_prompt: Optional[str] = None,
        tool_definitions: Optional[list[dict[str, Any]]] = None,
        file_path: Optional[str] = None,
        framework_hint: Optional[str] = None,
    ) -> AgentProfile:
        extraction = extract(
            agent=agent,
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
            file_path=file_path,
            framework_hint=framework_hint,
        )

        if extraction.confidence < CONFIDENCE_WARNING_THRESHOLD:
            extraction.warnings.append(
                f"Low extraction confidence ({extraction.confidence:.2f}). "
                "Pass system_prompt and tool_definitions explicitly for a more complete analysis."
            )

        # The four static dimensions run concurrently. Semantic runs after them because
        # it needs prompt-analysis output (constraint_count, ambiguous_phrases) injected
        # into its LLM prompt for context. Do not merge it back into the 5-way gather
        # without first removing that dependency.
        prompt_res, tools_res, memory_res, framework_res = await asyncio.gather(
            asyncio.to_thread(analyze_prompt, extraction.system_prompt),
            asyncio.to_thread(analyze_tools, extraction.tool_definitions),
            asyncio.to_thread(analyze_memory, extraction.source_object, extraction.source_code),
            asyncio.to_thread(analyze_framework, extraction.source_object, extraction.source_code),
            return_exceptions=True,
        )

        static_findings = {}
        if isinstance(prompt_res, PromptAnalysis):
            static_findings = {
                "ambiguous_phrases": prompt_res.ambiguous_phrases,
                "constraint_count": prompt_res.constraint_count,
            }
        try:
            semantic_res: Any = await self._run_semantic(extraction, static_findings)
        except Exception as exc:
            semantic_res = exc

        prompt = _unwrap(prompt_res, PromptAnalysis, extraction, "prompt")
        tools = _unwrap(tools_res, ToolsAnalysis, extraction, "tools")
        memory = _unwrap(memory_res, MemoryAnalysis, extraction, "memory")
        framework = _unwrap(framework_res, FrameworkAnalysis, extraction, "framework")
        semantic = _unwrap(semantic_res, SemanticAnalysis, extraction, "semantic")

        return aggregate(
            agent_id=agent_id,
            extraction=extraction,
            prompt=prompt,
            tools=tools,
            memory=memory,
            framework=framework,
            semantic=semantic,
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
    """Map gather-return into either the analysis object or None, recording errors."""
    if isinstance(value, BaseException):
        extraction.warnings.append(f"{label} analyzer failed: {type(value).__name__}: {value}")
        return None
    if isinstance(value, expected_type):
        return value
    return None
