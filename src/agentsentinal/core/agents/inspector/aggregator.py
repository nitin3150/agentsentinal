from agentsentinal.models.agent import InspectedAgentProfile
from typing import Optional

from agentsentinal.core.agents.inspector.analyzers.framework import FrameworkAnalysis
from agentsentinal.core.agents.inspector.analyzers.memory import MemoryAnalysis
from agentsentinal.core.agents.inspector.analyzers.prompt import PromptAnalysis
from agentsentinal.core.agents.inspector.analyzers.semantic import SemanticAnalysis
from agentsentinal.core.agents.inspector.analyzers.tools import ToolsAnalysis
from agentsentinal.intake.types import ExtractionResult
from agentsentinal.models import RiskFlag, RiskLevel


def _overall_risk(flags: list[RiskFlag]) -> RiskLevel:
    if any(f.severity == RiskLevel.HIGH for f in flags):
        return RiskLevel.HIGH
    if any(f.severity == RiskLevel.MEDIUM for f in flags):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def aggregate(
    agent_id: str,
    extraction: ExtractionResult,
    prompt: Optional[PromptAnalysis],
    tools: Optional[ToolsAnalysis],
    memory: Optional[MemoryAnalysis],
    framework: Optional[FrameworkAnalysis],
    semantic: Optional[SemanticAnalysis],
) -> InspectedAgentProfile:
    flags: list[RiskFlag] = []
    ambiguous: list[str] = []

    profile_kwargs: dict = {
        "agent_id": agent_id,
        "system_prompt": extraction.system_prompt,
        "tool_definitions": list(extraction.tool_definitions),
        "framework": extraction.framework,
        "warnings": list(extraction.warnings),
        "extraction_confidence": extraction.confidence,
        "extraction_warnings": list(extraction.warnings),
    }

    if prompt is not None:
        profile_kwargs.update(
            constraint_count=prompt.constraint_count,
            has_examples=prompt.has_examples,
            output_format_defined=prompt.output_format_defined,
            injection_surface=prompt.injection_surface,
        )
        ambiguous.extend(prompt.ambiguous_phrases)
        flags.extend(prompt.risk_flags)

    if tools is not None:
        profile_kwargs.update(
            tool_profiles=tools.tool_profiles,
            tool_count=tools.tool_count,
            avg_tool_quality=tools.avg_tool_quality,
        )
        flags.extend(tools.risk_flags)

    if memory is not None:
        profile_kwargs.update(
            has_memory=memory.has_memory,
            memory_type=memory.memory_type,
            memory_risks=memory.memory_risks,
        )
        flags.extend(memory.risk_flags)

    if framework is not None:
        profile_kwargs.update(
            estimated_depth=framework.estimated_depth,
            has_loops=framework.has_loops,
            has_conditional_edges=framework.has_conditional_edges,
            has_human_in_loop=framework.has_human_in_loop,
        )
        flags.extend(framework.risk_flags)
        if framework.framework and framework.framework != "unknown":
            profile_kwargs["framework"] = framework.framework

    if semantic is not None:
        profile_kwargs.update(
            persona_clarity_score=semantic.persona_clarity_score,
            scope_definition_score=semantic.scope_definition_score,
            tone_consistency_score=semantic.tone_consistency_score,
            estimated_baseline_score=semantic.estimated_baseline_score,
        )
        ambiguous.extend(semantic.ambiguous_phrases)
        flags.extend(semantic.risk_flags)

    # De-dupe ambiguous phrases while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for p in ambiguous:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    profile_kwargs["ambiguous_phrases"] = deduped
    profile_kwargs["risk_flags"] = flags
    profile_kwargs["overall_risk"] = _overall_risk(flags)

    return InspectedAgentProfile(**profile_kwargs)
