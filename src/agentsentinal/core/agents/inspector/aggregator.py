from agentsentinal.models.agent import InspectedAgentProfile
from typing import Optional

from agentsentinal.core.agents.inspector.analyzers.framework import FrameworkAnalysis
from agentsentinal.core.agents.inspector.analyzers.memory import MemoryAnalysis
from agentsentinal.core.agents.inspector.analyzers.policy import PolicyAnalysis
from agentsentinal.core.agents.inspector.analyzers.prompt import PromptAnalysis
from agentsentinal.core.agents.inspector.analyzers.semantic import SemanticAnalysis
from agentsentinal.core.agents.inspector.analyzers.tools import ToolsAnalysis
from agentsentinal.models.intake import ExtractionResult
from agentsentinal.models import RiskFlag, RiskLevel
import logging

logger = logging.getLogger(__name__)

def _fmt_results(k: dict) -> str:
    risk_flags: list[RiskFlag] = k.get("risk_flags") or []
    flag_lines = "\n".join(
        f"    [{f.severity.upper()}] {f.category} — {f.description}"
        for f in risk_flags
    ) or "    (none)"
    ambiguous = k.get("ambiguous_phrases") or []
    warnings  = k.get("warnings") or []
    tool_profiles = k.get("tool_profiles") or []
    tool_lines = "\n".join(
        f"    • {t.name} (quality: {t.quality_score}/10)"
        for t in tool_profiles
    ) or "    (none)"
    return (
        f"\n  agent_id            : {k.get('agent_id', '?')}"
        f"\n  framework           : {k.get('framework', '?')}"
        f"\n  confidence          : {k.get('extraction_confidence', 0):.2f}"
        f"\n  overall_risk        : {k.get('overall_risk', '?')}"
        f"\n  --- Scores ---"
        f"\n  persona_clarity     : {k.get('persona_clarity_score', '?')}/10"
        f"\n  scope_definition    : {k.get('scope_definition_score', '?')}/10"
        f"\n  tone_consistency    : {k.get('tone_consistency_score', '?')}/10"
        f"\n  baseline_score      : {k.get('estimated_baseline_score', '?')}/100"
        f"\n  --- Prompt ---"
        f"\n  constraints         : {k.get('constraint_count', 0)}"
        f"\n  has_examples        : {k.get('has_examples', False)}"
        f"\n  output_format       : {k.get('output_format_defined', False)}"
        f"\n  injection_surface   : {k.get('injection_surface', '?')}"
        f"\n  ambiguous_phrases   : {ambiguous or '(none)'}"
        f"\n  --- Tools ({k.get('tool_count', 0)}) ---\n{tool_lines}"
        f"\n  avg_tool_quality    : {k.get('avg_tool_quality', 0):.1f}/10"
        f"\n  --- Memory ---"
        f"\n  has_memory          : {k.get('has_memory', False)}"
        f"\n  memory_type         : {k.get('memory_type') or 'none'}"
        f"\n  --- Policy ---"
        f"\n  policy_compliance   : {k.get('policy_compliance_score', 100)}/100"
        f"\n  policy_violations   : {k.get('policy_violations') or '(none)'}"
        f"\n  --- Risk Flags ({len(risk_flags)}) ---\n{flag_lines}"
        f"\n  warnings            : {warnings or '(none)'}"
    )


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
    policy: Optional[PolicyAnalysis] = None,
) -> InspectedAgentProfile:
    flags: list[RiskFlag] = []
    ambiguous: list[str] = []

    profile_kwargs: dict = {
        "agent_id": agent_id,
        "system_prompt": extraction.system_prompt,
        "tool_definitions": list(extraction.tool_definitions),
        "framework": extraction.framework,
        "source_object": extraction.source_object,
        "warnings": list(extraction.warnings),
        "extraction_confidence": extraction.confidence,
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

    if policy is not None:
        profile_kwargs.update(
            policy_compliance_score=policy.compliance_score,
            policy_violations=[v.description for v in policy.violations],
        )
        flags.extend(policy.risk_flags)

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

    logger.info("Inspection Results: %s", _fmt_results(profile_kwargs))
    return InspectedAgentProfile(**profile_kwargs)
