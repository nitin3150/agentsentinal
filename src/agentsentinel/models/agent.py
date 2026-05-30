from __future__ import annotations

from typing import List
from typing import Any
from enum import Enum
from typing import Optional, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentsentinel.models.policies import ComplianceStandardResult


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskCategory(str, Enum):
    INJECTION_VULNERABLE = "injection_vulnerable"
    CONSTRAINT_MISSING = "constraint_missing"
    AMBIGUOUS_INSTRUCTIONS = "ambiguous_instructions"
    SCOPE_OVERFLOW = "scope_overflow"
    TOOL_QUALITY_LOW = "tool_quality_low"
    PERSONA_DRIFT = "persona_drift"
    MEMORY_RISK = "memory_risk"
    HALLUCINATION_PRONE = "hallucination_prone"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RiskFlag(BaseModel):
    category: RiskCategory
    description: str
    location: str
    severity: RiskLevel
    suggestion: str


class ToolProfile(BaseModel):
    name: str
    has_description: bool
    has_typed_params: bool
    has_error_handling: bool
    param_count: int = 0
    missing_fields: list[str] = Field(default_factory=list)
    quality_score: int = Field(ge=1, le=10)


class AgentProfile(BaseModel):
    domain: str = ""
    system_prompt: str = ""
    tool_definitions: List[dict] = Field(default_factory=list)
    framework: Any = "unknown"
    warnings: list[str] = Field(default_factory=list)
    source_object: Optional[Any] = None
    source: Optional[str | Path] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_log_str(self) -> str:
        tools = self.tool_definitions
        tool_lines = "\n".join(
            f"    • {t.get('name', '?')} — {t.get('description', 'no description')}"
            for t in tools
        ) or "    (none)"
        warnings = ", ".join(self.warnings) if self.warnings else "none"
        src = type(self.source_object).__name__ if self.source_object else "none"
        return (
            f"\n  domain        : {self.domain or '(unset)'}"
            f"\n  framework     : {self.framework}"
            f"\n  system_prompt : {self.system_prompt or '(unset)'}"
            f"\n  tools ({len(tools)})     :\n{tool_lines}"
            f"\n  warnings      : {warnings}"
            f"\n  source        : {src}"
        )


class InspectedAgentProfile(AgentProfile):
    """Extended profile produced by InspectorAgent after full static + semantic analysis."""

    # Identity
    agent_id: str = ""

    # Prompt quality
    persona_clarity_score: int = Field(ge=1, le=10, default=5)
    scope_definition_score: int = Field(ge=1, le=10, default=5)
    tone_consistency_score: int = Field(ge=1, le=10, default=5)
    constraint_count: int = 0
    has_examples: bool = False
    output_format_defined: bool = False

    # Risk
    ambiguous_phrases: list[str] = Field(default_factory=list)
    injection_surface: RiskLevel = RiskLevel.MEDIUM
    overall_risk: RiskLevel = RiskLevel.MEDIUM
    risk_flags: list[RiskFlag] = Field(default_factory=list)

    # Tools
    tool_profiles: list[ToolProfile] = Field(default_factory=list)
    tool_count: int = 0
    avg_tool_quality: float = 0.0

    # Memory
    has_memory: bool = False
    memory_type: Optional[str] = None
    memory_risks: list[str] = Field(default_factory=list)

    # Framework structure
    estimated_depth: int = 0
    has_loops: bool = False
    has_conditional_edges: bool = False
    has_human_in_loop: bool = False

    # Estimation
    estimated_baseline_score: int = Field(ge=0, le=100, default=50)

    # Policy compliance
    policy_compliance_score: int = Field(ge=0, le=100, default=100)
    policy_violations: list[str] = Field(default_factory=list)

    # Extraction metadata
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Compliance
    compliance_results: dict[str, "ComplianceStandardResult"] = Field(default_factory=dict)