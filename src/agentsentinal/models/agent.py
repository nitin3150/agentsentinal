from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
    # Identity
    agent_id: str
    framework: str = "unknown"

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

    # Extraction metadata
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    extraction_warnings: list[str] = Field(default_factory=list)
