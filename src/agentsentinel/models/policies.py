from pydantic import BaseModel, Field
from agentsentinel.models.agent import RiskFlag, RiskLevel


class PolicyViolation(BaseModel):
    description: str
    policy_reference: str
    severity: RiskLevel = RiskLevel.MEDIUM
    suggestion: str


class PolicyAnalysis(BaseModel):
    compliance_score: int = Field(ge=0, le=100, default=100)
    violations: list[PolicyViolation] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)


class ComplianceViolation(BaseModel):
    rule_id: str
    description: str
    severity: RiskLevel = RiskLevel.MEDIUM
    suggestion: str


class ComplianceStandardResult(BaseModel):
    compliant: bool
    violations: list[ComplianceViolation] = Field(default_factory=list)


class ComplianceAnalysis(BaseModel):
    standards_checked: list[str] = Field(default_factory=list)
    results: dict[str, ComplianceStandardResult] = Field(default_factory=dict)


# Resolve forward references in agent models that depend on compliance models
from agentsentinel.models.agent import InspectedAgentProfile  # noqa: E402

InspectedAgentProfile.model_rebuild()
