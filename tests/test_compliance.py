# tests/test_compliance.py
import pytest
from agentsentinel.models.policies import (
    ComplianceViolation,
    ComplianceStandardResult,
    ComplianceAnalysis,
)
from agentsentinel.models.agent import RiskCategory, RiskLevel, InspectedAgentProfile


def test_compliance_violation_fields():
    v = ComplianceViolation(
        rule_id="hipaa-001",
        description="PHI stored without consent",
        severity=RiskLevel.HIGH,
        suggestion="Add consent clause",
    )
    assert v.rule_id == "hipaa-001"
    assert v.severity == RiskLevel.HIGH


def test_compliance_standard_result_compliant_default():
    r = ComplianceStandardResult(compliant=True)
    assert r.violations == []


def test_compliance_standard_result_non_compliant():
    v = ComplianceViolation(
        rule_id="hipaa-001",
        description="PHI stored without consent",
        severity=RiskLevel.HIGH,
        suggestion="Add consent clause",
    )
    r = ComplianceStandardResult(compliant=False, violations=[v])
    assert len(r.violations) == 1


def test_compliance_analysis_defaults():
    a = ComplianceAnalysis()
    assert a.standards_checked == []
    assert a.results == {}


def test_riskcat_has_compliance_violation():
    assert RiskCategory.COMPLIANCE_VIOLATION == "compliance_violation"


def test_inspected_profile_has_compliance_results():
    p = InspectedAgentProfile(agent_id="test")
    assert p.compliance_results == {}


def test_inspected_profile_compliance_results_populated():
    from agentsentinel.models.policies import ComplianceStandardResult
    result = ComplianceStandardResult(compliant=True)
    p = InspectedAgentProfile(agent_id="test", compliance_results={"hipaa": result})
    assert p.compliance_results["hipaa"].compliant is True
