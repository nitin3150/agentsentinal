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


from agentsentinel.core.agents.inspector.analyzers.compliances import (
    resolve_standards,
    load_rules,
    SUPPORTED_STANDARDS,
)


# --- resolve_standards ---

def test_resolve_all_expands_to_all_supported():
    result = resolve_standards(["All"])
    assert set(result) == SUPPORTED_STANDARDS


def test_resolve_all_case_insensitive():
    result = resolve_standards(["all"])
    assert set(result) == SUPPORTED_STANDARDS


def test_resolve_single_valid():
    assert resolve_standards(["hipaa"]) == ["hipaa"]


def test_resolve_multiple_valid():
    result = resolve_standards(["hipaa", "soc2"])
    assert set(result) == {"hipaa", "soc2"}


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported compliance standard: 'completelyunknown'"):
        resolve_standards(["completelyunknown"])


def test_resolve_typo_warns_and_skips(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        result = resolve_standards(["hiipa"])
    assert result == []
    assert "hiipa" in caplog.text


def test_resolve_empty_list_returns_empty():
    assert resolve_standards([]) == []


# --- load_rules ---

def test_load_rules_hipaa_returns_rules():
    rules = load_rules("hipaa")
    assert len(rules) > 0
    assert all(hasattr(r, "id") for r in rules)


def test_load_rules_unknown_raises():
    with pytest.raises(ValueError, match="No rule file found for standard: 'fake'"):
        load_rules("fake")


def test_load_rules_rule_has_required_fields():
    rules = load_rules("hipaa")
    r = rules[0]
    assert r.id
    assert r.description
    assert r.severity in ("low", "medium", "high")
    assert isinstance(r.required_patterns, list)
    assert isinstance(r.forbidden_patterns, list)
