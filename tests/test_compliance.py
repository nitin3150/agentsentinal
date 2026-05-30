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


from agentsentinel.core.agents.inspector.analyzers.compliances import _check_rules_static


def test_static_check_finds_forbidden_pattern():
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-001",
        description="Must not store patient data",
        severity="high",
        suggestion="Fix it",
        required_patterns=[],
        forbidden_patterns=["store patient data"],
    )
    violations, ambiguous = _check_rules_static(
        "You are a helpful agent. You store patient data for analysis.",
        [],
        [rule],
    )
    assert len(violations) == 1
    assert violations[0].rule_id == "test-001"
    assert len(ambiguous) == 0


def test_static_check_finds_missing_required_pattern():
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-002",
        description="Must mention encryption",
        severity="medium",
        suggestion="Add encryption",
        required_patterns=["encrypt"],
        forbidden_patterns=[],
    )
    violations, ambiguous = _check_rules_static(
        "You are a helpful assistant.",
        [],
        [rule],
    )
    assert len(violations) == 0
    assert len(ambiguous) == 1
    assert ambiguous[0].id == "test-002"


def test_static_check_passes_compliant_prompt():
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-003",
        description="Must mention encryption",
        severity="medium",
        suggestion="Add encryption",
        required_patterns=["encrypt"],
        forbidden_patterns=["store patient data"],
    )
    violations, ambiguous = _check_rules_static(
        "You must encrypt all data. Never store patient data.",
        [],
        [rule],
    )
    assert violations == []
    assert ambiguous == []


import asyncio
from unittest.mock import AsyncMock, patch
from agentsentinel.core.agents.inspector.analyzers.compliances import _confirm_with_llm, ComplianceRule


def test_confirm_with_llm_returns_violations_for_confirmed():
    rule = ComplianceRule(
        id="hipaa-002",
        description="Must mention data minimization",
        severity="medium",
        suggestion="Add data minimization principle",
        required_patterns=["minimum necessary"],
        forbidden_patterns=[],
    )
    mock_response = '{"confirmed_violations": [{"rule_id": "hipaa-002", "confirmed": true, "description": "No data minimization mention", "suggestion": "Add data minimization principle"}]}'

    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value=mock_response),
    ):
        result = asyncio.run(_confirm_with_llm("You are a helpful agent.", [], [rule], "hipaa"))

    assert len(result) == 1
    assert result[0].rule_id == "hipaa-002"


def test_confirm_with_llm_dismisses_false_positives():
    rule = ComplianceRule(
        id="hipaa-002",
        description="Must mention data minimization",
        severity="medium",
        suggestion="Add data minimization principle",
        required_patterns=["minimum necessary"],
        forbidden_patterns=[],
    )
    mock_response = '{"confirmed_violations": [{"rule_id": "hipaa-002", "confirmed": false, "description": "", "suggestion": ""}]}'

    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value=mock_response),
    ):
        result = asyncio.run(_confirm_with_llm("You are a helpful agent.", [], [rule], "hipaa"))

    assert result == []


def test_confirm_with_llm_empty_ambiguous_skips_llm():
    result = asyncio.run(_confirm_with_llm("anything", [], [], "hipaa"))
    assert result == []


from agentsentinel.core.agents.inspector.analyzers.compliances import analyze_compliance
from agentsentinel.models.policies import ComplianceAnalysis


def test_analyze_compliance_empty_standards_returns_empty():
    result = asyncio.run(analyze_compliance("some prompt", [], []))
    assert isinstance(result, ComplianceAnalysis)
    assert result.standards_checked == []
    assert result.results == {}


def test_analyze_compliance_unknown_raises():
    with pytest.raises(ValueError):
        asyncio.run(analyze_compliance("prompt", [], ["unknownxyz"]))


def test_analyze_compliance_all_expands_and_runs():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value='{"confirmed_violations": []}'),
    ):
        result = asyncio.run(analyze_compliance(
            "You must encrypt all data. Access control required. Audit all actions.",
            [],
            ["All"],
        ))
    assert set(result.standards_checked) == {"hipaa", "soc2", "owasp", "pii"}
    assert set(result.results.keys()) == {"hipaa", "soc2", "owasp", "pii"}


def test_analyze_compliance_detects_forbidden_pattern():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value='{"confirmed_violations": []}'),
    ):
        result = asyncio.run(analyze_compliance(
            "store patient data for analysis",
            [],
            ["hipaa"],
        ))
    hipaa = result.results["hipaa"]
    assert not hipaa.compliant
    assert any(v.rule_id == "hipaa-001" for v in hipaa.violations)


def test_static_check_no_duplicate_when_both_lists_present():
    """Rule with both required and forbidden: if required absent AND forbidden matches,
    rule should be in violations only — not duplicated in ambiguous."""
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-004",
        description="Must mention encryption; must not store PHI",
        severity="high",
        suggestion="Add encryption, remove PHI storage",
        required_patterns=["encrypt"],
        forbidden_patterns=["store phi"],
    )
    violations, ambiguous = _check_rules_static(
        "You store phi for logging.",  # forbidden matches, required absent
        [],
        [rule],
    )
    assert len(violations) == 1
    assert len(ambiguous) == 0  # must NOT be in ambiguous too


def test_static_check_required_found_skips_forbidden():
    """Rule with both lists: if required pattern found, entire rule passes (no forbidden check)."""
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-005",
        description="Must mention encryption",
        severity="high",
        suggestion="Add encryption",
        required_patterns=["encrypt"],
        forbidden_patterns=["store phi"],
    )
    violations, ambiguous = _check_rules_static(
        "You must encrypt data. You store phi.",  # both present: required found → pass
        [],
        [rule],
    )
    assert violations == []
    assert ambiguous == []


def test_confirm_with_llm_returns_empty_on_none_llm_response():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value=None),
    ):
        from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
        rule = ComplianceRule(
            id="hipaa-002", description="test", severity="medium",
            suggestion="fix", required_patterns=["minimum necessary"],
        )
        result = asyncio.run(_confirm_with_llm("prompt", [], [rule], "hipaa"))
    assert result == []


def test_confirm_with_llm_handles_trailing_comma_json():
    """Malformed JSON with trailing comma should be recovered, not discarded."""
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="hipaa-002", description="test", severity="medium",
        suggestion="fix", required_patterns=["minimum necessary"],
    )
    trailing_comma_json = '{"confirmed_violations": [{"rule_id": "hipaa-002", "confirmed": true, "description": "missing", "suggestion": "fix"},]}'
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value=trailing_comma_json),
    ):
        result = asyncio.run(_confirm_with_llm("prompt", [], [rule], "hipaa"))
    assert len(result) == 1
    assert result[0].rule_id == "hipaa-002"


from agentsentinel.models import AgentProfile
from agentsentinel.core.agents.inspector.orchestrator import InspectorAgent


def test_orchestrator_compliance_surfaces_in_profile():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value='{"confirmed_violations": []}'),
    ), patch(
        "agentsentinel.core.agents.inspector.analyzers.semantic.analyze_semantic",
        new=AsyncMock(return_value=None),
    ), patch(
        "agentsentinel.core.agents.inspector.analyzers.policy.analyze_policy",
        new=AsyncMock(return_value=None),
    ):
        profile = AgentProfile(
            system_prompt="store patient data for analysis. You help users.",
        )
        inspector = InspectorAgent(semantic_enabled=False)
        result = asyncio.run(inspector.inspect(profile, compliance=["hipaa"]))

    assert "hipaa" in result.compliance_results
    assert not result.compliance_results["hipaa"].compliant


def test_orchestrator_empty_compliance_leaves_results_empty():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.semantic.analyze_semantic",
        new=AsyncMock(return_value=None),
    ), patch(
        "agentsentinel.core.agents.inspector.analyzers.policy.analyze_policy",
        new=AsyncMock(return_value=None),
    ):
        profile = AgentProfile(system_prompt="You are a helpful assistant.")
        inspector = InspectorAgent(semantic_enabled=False)
        result = asyncio.run(inspector.inspect(profile))

    assert result.compliance_results == {}
