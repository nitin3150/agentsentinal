import pytest
from io import StringIO
from rich.console import Console
from agentsentinel.models.agent import (
    InspectedAgentProfile, RiskFlag, RiskCategory, RiskLevel
)
from agentsentinel.models.prompt import OptimizedResult, ChangeLogEntry


def _make_profile(**kwargs):
    defaults = dict(
        framework="langgraph",
        domain="test",
        system_prompt="You are a test agent.",
        overall_risk=RiskLevel.LOW,
        persona_clarity_score=8,
        scope_definition_score=7,
        tone_consistency_score=9,
        injection_surface=RiskLevel.LOW,
        estimated_baseline_score=85,
        policy_compliance_score=100,
    )
    return InspectedAgentProfile(**{**defaults, **kwargs})


def _captured_console():
    buf = StringIO()
    return Console(file=buf, width=120, highlight=False), buf


def test_severity_color_returns_red_for_high():
    from agentsentinel.cli.output import severity_color
    assert severity_color("high") == "red"


def test_severity_color_returns_yellow_for_medium():
    from agentsentinel.cli.output import severity_color
    assert severity_color("medium") == "yellow"


def test_severity_color_returns_green_for_low():
    from agentsentinel.cli.output import severity_color
    assert severity_color("low") == "green"


def test_print_profile_panel_no_crash(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    out.print_profile_panel(_make_profile())
    assert "langgraph" in buf.getvalue()


def test_print_risk_table_empty_no_crash(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    out.print_risk_table([])  # must not raise


def test_print_risk_table_shows_category(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    flag = RiskFlag(
        category=RiskCategory.INJECTION_VULNERABLE,
        description="desc",
        location="system_prompt",
        severity=RiskLevel.HIGH,
        suggestion="Add guardrails",
    )
    out.print_risk_table([flag])
    assert "injection_vulnerable" in buf.getvalue()


def test_print_scores_panel_no_crash(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    out.print_scores_panel(_make_profile())
    assert "85" in buf.getvalue()


def test_print_compliance_panel_empty_no_crash(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    out.print_compliance_panel({})  # must not raise


def test_print_change_log_panel_no_crash(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    entry = ChangeLogEntry(field="system_prompt", before="old", after="new", reason="injection risk")
    out.print_change_log_panel([entry])
    assert "system_prompt" in buf.getvalue()


def test_print_stress_summary_panel_no_crash(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    report = {"summary": {"pass_rate_pct": 90.0, "passed": 9, "total": 10, "failed": 1, "skipped": 0}}
    out.print_stress_summary_panel(report)
    assert "90" in buf.getvalue()


def test_print_stress_failures_table_no_failures(monkeypatch):
    import agentsentinel.cli.output as out
    c, buf = _captured_console()
    monkeypatch.setattr(out, "console", c)
    out.print_stress_failures_table({"failures": [], "summary": {}})
    assert "No failures" in buf.getvalue()
