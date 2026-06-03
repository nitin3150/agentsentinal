import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from agentsentinel.models.agent import InspectedAgentProfile, RiskLevel


def _make_profile():
    return InspectedAgentProfile(
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


def _make_audit_result():
    return {
        "profile": _make_profile(),
        "report": {
            "summary": {"pass_rate_pct": 90.0, "passed": 9, "total": 10, "failed": 1, "skipped": 0},
            "by_category": {},
            "failures": [],
        },
        "iteration": 1,
    }


def test_audit_exits_when_llm_model_not_set(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    runner = CliRunner()
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(cli, ["audit", str(f)])
    assert result.exit_code != 0
    assert "LLM_MODEL" in result.output


def test_audit_rejects_unknown_compliance(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        result = runner.invoke(cli, ["audit", str(f), "--compliance", "bad_standard"])
    assert result.exit_code != 0
    assert "bad_standard" in result.output


def test_audit_calls_sentinel_audit(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    mock_agent = MagicMock()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.audit.load_agent", return_value=mock_agent):
            with patch("agentsentinel.cli.commands.audit.AgentSentinel") as MockS:
                MockS.return_value.audit.return_value = _make_audit_result()
                result = runner.invoke(cli, ["audit", str(f), "--threshold", "85", "--max-iterations", "2"])
    assert result.exit_code == 0, result.output
    MockS.return_value.audit.assert_called_once_with(
        mock_agent,
        compliance=[],
        policies="",
        pass_threshold=85.0,
        max_iterations=2,
    )


def test_audit_passes_compliance_to_sentinel(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    mock_agent = MagicMock()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.audit.load_agent", return_value=mock_agent):
            with patch("agentsentinel.cli.commands.audit.AgentSentinel") as MockS:
                MockS.return_value.audit.return_value = _make_audit_result()
                result = runner.invoke(cli, ["audit", str(f), "--compliance", "hipaa,owasp"])
    assert result.exit_code == 0, result.output
    call_kwargs = MockS.return_value.audit.call_args.kwargs
    assert call_kwargs["compliance"] == ["hipaa", "owasp"]
