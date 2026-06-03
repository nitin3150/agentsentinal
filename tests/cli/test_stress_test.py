import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from agentsentinel.models.agent import InspectedAgentProfile, RiskLevel


def _make_profile_json(tmp_path) -> Path:
    profile = InspectedAgentProfile(
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
    p = tmp_path / "profile.json"
    p.write_text(profile.model_dump_json())
    return p


def _make_report():
    return {
        "summary": {"pass_rate_pct": 85.0, "passed": 17, "total": 20, "failed": 3, "skipped": 0},
        "by_category": {},
        "failures": [],
    }


def test_stress_test_exits_when_llm_model_not_set(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    runner = CliRunner()
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(cli, ["stress-test", str(f)])
    assert result.exit_code != 0
    assert "LLM_MODEL" in result.output


def test_stress_test_with_saved_profile(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    pf = _make_profile_json(tmp_path)
    mock_agent = MagicMock()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.stress_test.load_agent", return_value=mock_agent):
            with patch("agentsentinel.cli.commands.stress_test.AgentSentinel") as MockS:
                MockS.return_value.stress_test.return_value = _make_report()
                result = runner.invoke(cli, ["stress-test", str(f), "--profile", str(pf)])
    assert result.exit_code == 0, result.output
    MockS.return_value.stress_test.assert_called_once()


def test_stress_test_without_profile_runs_inspect_first(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    mock_agent = MagicMock()
    mock_profile = InspectedAgentProfile(
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
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.stress_test.load_agent", return_value=mock_agent):
            with patch("agentsentinel.cli.commands.stress_test.AgentSentinel") as MockS:
                MockS.return_value.inspect.return_value = mock_profile
                MockS.return_value.stress_test.return_value = _make_report()
                result = runner.invoke(cli, ["stress-test", str(f)])
    assert result.exit_code == 0, result.output
    MockS.return_value.inspect.assert_called_once()
    MockS.return_value.stress_test.assert_called_once()
