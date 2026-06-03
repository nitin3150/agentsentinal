import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from agentsentinel.models.agent import InspectedAgentProfile, RiskLevel
from agentsentinel.models.prompt import OptimizedResult


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


def _make_optimize_result():
    return OptimizedResult(
        optimized_prompt="You are a hardened test agent.",
        optimized_tool_definitions=[],
        change_log=[],
        policy_violations=[],
    )


def test_inspect_exits_when_llm_model_not_set(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    runner = CliRunner()
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(cli, ["inspect", str(f)])
    assert result.exit_code != 0
    assert "LLM_MODEL" in result.output


def test_inspect_rejects_unknown_compliance(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        result = runner.invoke(cli, ["inspect", str(f), "--compliance", "not_a_standard"])
    assert result.exit_code != 0
    assert "not_a_standard" in result.output


def test_inspect_saves_profile_json(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    out = tmp_path / "profile.json"
    profile = _make_profile()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.inspect.AgentSentinel") as MockS:
            MockS.return_value.inspect.return_value = profile
            result = runner.invoke(cli, ["inspect", str(f), "--output", str(out)])
    assert out.exists(), result.output
    saved = json.loads(out.read_text())
    assert saved["framework"] == "langgraph"


def test_inspect_then_optimize_calls_optimize(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    profile = _make_profile()
    opt_result = _make_optimize_result()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.inspect.AgentSentinel") as MockS:
            MockS.return_value.inspect.return_value = profile
            MockS.return_value.optimize.return_value = opt_result
            result = runner.invoke(cli, ["inspect", str(f), "--then-optimize"])
    assert result.exit_code == 0
    MockS.return_value.optimize.assert_called_once()


def test_inspect_then_stress_test_calls_stress_test(tmp_path):
    from agentsentinel.cli.main import cli
    f = tmp_path / "agent.py"
    f.write_text("x = 1")
    profile = _make_profile()
    mock_agent = MagicMock()
    report = {"summary": {"pass_rate_pct": 90.0, "passed": 9, "total": 10, "failed": 1, "skipped": 0}, "failures": []}
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.inspect.AgentSentinel") as MockS:
            with patch("agentsentinel.cli.commands.inspect.load_agent", return_value=mock_agent):
                MockS.return_value.inspect.return_value = profile
                MockS.return_value.stress_test.return_value = report
                result = runner.invoke(cli, ["inspect", str(f), "--then-stress-test"])
    assert result.exit_code == 0
    MockS.return_value.stress_test.assert_called_once()
