import json
import os
from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner
from agentsentinel.models.agent import InspectedAgentProfile, RiskLevel
from agentsentinel.models.prompt import OptimizedResult, ChangeLogEntry


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


def _make_opt_result():
    return OptimizedResult(
        optimized_prompt="Hardened prompt.",
        optimized_tool_definitions=[],
        change_log=[ChangeLogEntry(field="system_prompt", before="old", after="new", reason="injection risk")],
        policy_violations=[],
    )


def test_optimize_exits_when_llm_model_not_set(tmp_path):
    from agentsentinel.cli.main import cli
    pf = _make_profile_json(tmp_path)
    runner = CliRunner()
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(cli, ["optimize", str(pf)])
    assert result.exit_code != 0
    assert "LLM_MODEL" in result.output


def test_optimize_invalid_profile_json_exits(tmp_path):
    from agentsentinel.cli.main import cli
    bad = tmp_path / "bad.json"
    bad.write_text("{not json}")
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        result = runner.invoke(cli, ["optimize", str(bad)])
    assert result.exit_code != 0


def test_optimize_calls_sentinel_optimize(tmp_path):
    from agentsentinel.cli.main import cli
    pf = _make_profile_json(tmp_path)
    opt_result = _make_opt_result()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.optimize.AgentSentinel") as MockS:
            MockS.return_value.optimize.return_value = opt_result
            result = runner.invoke(cli, ["optimize", str(pf)])
    assert result.exit_code == 0, result.output
    MockS.return_value.optimize.assert_called_once()


def test_optimize_saves_output_json(tmp_path):
    from agentsentinel.cli.main import cli
    pf = _make_profile_json(tmp_path)
    out = tmp_path / "optimized.json"
    opt_result = _make_opt_result()
    runner = CliRunner()
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama3", "LLM_API_KEY": "k"}):
        with patch("agentsentinel.cli.commands.optimize.AgentSentinel") as MockS:
            MockS.return_value.optimize.return_value = opt_result
            result = runner.invoke(cli, ["optimize", str(pf), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["optimized_prompt"] == "Hardened prompt."
