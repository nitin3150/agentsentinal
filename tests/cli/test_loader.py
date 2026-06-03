import json
import pytest
import click
from pathlib import Path
from unittest.mock import patch, MagicMock
from agentsentinel.models.agent import InspectedAgentProfile, RiskLevel


def _make_profile_dict():
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
    ).model_dump()


def test_load_profile_valid_json(tmp_path):
    from agentsentinel.cli.loader import load_profile
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(_make_profile_dict()))
    profile = load_profile(p)
    assert profile.framework == "langgraph"
    assert profile.domain == "test"


def test_load_profile_invalid_json_raises(tmp_path):
    from agentsentinel.cli.loader import load_profile
    p = tmp_path / "bad.json"
    p.write_text("not valid json {{{")
    with pytest.raises(click.ClickException):
        load_profile(p)


def test_load_profile_missing_required_field_raises(tmp_path):
    from agentsentinel.cli.loader import load_profile
    p = tmp_path / "incomplete.json"
    p.write_text(json.dumps({"persona_clarity_score": 999}))  # score out of range
    with pytest.raises(click.ClickException):
        load_profile(p)


def test_load_agent_raises_when_source_object_is_none(tmp_path):
    from agentsentinel.cli.loader import load_agent
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("x = 1")
    mock_profile = MagicMock()
    mock_profile.source_object = None
    with patch("agentsentinel.cli.loader.AgentIntake") as MockIntake:
        MockIntake.return_value.extract_profile.return_value = mock_profile
        with pytest.raises(click.ClickException, match="Could not load"):
            load_agent(agent_file)


def test_load_agent_returns_source_object(tmp_path):
    from agentsentinel.cli.loader import load_agent
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("x = 1")
    mock_agent = MagicMock()
    mock_profile = MagicMock()
    mock_profile.source_object = mock_agent
    with patch("agentsentinel.cli.loader.AgentIntake") as MockIntake:
        MockIntake.return_value.extract_profile.return_value = mock_profile
        result = load_agent(agent_file)
    assert result is mock_agent
