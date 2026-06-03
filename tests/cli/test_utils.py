import os
import pytest
import click
from unittest.mock import patch


def test_check_env_raises_when_no_llm_model():
    from agentsentinel.cli.utils import check_env
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(click.ClickException, match="LLM_MODEL"):
            check_env()


def test_check_env_passes_with_llm_model():
    from agentsentinel.cli.utils import check_env
    with patch.dict(os.environ, {"LLM_MODEL": "groq/llama-3.3-70b-versatile"}, clear=True):
        check_env()  # must not raise


def test_check_env_passes_with_groq_model():
    from agentsentinel.cli.utils import check_env
    with patch.dict(os.environ, {"GROQ_MODEL": "llama3"}, clear=True):
        check_env()  # must not raise


def test_parse_compliance_empty_string_returns_empty_list():
    from agentsentinel.cli.utils import parse_compliance
    assert parse_compliance(None, None, "") == []


def test_parse_compliance_valid_values():
    from agentsentinel.cli.utils import parse_compliance
    result = parse_compliance(None, None, "hipaa,owasp")
    assert result == ["hipaa", "owasp"]


def test_parse_compliance_all_is_valid():
    from agentsentinel.cli.utils import parse_compliance
    result = parse_compliance(None, None, "All")
    assert result == ["All"]


def test_parse_compliance_invalid_value_raises():
    from agentsentinel.cli.utils import parse_compliance
    with pytest.raises(click.BadParameter, match="unknown_standard"):
        parse_compliance(None, None, "hipaa,unknown_standard")


def test_parse_compliance_strips_whitespace():
    from agentsentinel.cli.utils import parse_compliance
    result = parse_compliance(None, None, " hipaa , owasp ")
    assert result == ["hipaa", "owasp"]
