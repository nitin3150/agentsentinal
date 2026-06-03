import json
from pathlib import Path
from typing import Any

import click

from agentsentinel.core.agents.intake.agent_intake import AgentIntake
from agentsentinel.models.agent import AgentProfile, InspectedAgentProfile


def load_agent(source_path: Path) -> Any:
    intake = AgentIntake()
    profile = intake.extract_profile(str(source_path), AgentProfile())
    if profile.source_object is None:
        raise click.ClickException(
            f"Could not load a live agent from '{source_path}'. "
            "Ensure the file contains a compiled agent (e.g. graph.compile()) "
            "and AGENTSENTINEL_SAFE_MODE is not set to true."
        )
    return profile.source_object


def load_profile(profile_json: Path) -> InspectedAgentProfile:
    from agentsentinel.models import policies as _  # noqa: F401 — triggers model_rebuild
    try:
        data = json.loads(profile_json.read_text())
        return InspectedAgentProfile.model_validate(data)
    except Exception as exc:
        raise click.ClickException(f"Invalid profile JSON at '{profile_json}': {exc}")
