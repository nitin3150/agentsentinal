from pathlib import Path

import click

from agentsentinel.cli.loader import load_agent, load_profile
from agentsentinel.cli.output import console, print_stress_failures_table, print_stress_summary_panel
from agentsentinel.cli.utils import check_env
from agentsentinel.sentinel import AgentSentinel


@click.command("stress-test")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "profile_json", default=None, type=click.Path(exists=True, path_type=Path),
              help="Path to saved InspectedAgentProfile JSON (re-inspects if omitted)")
@click.option("--policies", default="", type=click.Path(), help="Path to policy PDF")
@click.option("--output-dir", default=None, type=click.Path(), help="Directory to save results JSON files")
def stress_test_cmd(source, profile_json, policies, output_dir):
    """Run adversarial prompts against a live agent."""
    check_env()
    agent = load_agent(source)
    sentinel = AgentSentinel()

    if profile_json:
        profile = load_profile(profile_json)
    else:
        with console.status(f"Inspecting {source.name}...", spinner="dots"):
            profile = sentinel.inspect(source=source, policies=policies)

    with console.status("Running adversarial stress test...", spinner="dots"):
        report = sentinel.stress_test(agent, profile, policies=policies, output_dir=output_dir)

    print_stress_failures_table(report)
    print_stress_summary_panel(report)

    if output_dir:
        console.print(f"[dim]Results saved → {output_dir}[/dim]")
