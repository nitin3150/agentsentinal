from pathlib import Path

import click

from agentsentinel.cli.loader import load_profile
from agentsentinel.cli.output import console, print_change_log_panel
from agentsentinel.cli.utils import check_env
from agentsentinel.sentinel import AgentSentinel


@click.command("optimize")
@click.argument("profile_json", type=click.Path(exists=True, path_type=Path))
@click.option("--policies", default="", type=click.Path(), help="Path to policy PDF")
@click.option("--output", default=None, type=click.Path(), help="Save OptimizedResult JSON to this path")
def optimize_cmd(profile_json, policies, output):
    """Rewrite agent system prompt to fix all flagged risks."""
    check_env()
    profile = load_profile(profile_json)
    sentinel = AgentSentinel()

    with console.status("Optimizing prompt...", spinner="dots"):
        result = sentinel.optimize(profile, policies=policies)

    print_change_log_panel(result.change_log)

    if result.policy_violations:
        console.print("[bold red]Policy violations remain in optimized prompt:[/bold red]")
        for v in result.policy_violations:
            console.print(f"  • [red]{v}[/red]")

    if output:
        Path(output).write_text(result.model_dump_json(indent=2))
        console.print(f"[dim]Optimized result saved → {output}[/dim]")
