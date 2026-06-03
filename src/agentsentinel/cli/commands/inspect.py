from pathlib import Path

import click

from agentsentinel.cli.loader import load_agent
from agentsentinel.cli.output import (
    console,
    print_change_log_panel,
    print_compliance_panel,
    print_profile_panel,
    print_risk_table,
    print_scores_panel,
    print_stress_failures_table,
    print_stress_summary_panel,
)
from agentsentinel.cli.utils import check_env, parse_compliance
from agentsentinel.sentinel import AgentSentinel


@click.command("inspect")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--compliance", default="", callback=parse_compliance, is_eager=True,
              help="Comma-separated standards: hipaa,soc2,owasp,pii,All")
@click.option("--policies", default="", type=click.Path(), help="Path to policy PDF")
@click.option("--domain", default="", help="Agent domain description")
@click.option("--output", default=None, type=click.Path(), help="Save InspectedAgentProfile JSON to this path")
@click.option("--then-optimize", is_flag=True, help="Run optimize after inspect")
@click.option("--then-stress-test", is_flag=True, help="Run adversarial stress test after inspect")
def inspect_cmd(source, compliance, policies, domain, output, then_optimize, then_stress_test):
    """Inspect an agent source file for risks and compliance violations."""
    check_env()
    sentinel = AgentSentinel()

    with console.status(f"Inspecting {source.name}...", spinner="dots"):
        profile = sentinel.inspect(
            source=source,
            domain=domain,
            policies=policies,
            compliance=compliance,
        )

    print_profile_panel(profile)
    print_risk_table(profile.risk_flags)
    print_scores_panel(profile)
    print_compliance_panel(profile.compliance_results)

    if output:
        Path(output).write_text(profile.model_dump_json(indent=2))
        console.print(f"[dim]Profile saved → {output}[/dim]")

    if then_optimize:
        with console.status("Optimizing prompt...", spinner="dots"):
            opt_result = sentinel.optimize(profile, policies=policies)
        print_change_log_panel(opt_result.change_log)
        if output:
            opt_path = Path(output).with_stem(Path(output).stem + "_optimized")
            opt_path.write_text(opt_result.model_dump_json(indent=2))
            console.print(f"[dim]Optimized result saved → {opt_path}[/dim]")

    if then_stress_test:
        agent = load_agent(source)
        with console.status("Running adversarial stress test...", spinner="dots"):
            report = sentinel.stress_test(agent, profile, policies=policies)
        print_stress_failures_table(report)
        print_stress_summary_panel(report)
